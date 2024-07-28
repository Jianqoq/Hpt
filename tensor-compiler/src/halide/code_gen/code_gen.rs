#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use std::{
    alloc::Layout,
    collections::VecDeque,
    ffi::{ c_void, CStr },
    mem::MaybeUninit,
    sync::Arc,
};

use std::collections::{ HashMap, HashSet };
use itertools::izip;
use llvm_sys::{
    execution_engine::{
        LLVMCreateGenericValueOfInt,
        LLVMCreateGenericValueOfPointer,
        LLVMCreateJITCompilerForModule,
        LLVMGetExecutionEngineTargetData,
        LLVMLinkInMCJIT,
        LLVMOpaqueGenericValue,
        LLVMRunFunction,
    },
    target::{
        LLVM_InitializeNativeAsmParser,
        LLVM_InitializeNativeAsmPrinter,
        LLVM_InitializeNativeDisassembler,
        LLVM_InitializeNativeTarget,
    },
    LLVMIntPredicate,
    LLVMRealPredicate,
};
use tensor_allocator::CACHE;
use tensor_llvm::{
    builder::builder::Builder,
    context::context::Context,
    engine::engine::ExecutionEngine,
    types::values::{ dtype_to_llvm, BasicValue, FunctionValue, StructValue },
    utils::{ declare_printf, insert_printf, to_c_str },
};
use tensor_types::{
    convertion::Convertor,
    dtype::Dtype,
    type_promote::{ BitWiseOut, FloatOut, NormalOut },
};

use crate::{
    edges::Edges,
    halide::{
        alloca_stmt::AllocaStmt,
        code_gen::{
            pt_llvm::primitive_ty_to_llvm,
            type_utils::{ build_cast, general_types_to_primitive_type },
        },
        executable::executable::Executable,
        exprs::{ Int, Load },
        module::{ Function, Module },
        prime_expr::PrimeExpr,
        primitive_type::{ Array, PrimitiveType, Ptr },
        printer::IRPrinter,
        tensor_load::TensorLoad,
        traits::CodeGenVisitor,
    },
    te::hstrides::HStrides,
    tensor::{ Tensor, _Tensor },
    to_prim_expr::ToPrimeExpr,
};

use super::scope::ScopeStack;

pub struct CodeGen {
    ctx: Context,
    module: tensor_llvm::module::module::Module,
    builder: Builder,
    fns: HashMap<Arc<String>, FunctionValue>,
    id_fns: HashMap<usize, FunctionValue>,
    fns_id: HashMap<FunctionValue, usize>,
    bindings: HashMap<usize, ScopeStack>,
    current_fn: usize,
    halide_module: Module,
    tensor_type: StructValue,
    ee: ExecutionEngine,
}

impl CodeGen {
    pub fn new(ctx: Context, module: &Module, opt_lvl: u32) -> Self {
        let _module = tensor_llvm::module::module::Module::new(&module.name, &ctx);
        let global_tensor = _module.add_global_struct(&ctx.tensor_type().into(), 0, "Tensor");
        let builder = Builder::new(&ctx);

        unsafe {
            LLVMLinkInMCJIT();
            let code = LLVM_InitializeNativeTarget();
            if code == 1 {
                panic!("Failed to initialize native target");
            }
            let code = LLVM_InitializeNativeAsmPrinter();
            if code == 1 {
                panic!("Failed to initialize native asm printer");
            }
            let code = LLVM_InitializeNativeAsmParser();
            if code == 1 {
                panic!("Failed to initialize native asm parser");
            }
            let node = LLVM_InitializeNativeDisassembler();
            if node == 1 {
                panic!("Failed to initialize native asm printer");
            }
        }
        let mut execution_engine = MaybeUninit::uninit();
        let mut err_string = MaybeUninit::uninit();
        let code = unsafe {
            LLVMCreateJITCompilerForModule(
                execution_engine.as_mut_ptr(),
                _module.inner(),
                opt_lvl,
                err_string.as_mut_ptr()
            )
        };
        let ee = unsafe {
            if code == 1 {
                let msg = CStr::from_ptr(err_string.assume_init());
                panic!("Failed to create JIT compiler: {:?}", msg.to_string_lossy().into_owned());
            }
            let engine = ExecutionEngine::new(execution_engine.assume_init());
            declare_printf(&ctx, &_module);
            engine
        };
        let mut fns = HashMap::new();
        let mut fns_id = HashMap::new();
        let mut id_fns = HashMap::new();
        let mut cnt = 0;
        for fn_meta in module.fns.values() {
            let func = &fn_meta.function;
            let fn_val = _module.add_function(
                func.ty.to_llvm_func_type(&ctx, global_tensor),
                &func.name
            );
            fns.insert(func.name.clone(), fn_val.clone());
            fns_id.insert(fn_val.clone(), cnt);
            id_fns.insert(cnt, fn_val);
            cnt += 1;
        }
        CodeGen {
            ctx,
            module: _module,
            builder,
            fns,
            bindings: HashMap::new(),
            id_fns,
            current_fn: 0,
            fns_id,
            halide_module: module.clone(),
            tensor_type: global_tensor,
            ee,
        }
    }

    pub fn compile(&mut self) {
        for i in self.halide_module.fns.clone().values() {
            self.visit_function(&i.function);
        }
        self.module.print_to_file("module.ll").expect("failed to print to file");
    }

    pub fn into_executable(self, vars_map: HashMap<Arc<String>, i64>) -> Executable {
        let mut edges = Edges::new();
        let mut funcs = HashSet::new();
        let fns = self.halide_module.fns
            .values()
            .map(|fn_meta| {
                (
                    &fn_meta.function,
                    HashSet::from_iter(fn_meta.inputs_order.iter()),
                    HashSet::from_iter(fn_meta.outputs_order.iter()),
                    &fn_meta.outs_shape,
                )
            })
            .collect::<
                Vec<(&Function, HashSet<&usize>, HashSet<&usize>, &Vec<Arc<Vec<PrimeExpr>>>)>
            >();
        for (func, inputs, outputs, _) in fns.iter() {
            let input_funcs = inputs
                .iter()
                .map(|x|
                    fns
                        .iter()
                        .filter(|(_, _, outputs, _)| { outputs.contains(x) })
                        .map(|(inp_fn, _, _, _)| inp_fn)
                        .collect::<Vec<&&Function>>()
                )
                .flatten()
                .collect::<HashSet<&&Function>>();
            edges.insert(func, input_funcs);
            funcs.insert(func);
        }
        let order = topo(&edges, &funcs)
            .expect("cycle detected")
            .iter()
            .map(|&x| x.name.clone())
            .collect::<Vec<Arc<String>>>();
        let strides_cal_order = order
            .iter()
            .map(|x| self.halide_module.fns.get(x).unwrap().strides_cal.clone())
            .collect::<Vec<Arc<dyn Fn(&HashMap<Arc<String>, i64>) -> Vec<HStrides>>>>();
        let sorted_outs_shape = order
            .iter()
            .map(|x| self.halide_module.fns.get(x).unwrap().outs_shape.clone())
            .collect::<Vec<Vec<Arc<Vec<PrimeExpr>>>>>();
        let sorted_vars = order
            .iter()
            .map(|x| self.halide_module.fns.get(x).unwrap().vars_order.clone())
            .collect::<Vec<Vec<Arc<String>>>>();
        let sorted_inputs = order
            .iter()
            .map(|x| self.halide_module.fns.get(x).unwrap().inputs_order.clone())
            .collect::<Vec<Vec<usize>>>();
        let sorted_outputs = order
            .iter()
            .map(|x| self.halide_module.fns.get(x).unwrap().outputs_order.clone())
            .collect::<Vec<Vec<usize>>>();
        let e = Executable {
            ctx: self.ctx,
            module: self.module,
            builder: self.builder,
            ee: self.ee,
            sorted_fns: order,
            sorted_strides_cal: strides_cal_order,
            sorted_touse_vars: sorted_vars,
            sorted_out_shapes: sorted_outs_shape,
            touse_vars_layout: vec![],
            istrides_vec_layout: vec![],
            ostrides_vec_layout: vec![],
            strides_layout: vec![],
            ostrides_layouts: vec![],
            istrides: vec![],
            ostrides: vec![],
            vars: vec![],
            vars_map,
            sorted_inputs,
            sorted_outputs,
            sorted_ic_layouts: vec![],
            sorted_oc_layouts: vec![],
            sorted_inputs_container: vec![],
            sorted_outputs_container: vec![],
        };
        e.prepare()
    }

    pub fn printf(&self, fmt: &str, args: &[BasicValue]) {
        insert_printf(&self.ctx, &self.builder, &self.module, fmt, args)
    }
}

fn topo<'a>(
    edges: &'a Edges<&&Function>,
    nodes: &'a HashSet<&&Function>
) -> Option<VecDeque<&'a &'a Function>> {
    let mut in_degree = HashMap::new();
    let mut queue = VecDeque::new();
    let mut order = VecDeque::new();
    let edges = edges.invert();
    // calculate in degree
    for &func in nodes.iter() {
        in_degree.entry(func).or_insert(0);
        let edges = edges.get(func);
        if let Some(edges) = edges {
            for &target in edges {
                *in_degree.entry(target).or_insert(0) += 1;
            }
        }
    }

    // push nodes with in degree 0 to queue
    for (&node_id, &degree) in &in_degree {
        if degree == 0 {
            queue.push_back(node_id);
        }
    }

    // topological sort
    while let Some(node_id) = queue.pop_front() {
        order.push_back(node_id);
        if let Some(edges) = edges.get(node_id) {
            for &target in edges {
                let degree = in_degree.get_mut(&target).unwrap();
                *degree -= 1;
                if *degree == 0 {
                    queue.push_back(target);
                }
            }
        }
    }

    // check if there is a cycle
    if order.len() == nodes.len() {
        Some(order)
    } else {
        None // cycle detected
    }
}

impl CodeGenVisitor for CodeGen {
    fn visit_return(&mut self, return_: &crate::halide::return_stmt::ReturnStmt) {
        let return_block = self.ctx.append_basic_block(&self.id_fns[&self.current_fn], "return");
        self.builder.build_unconditional_branch(return_block);
        self.builder.position_at_end(return_block);
        self.builder.build_return(None)
    }

    fn visit_reduce(&mut self, reduce: &crate::halide::exprs::Reduce) -> BasicValue {
        todo!()
    }

    fn visit_variable(&mut self, var: &crate::halide::variable::Variable) -> BasicValue {
        self.bindings[&self.current_fn]
            .find_variable(&var.name)
            .expect(&format!("variable {} not found", var.name))
    }

    fn visit_str(&mut self, string: &crate::halide::exprs::Str) -> BasicValue {
        todo!()
    }

    fn visit_assign(&mut self, assign: &crate::halide::assign_stmt::AssignStmt) {
        todo!()
    }

    fn visit_cast(&mut self, cast: &crate::halide::exprs::Cast) -> BasicValue {
        let expr = self.visit_expr(cast.expr());
        let from = self.bindings[&self.current_fn].find_type(&expr).unwrap();
        let to = cast.dtype();
        todo!()
    }

    fn visit_bitcast(&mut self, bit_cast: &crate::halide::exprs::BitCast) -> BasicValue {
        let expr = self.visit_expr(bit_cast.expr());
        let from = self.bindings[&self.current_fn].find_type(&expr).unwrap();
        let to = bit_cast.dtype();
        let val = self.builder.build_bitcast(
            expr,
            primitive_ty_to_llvm(&self.ctx, to, self.tensor_type),
            "bit_cast"
        );
        self.bindings.get_mut(&self.current_fn).expect("fn not find").insert_type(val, to.clone());
        val
    }

    fn visit_add(&mut self, add: &crate::halide::exprs::Add) -> BasicValue {
        let lhs = self.visit_expr(add.e1());
        let rhs = self.visit_expr(add.e2());
        let lhs_type = self.bindings[&self.current_fn].find_type(&lhs).unwrap().dtype();
        let rhs_type = self.bindings[&self.current_fn].find_type(&rhs).unwrap().dtype();
        let res_type = lhs_type._add(rhs_type);
        let casted_lhs = build_cast(
            lhs_type,
            res_type,
            lhs,
            "lhs_casted",
            &self.ctx,
            &self.builder
        );
        let casted_rhs = build_cast(
            rhs_type,
            res_type,
            rhs,
            "rhs_casted",
            &self.ctx,
            &self.builder
        );
        let res = match res_type {
            | Dtype::Bool
            | Dtype::I8
            | Dtype::U8
            | Dtype::I16
            | Dtype::U16
            | Dtype::I32
            | Dtype::U32
            | Dtype::I64
            | Dtype::U64
            | Dtype::Isize
            | Dtype::Usize => {
                self.builder.build_int_add(casted_lhs, casted_rhs, "add")
            }
            Dtype::BF16 | Dtype::F16 | Dtype::F32 | Dtype::F64 => {
                self.builder.build_float_add(casted_lhs, casted_rhs, "add")
            }
            _ => unimplemented!("unsupported dtype, {}", res_type),
        };
        self.bindings
            .get_mut(&self.current_fn)
            .expect("fn not find")
            .insert_type(res, PrimitiveType::Dtype(res_type));
        res
    }

    fn visit_sub(&mut self, sub: &crate::halide::exprs::Sub) -> BasicValue {
        let lhs = self.visit_expr(sub.e1());
        let rhs = self.visit_expr(sub.e2());
        let lhs_type = self.bindings[&self.current_fn].find_type(&lhs).unwrap().dtype();
        let rhs_type = self.bindings[&self.current_fn].find_type(&rhs).unwrap().dtype();
        let res_type = lhs_type._sub(rhs_type);
        let casted_lhs = build_cast(
            lhs_type,
            res_type,
            lhs,
            "lhs_casted",
            &self.ctx,
            &self.builder
        );
        let casted_rhs = build_cast(
            rhs_type,
            res_type,
            rhs,
            "rhs_casted",
            &self.ctx,
            &self.builder
        );
        let res = match res_type {
            | Dtype::Bool
            | Dtype::I8
            | Dtype::U8
            | Dtype::I16
            | Dtype::U16
            | Dtype::I32
            | Dtype::U32
            | Dtype::I64
            | Dtype::U64
            | Dtype::Isize
            | Dtype::Usize => {
                self.builder.build_int_sub(casted_lhs, casted_rhs, "sub")
            }
            Dtype::BF16 | Dtype::F16 | Dtype::F32 | Dtype::F64 => {
                self.builder.build_float_sub(casted_lhs, casted_rhs, "sub")
            }
            _ => unimplemented!("unsupported dtype, {}", res_type),
        };
        self.bindings
            .get_mut(&self.current_fn)
            .expect("fn not find")
            .insert_type(res, PrimitiveType::Dtype(res_type));
        res
    }

    fn visit_mul(&mut self, mul: &crate::halide::exprs::Mul) -> BasicValue {
        let lhs = self.visit_expr(mul.e1());
        let rhs = self.visit_expr(mul.e2());
        let lhs_type = self.bindings[&self.current_fn].find_type(&lhs).unwrap().dtype();
        let rhs_type = self.bindings[&self.current_fn].find_type(&rhs).unwrap().dtype();
        let res_type = lhs_type._mul(rhs_type);
        let casted_lhs = build_cast(
            lhs_type,
            res_type,
            lhs,
            "lhs_casted",
            &self.ctx,
            &self.builder
        );
        let casted_rhs = build_cast(
            rhs_type,
            res_type,
            rhs,
            "rhs_casted",
            &self.ctx,
            &self.builder
        );
        let res = match res_type {
            | Dtype::Bool
            | Dtype::I8
            | Dtype::U8
            | Dtype::I16
            | Dtype::U16
            | Dtype::I32
            | Dtype::U32
            | Dtype::I64
            | Dtype::U64
            | Dtype::Isize
            | Dtype::Usize => {
                self.builder.build_int_mul(casted_lhs, casted_rhs, "mul")
            }
            Dtype::BF16 | Dtype::F16 | Dtype::F32 | Dtype::F64 => {
                self.builder.build_float_mul(casted_lhs, casted_rhs, "mul")
            }
            _ => unimplemented!("unsupported dtype, {}", res_type),
        };
        self.bindings
            .get_mut(&self.current_fn)
            .expect("fn not find")
            .insert_type(res, PrimitiveType::Dtype(res_type));
        res
    }

    fn visit_div(&mut self, div: &crate::halide::exprs::Div) -> BasicValue {
        let lhs = self.visit_expr(div.e1());
        let rhs = self.visit_expr(div.e2());
        if div.e2() == &PrimeExpr::Int(Int::make(Dtype::I64, 0)) {
            panic!("division by zero")
        }
        let lhs_type = self.bindings[&self.current_fn].find_type(&lhs).unwrap().dtype();
        let rhs_type = self.bindings[&self.current_fn].find_type(&rhs).unwrap().dtype();
        let res_type = lhs_type._div(rhs_type);
        let casted_lhs = build_cast(
            lhs_type,
            res_type,
            lhs,
            "lhs_casted",
            &self.ctx,
            &self.builder
        );
        let casted_rhs = build_cast(
            rhs_type,
            res_type,
            rhs,
            "rhs_casted",
            &self.ctx,
            &self.builder
        );
        let res = match res_type {
            Dtype::BF16 | Dtype::F16 | Dtype::F32 | Dtype::F64 => {
                self.builder.build_float_div(casted_lhs, casted_rhs, "div")
            }
            _ => unimplemented!("unsupported dtype, {}", res_type),
        };
        self.bindings
            .get_mut(&self.current_fn)
            .expect("fn not find")
            .insert_type(res, PrimitiveType::Dtype(res_type));
        res
    }

    fn visit_floor_div(&mut self, floor_div: &crate::halide::exprs::FloorDiv) -> BasicValue {
        let lhs = self.visit_expr(floor_div.e1());
        let rhs = self.visit_expr(floor_div.e2());
        let lhs_type = self.bindings[&self.current_fn].find_type(&lhs).unwrap().dtype();
        let rhs_type = self.bindings[&self.current_fn].find_type(&rhs).unwrap().dtype();
        let res_type = lhs_type._div(rhs_type);
        let casted_lhs = build_cast(
            lhs_type,
            res_type,
            lhs,
            "lhs_casted",
            &self.ctx,
            &self.builder
        );
        let casted_rhs = build_cast(
            rhs_type,
            res_type,
            rhs,
            "rhs_casted",
            &self.ctx,
            &self.builder
        );
        let res = match res_type {
            Dtype::F32 | Dtype::F64 => {
                let res = self.builder.build_float_div(casted_lhs, casted_rhs, "div");
                match res_type {
                    Dtype::F32 => {
                        if let Some(floor_div) = self.fns.get(&"floorf".to_string()) {
                            self.builder.build_call(floor_div, &[res], "floor")
                        } else {
                            let floor_div = self.module.add_function(
                                self.ctx.f32_type().fn_type(&[self.ctx.f32_type().into()], false),
                                "floorf"
                            );
                            self.fns.insert("floorf".to_string().into(), floor_div.clone());
                            self.builder.build_call(&floor_div, &[res], "floor")
                        }
                    }
                    Dtype::F64 => {
                        if let Some(fllor) = self.fns.get(&"floor".to_string()) {
                            self.builder.build_call(fllor, &[res], "floor")
                        } else {
                            let floor = self.module.add_function(
                                self.ctx.f64_type().fn_type(&[self.ctx.f64_type().into()], false),
                                "floor"
                            );
                            self.fns.insert("floor".to_string().into(), floor.clone());
                            self.builder.build_call(&floor, &[res], "floor")
                        }
                    }
                    _ => unimplemented!(),
                }
            }
            _ => unimplemented!("unsupported dtype, {}", res_type),
        };
        let to_cast_dtype = lhs_type._add(rhs_type);
        let res = build_cast(
            res_type,
            to_cast_dtype,
            res,
            "floor_casted",
            &self.ctx,
            &self.builder
        );
        self.bindings
            .get_mut(&self.current_fn)
            .expect("fn not find")
            .insert_type(res, PrimitiveType::Dtype(to_cast_dtype));
        res
    }

    fn visit_mod(&mut self, mod_: &crate::halide::exprs::Rem) -> BasicValue {
        let lhs = self.visit_expr(mod_.e1());
        let rhs = self.visit_expr(mod_.e2());
        let lhs_type = self.bindings[&self.current_fn].find_type(&lhs).unwrap().dtype();
        let rhs_type = self.bindings[&self.current_fn].find_type(&rhs).unwrap().dtype();
        let res_type = lhs_type._rem(rhs_type);
        let casted_lhs = build_cast(
            lhs_type,
            res_type,
            lhs,
            "lhs_casted",
            &self.ctx,
            &self.builder
        );
        let casted_rhs = build_cast(
            rhs_type,
            res_type,
            rhs,
            "rhs_casted",
            &self.ctx,
            &self.builder
        );
        let res = match res_type {
            Dtype::I8 | Dtype::I16 | Dtype::I32 | Dtype::I64 | Dtype::Isize => {
                self.builder.build_int_rem(casted_lhs, casted_rhs, "srem")
            }
            Dtype::U8 | Dtype::U16 | Dtype::U32 | Dtype::U64 | Dtype::Usize => {
                self.builder.build_uint_rem(casted_lhs, casted_rhs, "urem")
            }
            Dtype::BF16 | Dtype::F16 | Dtype::F32 | Dtype::F64 => {
                self.builder.build_float_rem(casted_lhs, casted_rhs, "frem")
            }
            _ => unimplemented!("unsupported dtype, {}", res_type),
        };
        self.bindings
            .get_mut(&self.current_fn)
            .expect("fn not find")
            .insert_type(res, PrimitiveType::Dtype(res_type));
        res
    }

    fn visit_min(&mut self, min: &crate::halide::exprs::Min) -> BasicValue {
        let lhs = self.visit_expr(min.e1());
        let rhs = self.visit_expr(min.e2());
        let lhs_type = self.bindings[&self.current_fn].find_type(&lhs).unwrap().dtype();
        let rhs_type = self.bindings[&self.current_fn].find_type(&rhs).unwrap().dtype();
        let res_type = lhs_type._add(rhs_type);
        let casted_lhs = build_cast(
            lhs_type,
            res_type,
            lhs,
            "lhs_casted",
            &self.ctx,
            &self.builder
        );
        let casted_rhs = build_cast(
            rhs_type,
            res_type,
            rhs,
            "rhs_casted",
            &self.ctx,
            &self.builder
        );
        let res = match res_type {
            Dtype::I8 | Dtype::I16 | Dtype::I32 | Dtype::I64 | Dtype::Isize => {
                let cond = self.builder.build_int_cmp(
                    llvm_sys::LLVMIntPredicate::LLVMIntSLT,
                    casted_lhs,
                    casted_rhs,
                    "min_cond"
                );
                self.builder.build_select(cond, casted_lhs, casted_rhs, "min")
            }
            Dtype::Bool | Dtype::U8 | Dtype::U16 | Dtype::U32 | Dtype::U64 | Dtype::Usize => {
                let cond = self.builder.build_int_cmp(
                    llvm_sys::LLVMIntPredicate::LLVMIntULT,
                    casted_lhs,
                    casted_rhs,
                    "min_cond"
                );
                self.builder.build_select(cond, casted_lhs, casted_rhs, "min")
            }
            Dtype::BF16 | Dtype::F16 | Dtype::F32 | Dtype::F64 => {
                let cond = self.builder.build_float_cmp(
                    llvm_sys::LLVMRealPredicate::LLVMRealOLT,
                    casted_lhs,
                    casted_rhs,
                    "min_cond"
                );
                self.builder.build_select(cond, casted_lhs, casted_rhs, "min")
            }
            _ => unimplemented!("unsupported dtype, {}", res_type),
        };
        self.bindings
            .get_mut(&self.current_fn)
            .expect("fn not find")
            .insert_type(res, PrimitiveType::Dtype(res_type));
        res
    }

    fn visit_max(&mut self, max: &crate::halide::exprs::Max) -> BasicValue {
        let lhs = self.visit_expr(max.e1());
        let rhs = self.visit_expr(max.e2());
        let lhs_type = self.bindings[&self.current_fn].find_type(&lhs).unwrap().dtype();
        let rhs_type = self.bindings[&self.current_fn].find_type(&rhs).unwrap().dtype();
        let res_type = lhs_type._add(rhs_type);
        let casted_lhs = build_cast(
            lhs_type,
            res_type,
            lhs,
            "lhs_casted",
            &self.ctx,
            &self.builder
        );
        let casted_rhs = build_cast(
            rhs_type,
            res_type,
            rhs,
            "rhs_casted",
            &self.ctx,
            &self.builder
        );
        let res = match res_type {
            Dtype::I8 | Dtype::I16 | Dtype::I32 | Dtype::I64 | Dtype::Isize => {
                let cond = self.builder.build_int_cmp(
                    llvm_sys::LLVMIntPredicate::LLVMIntSLT,
                    casted_lhs,
                    casted_rhs,
                    "max_cond"
                );
                self.builder.build_select(cond, casted_lhs, casted_rhs, "max")
            }
            Dtype::Bool | Dtype::U8 | Dtype::U16 | Dtype::U32 | Dtype::U64 | Dtype::Usize => {
                let cond = self.builder.build_int_cmp(
                    llvm_sys::LLVMIntPredicate::LLVMIntULT,
                    casted_lhs,
                    casted_rhs,
                    "max_cond"
                );
                self.builder.build_select(cond, casted_lhs, casted_rhs, "max")
            }
            Dtype::BF16 | Dtype::F16 | Dtype::F32 | Dtype::F64 => {
                let cond = self.builder.build_float_cmp(
                    llvm_sys::LLVMRealPredicate::LLVMRealOLT,
                    casted_lhs,
                    casted_rhs,
                    "max_cond"
                );
                self.builder.build_select(cond, casted_lhs, casted_rhs, "max")
            }
            _ => unimplemented!("unsupported dtype, {}", res_type),
        };
        self.bindings
            .get_mut(&self.current_fn)
            .expect("fn not find")
            .insert_type(res, PrimitiveType::Dtype(res_type));
        res
    }

    fn visit_ge(&mut self, ge: &crate::halide::exprs::Ge) -> BasicValue {
        let lhs = self.visit_expr(ge.e1());
        let rhs = self.visit_expr(ge.e2());
        let lhs_type = self.bindings[&self.current_fn].find_type(&lhs).unwrap().dtype();
        let rhs_type = self.bindings[&self.current_fn].find_type(&rhs).unwrap().dtype();
        let res_type = lhs_type._add(rhs_type);
        let casted_lhs = build_cast(
            lhs_type,
            res_type,
            lhs,
            "lhs_casted",
            &self.ctx,
            &self.builder
        );
        let casted_rhs = build_cast(
            rhs_type,
            res_type,
            rhs,
            "rhs_casted",
            &self.ctx,
            &self.builder
        );
        let res = match res_type {
            Dtype::I8 | Dtype::I16 | Dtype::I32 | Dtype::I64 | Dtype::Isize => {
                self.builder.build_int_cmp(
                    LLVMIntPredicate::LLVMIntSGE,
                    casted_lhs,
                    casted_rhs,
                    "ge"
                )
            }
            Dtype::Bool | Dtype::U8 | Dtype::U16 | Dtype::U32 | Dtype::U64 | Dtype::Usize => {
                self.builder.build_int_cmp(
                    LLVMIntPredicate::LLVMIntUGE,
                    casted_lhs,
                    casted_rhs,
                    "ge"
                )
            }
            Dtype::BF16 | Dtype::F16 | Dtype::F32 | Dtype::F64 => {
                self.builder.build_float_cmp(
                    LLVMRealPredicate::LLVMRealOGE,
                    casted_lhs,
                    casted_rhs,
                    "ge"
                )
            }
            _ => unimplemented!("unsupported dtype, {}", res_type),
        };
        self.bindings
            .get_mut(&self.current_fn)
            .expect("fn not find")
            .insert_type(res, PrimitiveType::Dtype(res_type));
        res
    }

    fn visit_gt(&mut self, gt: &crate::halide::exprs::Gt) -> BasicValue {
        let lhs = self.visit_expr(gt.e1());
        let rhs = self.visit_expr(gt.e2());
        let lhs_type = self.bindings[&self.current_fn].find_type(&lhs).unwrap().dtype();
        let rhs_type = self.bindings[&self.current_fn].find_type(&rhs).unwrap().dtype();
        let res_type = lhs_type._add(rhs_type);
        let casted_lhs = build_cast(
            lhs_type,
            res_type,
            lhs,
            "lhs_casted",
            &self.ctx,
            &self.builder
        );
        let casted_rhs = build_cast(
            rhs_type,
            res_type,
            rhs,
            "rhs_casted",
            &self.ctx,
            &self.builder
        );
        let res = match res_type {
            Dtype::I8 | Dtype::I16 | Dtype::I32 | Dtype::I64 | Dtype::Isize => {
                self.builder.build_int_cmp(
                    LLVMIntPredicate::LLVMIntSGT,
                    casted_lhs,
                    casted_rhs,
                    "gt"
                )
            }
            Dtype::Bool | Dtype::U8 | Dtype::U16 | Dtype::U32 | Dtype::U64 | Dtype::Usize => {
                self.builder.build_int_cmp(
                    LLVMIntPredicate::LLVMIntUGT,
                    casted_lhs,
                    casted_rhs,
                    "gt"
                )
            }
            Dtype::BF16 | Dtype::F16 | Dtype::F32 | Dtype::F64 => {
                self.builder.build_float_cmp(
                    LLVMRealPredicate::LLVMRealOGT,
                    casted_lhs,
                    casted_rhs,
                    "gt"
                )
            }
            _ => unimplemented!("unsupported dtype, {}", res_type),
        };
        self.bindings
            .get_mut(&self.current_fn)
            .expect("fn not find")
            .insert_type(res, PrimitiveType::Dtype(res_type));
        res
    }

    fn visit_le(&mut self, le: &crate::halide::exprs::Le) -> BasicValue {
        let lhs = self.visit_expr(le.e1());
        let rhs = self.visit_expr(le.e2());
        let lhs_type = self.bindings[&self.current_fn].find_type(&lhs).unwrap().dtype();
        let rhs_type = self.bindings[&self.current_fn].find_type(&rhs).unwrap().dtype();
        let res_type = lhs_type._add(rhs_type);
        let casted_lhs = build_cast(
            lhs_type,
            res_type,
            lhs,
            "lhs_casted",
            &self.ctx,
            &self.builder
        );
        let casted_rhs = build_cast(
            rhs_type,
            res_type,
            rhs,
            "rhs_casted",
            &self.ctx,
            &self.builder
        );
        let res = match res_type {
            Dtype::I8 | Dtype::I16 | Dtype::I32 | Dtype::I64 | Dtype::Isize => {
                self.builder.build_int_cmp(
                    LLVMIntPredicate::LLVMIntSLE,
                    casted_lhs,
                    casted_rhs,
                    "le"
                )
            }
            Dtype::Bool | Dtype::U8 | Dtype::U16 | Dtype::U32 | Dtype::U64 | Dtype::Usize => {
                self.builder.build_int_cmp(
                    LLVMIntPredicate::LLVMIntULE,
                    casted_lhs,
                    casted_rhs,
                    "le"
                )
            }
            Dtype::BF16 | Dtype::F16 | Dtype::F32 | Dtype::F64 => {
                self.builder.build_float_cmp(
                    LLVMRealPredicate::LLVMRealOLE,
                    casted_lhs,
                    casted_rhs,
                    "le"
                )
            }
            _ => unimplemented!("unsupported dtype, {}", res_type),
        };
        self.bindings
            .get_mut(&self.current_fn)
            .expect("fn not find")
            .insert_type(res, PrimitiveType::Dtype(res_type));
        res
    }

    fn visit_lt(&mut self, lt: &crate::halide::exprs::Lt) -> BasicValue {
        let lhs = self.visit_expr(lt.e1());
        let rhs = self.visit_expr(lt.e2());
        let lhs_type = self.bindings[&self.current_fn].find_type(&lhs).unwrap().dtype();
        let rhs_type = self.bindings[&self.current_fn].find_type(&rhs).unwrap().dtype();
        let res_type = lhs_type._add(rhs_type);
        let casted_lhs = build_cast(
            lhs_type,
            res_type,
            lhs,
            "lhs_casted",
            &self.ctx,
            &self.builder
        );
        let casted_rhs = build_cast(
            rhs_type,
            res_type,
            rhs,
            "rhs_casted",
            &self.ctx,
            &self.builder
        );
        let res = match res_type {
            Dtype::I8 | Dtype::I16 | Dtype::I32 | Dtype::I64 | Dtype::Isize => {
                self.builder.build_int_cmp(
                    LLVMIntPredicate::LLVMIntSLT,
                    casted_lhs,
                    casted_rhs,
                    "lt"
                )
            }
            Dtype::Bool | Dtype::U8 | Dtype::U16 | Dtype::U32 | Dtype::U64 | Dtype::Usize => {
                self.builder.build_int_cmp(
                    LLVMIntPredicate::LLVMIntULT,
                    casted_lhs,
                    casted_rhs,
                    "lt"
                )
            }
            Dtype::BF16 | Dtype::F16 | Dtype::F32 | Dtype::F64 => {
                self.builder.build_float_cmp(
                    LLVMRealPredicate::LLVMRealOLT,
                    casted_lhs,
                    casted_rhs,
                    "lt"
                )
            }
            _ => unimplemented!("unsupported dtype, {}", res_type),
        };
        self.bindings
            .get_mut(&self.current_fn)
            .expect("fn not find")
            .insert_type(res, PrimitiveType::Dtype(res_type));
        res
    }

    fn visit_eq(&mut self, eq: &crate::halide::exprs::Eq) -> BasicValue {
        let lhs = self.visit_expr(eq.e1());
        let rhs = self.visit_expr(eq.e2());
        let lhs_type = self.bindings[&self.current_fn].find_type(&lhs).unwrap().dtype();
        let rhs_type = self.bindings[&self.current_fn].find_type(&rhs).unwrap().dtype();
        let res_type = lhs_type._add(rhs_type);
        let casted_lhs = build_cast(
            lhs_type,
            res_type,
            lhs,
            "lhs_casted",
            &self.ctx,
            &self.builder
        );
        let casted_rhs = build_cast(
            rhs_type,
            res_type,
            rhs,
            "rhs_casted",
            &self.ctx,
            &self.builder
        );
        let res = match res_type {
            | Dtype::I8
            | Dtype::I16
            | Dtype::I32
            | Dtype::I64
            | Dtype::Isize
            | Dtype::Bool
            | Dtype::U8
            | Dtype::U16
            | Dtype::U32
            | Dtype::U64
            | Dtype::Usize => {
                self.builder.build_int_cmp(
                    LLVMIntPredicate::LLVMIntEQ,
                    casted_lhs,
                    casted_rhs,
                    "eq"
                )
            }
            Dtype::BF16 | Dtype::F16 | Dtype::F32 | Dtype::F64 => {
                self.builder.build_float_cmp(
                    LLVMRealPredicate::LLVMRealOEQ,
                    casted_lhs,
                    casted_rhs,
                    "eq"
                )
            }
            _ => unimplemented!("unsupported dtype, {}", res_type),
        };
        self.bindings
            .get_mut(&self.current_fn)
            .expect("fn not find")
            .insert_type(res, PrimitiveType::Dtype(res_type));
        res
    }

    fn visit_ne(&mut self, ne: &crate::halide::exprs::Ne) -> BasicValue {
        let lhs = self.visit_expr(ne.e1());
        let rhs = self.visit_expr(ne.e2());
        let lhs_type = self.bindings[&self.current_fn].find_type(&lhs).unwrap().dtype();
        let rhs_type = self.bindings[&self.current_fn].find_type(&rhs).unwrap().dtype();
        let res_type = lhs_type._add(rhs_type);
        let casted_lhs = build_cast(
            lhs_type,
            res_type,
            lhs,
            "lhs_casted",
            &self.ctx,
            &self.builder
        );
        let casted_rhs = build_cast(
            rhs_type,
            res_type,
            rhs,
            "rhs_casted",
            &self.ctx,
            &self.builder
        );
        let res = match res_type {
            | Dtype::I8
            | Dtype::I16
            | Dtype::I32
            | Dtype::I64
            | Dtype::Isize
            | Dtype::Bool
            | Dtype::U8
            | Dtype::U16
            | Dtype::U32
            | Dtype::U64
            | Dtype::Usize => {
                self.builder.build_int_cmp(
                    LLVMIntPredicate::LLVMIntNE,
                    casted_lhs,
                    casted_rhs,
                    "ne"
                )
            }
            Dtype::BF16 | Dtype::F16 | Dtype::F32 | Dtype::F64 => {
                self.builder.build_float_cmp(
                    LLVMRealPredicate::LLVMRealONE,
                    casted_lhs,
                    casted_rhs,
                    "ne"
                )
            }
            _ => unimplemented!("unsupported dtype, {}", res_type),
        };
        self.bindings
            .get_mut(&self.current_fn)
            .expect("fn not find")
            .insert_type(res, PrimitiveType::Dtype(res_type));
        res
    }

    fn visit_and(&mut self, and: &crate::halide::exprs::BitAnd) -> BasicValue {
        let lhs = self.visit_expr(and.e1());
        let rhs = self.visit_expr(and.e2());
        let lhs_type = self.bindings[&self.current_fn].find_type(&lhs).unwrap().dtype();
        let rhs_type = self.bindings[&self.current_fn].find_type(&rhs).unwrap().dtype();
        let res_type = lhs_type._bitand(rhs_type);
        let casted_lhs = build_cast(
            lhs_type,
            res_type,
            lhs,
            "lhs_casted",
            &self.ctx,
            &self.builder
        );
        let casted_rhs = build_cast(
            rhs_type,
            res_type,
            rhs,
            "rhs_casted",
            &self.ctx,
            &self.builder
        );
        let res = match res_type {
            | Dtype::I8
            | Dtype::I16
            | Dtype::I32
            | Dtype::I64
            | Dtype::Isize
            | Dtype::Bool
            | Dtype::U8
            | Dtype::U16
            | Dtype::U32
            | Dtype::U64
            | Dtype::Usize => {
                self.builder.build_int_and(casted_lhs, casted_rhs, "bitand")
            }
            _ => unimplemented!("unsupported dtype, {}", res_type),
        };
        self.bindings
            .get_mut(&self.current_fn)
            .expect("fn not find")
            .insert_type(res, PrimitiveType::Dtype(res_type));
        res
    }

    fn visit_xor(&mut self, xor: &crate::halide::exprs::BitXor) -> BasicValue {
        let lhs = self.visit_expr(xor.e1());
        let rhs = self.visit_expr(xor.e2());
        let lhs_type = self.bindings[&self.current_fn].find_type(&lhs).unwrap().dtype();
        let rhs_type = self.bindings[&self.current_fn].find_type(&rhs).unwrap().dtype();
        let res_type = lhs_type._bitxor(rhs_type);
        let casted_lhs = build_cast(
            lhs_type,
            res_type,
            lhs,
            "lhs_casted",
            &self.ctx,
            &self.builder
        );
        let casted_rhs = build_cast(
            rhs_type,
            res_type,
            rhs,
            "rhs_casted",
            &self.ctx,
            &self.builder
        );
        let res = match res_type {
            | Dtype::I8
            | Dtype::I16
            | Dtype::I32
            | Dtype::I64
            | Dtype::Isize
            | Dtype::Bool
            | Dtype::U8
            | Dtype::U16
            | Dtype::U32
            | Dtype::U64
            | Dtype::Usize => {
                self.builder.build_int_xor(casted_lhs, casted_rhs, "xor")
            }
            _ => unimplemented!("unsupported dtype, {}", res_type),
        };
        self.bindings
            .get_mut(&self.current_fn)
            .expect("fn not find")
            .insert_type(res, PrimitiveType::Dtype(res_type));
        res
    }

    fn visit_or(&mut self, or: &crate::halide::exprs::BitOr) -> BasicValue {
        let lhs = self.visit_expr(or.e1());
        let rhs = self.visit_expr(or.e2());
        let lhs_type = self.bindings[&self.current_fn].find_type(&lhs).unwrap().dtype();
        let rhs_type = self.bindings[&self.current_fn].find_type(&rhs).unwrap().dtype();
        let res_type = lhs_type._bitor(rhs_type);
        let casted_lhs = build_cast(
            lhs_type,
            res_type,
            lhs,
            "lhs_casted",
            &self.ctx,
            &self.builder
        );
        let casted_rhs = build_cast(
            rhs_type,
            res_type,
            rhs,
            "rhs_casted",
            &self.ctx,
            &self.builder
        );
        let res = match res_type {
            | Dtype::I8
            | Dtype::I16
            | Dtype::I32
            | Dtype::I64
            | Dtype::Isize
            | Dtype::Bool
            | Dtype::U8
            | Dtype::U16
            | Dtype::U32
            | Dtype::U64
            | Dtype::Usize => {
                self.builder.build_int_or(casted_lhs, casted_rhs, "or")
            }
            _ => unimplemented!("unsupported dtype, {}", res_type),
        };
        self.bindings
            .get_mut(&self.current_fn)
            .expect("fn not find")
            .insert_type(res, PrimitiveType::Dtype(res_type));
        res
    }

    fn visit_not(&mut self, not: &crate::halide::exprs::Not) -> BasicValue {
        let val = self.visit_expr(not.e());
        let val_type = self.bindings[&self.current_fn].find_type(&val).unwrap().dtype();
        let res_type = val_type._not();
        let casted_val = build_cast(
            val_type,
            res_type,
            val,
            "val_casted",
            &self.ctx,
            &self.builder
        );
        let res = match res_type {
            | Dtype::Bool
            | Dtype::I8
            | Dtype::U8
            | Dtype::I16
            | Dtype::U16
            | Dtype::I32
            | Dtype::U32
            | Dtype::I64
            | Dtype::U64
            | Dtype::Isize
            | Dtype::Usize => {
                self.builder.build_int_not(casted_val, "not")
            }
            _ => unimplemented!("unsupported dtype, {}", res_type),
        };
        self.bindings
            .get_mut(&self.current_fn)
            .expect("fn not find")
            .insert_type(res, PrimitiveType::Dtype(res_type));
        res
    }

    fn visit_let_stmt(&mut self, let_stmt: &crate::halide::let_stmt::LetStmt) {
        let var = let_stmt.var();
        let val = self.visit_expr(&let_stmt.value());
        if let_stmt.mutable() {
            self.bindings.get_mut(&self.current_fn).expect("fn not find").insert_mutable(&var.name);
            let ptr = self.builder.build_alloca(val.to_basic_type(), &var.name);
            self.builder.build_store(ptr, val);
            self.bindings
                .get_mut(&self.current_fn)
                .expect("fn not find")
                .insert_variable(&var.name, ptr.into());
        } else {
            self.bindings
                .get_mut(&self.current_fn)
                .expect("fn not find")
                .insert_variable(&var.name, val);
        }
        self.visit_stmt(&let_stmt.body())
    }

    fn visit_let(&mut self, let_: &crate::halide::exprs::Let) -> BasicValue {
        let var = let_.name();
        let val = self.visit_expr(&let_.value_());
        self.bindings
            .get_mut(&self.current_fn)
            .expect("fn not find")
            .insert_variable(&var.name, val);
        self.visit_expr(&let_.body_())
    }

    fn visit_for(&mut self, for_stmt: &crate::halide::for_stmt::For) {
        let index = for_stmt.var();
        let entry_block = self.ctx.append_basic_block(&self.id_fns[&self.current_fn], "for_entry");
        let cond_block = self.ctx.append_basic_block(&self.id_fns[&self.current_fn], "for_cond");
        let body_block = self.ctx.append_basic_block(&self.id_fns[&self.current_fn], "for_body");
        let end_block = self.ctx.append_basic_block(&self.id_fns[&self.current_fn], "for_exit");

        self.builder.build_unconditional_branch(entry_block);
        self.builder.position_at_end(entry_block);
        let start = self.visit_expr(for_stmt.start());
        let end = self.visit_expr(for_stmt.end());
        let end_ty = self.bindings[&self.current_fn].find_type(&end).unwrap().dtype();
        let end = if end_ty == Dtype::F64 {
            self.builder.build_float_to_signed_int(self.ctx.i64_type().into(), end, "casted_end")
        } else {
            end
        };
        let step = self.visit_expr(for_stmt.step());
        self.builder.build_unconditional_branch(cond_block);
        self.builder.position_at_end(cond_block);
        self.bindings.get_mut(&self.current_fn).expect("fn not find").push_scope();
        let phi = self.builder.build_phi(self.ctx.i64_type().into(), "phi");
        self.bindings
            .get_mut(&self.current_fn)
            .expect("fn not find")
            .insert_variable(&index.name, phi.into_int_value().into());
        self.bindings
            .get_mut(&self.current_fn)
            .expect("fn not find")
            .insert_type(phi.into_int_value().into(), PrimitiveType::Dtype(Dtype::I64));
        phi.add_incoming(&[(&start, entry_block)]);

        let cond = self.builder.build_int_cmp(
            llvm_sys::LLVMIntPredicate::LLVMIntSLT,
            phi.into(),
            end,
            "cond"
        );
        self.builder.build_conditional_branch(cond, body_block, end_block);

        self.builder.position_at_end(body_block);
        let new_phi = self.builder.build_int_add(phi.into_int_value().into(), step, "new_phi");

        self.visit_stmt(for_stmt.stmt());

        self.builder.build_unconditional_branch(cond_block);
        phi.add_incoming(&[(&new_phi, self.builder.get_insert_block())]);

        self.builder.position_at_end(end_block);
        self.bindings.get_mut(&self.current_fn).expect("fn not find").pop_scope();
    }

    #[rustfmt::skip]
    fn visit_int(&mut self, int: &crate::halide::exprs::Int) -> BasicValue {
        let res: BasicValue = match int.dtype() {
            Dtype::I8 => self.ctx.i8_type().const_int(int.value() as u64, false).into(),
            Dtype::I16 =>   self.ctx.i16_type().const_int(int.value() as u64, false).into(),
            Dtype::I32 => self.ctx.i32_type().const_int(int.value() as u64, false).into(),
            Dtype::I64 => self.ctx.i64_type().const_int(int.value() as u64, false).into(),
            Dtype::Isize => self.ctx.isize_type().const_int(int.value() as u64, false).into(),
            _ => unimplemented!(),
        };
        self.bindings
            .get_mut(&self.current_fn)
            .expect("fn not find")
            .insert_type(res, PrimitiveType::Dtype(*int.dtype()));
        res
    }

    #[rustfmt::skip]
    fn visit_uint(&mut self, uint: &crate::halide::exprs::UInt) -> BasicValue {
        let res: BasicValue = match uint.dtype() {
            Dtype::U8 => self.ctx.u8_type().const_int(uint.value() as u64, false).into(),
            Dtype::U16 =>   self.ctx.u16_type().const_int(uint.value() as u64, false).into(),
            Dtype::U32 => self.ctx.u32_type().const_int(uint.value() as u64, false).into(),
            Dtype::U64 => self.ctx.u64_type().const_int(uint.value() as u64, false).into(),
            Dtype::Usize => self.ctx.usize_type().const_int(uint.value() as u64, false).into(),
            _ => unimplemented!("unsupported dtype, {}", uint.dtype()),
        };
        self.bindings
            .get_mut(&self.current_fn)
            .expect("fn not find")
            .insert_type(res, PrimitiveType::Dtype(*uint.dtype()));
        res
    }

    #[rustfmt::skip]
    fn visit_float(&mut self, float: &crate::halide::exprs::Float) -> BasicValue {
        let res: BasicValue = match float.dtype() {
            Dtype::F32 => self.ctx.f32_type().const_float(float.value().to_f64()).into(),
            Dtype::F64 => self.ctx.f64_type().const_float(float.value()).into(),
            _ => unimplemented!(),
        };
        self.bindings
            .get_mut(&self.current_fn)
            .expect("fn not find")
            .insert_type(res, PrimitiveType::Dtype(*float.dtype()));
        res
    }

    fn visit_call(&mut self, call: &crate::halide::exprs::Call) -> BasicValue {
        let fn_name = call.name();
        if let Some(fn_val) = self.fns.get(fn_name).cloned() {
            let args = call
                .args()
                .iter()
                .map(|arg| self.visit_expr(arg))
                .collect::<Vec<_>>();
            let res = self.builder.build_call(&fn_val, &args, "call");
            self.bindings
                .get_mut(&self.current_fn)
                .expect("fn not find")
                .insert_type(res, general_types_to_primitive_type(fn_val.ret_type()));
            res
        } else {
            match fn_name.as_str() {
                "sin" => {
                    let arg = self.visit_expr(call.args().first().unwrap());
                    let arg_ty = self.bindings[&self.current_fn].find_type(&arg).unwrap().dtype();
                    let casted_ty = arg_ty._sin();
                    if casted_ty == arg_ty {
                        let res = match casted_ty {
                            Dtype::F32 => {
                                let ret_ty = self.ctx.f32_type();
                                let fn_ty = ret_ty.fn_type(&[ret_ty.into()], false);
                                let sinf = self.module.add_function(fn_ty, "sinf");
                                self.fns.insert("sinf".to_string().into(), sinf.clone());
                                self.builder.build_call(&sinf, &[arg], "sinf")
                            }
                            Dtype::F64 => {
                                let ret_ty = self.ctx.f64_type();
                                let fn_ty = ret_ty.fn_type(&[ret_ty.into()], false);
                                let sin = self.module.add_function(fn_ty, "sin");
                                self.fns.insert("sin".to_string().into(), sin.clone());
                                self.builder.build_call(&sin, &[arg], "sin")
                            }
                            _ => unimplemented!("unsupported dtype, {}", casted_ty),
                        };
                        self.bindings
                            .get_mut(&self.current_fn)
                            .expect("fn not find")
                            .insert_type(res, PrimitiveType::Dtype(casted_ty));
                        res
                    } else {
                        let casted_arg = build_cast(
                            arg_ty,
                            casted_ty,
                            arg,
                            "casted_arg",
                            &self.ctx,
                            &self.builder
                        );
                        let res = match casted_ty {
                            Dtype::F32 => {
                                let ret_ty = self.ctx.f32_type();
                                let fn_ty = ret_ty.fn_type(&[ret_ty.into()], false);
                                let sinf = self.module.add_function(fn_ty, "sinf");
                                self.fns.insert("sinf".to_string().into(), sinf.clone());
                                self.builder.build_call(&sinf, &[casted_arg], "sinf")
                            }
                            Dtype::F64 => {
                                let ret_ty = self.ctx.f64_type();
                                let fn_ty = ret_ty.fn_type(&[ret_ty.into()], false);
                                let sin = self.module.add_function(fn_ty, "sin");
                                self.fns.insert("sin".to_string().into(), sin.clone());
                                self.builder.build_call(&sin, &[casted_arg], "sin")
                            }
                            _ => unimplemented!("unsupported dtype, {}", casted_ty),
                        };
                        self.bindings
                            .get_mut(&self.current_fn)
                            .expect("fn not find")
                            .insert_type(res, PrimitiveType::Dtype(casted_ty));
                        res
                    }
                }
                "ceil" => {
                    let arg = self.visit_expr(call.args().first().unwrap());
                    let arg_ty = self.bindings[&self.current_fn].find_type(&arg).unwrap().dtype();
                    let ret = match arg_ty {
                        Dtype::I8 | Dtype::I16 | Dtype::I32 | Dtype::I64 | Dtype::Isize => {
                            arg
                        }
                        Dtype::U8 | Dtype::U16 | Dtype::U32 | Dtype::U64 | Dtype::Usize => {
                            arg
                        }
                        Dtype::F32 => {
                            let ret_ty = self.ctx.f32_type();
                            let fn_ty = ret_ty.fn_type(&[ret_ty.into()], false);
                            let ceilf = self.module.add_function(fn_ty, "ceilf");
                            self.fns.insert("ceilf".to_string().into(), ceilf.clone());
                            let res = self.builder.build_call(&ceilf, &[arg], "ceilf");
                            self.bindings
                                .get_mut(&self.current_fn)
                                .expect("fn not find")
                                .insert_type(res, PrimitiveType::Dtype(Dtype::F32));
                            res
                        }
                        Dtype::F64 => {
                            let ret_ty = self.ctx.f64_type();
                            let fn_ty = ret_ty.fn_type(&[ret_ty.into()], false);
                            let ceil = self.module.add_function(fn_ty, "ceil");
                            self.fns.insert("ceil".to_string().into(), ceil.clone());
                            let res = self.builder.build_call(&ceil, &[arg], "ceil");
                            self.bindings
                                .get_mut(&self.current_fn)
                                .expect("fn not find")
                                .insert_type(res, PrimitiveType::Dtype(Dtype::F64));
                            res
                        }
                        _ => unimplemented!("unsupported dtype, {}", arg_ty),
                    };
                    self.bindings
                        .get_mut(&self.current_fn)
                        .expect("fn not find")
                        .insert_type(ret, PrimitiveType::Dtype(arg_ty));
                    ret
                }
                _ => unimplemented!("function {} not found", fn_name),
            }
        }
    }

    fn visit_select(&mut self, select: &crate::halide::exprs::Select) -> BasicValue {
        let cond = self.visit_expr(select.cond());
        let true_val = self.visit_expr(select.true_expr());
        let false_val = self.visit_expr(select.false_expr());
        let true_ty = self.bindings[&self.current_fn].find_type(&true_val).unwrap().clone();
        let false_ty = self.bindings[&self.current_fn].find_type(&false_val).unwrap().clone();
        assert!(true_ty == false_ty);

        let res = self.builder.build_select(cond, true_val, false_val, "select");
        self.bindings.get_mut(&self.current_fn).expect("fn not find").insert_type(res, true_ty);
        res
    }

    fn visit_load(&mut self, _load: &crate::halide::exprs::Load) -> BasicValue {
        let var = _load.name();
        let basic_val = self.bindings[&self.current_fn]
            .find_variable(&var.name)
            .expect(&format!("variable {} not found", var.name));
        let var_type = self.bindings[&self.current_fn]
            .find_type(&basic_val)
            .expect(&format!("type not found for variable {}", var.name))
            .clone();
        let indices = self.visit_expr(_load.indices());
        let (loaded, loaded_type) = load(
            &var_type,
            basic_val,
            indices,
            &self.builder,
            &self.module,
            self.tensor_type,
            &self.ctx
        );
        self.bindings
            .get_mut(&self.current_fn)
            .expect("fn not find")
            .insert_type(loaded, loaded_type);
        loaded
    }

    fn visit_store(&mut self, store: &crate::halide::store_stmt::StoreStmt) {
        let var = store.var();
        let basic_val = self.bindings[&self.current_fn]
            .find_variable(&var.name)
            .expect("var not find");
        let var_type = self.bindings[&self.current_fn]
            .find_type(&basic_val)
            .expect("type not find")
            .clone();
        let indices = self.visit_expr(store.indices());
        let new_ptr = match &var_type {
            PrimitiveType::Ptr(ptr) =>
                match ptr.inner.as_ref() {
                    PrimitiveType::Dtype(dtype) => {
                        match dtype {
                            Dtype::Bool => {
                                self.builder.build_gep(
                                    self.ctx.bool_type(),
                                    basic_val.to_ptr_value(),
                                    &[indices],
                                    "gep"
                                )
                            }
                            Dtype::I8 => {
                                self.builder.build_gep(
                                    self.ctx.i8_type(),
                                    basic_val.to_ptr_value(),
                                    &[indices],
                                    "gep"
                                )
                            }
                            Dtype::U8 => {
                                self.builder.build_gep(
                                    self.ctx.u8_type(),
                                    basic_val.to_ptr_value(),
                                    &[indices],
                                    "gep"
                                )
                            }
                            Dtype::I16 => {
                                self.builder.build_gep(
                                    self.ctx.i16_type(),
                                    basic_val.to_ptr_value(),
                                    &[indices],
                                    "gep"
                                )
                            }
                            Dtype::U16 => {
                                self.builder.build_gep(
                                    self.ctx.u16_type(),
                                    basic_val.to_ptr_value(),
                                    &[indices],
                                    "gep"
                                )
                            }
                            Dtype::I32 => {
                                self.builder.build_gep(
                                    self.ctx.i32_type(),
                                    basic_val.to_ptr_value(),
                                    &[indices],
                                    "gep"
                                )
                            }
                            Dtype::U32 => {
                                self.builder.build_gep(
                                    self.ctx.u32_type(),
                                    basic_val.to_ptr_value(),
                                    &[indices],
                                    "gep"
                                )
                            }
                            Dtype::I64 => {
                                self.builder.build_gep(
                                    self.ctx.i64_type(),
                                    basic_val.to_ptr_value(),
                                    &[indices],
                                    "gep"
                                )
                            }
                            Dtype::U64 => {
                                self.builder.build_gep(
                                    self.ctx.u64_type(),
                                    basic_val.to_ptr_value(),
                                    &[indices],
                                    "gep"
                                )
                            }
                            Dtype::BF16 => todo!(),
                            Dtype::F16 => {
                                self.builder.build_gep(
                                    self.ctx.f16_type(),
                                    basic_val.to_ptr_value(),
                                    &[indices],
                                    "gep"
                                )
                            }
                            Dtype::F32 => {
                                self.builder.build_gep(
                                    self.ctx.f32_type(),
                                    basic_val.to_ptr_value(),
                                    &[indices],
                                    "gep"
                                )
                            }
                            Dtype::F64 => {
                                self.builder.build_gep(
                                    self.ctx.f64_type(),
                                    basic_val.to_ptr_value(),
                                    &[indices],
                                    "gep"
                                )
                            }
                            Dtype::C32 => todo!(),
                            Dtype::C64 => todo!(),
                            Dtype::Isize => {
                                self.builder.build_gep(
                                    self.ctx.isize_type(),
                                    basic_val.to_ptr_value(),
                                    &[indices],
                                    "gep"
                                )
                            }
                            Dtype::Usize => todo!(),
                        }
                    }
                    PrimitiveType::Ptr(_) => {
                        self.builder.build_gep(
                            self.ctx.void_type().ptr_type(0),
                            basic_val.to_ptr_value(),
                            &[indices],
                            "gep"
                        )
                    }
                    PrimitiveType::Tuple(_) => todo!(),
                    PrimitiveType::Array(_) => todo!(),
                    PrimitiveType::Str => todo!(),
                    PrimitiveType::Void => todo!(),
                    PrimitiveType::Tensor(tensor) => {
                        let basic_val = self.bindings[&self.current_fn]
                            .find_variable(&format!("{}.data", var.name))
                            .expect("var not find");
                        match tensor.dtype {
                            Dtype::Bool => {
                                self.builder.build_gep(
                                    self.ctx.bool_type(),
                                    basic_val.to_ptr_value(),
                                    &[indices],
                                    "gep"
                                )
                            }
                            Dtype::I8 => {
                                self.builder.build_gep(
                                    self.ctx.i8_type(),
                                    basic_val.to_ptr_value(),
                                    &[indices],
                                    "gep"
                                )
                            }
                            Dtype::U8 => {
                                self.builder.build_gep(
                                    self.ctx.u8_type(),
                                    basic_val.to_ptr_value(),
                                    &[indices],
                                    "gep"
                                )
                            }
                            Dtype::I16 => {
                                self.builder.build_gep(
                                    self.ctx.i16_type(),
                                    basic_val.to_ptr_value(),
                                    &[indices],
                                    "gep"
                                )
                            }
                            Dtype::U16 => {
                                self.builder.build_gep(
                                    self.ctx.u16_type(),
                                    basic_val.to_ptr_value(),
                                    &[indices],
                                    "gep"
                                )
                            }
                            Dtype::I32 => {
                                self.builder.build_gep(
                                    self.ctx.i32_type(),
                                    basic_val.to_ptr_value(),
                                    &[indices],
                                    "gep"
                                )
                            }
                            Dtype::U32 => {
                                self.builder.build_gep(
                                    self.ctx.u32_type(),
                                    basic_val.to_ptr_value(),
                                    &[indices],
                                    "gep"
                                )
                            }
                            Dtype::I64 => {
                                self.builder.build_gep(
                                    self.ctx.i64_type(),
                                    basic_val.to_ptr_value(),
                                    &[indices],
                                    "gep"
                                )
                            }
                            Dtype::U64 => {
                                self.builder.build_gep(
                                    self.ctx.u64_type(),
                                    basic_val.to_ptr_value(),
                                    &[indices],
                                    "gep"
                                )
                            }
                            Dtype::F32 => {
                                self.builder.build_gep(
                                    self.ctx.f32_type(),
                                    basic_val.to_ptr_value(),
                                    &[indices],
                                    "gep"
                                )
                            }
                            Dtype::F64 => {
                                self.builder.build_gep(
                                    self.ctx.f64_type(),
                                    basic_val.to_ptr_value(),
                                    &[indices],
                                    "gep"
                                )
                            }
                            Dtype::Isize => {
                                self.builder.build_gep(
                                    self.ctx.isize_type(),
                                    basic_val.to_ptr_value(),
                                    &[indices],
                                    "gep"
                                )
                            }
                            Dtype::Usize => {
                                self.builder.build_gep(
                                    self.ctx.usize_type(),
                                    basic_val.to_ptr_value(),
                                    &[indices],
                                    "gep"
                                )
                            }
                            _ => unimplemented!(),
                        }
                    }
                }
            _ => unimplemented!(),
        };
        let val = self.visit_expr(store.val());
        let val_type = self.bindings[&self.current_fn]
            .find_type(&val)
            .expect("type not find")
            .clone();
        self.builder.build_store(new_ptr, val);
    }

    fn visit_if_then_else(&mut self, if_then_else: &crate::halide::if_stmt::IfThenElse) {
        let cond_block = self.ctx.append_basic_block(&self.id_fns[&self.current_fn], "if_cond");
        let then_block = self.ctx.append_basic_block(&self.id_fns[&self.current_fn], "if_then");
        let else_block = self.ctx.append_basic_block(&self.id_fns[&self.current_fn], "if_else");
        let end_block = self.ctx.append_basic_block(&self.id_fns[&self.current_fn], "if_end");
        self.builder.build_unconditional_branch(cond_block);
        self.builder.position_at_end(cond_block);
        let cond = self.visit_expr(if_then_else.cond());
        self.builder.build_conditional_branch(cond, then_block, else_block);
        self.bindings.get_mut(&self.current_fn).expect("fn not find").push_scope();
        self.builder.position_at_end(then_block);
        self.visit_stmt(if_then_else.then_case());
        self.builder.build_unconditional_branch(end_block);
        self.bindings.get_mut(&self.current_fn).expect("fn not find").pop_scope();
        self.builder.position_at_end(else_block);
        self.bindings.get_mut(&self.current_fn).expect("fn not find").push_scope();
        self.visit_stmt(if_then_else.else_case());
        self.builder.build_unconditional_branch(end_block);
        self.bindings.get_mut(&self.current_fn).expect("fn not find").pop_scope();
        self.builder.position_at_end(end_block);
    }

    fn visit_inplace_store(
        &mut self,
        inplace_store: &crate::halide::inplace_store_stmt::InplaceStore
    ) {
        todo!()
    }

    fn visit_inplace_add(&mut self, inplace_add: &crate::halide::inplace_store_stmt::InplaceAdd) {
        let to_store = self.visit_expr(&inplace_add.to_store());
        let val = self.visit_expr(&inplace_add.val());
        let to_store_type = self.bindings[&self.current_fn].find_type(&to_store).unwrap().dtype();
        let casted_val = build_cast(
            self.bindings[&self.current_fn].find_type(&val).unwrap().dtype(),
            to_store_type,
            val,
            "val_casted",
            &self.ctx,
            &self.builder
        );
        let res = match to_store_type {
            | Dtype::Bool
            | Dtype::I8
            | Dtype::U8
            | Dtype::I16
            | Dtype::U16
            | Dtype::I32
            | Dtype::U32
            | Dtype::I64
            | Dtype::U64
            | Dtype::Isize
            | Dtype::Usize => {
                self.builder.build_int_add(to_store, casted_val, "add")
            }
            Dtype::BF16 | Dtype::F16 | Dtype::F32 | Dtype::F64 => {
                self.builder.build_float_add(to_store, casted_val, "add")
            }
            _ => unimplemented!("unsupported dtype, {}", to_store_type),
        };

        match inplace_add.to_store() {
            PrimeExpr::Load(load) => {
                let indices = load.indices();
                let indices_val = self.visit_expr(&indices);
                let ptr = inplace_add.to_store().to_load().expect("expected load").name();
                let ptr_val = self.bindings[&self.current_fn]
                    .find_variable(&ptr.name)
                    .unwrap()
                    .to_ptr_value();
                let gep = self.builder.build_gep(
                    dtype_to_llvm(to_store_type, &self.ctx),
                    ptr_val,
                    &[indices_val],
                    "gep"
                );
                self.builder.build_store(gep, res)
            }
            _ => {
                let val = self.visit_expr(&inplace_add.to_store());
                let val_type = self.bindings[&self.current_fn].find_type(&val).unwrap().dtype();
                self.builder.build_store(val.to_ptr_value(), res)
            }
        };
    }

    fn visit_inplace_sub(&mut self, inplace_sub: &crate::halide::inplace_store_stmt::InplaceSub) {
        let to_store = self.visit_expr(&inplace_sub.to_store());
        let val = self.visit_expr(&inplace_sub.val());
        let to_store_type = self.bindings[&self.current_fn].find_type(&to_store).unwrap().dtype();
        let casted_val = build_cast(
            self.bindings[&self.current_fn].find_type(&val).unwrap().dtype(),
            to_store_type,
            val,
            "val_casted",
            &self.ctx,
            &self.builder
        );
        let res = match to_store_type {
            | Dtype::Bool
            | Dtype::I8
            | Dtype::U8
            | Dtype::I16
            | Dtype::U16
            | Dtype::I32
            | Dtype::U32
            | Dtype::I64
            | Dtype::U64
            | Dtype::Isize
            | Dtype::Usize => {
                self.builder.build_int_sub(to_store, casted_val, "add")
            }
            Dtype::BF16 | Dtype::F16 | Dtype::F32 | Dtype::F64 => {
                self.builder.build_float_sub(to_store, casted_val, "add")
            }
            _ => unimplemented!("unsupported dtype, {}", to_store_type),
        };
        let indices = inplace_sub.to_store().to_load().expect("expected load").indices();
        let indices_val = self.visit_expr(&indices);
        let ptr = inplace_sub.to_store().to_load().expect("expected load").name();
        let ptr_val = self.bindings[&self.current_fn]
            .find_variable(&ptr.name)
            .unwrap()
            .to_ptr_value();
        let gep = self.builder.build_gep(
            dtype_to_llvm(to_store_type, &self.ctx),
            ptr_val,
            &[indices_val],
            "gep"
        );
        self.builder.build_store(gep, res)
    }

    fn visit_inplace_mul(&mut self, inplace_mul: &crate::halide::inplace_store_stmt::InplaceMul) {
        let to_store = self.visit_expr(&inplace_mul.to_store());
        let val = self.visit_expr(&inplace_mul.val());
        let to_store_type = self.bindings[&self.current_fn].find_type(&to_store).unwrap().dtype();
        let casted_val = build_cast(
            self.bindings[&self.current_fn].find_type(&val).unwrap().dtype(),
            to_store_type,
            val,
            "val_casted",
            &self.ctx,
            &self.builder
        );
        let res = match to_store_type {
            | Dtype::Bool
            | Dtype::I8
            | Dtype::U8
            | Dtype::I16
            | Dtype::U16
            | Dtype::I32
            | Dtype::U32
            | Dtype::I64
            | Dtype::U64
            | Dtype::Isize
            | Dtype::Usize => {
                self.builder.build_int_mul(to_store, casted_val, "add")
            }
            Dtype::BF16 | Dtype::F16 | Dtype::F32 | Dtype::F64 => {
                self.builder.build_float_mul(to_store, casted_val, "add")
            }
            _ => unimplemented!("unsupported dtype, {}", to_store_type),
        };
        let indices = inplace_mul.to_store().to_load().expect("expected load").indices();
        let indices_val = self.visit_expr(&indices);
        let ptr = inplace_mul.to_store().to_load().expect("expected load").name();
        let ptr_val = self.bindings[&self.current_fn]
            .find_variable(&ptr.name)
            .unwrap()
            .to_ptr_value();
        let gep = self.builder.build_gep(
            dtype_to_llvm(to_store_type, &self.ctx),
            ptr_val,
            &[indices_val],
            "gep"
        );
        self.builder.build_store(gep, res)
    }

    fn visit_inplace_div(&mut self, inplace_div: &crate::halide::inplace_store_stmt::InplaceDiv) {
        let to_store = self.visit_expr(&inplace_div.to_store());
        let val = self.visit_expr(&inplace_div.val());
        let to_store_type = self.bindings[&self.current_fn].find_type(&to_store).unwrap().dtype();
        let casted_val = build_cast(
            self.bindings[&self.current_fn].find_type(&val).unwrap().dtype(),
            to_store_type,
            val,
            "val_casted",
            &self.ctx,
            &self.builder
        );
        let res = match to_store_type {
            Dtype::BF16 | Dtype::F16 | Dtype::F32 | Dtype::F64 => {
                self.builder.build_float_div(to_store, casted_val, "add")
            }
            _ => unimplemented!("unsupported dtype, {}", to_store_type),
        };
        let indices = inplace_div.to_store().to_load().expect("expected load").indices();
        let indices_val = self.visit_expr(&indices);
        let ptr = inplace_div.to_store().to_load().expect("expected load").name();
        let ptr_val = self.bindings[&self.current_fn]
            .find_variable(&ptr.name)
            .unwrap()
            .to_ptr_value();
        let gep = self.builder.build_gep(
            dtype_to_llvm(to_store_type, &self.ctx),
            ptr_val,
            &[indices_val],
            "gep"
        );
        self.builder.build_store(gep, res)
    }

    fn visit_function(&mut self, function: &crate::halide::module::Function) {
        let id = self.fns_id[&self.fns[&function.name]];
        assert!(self.bindings.get(&id).is_none());
        let mut scope_stack = ScopeStack::new();
        let block = self.ctx.append_basic_block(&self.fns[&function.name], "entry");
        self.builder.position_at_end(block);
        for (args_idx, (arg_name, arg_ty)) in function.ty.args.iter().enumerate() {
            let arg = self.fns[&function.name].get_nth_param(args_idx);
            let val = arg.inner() as usize;
            scope_stack.insert_variable(&Arc::new(arg_name.clone()), arg);
            scope_stack.insert_type(arg, arg_ty.clone());
        }
        self.bindings.insert(id, scope_stack);
        self.current_fn = id;
        self.visit_stmt(&function.body);
        self.builder.position_at_end(block);
    }

    fn visit_shl(&mut self, shl: &crate::halide::exprs::Shl) -> BasicValue {
        todo!()
    }

    fn visit_shr(&mut self, shr: &crate::halide::exprs::Shr) -> BasicValue {
        todo!()
    }

    fn visit_layout(&mut self, layout: &crate::halide::exprs::Layout) -> BasicValue {
        todo!()
    }

    fn visit_malloc(&mut self, malloc: &crate::halide::exprs::Malloc) -> BasicValue {
        todo!()
    }

    fn visit_alloca(&mut self, alloca: &AllocaStmt) {
        let var = alloca.var();
        self.bindings.get_mut(&self.current_fn).expect("fn not find").insert_mutable(&var.name);
        let alloca_ty = PrimitiveType::Ptr(Ptr {
            inner: Arc::new(alloca.dtype().clone()),
        });
        let llvm_ty = primitive_ty_to_llvm(&self.ctx, &alloca_ty, self.tensor_type);
        let ptr = self.builder.build_alloca(llvm_ty, &var.name);
        self.bindings
            .get_mut(&self.current_fn)
            .expect("fn not find")
            .insert_variable(&var.name, ptr.into());
        self.bindings
            .get_mut(&self.current_fn)
            .expect("fn not find")
            .insert_type(ptr.into(), alloca_ty);
        self.visit_stmt(&alloca.body())
    }

    fn visit_tensor_load(&mut self, tensor_load: &TensorLoad) -> BasicValue {
        let one = PrimeExpr::Int(Int::make(Dtype::I64, 1i64));
        let zero = PrimeExpr::Int(Int::make(Dtype::I64, 0i64));
        assert!(tensor_load.begins.len() == tensor_load.axes.len());
        assert!(tensor_load.begins.len() == tensor_load.steps.len());
        assert!(tensor_load.begins.len() == tensor_load.strides.len());
        let mut indices = PrimeExpr::None;
        for (begin, axes, step, stride) in izip!(
            tensor_load.begins.iter(),
            tensor_load.axes.iter(),
            tensor_load.steps.iter(),
            tensor_load.strides.iter()
        ) {
            if begin == &zero && step == &one {
                if indices.is_none() {
                    indices = axes.clone() * stride.clone();
                } else {
                    indices = indices + axes.clone() * stride.clone();
                }
            } else if step == &one {
                if indices.is_none() {
                    indices = (axes.clone() + begin.clone()) * stride.clone();
                } else {
                    indices = indices + (axes.clone() + begin.clone()) * stride.clone();
                }
            } else {
                if indices.is_none() {
                    indices = (axes.clone() * step.clone() + begin.clone()) * stride.clone();
                } else {
                    indices =
                        indices + (axes.clone() * step.clone() + begin.clone()) * stride.clone();
                }
            }
        }
        let load = Load::make(tensor_load.var.as_ref(), indices);
        self.visit_load(&load)
    }

    fn visit_neg(&mut self, neg: &crate::halide::exprs::Neg) -> BasicValue {
        todo!()
    }
}

pub fn load(
    primitive_type: &PrimitiveType,
    to_load: BasicValue,
    indices: BasicValue,
    builder: &Builder,
    module: &tensor_llvm::module::module::Module,
    tensor_type: StructValue,
    ctx: &Context
) -> (BasicValue, PrimitiveType) {
    match primitive_type {
        PrimitiveType::Ptr(ptr) => {
            let pointing_ty = &ptr.inner;
            match pointing_ty.as_ref() {
                PrimitiveType::Dtype(val) => {
                    match val {
                        Dtype::Bool => {
                            let gep = builder.build_gep(
                                ctx.bool_type(),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.bool_type(), gep, "load"),
                                PrimitiveType::Dtype(Dtype::Bool),
                            )
                        }
                        Dtype::I8 => {
                            let gep = builder.build_gep(
                                ctx.i8_type(),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.i8_type(), gep, "load"),
                                PrimitiveType::Dtype(Dtype::I8),
                            )
                        }
                        Dtype::U8 => {
                            let gep = builder.build_gep(
                                ctx.u8_type(),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.u8_type(), gep, "load"),
                                PrimitiveType::Dtype(Dtype::U8),
                            )
                        }
                        Dtype::I16 => {
                            let gep = builder.build_gep(
                                ctx.i16_type(),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.i16_type(), gep, "load"),
                                PrimitiveType::Dtype(Dtype::I16),
                            )
                        }
                        Dtype::U16 => {
                            let gep = builder.build_gep(
                                ctx.u16_type(),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.u16_type(), gep, "load"),
                                PrimitiveType::Dtype(Dtype::U16),
                            )
                        }
                        Dtype::I32 => {
                            let gep = builder.build_gep(
                                ctx.i32_type(),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.i32_type(), gep, "load"),
                                PrimitiveType::Dtype(Dtype::I32),
                            )
                        }
                        Dtype::U32 => {
                            let gep = builder.build_gep(
                                ctx.u32_type(),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.u32_type(), gep, "load"),
                                PrimitiveType::Dtype(Dtype::U32),
                            )
                        }
                        Dtype::I64 => {
                            let gep = builder.build_gep(
                                ctx.i64_type(),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.i64_type(), gep, "load"),
                                PrimitiveType::Dtype(Dtype::I64),
                            )
                        }
                        Dtype::U64 => {
                            let gep = builder.build_gep(
                                ctx.u64_type(),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.u64_type(), gep, "load"),
                                PrimitiveType::Dtype(Dtype::U64),
                            )
                        }
                        Dtype::BF16 => {
                            let gep = builder.build_gep(
                                ctx.i16_type(),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.i16_type(), gep, "load"),
                                PrimitiveType::Dtype(Dtype::BF16),
                            )
                        }
                        Dtype::F16 => {
                            let gep = builder.build_gep(
                                ctx.f16_type(),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.f16_type(), gep, "load"),
                                PrimitiveType::Dtype(Dtype::F16),
                            )
                        }
                        Dtype::F32 => {
                            let gep = builder.build_gep(
                                ctx.f32_type(),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.f32_type(), gep, "load"),
                                PrimitiveType::Dtype(Dtype::F32),
                            )
                        }
                        Dtype::F64 => {
                            let gep = builder.build_gep(
                                ctx.f64_type(),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.f64_type(), gep, "load"),
                                PrimitiveType::Dtype(Dtype::F64),
                            )
                        }
                        Dtype::C32 => todo!(),
                        Dtype::C64 => todo!(),
                        Dtype::Isize => {
                            let gep = builder.build_gep(
                                ctx.isize_type(),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.isize_type(), gep, "load"),
                                PrimitiveType::Dtype(Dtype::Isize),
                            )
                        }
                        Dtype::Usize => todo!(),
                    }
                }
                PrimitiveType::Tuple(_) => todo!(),
                PrimitiveType::Array(arr) => {
                    match arr.inner.as_ref() {
                        PrimitiveType::Dtype(dtype) => {
                            macro_rules! impl_array_load {
                                ($dtype:ident) => {
                                    (
                                        builder.build_load(
                                        ctx.$dtype(), 
                                        builder.build_gep(
                                            ctx.$dtype(),
                                            to_load.into(),
                                            &[indices],
                                            "load"),
                                        "load"),
                                    PrimitiveType::Dtype(*dtype)
                                    )
                                };
                            }
                            match dtype {
                                Dtype::Bool => { impl_array_load!(bool_type) }
                                Dtype::I8 => { impl_array_load!(i8_type) }
                                Dtype::U8 => { impl_array_load!(u8_type) }
                                Dtype::I16 => { impl_array_load!(i16_type) }
                                Dtype::U16 => { impl_array_load!(u16_type) }
                                Dtype::I32 => { impl_array_load!(i32_type) }
                                Dtype::U32 => { impl_array_load!(u32_type) }
                                Dtype::I64 => { impl_array_load!(i64_type) }
                                Dtype::U64 => { impl_array_load!(u64_type) }
                                Dtype::BF16 => todo!(),
                                Dtype::F16 => { impl_array_load!(f16_type) }
                                Dtype::F32 => { impl_array_load!(f32_type) }
                                Dtype::F64 => { impl_array_load!(f64_type) }
                                Dtype::C32 => todo!(),
                                Dtype::C64 => todo!(),
                                Dtype::Isize => { impl_array_load!(isize_type) }
                                Dtype::Usize => todo!(),
                            }
                        }
                        PrimitiveType::Tuple(_) => todo!(),
                        PrimitiveType::Array(_) => todo!(),
                        PrimitiveType::Ptr(_) => todo!(),
                        PrimitiveType::Tensor(_) => todo!(),
                        PrimitiveType::Str => todo!(),
                        PrimitiveType::Void => todo!(),
                    }
                }
                PrimitiveType::Ptr(ptr) => {
                    match ptr.inner.as_ref() {
                        PrimitiveType::Dtype(dtype) => {
                            match dtype {
                                Dtype::Bool => {
                                    let gep = builder.build_gep(
                                        ctx.bool_type().ptr_type(0),
                                        to_load.to_ptr_value(),
                                        &[indices],
                                        "gep"
                                    );
                                    (
                                        builder.build_load(
                                            ctx.bool_type().ptr_type(0),
                                            gep,
                                            "load"
                                        ),
                                        PrimitiveType::Ptr(Ptr {
                                            inner: Arc::new(PrimitiveType::Dtype(Dtype::Bool)),
                                        }),
                                    )
                                }
                                Dtype::I8 => {
                                    let gep = builder.build_gep(
                                        ctx.i8_type().ptr_type(0),
                                        to_load.to_ptr_value(),
                                        &[indices],
                                        "gep"
                                    );
                                    (
                                        builder.build_load(ctx.i8_type().ptr_type(0), gep, "load"),
                                        PrimitiveType::Ptr(Ptr {
                                            inner: Arc::new(PrimitiveType::Dtype(Dtype::I8)),
                                        }),
                                    )
                                }
                                Dtype::U8 => {
                                    let gep = builder.build_gep(
                                        ctx.u8_type().ptr_type(0),
                                        to_load.to_ptr_value(),
                                        &[indices],
                                        "gep"
                                    );
                                    (
                                        builder.build_load(ctx.u8_type().ptr_type(0), gep, "load"),
                                        PrimitiveType::Ptr(Ptr {
                                            inner: Arc::new(PrimitiveType::Dtype(Dtype::U8)),
                                        }),
                                    )
                                }
                                Dtype::I16 => {
                                    let gep = builder.build_gep(
                                        ctx.i16_type().ptr_type(0),
                                        to_load.to_ptr_value(),
                                        &[indices],
                                        "gep"
                                    );
                                    (
                                        builder.build_load(ctx.i16_type().ptr_type(0), gep, "load"),
                                        PrimitiveType::Ptr(Ptr {
                                            inner: Arc::new(PrimitiveType::Dtype(Dtype::I16)),
                                        }),
                                    )
                                }
                                Dtype::U16 => {
                                    let gep = builder.build_gep(
                                        ctx.u16_type().ptr_type(0),
                                        to_load.to_ptr_value(),
                                        &[indices],
                                        "gep"
                                    );
                                    (
                                        builder.build_load(ctx.u16_type().ptr_type(0), gep, "load"),
                                        PrimitiveType::Ptr(Ptr {
                                            inner: Arc::new(PrimitiveType::Dtype(Dtype::U16)),
                                        }),
                                    )
                                }
                                Dtype::I32 => {
                                    let gep = builder.build_gep(
                                        ctx.i32_type().ptr_type(0),
                                        to_load.to_ptr_value(),
                                        &[indices],
                                        "gep"
                                    );
                                    (
                                        builder.build_load(ctx.i32_type().ptr_type(0), gep, "load"),
                                        PrimitiveType::Ptr(Ptr {
                                            inner: Arc::new(PrimitiveType::Dtype(Dtype::I32)),
                                        }),
                                    )
                                }
                                Dtype::U32 => {
                                    let gep = builder.build_gep(
                                        ctx.u32_type().ptr_type(0),
                                        to_load.to_ptr_value(),
                                        &[indices],
                                        "gep"
                                    );
                                    (
                                        builder.build_load(ctx.u32_type().ptr_type(0), gep, "load"),
                                        PrimitiveType::Ptr(Ptr {
                                            inner: Arc::new(PrimitiveType::Dtype(Dtype::U32)),
                                        }),
                                    )
                                }
                                Dtype::I64 => {
                                    let gep = builder.build_gep(
                                        ctx.i64_type().ptr_type(0),
                                        to_load.to_ptr_value(),
                                        &[indices],
                                        "gep"
                                    );
                                    (
                                        builder.build_load(ctx.i64_type().ptr_type(0), gep, "load"),
                                        PrimitiveType::Ptr(Ptr {
                                            inner: Arc::new(PrimitiveType::Dtype(Dtype::I64)),
                                        }),
                                    )
                                }
                                Dtype::U64 => {
                                    let gep = builder.build_gep(
                                        ctx.u64_type().ptr_type(0),
                                        to_load.to_ptr_value(),
                                        &[indices],
                                        "gep"
                                    );
                                    (
                                        builder.build_load(ctx.u64_type().ptr_type(0), gep, "load"),
                                        PrimitiveType::Ptr(Ptr {
                                            inner: Arc::new(PrimitiveType::Dtype(Dtype::U64)),
                                        }),
                                    )
                                }
                                Dtype::BF16 => {
                                    let gep = builder.build_gep(
                                        ctx.i16_type().ptr_type(0),
                                        to_load.to_ptr_value(),
                                        &[indices],
                                        "gep"
                                    );
                                    (
                                        builder.build_load(ctx.i16_type().ptr_type(0), gep, "load"),
                                        PrimitiveType::Ptr(Ptr {
                                            inner: Arc::new(PrimitiveType::Dtype(Dtype::BF16)),
                                        }),
                                    )
                                }
                                Dtype::F16 => {
                                    let gep = builder.build_gep(
                                        ctx.f16_type().ptr_type(0),
                                        to_load.to_ptr_value(),
                                        &[indices],
                                        "gep"
                                    );
                                    (
                                        builder.build_load(ctx.f16_type().ptr_type(0), gep, "load"),
                                        PrimitiveType::Ptr(Ptr {
                                            inner: Arc::new(PrimitiveType::Dtype(Dtype::F32)),
                                        }),
                                    )
                                }
                                Dtype::F32 => {
                                    let gep = builder.build_gep(
                                        ctx.f32_type().ptr_type(0),
                                        to_load.to_ptr_value(),
                                        &[indices],
                                        "gep"
                                    );
                                    (
                                        builder.build_load(ctx.f32_type().ptr_type(0), gep, "load"),
                                        PrimitiveType::Ptr(Ptr {
                                            inner: Arc::new(PrimitiveType::Dtype(Dtype::F32)),
                                        }),
                                    )
                                }
                                Dtype::F64 => {
                                    let gep = builder.build_gep(
                                        ctx.f64_type().ptr_type(0),
                                        to_load.to_ptr_value(),
                                        &[indices],
                                        "gep"
                                    );
                                    (
                                        builder.build_load(ctx.f64_type().ptr_type(0), gep, "load"),
                                        PrimitiveType::Ptr(Ptr {
                                            inner: Arc::new(PrimitiveType::Dtype(Dtype::F64)),
                                        }),
                                    )
                                }
                                Dtype::C32 => todo!(),
                                Dtype::C64 => todo!(),
                                Dtype::Isize => {
                                    let gep = builder.build_gep(
                                        ctx.isize_type().ptr_type(0),
                                        to_load.to_ptr_value(),
                                        &[indices],
                                        "gep"
                                    );
                                    (
                                        builder.build_load(
                                            ctx.isize_type().ptr_type(0),
                                            gep,
                                            "load"
                                        ),
                                        PrimitiveType::Ptr(Ptr {
                                            inner: Arc::new(PrimitiveType::Dtype(Dtype::Isize)),
                                        }),
                                    )
                                }
                                Dtype::Usize => todo!(),
                            }
                        }
                        PrimitiveType::Tuple(tuple) => {
                            let types = tuple.inner
                                .iter()
                                .map(|x| x.to_llvm_type(ctx, tensor_type))
                                .collect::<Vec<_>>();
                            let gep = builder.build_gep(
                                ctx.struct_type(&types, false),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.struct_type(&types, false), gep, "load"),
                                PrimitiveType::Tuple(tuple.clone()),
                            )
                        }
                        PrimitiveType::Array(_) => todo!(),
                        PrimitiveType::Ptr(ptr) => {
                            let gep = builder.build_gep(
                                ctx.void_type().ptr_type(0),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.void_type().ptr_type(0), gep, "load"),
                                PrimitiveType::Ptr(Ptr {
                                    inner: Arc::new(PrimitiveType::Ptr(ptr.clone())),
                                }),
                            )
                        }
                        PrimitiveType::Tensor(_) => todo!(),
                        PrimitiveType::Str => {
                            let gep = builder.build_gep(
                                ctx.str_ptr_type(),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.str_ptr_type(), gep, "load"),
                                PrimitiveType::Ptr(Ptr {
                                    inner: Arc::new(PrimitiveType::Str),
                                }),
                            )
                        }
                        PrimitiveType::Void => {
                            let gep = builder.build_gep(
                                ctx.void_type().ptr_type(0),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.void_type().ptr_type(0), gep, "load"),
                                PrimitiveType::Ptr(Ptr {
                                    inner: Arc::new(PrimitiveType::Void),
                                }),
                            )
                        }
                    }
                }
                PrimitiveType::Str => {
                    (
                        builder.build_load(ctx.str_ptr_type(), to_load.to_ptr_value(), "load"),
                        PrimitiveType::Str,
                    )
                }
                PrimitiveType::Void => {
                    (
                        builder.build_load(
                            ctx.void_type().ptr_type(0),
                            to_load.to_ptr_value(),
                            "load"
                        ),
                        PrimitiveType::Void,
                    )
                }
                PrimitiveType::Tensor(tensor) => {
                    match tensor.dtype {
                        Dtype::Bool => {
                            let gep = builder.build_gep(
                                ctx.bool_type().ptr_type(0),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.bool_type(), gep, "load"),
                                PrimitiveType::Dtype(Dtype::Bool),
                            )
                        }
                        Dtype::I8 => {
                            let gep = builder.build_gep(
                                ctx.i8_type().ptr_type(0),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.i8_type(), gep, "load"),
                                PrimitiveType::Dtype(Dtype::I8),
                            )
                        }
                        Dtype::U8 => {
                            let gep = builder.build_gep(
                                ctx.u8_type().ptr_type(0),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.u8_type(), gep, "load"),
                                PrimitiveType::Dtype(Dtype::U8),
                            )
                        }
                        Dtype::I16 => {
                            let gep = builder.build_gep(
                                ctx.i16_type().ptr_type(0),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.i16_type(), gep, "load"),
                                PrimitiveType::Dtype(Dtype::I16),
                            )
                        }
                        Dtype::U16 => {
                            let gep = builder.build_gep(
                                ctx.u16_type().ptr_type(0),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.u16_type(), gep, "load"),
                                PrimitiveType::Dtype(Dtype::U16),
                            )
                        }
                        Dtype::I32 => {
                            let gep = builder.build_gep(
                                ctx.i32_type().ptr_type(0),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.i32_type(), gep, "load"),
                                PrimitiveType::Dtype(Dtype::I32),
                            )
                        }
                        Dtype::U32 => {
                            let gep = builder.build_gep(
                                ctx.u32_type().ptr_type(0),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.u32_type(), gep, "load"),
                                PrimitiveType::Dtype(Dtype::U32),
                            )
                        }
                        Dtype::I64 => {
                            let gep = builder.build_gep(
                                ctx.i64_type().ptr_type(0),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.i64_type(), gep, "load"),
                                PrimitiveType::Dtype(Dtype::I64),
                            )
                        }
                        Dtype::U64 => {
                            let gep = builder.build_gep(
                                ctx.u64_type().ptr_type(0),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.u64_type(), gep, "load"),
                                PrimitiveType::Dtype(Dtype::U64),
                            )
                        }
                        Dtype::BF16 => {
                            let gep = builder.build_gep(
                                ctx.i16_type().ptr_type(0),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.i16_type(), gep, "load"),
                                PrimitiveType::Dtype(Dtype::BF16),
                            )
                        }
                        Dtype::F16 => {
                            let gep = builder.build_gep(
                                ctx.f16_type().ptr_type(0),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.f16_type(), gep, "load"),
                                PrimitiveType::Dtype(Dtype::F16),
                            )
                        }
                        Dtype::F32 => {
                            let gep = builder.build_gep(
                                ctx.f32_type().ptr_type(0),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.f32_type(), gep, "load"),
                                PrimitiveType::Dtype(Dtype::F32),
                            )
                        }
                        Dtype::F64 => {
                            let gep = builder.build_gep(
                                ctx.f64_type().ptr_type(0),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.f64_type(), gep, "load"),
                                PrimitiveType::Dtype(Dtype::F64),
                            )
                        }
                        Dtype::C32 => todo!(),
                        Dtype::C64 => todo!(),
                        Dtype::Isize => {
                            let gep = builder.build_gep(
                                ctx.isize_type().ptr_type(0),
                                to_load.to_ptr_value(),
                                &[indices],
                                "gep"
                            );
                            (
                                builder.build_load(ctx.isize_type(), gep, "load"),
                                PrimitiveType::Dtype(Dtype::Isize),
                            )
                        }
                        Dtype::Usize => todo!(),
                    }
                }
            }
        }
        _ => unimplemented!("unsupported type, {}", primitive_type),
    }
}
