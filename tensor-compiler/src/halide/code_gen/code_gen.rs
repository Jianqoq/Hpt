use std::sync::Arc;

use hashbrown::HashMap;
use tensor_llvm::{
    builder::builder::Builder,
    context::context::Context,
    types::values::{ BasicValue, FunctionValue },
};

use crate::halide::{
    module::Module,
    primitive_type::PrimitiveType,
    traits::{ AccepterMut, CodeGenVisitor },
};

pub struct CodeGen {
    ctx: Context,
    module: tensor_llvm::module::module::Module,
    builder: Builder,
    fns: HashMap<Arc<String>, FunctionValue>,
    fns_id: HashMap<FunctionValue, usize>,
    bindings: HashMap<usize, HashMap<Arc<String>, BasicValue>>,
    type_bindings: HashMap<usize, HashMap<BasicValue, PrimitiveType>>,
    current_fn: usize,
}

impl CodeGen {
    pub fn new(ctx: Context, module: &Module) -> Self {
        let _module = tensor_llvm::module::module::Module::new(&module.name, &ctx);
        let builder = Builder::new(&ctx);
        let mut fns = HashMap::new();
        let mut fns_id = HashMap::new();
        let mut cnt = 0;
        for func in module.fns.values() {
            let fn_val = _module.add_function(func.ty.to_llvm_func_type(&ctx), &func.name);
            fns.insert(func.name.clone(), fn_val.clone());
            fns_id.insert(fn_val, cnt);
            cnt += 1;
        }
        CodeGen {
            ctx,
            module: _module,
            builder,
            fns,
            bindings: HashMap::new(),
            type_bindings: HashMap::new(),
            current_fn: 0,
            fns_id,
        }
    }
}

impl CodeGenVisitor for CodeGen {
    fn visit_return(&mut self, return_: &crate::halide::return_stmt::ReturnStmt) {
        todo!()
    }

    fn visit_tensor_slice(&mut self, slice: &crate::hlir::tensor_slice::TensorSlice) -> BasicValue {
        todo!()
    }

    fn visit_reduce(&mut self, reduce: &crate::halide::exprs::Reduce) -> BasicValue {
        todo!()
    }

    fn visit_variable(&mut self, var: &crate::halide::variable::Variable) -> BasicValue {
        todo!()
    }

    fn visit_str(&mut self, string: &crate::halide::exprs::Str) -> BasicValue {
        todo!()
    }

    fn visit_assign(&mut self, assign: &crate::halide::assign_stmt::AssignStmt) {
        todo!()
    }

    fn visit_cast(&mut self, cast: &crate::halide::exprs::Cast) -> BasicValue {
        todo!()
    }

    fn visit_add(&mut self, add: &crate::halide::exprs::Add) -> BasicValue {
        todo!()
    }

    fn visit_sub(&mut self, sub: &crate::halide::exprs::Sub) -> BasicValue {
        todo!()
    }

    fn visit_mul(&mut self, mul: &crate::halide::exprs::Mul) -> BasicValue {
        todo!()
    }

    fn visit_div(&mut self, div: &crate::halide::exprs::Div) -> BasicValue {
        todo!()
    }

    fn visit_floor_div(&mut self, floor_div: &crate::halide::exprs::FloorDiv) -> BasicValue {
        todo!()
    }

    fn visit_mod(&mut self, mod_: &crate::halide::exprs::Mod) -> BasicValue {
        todo!()
    }

    fn visit_min(&mut self, min: &crate::halide::exprs::Min) -> BasicValue {
        todo!()
    }

    fn visit_max(&mut self, max: &crate::halide::exprs::Max) -> BasicValue {
        todo!()
    }

    fn visit_ge(&mut self, ge: &crate::halide::exprs::Ge) -> BasicValue {
        todo!()
    }

    fn visit_gt(&mut self, gt: &crate::halide::exprs::Gt) -> BasicValue {
        todo!()
    }

    fn visit_le(&mut self, le: &crate::halide::exprs::Le) -> BasicValue {
        todo!()
    }

    fn visit_lt(&mut self, lt: &crate::halide::exprs::Lt) -> BasicValue {
        todo!()
    }

    fn visit_eq(&mut self, eq: &crate::halide::exprs::Eq) -> BasicValue {
        todo!()
    }

    fn visit_ne(&mut self, ne: &crate::halide::exprs::Ne) -> BasicValue {
        todo!()
    }

    fn visit_and(&mut self, and: &crate::halide::exprs::And) -> BasicValue {
        todo!()
    }

    fn visit_xor(&mut self, xor: &crate::halide::exprs::Xor) -> BasicValue {
        todo!()
    }

    fn visit_or(&mut self, or: &crate::halide::exprs::Or) -> BasicValue {
        todo!()
    }

    fn visit_not(&mut self, not: &crate::halide::exprs::Not) -> BasicValue {
        todo!()
    }

    fn visit_let_stmt(&mut self, let_stmt: &crate::halide::let_stmt::LetStmt) {
        todo!()
    }

    fn visit_let(&mut self, let_: &crate::halide::exprs::Let) -> BasicValue {
        let var = let_.name();
        self.visit_expr(&let_.value_());
        todo!()
    }

    fn visit_for(&mut self, for_stmt: &crate::halide::for_stmt::For) {
        todo!()
    }

    fn visit_int(&mut self, int: &crate::halide::exprs::Int) -> BasicValue {
        todo!()
    }

    fn visit_uint(&mut self, uint: &crate::halide::exprs::UInt) -> BasicValue {
        todo!()
    }

    fn visit_float(&mut self, float: &crate::halide::exprs::Float) -> BasicValue {
        todo!()
    }

    fn visit_call(&mut self, call: &crate::halide::exprs::Call) -> BasicValue {
        todo!()
    }

    fn visit_select(&mut self, select: &crate::halide::exprs::Select) -> BasicValue {
        todo!()
    }

    fn visit_load(&mut self, load: &crate::halide::exprs::Load) -> BasicValue {
        todo!()
    }

    fn visit_store(&mut self, store: &crate::halide::store_stmt::StoreStmt) {
        todo!()
    }

    fn visit_if_then_else(&mut self, if_then_else: &crate::halide::if_stmt::IfThenElse) {
        todo!()
    }

    fn visit_inplace_store(
        &mut self,
        inplace_store: &crate::halide::inplace_store_stmt::InplaceStore
    ) {
        todo!()
    }

    fn visit_inplace_add(&mut self, inplace_add: &crate::halide::inplace_store_stmt::InplaceAdd) {
        todo!()
    }

    fn visit_inplace_sub(&mut self, inplace_sub: &crate::halide::inplace_store_stmt::InplaceSub) {
        todo!()
    }

    fn visit_inplace_mul(&mut self, inplace_mul: &crate::halide::inplace_store_stmt::InplaceMul) {
        todo!()
    }

    fn visit_inplace_div(&mut self, inplace_div: &crate::halide::inplace_store_stmt::InplaceDiv) {
        todo!()
    }

    fn visit_function(&mut self, function: &crate::halide::module::Function) -> BasicValue {
        let id = self.fns_id[&self.fns[&function.name]];
        assert!(self.bindings.get(&id).is_none());
        let mut type_bindings = HashMap::new();
        let mut bindings = HashMap::new();
        for (idx, (name, arg_ty)) in function.ty.args.iter().flatten().enumerate() {
            type_bindings.insert(self.fns[&function.name].get_nth_param(idx), arg_ty.clone());
            bindings.insert(Arc::new(name.clone()), self.fns[&function.name].get_nth_param(idx));
        }
        self.bindings.insert(id, bindings);
        self.type_bindings.insert(id, type_bindings);
        self.current_fn = id;
        let block = self.ctx.append_basic_block(&self.fns[&function.name], "entry");
        self.builder.position_at_end(block);
        self.visit_stmt(&function.body);
        self.builder.position_at_end(block);

        match &function.ty.ret_ty {
            PrimitiveType::Dtype(_) | PrimitiveType::Array(_) | PrimitiveType::Ptr(_) => {
                assert!(function.ty.args[2].len() == 1);
                let arg = self.bindings[&id][&function.ty.args[2][0].0];
                self.builder.build_return(Some(arg));
                arg
            }
            PrimitiveType::Tuple(tuple) => {
                let args = function.ty.args[2]
                    .iter()
                    .map(|(name, _)| self.bindings[&id][name])
                    .collect::<Vec<_>>();
                let struct_ty = tuple.to_llvm_type(&self.ctx);
                let tuple = BasicValue::Struct(struct_ty.const_named_struct(&args));
                self.builder.build_return(Some(tuple));
                tuple
            }
            PrimitiveType::Str => todo!(),
            PrimitiveType::Void => todo!(),
        }
    }
}
