use hashbrown::{ HashMap, HashSet };
use llvm_sys::{ LLVMIntPredicate, LLVMRealPredicate };
use tensor_llvm::{
    builder::builder::Builder,
    context::context::Context,
    module::module::Module,
    types::values::{ BasicValue, FunctionValue },
};
use tensor_types::{ dtype::Dtype, type_promote::{ BitWiseOut, FloatOut, NormalOut } };

use crate::{
    halide::{
        assign_stmt::AssignStmt,
        exprs::{ Call, Str },
        if_stmt::IfThenElse,
        inplace_store_stmt::{ InplaceAdd, InplaceDiv, InplaceMul, InplaceStore, InplaceSub },
        prime_expr::PrimeExpr,
        stmt::Stmt,
        variable::Variable,
    },
    hlir::tensor_slice::TensorSlice,
};

pub struct CodeGen {
    ctx: Context,
    module: Module,
    builder: Builder,
    values: HashMap<String, BasicValue>,
    ptrs: HashMap<Variable, BasicValue>,
    casted_values: HashSet<BasicValue>,
    current_var: String,
    functions: HashMap<Call, FunctionValue>,
    bindings: HashMap<Variable, BasicValue>,
    current_func: FunctionValue,
}

impl CodeGen {
    fn visit_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::LetStmt(let_stmt) => self.visit_let_stmt(&let_stmt),
            Stmt::For(for_stmt) => self.visit_for(&for_stmt),
            Stmt::StoreStmt(store) => self.visit_store(&store),
            Stmt::Seq(stmts) => {
                for stmt in stmts.stmts() {
                    self.visit_stmt(stmt);
                }
            }
            Stmt::IfThenElse(if_then_else) => self.visit_if_then_else(&if_then_else),
            Stmt::InplaceStore(inplace_store) => self.visit_inplace_store(&inplace_store),
            Stmt::InplaceAdd(inplace_add) => self.visit_inplace_add(&inplace_add),
            Stmt::InplaceSub(inplace_sub) => self.visit_inplace_sub(&inplace_sub),
            Stmt::InplaceMul(inplace_mul) => self.visit_inplace_mul(&inplace_mul),
            Stmt::InplaceDiv(inplace_div) => self.visit_inplace_div(&inplace_div),
            Stmt::AssignStmt(assign) => self.visit_assign(&assign),
            Stmt::None => {}
        }
    }
    fn visit_let_stmt(&mut self, let_stmt: &crate::halide::let_stmt::LetStmt) {
        let var = let_stmt.var().name();
        self.current_var = var.to_string();
        let value = self.visit_expr(let_stmt.value());
        self.bindings.insert(let_stmt.var().clone(), value);
        self.visit_stmt(let_stmt.body());
    }
    fn visit_for(&mut self, for_stmt: &crate::halide::for_stmt::For) {
        let start = self.visit_expr(for_stmt.start());
        let end = self.visit_expr(for_stmt.end());
        let step = self.visit_expr(for_stmt.step());
        let var = for_stmt.var().name();
        let start = start.cast(Dtype::I64, "", &self.ctx, &self.builder);
        let end = end.cast(Dtype::I64, "", &self.ctx, &self.builder);
        let step = step.cast(Dtype::I64, "", &self.ctx, &self.builder);
        let pre_header = self.ctx.append_basic_block(&self.current_func, "preheader");
        let loop_block = self.ctx.append_basic_block(&self.current_func, "loop");
        let after_block = self.ctx.append_basic_block(&self.current_func, "afterloop");
        self.builder.build_unconditional_branch(pre_header);
        self.builder.position_at_end(pre_header);
        let phi = self.builder.build_phi(self.ctx.i64_type().into(), "indvar");
        phi.add_incoming(&[(&start, pre_header)]);
        let loop_cond = self.builder.build_int_cmp(
            LLVMIntPredicate::LLVMIntSLT,
            phi.into(),
            end,
            "loopcond"
        );
        self.builder.build_conditional_branch(loop_cond, loop_block, after_block);
        self.builder.position_at_end(loop_block);
        let body = for_stmt.stmt();
        self.visit_stmt(body);
        self.values.insert(var.to_string(), phi.into());
        self.visit_stmt(for_stmt.stmt());
        let next_var = self.builder.build_int_add(phi.into(), step, "nextvar");
        phi.add_incoming(&[(&next_var, loop_block)]);
        self.builder.build_unconditional_branch(pre_header);
        self.builder.position_at_end(after_block);
    }
    fn visit_store(&mut self, store: &crate::halide::store_stmt::StoreStmt) {
        let var = store.var().name();
        let val = self.visit_expr(store.val());
        let indices = store.indices();
        let var_ptr = self.ptrs[store.var()].clone();
    }
    fn visit_if_then_else(&mut self, if_then_else: &IfThenElse) {
        let cond = self.visit_expr(if_then_else.cond());
        let cond = cond.cast(Dtype::Bool, "", &self.ctx, &self.builder);
        let then_block = self.ctx.append_basic_block(&self.current_func, "then");
        let else_block = self.ctx.append_basic_block(&self.current_func, "else");
        let after_block = self.ctx.append_basic_block(&self.current_func, "after");
        self.builder.build_conditional_branch(cond, then_block, else_block);
        self.builder.position_at_end(then_block);
        self.visit_stmt(if_then_else.then_case());
        self.builder.build_unconditional_branch(after_block);
        self.builder.position_at_end(else_block);
        self.visit_stmt(if_then_else.else_case());
        self.builder.build_unconditional_branch(after_block);
        self.builder.position_at_end(after_block);
    }
    fn visit_inplace_add(&mut self, inplace_add: &InplaceAdd) {}
    fn visit_inplace_div(&mut self, inplace_div: &InplaceDiv) {}
    fn visit_inplace_mul(&mut self, inplace_mul: &InplaceMul) {}
    fn visit_inplace_sub(&mut self, inplace_sub: &InplaceSub) {}
    fn visit_inplace_store(&mut self, inplace_store: &InplaceStore) {}
    fn visit_assign(&mut self, assign: &AssignStmt) {}
    fn visit_expr(&mut self, expr: &PrimeExpr) -> BasicValue {
        match expr {
            PrimeExpr::Int(int) => self.visit_int(&int),
            PrimeExpr::Float(float) => self.visit_float(&float),
            PrimeExpr::UInt(uint) => self.visit_uint(&uint),
            PrimeExpr::Str(string) => self.visit_str(&string),
            PrimeExpr::Cast(cast) => self.visit_cast(&cast),
            PrimeExpr::Add(add) => self.visit_add(&add),
            PrimeExpr::Sub(sub) => self.visit_sub(&sub),
            PrimeExpr::Mul(mul) => self.visit_mul(&mul),
            PrimeExpr::Div(div) => self.visit_div(&div),
            PrimeExpr::FloorDiv(floor_div) => self.visit_floor_div(&floor_div),
            PrimeExpr::Mod(r#mod) => self.visit_mod(&r#mod),
            PrimeExpr::Min(min) => self.visit_min(&min),
            PrimeExpr::Max(max) => self.visit_max(&max),
            PrimeExpr::Eq(eq) => self.visit_eq(&eq),
            PrimeExpr::Ne(ne) => self.visit_ne(&ne),
            PrimeExpr::Lt(lt) => self.visit_lt(&lt),
            PrimeExpr::Le(le) => self.visit_le(&le),
            PrimeExpr::Gt(gt) => self.visit_gt(&gt),
            PrimeExpr::Ge(ge) => self.visit_ge(&ge),
            PrimeExpr::And(and) => self.visit_and(&and),
            PrimeExpr::Xor(or) => self.visit_xor(&or),
            PrimeExpr::Or(or) => self.visit_or(&or),
            PrimeExpr::Not(not) => self.visit_not(&not),
            PrimeExpr::Call(call) => self.visit_call(&call),
            PrimeExpr::Select(select) => self.visit_select(&select),
            PrimeExpr::Load(load) => self.visit_load(&load),
            PrimeExpr::Let(let_) => self.visit_let(&let_),
            PrimeExpr::Reduce(reduce) => self.visit_reduce(&reduce),
            PrimeExpr::TensorSlice(slice) => self.visit_tensor_slice(&slice),
            PrimeExpr::None => { BasicValue::None }
            _ => unreachable!("unsupported expr, {:?}", expr),
        }
    }
    fn visit_int(&mut self, int: &crate::halide::exprs::Int) -> BasicValue {
        match int.dtype() {
            tensor_types::dtype::Dtype::I8 => {
                let i8_type = self.ctx.i8_type();
                let i8_val = i8_type.const_int(int.value() as u64, false);
                i8_val.into()
            }
            tensor_types::dtype::Dtype::I16 => {
                let i16_type = self.ctx.i16_type();
                let i16_val = i16_type.const_int(int.value() as u64, false);
                i16_val.into()
            }
            tensor_types::dtype::Dtype::I32 => {
                let i32_type = self.ctx.i32_type();
                let i32_val = i32_type.const_int(int.value() as u64, false);
                i32_val.into()
            }
            tensor_types::dtype::Dtype::I64 => {
                let i64_type = self.ctx.i64_type();
                let i64_val = i64_type.const_int(int.value() as u64, false);
                i64_val.into()
            }
            tensor_types::dtype::Dtype::Isize => {
                let isize_type = self.ctx.isize_type();
                let isize_val = isize_type.const_int(int.value() as u64, false);
                isize_val.into()
            }
            _ => unreachable!("unsupported dtype, {}", int.dtype()),
        }
    }
    fn visit_float(&mut self, float: &crate::halide::exprs::Float) -> BasicValue {
        match float.dtype() {
            tensor_types::dtype::Dtype::F32 => {
                let f32_type = self.ctx.f32_type();
                let f32_val = f32_type.const_float(float.value());
                f32_val.into()
            }
            tensor_types::dtype::Dtype::F64 => {
                let f64_type = self.ctx.f64_type();
                let f64_val = f64_type.const_float(float.value());
                f64_val.into()
            }
            tensor_types::dtype::Dtype::F16 => {
                let f16_type = self.ctx.f16_type();
                let f16_val = f16_type.const_float(float.value());
                f16_val.into()
            }
            _ => unreachable!("unsupported dtype, {}", float.dtype()),
        }
    }
    fn visit_uint(&mut self, uint: &crate::halide::exprs::UInt) -> BasicValue {
        match uint.dtype() {
            tensor_types::dtype::Dtype::U8 => {
                let u8_type = self.ctx.u8_type();
                let u8_val = u8_type.const_int(uint.value(), false);
                u8_val.into()
            }
            tensor_types::dtype::Dtype::U16 => {
                let u16_type = self.ctx.u16_type();
                let u16_val = u16_type.const_int(uint.value(), false);
                u16_val.into()
            }
            tensor_types::dtype::Dtype::U32 => {
                let u32_type = self.ctx.u32_type();
                let u32_val = u32_type.const_int(uint.value(), false);
                u32_val.into()
            }
            tensor_types::dtype::Dtype::U64 => {
                let u64_type = self.ctx.u64_type();
                let u64_val = u64_type.const_int(uint.value(), false);
                u64_val.into()
            }
            _ => unreachable!("unsupported dtype, {}", uint.dtype()),
        }
    }
    fn visit_add(&mut self, add: &crate::halide::exprs::Add) -> BasicValue {
        let lhs = self.visit_expr(&add.e1_());
        let rhs = self.visit_expr(&add.e2_());
        let res_type = lhs.to_dtype()._add(rhs.to_dtype());
        let casted_lhs = lhs.cast(res_type, "", &self.ctx, &self.builder);
        let casted_rhs = rhs.cast(res_type, "", &self.ctx, &self.builder);
        self.casted_values.insert(casted_lhs);
        self.casted_values.insert(casted_rhs);
        match res_type {
            | Dtype::Bool
            | Dtype::I8
            | Dtype::U8
            | Dtype::I16
            | Dtype::U16
            | Dtype::I32
            | Dtype::U32
            | Dtype::U64
            | Dtype::I64
            | Dtype::Isize
            | Dtype::Usize => self.builder.build_int_add(casted_lhs, casted_rhs, &self.current_var),
            Dtype::F16 | Dtype::F32 | Dtype::F64 =>
                self.builder.build_float_add(casted_lhs, casted_rhs, &self.current_var),
            _ => unreachable!("unsupported dtype, {}", res_type),
        }
    }
    fn visit_sub(&mut self, sub: &crate::halide::exprs::Sub) -> BasicValue {
        let lhs = self.visit_expr(&sub.e1_());
        let rhs = self.visit_expr(&sub.e2_());
        let res_type = lhs.to_dtype()._sub(rhs.to_dtype());
        let casted_lhs = lhs.cast(res_type, "", &self.ctx, &self.builder);
        let casted_rhs = rhs.cast(res_type, "", &self.ctx, &self.builder);
        self.casted_values.insert(casted_lhs);
        self.casted_values.insert(casted_rhs);
        match res_type {
            | Dtype::Bool
            | Dtype::I8
            | Dtype::U8
            | Dtype::I16
            | Dtype::U16
            | Dtype::I32
            | Dtype::U32
            | Dtype::U64
            | Dtype::I64
            | Dtype::Isize
            | Dtype::Usize => self.builder.build_int_sub(casted_lhs, casted_rhs, &self.current_var),
            Dtype::F16 | Dtype::F32 | Dtype::F64 =>
                self.builder.build_float_sub(casted_lhs, casted_rhs, &self.current_var),
            _ => unreachable!("unsupported dtype, {}", res_type),
        }
    }
    fn visit_mul(&mut self, mul: &crate::halide::exprs::Mul) -> BasicValue {
        let lhs = self.visit_expr(&mul.e1_());
        let rhs = self.visit_expr(&mul.e2_());
        let res_type = lhs.to_dtype()._mul(rhs.to_dtype());
        let casted_lhs = lhs.cast(res_type, "", &self.ctx, &self.builder);
        let casted_rhs = rhs.cast(res_type, "", &self.ctx, &self.builder);
        self.casted_values.insert(casted_lhs);
        self.casted_values.insert(casted_rhs);
        match res_type {
            | Dtype::Bool
            | Dtype::I8
            | Dtype::U8
            | Dtype::I16
            | Dtype::U16
            | Dtype::I32
            | Dtype::U32
            | Dtype::U64
            | Dtype::I64
            | Dtype::Isize
            | Dtype::Usize => self.builder.build_int_mul(casted_lhs, casted_rhs, &self.current_var),
            Dtype::F16 | Dtype::F32 | Dtype::F64 =>
                self.builder.build_float_mul(casted_lhs, casted_rhs, &self.current_var),
            _ => unreachable!("unsupported dtype, {}", res_type),
        }
    }
    fn visit_div(&mut self, div: &crate::halide::exprs::Div) -> BasicValue {
        let lhs = self.visit_expr(&div.e1_());
        let rhs = self.visit_expr(&div.e2_());
        let res_type = lhs.to_dtype()._div(rhs.to_dtype());
        let casted_lhs = lhs.cast(res_type, "", &self.ctx, &self.builder);
        let casted_rhs = rhs.cast(res_type, "", &self.ctx, &self.builder);
        self.casted_values.insert(casted_lhs);
        self.casted_values.insert(casted_rhs);
        match res_type {
            Dtype::F16 | Dtype::F32 | Dtype::F64 =>
                self.builder.build_float_div(casted_lhs, casted_rhs, &self.current_var),
            _ => unreachable!("unsupported dtype, {}", res_type),
        }
    }
    fn visit_floor_div(&mut self, floor_div: &crate::halide::exprs::FloorDiv) -> BasicValue {
        let lhs = self.visit_expr(&floor_div.e1_());
        let rhs = self.visit_expr(&floor_div.e2_());
        todo!()
    }
    fn visit_mod(&mut self, r#mod: &crate::halide::exprs::Mod) -> BasicValue {
        todo!()
    }
    fn visit_min(&mut self, min: &crate::halide::exprs::Min) -> BasicValue {
        let lhs = self.visit_expr(&min.e1_());
        let rhs = self.visit_expr(&min.e2_());
        let res_type = lhs.to_dtype()._add(rhs.to_dtype());
        let casted_lhs = lhs.cast(res_type, "", &self.ctx, &self.builder);
        let casted_rhs = rhs.cast(res_type, "", &self.ctx, &self.builder);
        self.casted_values.insert(casted_lhs);
        self.casted_values.insert(casted_rhs);
        let cond = match res_type {
            Dtype::I8 | Dtype::I16 | Dtype::I32 | Dtype::I64 | Dtype::Isize =>
                self.builder.build_int_cmp(
                    LLVMIntPredicate::LLVMIntSLT,
                    casted_lhs,
                    casted_rhs,
                    &self.current_var
                ),
            Dtype::U8 | Dtype::U16 | Dtype::U32 | Dtype::U64 | Dtype::Usize =>
                self.builder.build_int_cmp(
                    LLVMIntPredicate::LLVMIntULT,
                    casted_lhs,
                    casted_rhs,
                    &self.current_var
                ),
            Dtype::F16 | Dtype::F32 | Dtype::F64 =>
                self.builder.build_float_cmp(
                    LLVMRealPredicate::LLVMRealULT,
                    casted_lhs,
                    casted_rhs,
                    &self.current_var
                ),
            _ => unreachable!("unsupported dtype, {}", res_type),
        };
        self.builder.build_select(cond, casted_lhs, casted_rhs, &self.current_var)
    }
    fn visit_max(&mut self, max: &crate::halide::exprs::Max) -> BasicValue {
        let lhs = self.visit_expr(&max.e1_());
        let rhs = self.visit_expr(&max.e2_());
        let res_type = lhs.to_dtype()._add(rhs.to_dtype());
        let casted_lhs = lhs.cast(res_type, "", &self.ctx, &self.builder);
        let casted_rhs = rhs.cast(res_type, "", &self.ctx, &self.builder);
        self.casted_values.insert(casted_lhs);
        self.casted_values.insert(casted_rhs);
        let cond = match res_type {
            Dtype::I8 | Dtype::I16 | Dtype::I32 | Dtype::I64 | Dtype::Isize =>
                self.builder.build_int_cmp(
                    LLVMIntPredicate::LLVMIntSGT,
                    casted_lhs,
                    casted_rhs,
                    &self.current_var
                ),
            Dtype::U8 | Dtype::U16 | Dtype::U32 | Dtype::U64 | Dtype::Usize =>
                self.builder.build_int_cmp(
                    LLVMIntPredicate::LLVMIntUGT,
                    casted_lhs,
                    casted_rhs,
                    &self.current_var
                ),
            Dtype::F16 | Dtype::F32 | Dtype::F64 =>
                self.builder.build_float_cmp(
                    LLVMRealPredicate::LLVMRealUGT,
                    casted_lhs,
                    casted_rhs,
                    &self.current_var
                ),
            _ => unreachable!("unsupported dtype, {}", res_type),
        };
        self.builder.build_select(cond, casted_lhs, casted_rhs, &self.current_var)
    }

    fn visit_eq(&mut self, eq: &crate::halide::exprs::Eq) -> BasicValue {
        let lhs = self.visit_expr(&eq.e1_());
        let rhs = self.visit_expr(&eq.e2_());
        let res_type = lhs.to_dtype()._add(rhs.to_dtype());
        let casted_lhs = lhs.cast(res_type, "", &self.ctx, &self.builder);
        let casted_rhs = rhs.cast(res_type, "", &self.ctx, &self.builder);
        self.casted_values.insert(casted_lhs);
        self.casted_values.insert(casted_rhs);
        match res_type {
            Dtype::I8 | Dtype::I16 | Dtype::I32 | Dtype::I64 | Dtype::Isize =>
                self.builder.build_int_cmp(
                    LLVMIntPredicate::LLVMIntEQ,
                    casted_lhs,
                    casted_rhs,
                    &self.current_var
                ),
            Dtype::U8 | Dtype::U16 | Dtype::U32 | Dtype::U64 | Dtype::Usize =>
                self.builder.build_int_cmp(
                    LLVMIntPredicate::LLVMIntEQ,
                    casted_lhs,
                    casted_rhs,
                    &self.current_var
                ),
            Dtype::F16 | Dtype::F32 | Dtype::F64 =>
                self.builder.build_float_cmp(
                    LLVMRealPredicate::LLVMRealOEQ,
                    casted_lhs,
                    casted_rhs,
                    &self.current_var
                ),
            _ => unreachable!("unsupported dtype, {}", res_type),
        }
    }
    fn visit_ne(&mut self, ne: &crate::halide::exprs::Ne) -> BasicValue {
        let lhs = self.visit_expr(&ne.e1_());
        let rhs = self.visit_expr(&ne.e2_());
        let res_type = lhs.to_dtype()._add(rhs.to_dtype());
        let casted_lhs = lhs.cast(res_type, "", &self.ctx, &self.builder);
        let casted_rhs = rhs.cast(res_type, "", &self.ctx, &self.builder);
        self.casted_values.insert(casted_lhs);
        self.casted_values.insert(casted_rhs);
        match res_type {
            | Dtype::I8
            | Dtype::I16
            | Dtype::I32
            | Dtype::I64
            | Dtype::Isize
            | Dtype::U8
            | Dtype::U16
            | Dtype::U32
            | Dtype::U64
            | Dtype::Usize =>
                self.builder.build_int_cmp(
                    LLVMIntPredicate::LLVMIntNE,
                    casted_lhs,
                    casted_rhs,
                    &self.current_var
                ),
            Dtype::F16 | Dtype::F32 | Dtype::F64 =>
                self.builder.build_float_cmp(
                    LLVMRealPredicate::LLVMRealUNE,
                    casted_lhs,
                    casted_rhs,
                    &self.current_var
                ),
            _ => unreachable!("unsupported dtype, {}", res_type),
        }
    }
    fn visit_lt(&mut self, lt: &crate::halide::exprs::Lt) -> BasicValue {
        let lhs = self.visit_expr(&lt.e1_());
        let rhs = self.visit_expr(&lt.e2_());
        let res_type = lhs.to_dtype()._add(rhs.to_dtype());
        let casted_lhs = lhs.cast(res_type, "", &self.ctx, &self.builder);
        let casted_rhs = rhs.cast(res_type, "", &self.ctx, &self.builder);
        self.casted_values.insert(casted_lhs);
        self.casted_values.insert(casted_rhs);
        match res_type {
            Dtype::I8 | Dtype::I16 | Dtype::I32 | Dtype::I64 | Dtype::Isize =>
                self.builder.build_int_cmp(
                    LLVMIntPredicate::LLVMIntSLT,
                    casted_lhs,
                    casted_rhs,
                    &self.current_var
                ),
            Dtype::U8 | Dtype::U16 | Dtype::U32 | Dtype::U64 | Dtype::Usize =>
                self.builder.build_int_cmp(
                    LLVMIntPredicate::LLVMIntULT,
                    casted_lhs,
                    casted_rhs,
                    &self.current_var
                ),
            Dtype::F16 | Dtype::F32 | Dtype::F64 =>
                self.builder.build_float_cmp(
                    LLVMRealPredicate::LLVMRealULT,
                    casted_lhs,
                    casted_rhs,
                    &self.current_var
                ),
            _ => unreachable!("unsupported dtype, {}", res_type),
        }
    }
    fn visit_le(&mut self, le: &crate::halide::exprs::Le) -> BasicValue {
        let lhs = self.visit_expr(&le.e1_());
        let rhs = self.visit_expr(&le.e2_());
        let res_type = lhs.to_dtype()._add(rhs.to_dtype());
        let casted_lhs = lhs.cast(res_type, "", &self.ctx, &self.builder);
        let casted_rhs = rhs.cast(res_type, "", &self.ctx, &self.builder);
        self.casted_values.insert(casted_lhs);
        self.casted_values.insert(casted_rhs);
        match res_type {
            Dtype::I8 | Dtype::I16 | Dtype::I32 | Dtype::I64 | Dtype::Isize =>
                self.builder.build_int_cmp(
                    LLVMIntPredicate::LLVMIntSLE,
                    casted_lhs,
                    casted_rhs,
                    &self.current_var
                ),
            Dtype::U8 | Dtype::U16 | Dtype::U32 | Dtype::U64 | Dtype::Usize =>
                self.builder.build_int_cmp(
                    LLVMIntPredicate::LLVMIntULE,
                    casted_lhs,
                    casted_rhs,
                    &self.current_var
                ),
            Dtype::F16 | Dtype::F32 | Dtype::F64 =>
                self.builder.build_float_cmp(
                    LLVMRealPredicate::LLVMRealULE,
                    casted_lhs,
                    casted_rhs,
                    &self.current_var
                ),
            _ => unreachable!("unsupported dtype, {}", res_type),
        }
    }
    fn visit_gt(&mut self, gt: &crate::halide::exprs::Gt) -> BasicValue {
        let lhs = self.visit_expr(&gt.e1_());
        let rhs = self.visit_expr(&gt.e2_());
        let res_type = lhs.to_dtype()._add(rhs.to_dtype());
        let casted_lhs = lhs.cast(res_type, "", &self.ctx, &self.builder);
        let casted_rhs = rhs.cast(res_type, "", &self.ctx, &self.builder);
        self.casted_values.insert(casted_lhs);
        self.casted_values.insert(casted_rhs);
        match res_type {
            Dtype::I8 | Dtype::I16 | Dtype::I32 | Dtype::I64 | Dtype::Isize =>
                self.builder.build_int_cmp(
                    LLVMIntPredicate::LLVMIntSGT,
                    casted_lhs,
                    casted_rhs,
                    &self.current_var
                ),
            Dtype::U8 | Dtype::U16 | Dtype::U32 | Dtype::U64 | Dtype::Usize =>
                self.builder.build_int_cmp(
                    LLVMIntPredicate::LLVMIntUGT,
                    casted_lhs,
                    casted_rhs,
                    &self.current_var
                ),
            Dtype::F16 | Dtype::F32 | Dtype::F64 =>
                self.builder.build_float_cmp(
                    LLVMRealPredicate::LLVMRealUGT,
                    casted_lhs,
                    casted_rhs,
                    &self.current_var
                ),
            _ => unreachable!("unsupported dtype, {}", res_type),
        }
    }
    fn visit_ge(&mut self, ge: &crate::halide::exprs::Ge) -> BasicValue {
        let lhs = self.visit_expr(&ge.e1_());
        let rhs = self.visit_expr(&ge.e2_());
        let res_type = lhs.to_dtype()._add(rhs.to_dtype());
        let casted_lhs = lhs.cast(res_type, "", &self.ctx, &self.builder);
        let casted_rhs = rhs.cast(res_type, "", &self.ctx, &self.builder);
        self.casted_values.insert(casted_lhs);
        self.casted_values.insert(casted_rhs);
        match res_type {
            Dtype::I8 | Dtype::I16 | Dtype::I32 | Dtype::I64 | Dtype::Isize =>
                self.builder.build_int_cmp(
                    LLVMIntPredicate::LLVMIntSGE,
                    casted_lhs,
                    casted_rhs,
                    &self.current_var
                ),
            Dtype::U8 | Dtype::U16 | Dtype::U32 | Dtype::U64 | Dtype::Usize =>
                self.builder.build_int_cmp(
                    LLVMIntPredicate::LLVMIntUGE,
                    casted_lhs,
                    casted_rhs,
                    &self.current_var
                ),
            Dtype::F16 | Dtype::F32 | Dtype::F64 =>
                self.builder.build_float_cmp(
                    LLVMRealPredicate::LLVMRealUGE,
                    casted_lhs,
                    casted_rhs,
                    &self.current_var
                ),
            _ => unreachable!("unsupported dtype, {}", res_type),
        }
    }
    fn visit_and(&mut self, and: &crate::halide::exprs::And) -> BasicValue {
        let lhs = self.visit_expr(&and.e1_());
        let rhs = self.visit_expr(&and.e2_());
        let res_type = lhs.to_dtype()._and(rhs.to_dtype());
        let casted_lhs = lhs.cast(res_type, "", &self.ctx, &self.builder);
        let casted_rhs = rhs.cast(res_type, "", &self.ctx, &self.builder);
        self.casted_values.insert(casted_lhs);
        self.casted_values.insert(casted_rhs);
        match res_type {
            | Dtype::Bool
            | Dtype::I8
            | Dtype::U8
            | Dtype::I16
            | Dtype::U16
            | Dtype::I32
            | Dtype::U32
            | Dtype::U64
            | Dtype::I64
            | Dtype::Isize
            | Dtype::Usize => self.builder.build_int_and(casted_lhs, casted_rhs, &self.current_var),
            _ => unreachable!("unsupported dtype, {}", res_type),
        }
    }
    fn visit_xor(&mut self, xor: &crate::halide::exprs::Xor) -> BasicValue {
        let lhs = self.visit_expr(&xor.e1_());
        let rhs = self.visit_expr(&xor.e2_());
        let res_type = lhs.to_dtype()._xor(rhs.to_dtype());
        let casted_lhs = lhs.cast(res_type, "", &self.ctx, &self.builder);
        let casted_rhs = rhs.cast(res_type, "", &self.ctx, &self.builder);
        self.casted_values.insert(casted_lhs);
        self.casted_values.insert(casted_rhs);
        match res_type {
            | Dtype::Bool
            | Dtype::I8
            | Dtype::U8
            | Dtype::I16
            | Dtype::U16
            | Dtype::I32
            | Dtype::U32
            | Dtype::U64
            | Dtype::I64
            | Dtype::Isize
            | Dtype::Usize => self.builder.build_int_xor(casted_lhs, casted_rhs, &self.current_var),
            _ => unreachable!("unsupported dtype, {}", res_type),
        }
    }
    fn visit_or(&mut self, or: &crate::halide::exprs::Or) -> BasicValue {
        let lhs = self.visit_expr(&or.e1_());
        let rhs = self.visit_expr(&or.e2_());
        let res_type = lhs.to_dtype()._or(rhs.to_dtype());
        let casted_lhs = lhs.cast(res_type, "", &self.ctx, &self.builder);
        let casted_rhs = rhs.cast(res_type, "", &self.ctx, &self.builder);
        self.casted_values.insert(casted_lhs);
        self.casted_values.insert(casted_rhs);
        match res_type {
            | Dtype::Bool
            | Dtype::I8
            | Dtype::U8
            | Dtype::I16
            | Dtype::U16
            | Dtype::I32
            | Dtype::U32
            | Dtype::U64
            | Dtype::I64
            | Dtype::Isize
            | Dtype::Usize => self.builder.build_int_or(casted_lhs, casted_rhs, &self.current_var),
            _ => unreachable!("unsupported dtype, {}", res_type),
        }
    }
    fn visit_not(&mut self, not: &crate::halide::exprs::Not) -> BasicValue {
        let val = self.visit_expr(&not.e_());
        let res_type = val.to_dtype()._not();
        let casted_val = val.cast(res_type, "", &self.ctx, &self.builder);
        self.casted_values.insert(casted_val);
        match res_type {
            | Dtype::Bool
            | Dtype::I8
            | Dtype::U8
            | Dtype::I16
            | Dtype::U16
            | Dtype::I32
            | Dtype::U32
            | Dtype::U64
            | Dtype::I64
            | Dtype::Isize
            | Dtype::Usize => self.builder.build_int_not(casted_val, &self.current_var),
            _ => unreachable!("unsupported dtype, {}", res_type),
        }
    }
    fn visit_call(&mut self, call: &crate::halide::exprs::Call) -> BasicValue {
        if let Some(fn_value) = self.functions.get(call).cloned() {
            let args = call
                .args()
                .iter()
                .map(|arg| self.visit_expr(arg))
                .collect::<Vec<_>>();
            self.builder.build_call(&fn_value, &args, &self.current_var)
        } else {
            todo!()
        }
    }
    fn visit_select(&mut self, select: &crate::halide::exprs::Select) -> BasicValue {
        let cond = self.visit_expr(&select.cond());
        let true_val = self.visit_expr(&select.true_expr());
        let false_val = self.visit_expr(&select.false_expr());
        let res_type = true_val.to_dtype()._add(false_val.to_dtype());
        let casted_cond = cond.cast(res_type, "", &self.ctx, &self.builder);
        let casted_true_val = true_val.cast(res_type, "", &self.ctx, &self.builder);
        let casted_false_val = false_val.cast(res_type, "", &self.ctx, &self.builder);
        self.casted_values.insert(casted_cond);
        self.casted_values.insert(casted_true_val);
        self.casted_values.insert(casted_false_val);
        match res_type {
            | Dtype::Bool
            | Dtype::I8
            | Dtype::U8
            | Dtype::I16
            | Dtype::U16
            | Dtype::I32
            | Dtype::U32
            | Dtype::U64
            | Dtype::I64
            | Dtype::Isize
            | Dtype::Usize
            | Dtype::F16
            | Dtype::F32
            | Dtype::F64 =>
                self.builder.build_select(
                    casted_cond,
                    casted_true_val,
                    casted_false_val,
                    &self.current_var
                ),
            _ => unreachable!("unsupported dtype, {}", res_type),
        }
    }
    fn visit_load(&mut self, load: &crate::halide::exprs::Load) -> BasicValue {
        todo!()
    }
    fn visit_let(&mut self, let_: &crate::halide::exprs::Let) -> BasicValue {
        let var = let_.name();
        let val = self.visit_expr(let_.value());
        self.bindings.insert(var.clone(), val);
        self.visit_expr(let_.body());
        val
    }
    fn visit_reduce(&mut self, reduce: &crate::halide::exprs::Reduce) -> BasicValue {
        todo!()
    }
    fn visit_tensor_slice(&mut self, slice: &TensorSlice) -> BasicValue {
        todo!()
    }
    fn visit_cast(&mut self, cast: &crate::halide::exprs::Cast) -> BasicValue {
        todo!()
    }
    fn visit_str(&mut self, string: &Str) -> BasicValue {
        todo!()
    }
}
