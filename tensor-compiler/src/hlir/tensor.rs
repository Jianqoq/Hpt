#![allow(unused_imports)]
use std::{ fmt::Display, sync::Arc };

use half::{ bf16, f16 };
use hashbrown::HashMap;
use tensor_common::{ axis::{ process_axes, Axis }, shape::Shape };
use tensor_types::dtype::Dtype;

use crate::{
    halide::{
        assign_stmt::AssignStmt,
        exprs::{ Add, Float, Gt, Int, Load, Lt, Max, Min, Mul, UInt },
        if_stmt::IfThenElse,
        inplace_store_stmt::InplaceAdd,
        let_stmt::LetStmt,
        loop_utils::{ build_nested::build_nested_for, reduction::* },
        prime_expr::PrimeExpr,
        seq_stmt::Seq,
        stmt::Stmt,
        store_stmt::StoreStmt,
        variable::Variable,
    },
    iter_val::IterVar,
};
use tensor_types::dtype::TypeCommon;

pub fn dtype_inf(dtype: Dtype) -> PrimeExpr {
    match dtype {
        Dtype::Bool => PrimeExpr::Int(Int::make(Dtype::Bool, 1)),
        Dtype::I8 => PrimeExpr::Int(Int::make(Dtype::I8, i8::MAX as i64)),
        Dtype::U8 => PrimeExpr::UInt(UInt::make(Dtype::U8, u8::MAX as u64)),
        Dtype::I16 => PrimeExpr::Int(Int::make(Dtype::I64, i16::MAX as i64)),
        Dtype::U16 => PrimeExpr::UInt(UInt::make(Dtype::U16, u16::MAX as u64)),
        Dtype::I32 => PrimeExpr::Int(Int::make(Dtype::I32, i32::MAX as i64)),
        Dtype::U32 => PrimeExpr::UInt(UInt::make(Dtype::U32, u32::MAX as u64)),
        Dtype::I64 => PrimeExpr::Int(Int::make(Dtype::I64, i64::MAX)),
        Dtype::U64 => PrimeExpr::UInt(UInt::make(Dtype::U64, u64::MAX as u64)),
        Dtype::BF16 => PrimeExpr::Float(Float::make(Dtype::BF16, bf16::INFINITY.to_f64())),
        Dtype::F16 => PrimeExpr::Float(Float::make(Dtype::F16, f16::INFINITY.to_f64())),
        Dtype::F32 => PrimeExpr::Float(Float::make(Dtype::F32, f32::INFINITY as f64)),
        Dtype::F64 => PrimeExpr::Float(Float::make(Dtype::F64, f64::INFINITY)),
        Dtype::C32 => todo!(),
        Dtype::C64 => todo!(),
        Dtype::Isize => PrimeExpr::Int(Int::make(Dtype::Isize, isize::MAX as i64)),
        Dtype::Usize => PrimeExpr::UInt(UInt::make(Dtype::Usize, usize::MAX as u64)),
    }
}

pub fn dtype_neg_inf(dtype: Dtype) -> PrimeExpr {
    match dtype {
        Dtype::Bool => PrimeExpr::Int(Int::make(Dtype::Bool, 0)),
        Dtype::I8 => PrimeExpr::Int(Int::make(Dtype::I8, i8::MIN as i64)),
        Dtype::U8 => PrimeExpr::UInt(UInt::make(Dtype::U8, u8::MIN as u64)),
        Dtype::I16 => PrimeExpr::Int(Int::make(Dtype::I64, i16::MIN as i64)),
        Dtype::U16 => PrimeExpr::UInt(UInt::make(Dtype::U16, u16::MIN as u64)),
        Dtype::I32 => PrimeExpr::Int(Int::make(Dtype::I32, i32::MIN as i64)),
        Dtype::U32 => PrimeExpr::UInt(UInt::make(Dtype::U32, u32::MIN as u64)),
        Dtype::I64 => PrimeExpr::Int(Int::make(Dtype::I64, i64::MIN)),
        Dtype::U64 => PrimeExpr::UInt(UInt::make(Dtype::U64, u64::MIN as u64)),
        Dtype::BF16 => PrimeExpr::Float(Float::make(Dtype::BF16, bf16::NEG_INFINITY.to_f64())),
        Dtype::F16 => PrimeExpr::Float(Float::make(Dtype::F16, f16::NEG_INFINITY.to_f64())),
        Dtype::F32 => PrimeExpr::Float(Float::make(Dtype::F32, f32::NEG_INFINITY as f64)),
        Dtype::F64 => PrimeExpr::Float(Float::make(Dtype::F64, f64::NEG_INFINITY)),
        Dtype::C32 => todo!(),
        Dtype::C64 => todo!(),
        Dtype::Isize => PrimeExpr::Int(Int::make(Dtype::Isize, isize::MIN as i64)),
        Dtype::Usize => PrimeExpr::UInt(UInt::make(Dtype::Usize, usize::MIN as u64)),
    }
}

#[derive(Clone)]
pub struct Tensor {
    shape: Arc<Vec<IterVar>>,
    op: Arc<dyn Fn(Arc<Vec<Tensor>>, Vec<PrimeExpr>) -> PrimeExpr>,
    name: Arc<String>,
    inputs: Arc<Vec<Tensor>>,
    dtype: Dtype,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("name", &self.name)
            .field("inputs", &self.inputs)
            .finish()
    }
}

impl Tensor {
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    pub fn sum<T: Into<PrimeExpr> + Clone>(&self, init: T, axes: i64) -> Self {
        let axes: Vec<usize> = process_axes([axes], self.ndim()).unwrap();
        let axis = axes[0];
        let _a = self.clone();
        let res_shape = self.shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !axes.contains(i))
            .map(|(_, x)| x.clone())
            .collect::<Vec<_>>();
        let init: PrimeExpr = init.clone().into();
        _compute_known_iter(
            self.dtype,
            res_shape,
            vec![self],
            &format!("{}_red", self.name),
            move |inputs, indices| {
                let mut indices = indices
                    .into_iter()
                    .enumerate()
                    .filter(|(i, _)| !axes.contains(&i))
                    .map(|(_, x)| x)
                    .collect::<Vec<_>>();
                let var = Variable::new(format!("red_{}", inputs[0].name));
                indices.push(var.clone().into());
                let mut reduce_iter_var = inputs[0].shape[axis].clone();
                reduce_iter_var.set_var(var);
                sum([inputs[0].slice(indices)], [init.clone()], [reduce_iter_var])
            }
        )
    }
    pub fn prod<T: Into<PrimeExpr> + Clone>(&self, init: T, axes: i64) -> Self {
        let axes: Vec<usize> = process_axes([axes], self.ndim()).unwrap();
        let axis = axes[0];
        let _a = self.clone();
        let res_shape = self.shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !axes.contains(i))
            .map(|(_, x)| x.clone())
            .collect::<Vec<_>>();
        let init: PrimeExpr = init.clone().into();
        _compute_known_iter(
            self.dtype,
            res_shape,
            vec![self],
            &format!("{}_red", self.name),
            move |inputs, indices| {
                let mut indices = indices
                    .into_iter()
                    .enumerate()
                    .filter(|(i, _)| !axes.contains(&i))
                    .map(|(_, x)| x)
                    .collect::<Vec<_>>();
                let var = Variable::new(format!("red_{}", inputs[0].name));
                indices.push(var.clone().into());
                let mut reduce_iter_var = inputs[0].shape[axis].clone();
                reduce_iter_var.set_var(var);
                prod([inputs[0].slice(indices)], [init.clone()], [reduce_iter_var])
            }
        )
    }
    pub fn argmax<T: Into<PrimeExpr> + Clone + TypeCommon>(&self, init: T, axes: i64) -> Self {
        let axes: Vec<usize> = process_axes([axes], self.ndim()).unwrap();
        let axis = axes[0];
        let _a = self.clone();
        let res_shape = self.shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !axes.contains(i))
            .map(|(_, x)| x.clone())
            .collect::<Vec<_>>();
        let init: PrimeExpr = init.clone().into();
        _compute_known_iter(
            Dtype::I64,
            res_shape,
            vec![self],
            &format!("{}_red", self.name),
            move |inputs, indices| {
                let mut indices = indices
                    .into_iter()
                    .enumerate()
                    .filter(|(i, _)| !axes.contains(&i))
                    .map(|(_, x)| x)
                    .collect::<Vec<_>>();
                let var = Variable::new(format!("red_{}", inputs[0].name));
                indices.push(var.clone().into());
                let mut reduce_iter_var = inputs[0].shape[axis].clone();
                reduce_iter_var.set_var(var);
                argmax(
                    [inputs[0].slice(indices)],
                    [init.clone(), T::NEG_INF.into()],
                    [reduce_iter_var]
                )
            }
        )
    }
    pub fn argmin<T: Into<PrimeExpr> + Clone + TypeCommon>(&self, init: T, axes: i64) -> Self {
        let axes: Vec<usize> = process_axes([axes], self.ndim()).unwrap();
        let axis = axes[0];
        let _a = self.clone();
        let res_shape = self.shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !axes.contains(i))
            .map(|(_, x)| x.clone())
            .collect::<Vec<_>>();
        let init: PrimeExpr = init.clone().into();
        _compute_known_iter(
            Dtype::I64,
            res_shape,
            vec![self],
            &format!("{}_red", self.name),
            move |inputs, indices| {
                let mut indices = indices
                    .into_iter()
                    .enumerate()
                    .filter(|(i, _)| !axes.contains(&i))
                    .map(|(_, x)| x)
                    .collect::<Vec<_>>();
                let var = Variable::new(format!("red_{}", inputs[0].name));
                indices.push(var.clone().into());
                let mut reduce_iter_var = inputs[0].shape[axis].clone();
                reduce_iter_var.set_var(var);
                argmin([inputs[0].slice(indices)], [init.clone(), T::INF.into()], [reduce_iter_var])
            }
        )
    }
    pub fn max(&self, axes: i64) -> Self {
        let axes: Vec<usize> = process_axes([axes], self.ndim()).unwrap();
        let axis = axes[0];
        let _a = self.clone();
        let res_shape = self.shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !axes.contains(i))
            .map(|(_, x)| x.clone())
            .collect::<Vec<_>>();
        _compute_known_iter(
            self.dtype,
            res_shape,
            vec![self],
            &format!("{}_red", self.name),
            move |inputs, indices| {
                let mut indices = indices
                    .into_iter()
                    .enumerate()
                    .filter(|(i, _)| !axes.contains(&i))
                    .map(|(_, x)| x)
                    .collect::<Vec<_>>();
                let var = Variable::new(format!("red_{}", inputs[0].name));
                indices.push(var.clone().into());
                let mut reduce_iter_var = inputs[0].shape[axis].clone();
                reduce_iter_var.set_var(var);
                max([inputs[0].slice(indices)], [dtype_neg_inf(inputs[0].dtype)], [reduce_iter_var])
            }
        )
    }
    pub fn min(&self, axes: i64) -> Self {
        let axes: Vec<usize> = process_axes([axes], self.ndim()).unwrap();
        let axis = axes[0];
        let _a = self.clone();
        let res_shape = self.shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !axes.contains(i))
            .map(|(_, x)| x.clone())
            .collect::<Vec<_>>();
        _compute_known_iter(
            self.dtype,
            res_shape,
            vec![self],
            &format!("{}_red", self.name),
            move |inputs, indices| {
                let mut indices = indices
                    .into_iter()
                    .enumerate()
                    .filter(|(i, _)| !axes.contains(&i))
                    .map(|(_, x)| x)
                    .collect::<Vec<_>>();
                let var = Variable::new(format!("red_{}", inputs[0].name));
                indices.push(var.clone().into());
                let mut reduce_iter_var = inputs[0].shape[axis].clone();
                reduce_iter_var.set_var(var);
                min([inputs[0].slice(indices)], [dtype_inf(inputs[0].dtype)], [reduce_iter_var])
            }
        )
    }
}

impl Eq for Tensor {}

impl std::hash::Hash for Tensor {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.shape.hash(state);
        let ptr = Arc::as_ptr(&self.op);
        ptr.hash(state);
        self.name.hash(state);
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && Arc::ptr_eq(&self.op, &other.op) && self.name == other.name
    }
}

impl Tensor {
    pub fn placeholder<T: IntoIterator<Item: Into<PrimeExpr>>>(
        shape: T,
        dtype: Dtype,
        name: &str
    ) -> Self {
        let tensor_name = name.to_string();
        let iter_vars = shape
            .into_iter()
            .map(|x| x.into())
            .enumerate()
            .map(|(i, x)| {
                IterVar::new(
                    Int::make(Dtype::I64, 0),
                    x,
                    Int::make(Dtype::I64, 1),
                    Variable::new(format!("ax{}", i))
                )
            })
            .collect::<Vec<_>>();
        Self {
            shape: Arc::new(iter_vars),
            op: Arc::new(move |_, vec| {
                Load::make(
                    Variable::new(tensor_name.to_string()),
                    vec
                        .iter()
                        .map(|x| x.clone())
                        .reduce(|acc, x| acc + x)
                        .unwrap()
                ).into()
            }),
            name: name.to_string().into(),
            inputs: vec![].into(),
            dtype,
        }
    }
    pub fn slice<T: IntoIterator<Item: Into<PrimeExpr>>>(&self, indices: T) -> PrimeExpr {
        Load::make(
            Variable::make(&self.name),
            indices
                .into_iter()
                .map(|x| x.into())
                .reduce(|acc, x| acc + x)
                .unwrap()
        ).into()
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl Into<Tensor> for &Tensor {
    fn into(self) -> Tensor {
        self.clone()
    }
}

pub fn compute<
    const M: usize,
    const N: usize,
    F,
    T: Into<PrimeExpr> + Clone,
    A: Into<Tensor> + Clone
    >(dtype: Dtype, res_shape: [T; N], inputs: [A; M], name: &str, op: F) -> Tensor
    where F: Fn([Tensor; M], [PrimeExpr; N]) -> PrimeExpr + 'static
{
    let new_fn = move |inputs: Arc<Vec<Tensor>>, vec: Vec<PrimeExpr>| -> PrimeExpr {
        op(inputs.as_ref().clone().try_into().unwrap(), vec.try_into().unwrap())
    };
    let iter_vars = res_shape
        .iter()
        .enumerate()
        .map(|(i, x)| {
            IterVar::new(
                Int::make(Dtype::I64, 0),
                x.clone(),
                Int::make(Dtype::I64, 1),
                Variable::new(format!("ax{}", i))
            )
        })
        .collect::<Vec<_>>();
    let inputs = inputs
        .into_iter()
        .map(|x| x.into())
        .collect::<Vec<Tensor>>();
    Tensor {
        shape: Arc::new(iter_vars),
        op: Arc::new(new_fn),
        name: name.to_string().into(),
        inputs: inputs.into(),
        dtype,
    }
}

pub fn _compute<F, T: Into<PrimeExpr> + Clone, A: Into<Tensor> + Clone>(
    dtype: Dtype,
    res_shape: Vec<T>,
    inputs: Vec<A>,
    name: &str,
    op: F
) -> Tensor
    where F: Fn(Arc<Vec<Tensor>>, Vec<PrimeExpr>) -> PrimeExpr + 'static
{
    let iter_vars = res_shape
        .iter()
        .enumerate()
        .map(|(i, x)| {
            IterVar::new(
                Int::make(Dtype::I64, 0),
                x.clone(),
                Int::make(Dtype::I64, 1),
                Variable::new(format!("ax{}", i))
            )
        })
        .collect::<Vec<_>>();
    let inputs = inputs
        .into_iter()
        .map(|x| x.into())
        .collect::<Vec<Tensor>>();
    Tensor {
        shape: Arc::new(iter_vars),
        op: Arc::new(op),
        name: name.to_string().into(),
        inputs: inputs.into(),
        dtype,
    }
}

pub fn _compute_known_iter<F, A: Into<Tensor> + Clone>(
    dtype: Dtype,
    iter_vars: Vec<IterVar>,
    inputs: Vec<A>,
    name: &str,
    op: F
) -> Tensor
    where F: Fn(Arc<Vec<Tensor>>, Vec<PrimeExpr>) -> PrimeExpr + 'static
{
    let inputs = inputs
        .into_iter()
        .map(|x| x.into())
        .collect::<Vec<Tensor>>();
    Tensor {
        shape: Arc::new(iter_vars),
        op: Arc::new(op),
        name: name.to_string().into(),
        inputs: inputs.into(),
        dtype,
    }
}

pub struct Schedule {
    ops: HashMap<Tensor, Arc<dyn Fn(Arc<Vec<Tensor>>, Vec<PrimeExpr>) -> PrimeExpr>>,
}

impl Schedule {
    pub fn create(tensors: Vec<Tensor>) -> Self {
        let mut ops = HashMap::new();
        for tensor in tensors {
            let op = tensor.op.clone();
            ops.insert(tensor, op);
        }
        Self { ops }
    }
    pub fn lower(&self) -> Vec<Stmt> {
        let mut stmts = Vec::new();
        for (tensor, op) in &self.ops {
            let shape = tensor.shape.clone();
            let name = tensor.name.clone();
            let expr = op(
                tensor.inputs.clone(),
                shape
                    .iter()
                    .map(|x| x.var().clone().into())
                    .collect()
            );
            let mut main_stmt: Vec<Stmt> = vec![];
            match expr {
                PrimeExpr::Reduce(reduce) => {
                    let indices = &shape
                        .iter()
                        .map(|x| x.var().into())
                        .reduce(|acc: PrimeExpr, x| acc + x)
                        .unwrap();
                    let iter_vars = reduce.iter_vars();
                    let fors = match reduce.op() {
                        "sum" => {
                            assert!(reduce.identity().len() == 1);
                            assert!(reduce.expr().len() == 1);
                            let out_name = Variable::make(&format!("{}", name));
                            main_stmt.push(
                                StoreStmt::make(
                                    &out_name,
                                    indices,
                                    &reduce.identity()[0].clone()
                                ).into()
                            );
                            build_nested_for(
                                iter_vars,
                                InplaceAdd::make(Load::make(&out_name, indices), &reduce.expr()[0])
                            )
                        }
                        "min" => {
                            assert!(reduce.identity().len() == 1);
                            assert!(reduce.expr().len() == 1);
                            main_stmt.push(
                                StoreStmt::make(
                                    &Variable::make(&format!("{}", name)),
                                    indices,
                                    &reduce.identity()[0].clone()
                                ).into()
                            );
                            build_nested_for(
                                iter_vars,
                                StoreStmt::make(
                                    &Variable::make(&format!("{}", name)),
                                    indices,
                                    &Min::make(
                                        &Load::make(Variable::make(&format!("{}", name)), indices),
                                        &reduce.expr()[0]
                                    )
                                )
                            )
                        }
                        "max" => {
                            assert!(reduce.identity().len() == 1);
                            assert!(reduce.expr().len() == 1);
                            main_stmt.push(
                                StoreStmt::make(
                                    &Variable::make(&format!("{}", name)),
                                    indices,
                                    &reduce.identity()[0].clone()
                                ).into()
                            );
                            build_nested_for(
                                iter_vars,
                                StoreStmt::make(
                                    &Variable::make(&format!("{}", name)),
                                    indices,
                                    &Max::make(
                                        &Load::make(Variable::make(&format!("{}", name)), indices),
                                        &reduce.expr()[0]
                                    )
                                )
                            )
                        }
                        "prod" => {
                            assert!(reduce.identity().len() == 1);
                            assert!(reduce.expr().len() == 1);
                            main_stmt.push(
                                StoreStmt::make(
                                    &Variable::make(&format!("{}", name)),
                                    indices,
                                    &reduce.identity()[0].clone()
                                ).into()
                            );
                            build_nested_for(
                                iter_vars,
                                StoreStmt::make(
                                    &Variable::make(&format!("{}", name)),
                                    indices,
                                    &Mul::make(
                                        &Load::make(Variable::make(&format!("{}", name)), indices),
                                        &reduce.expr()[0]
                                    )
                                )
                            )
                        }
                        "argmin" => {
                            assert!(reduce.identity().len() == 2);
                            assert!(reduce.expr().len() == 1);
                            let idx = Variable::make(&format!("idx_{}", name));
                            main_stmt.push(LetStmt::make(&idx, Int::make(Dtype::I64, 0)).into());
                            let min_val = Variable::make(&format!("min_val_{}", name));
                            main_stmt.push(LetStmt::make(&min_val, &reduce.identity()[1]).into());
                            let name = Variable::make(&format!("{}", name));
                            main_stmt.push(
                                StoreStmt::make(&name, &indices, Int::make(Dtype::I64, 0)).into()
                            );
                            let mut body: Vec<Stmt> = vec![];
                            let cond = Lt::make(&reduce.expr()[0], &min_val);
                            let then_case = Seq::make([
                                Stmt::AssignStmt(AssignStmt::make(min_val, &reduce.expr()[0])),
                                Stmt::StoreStmt(StoreStmt::make(&name, &indices, &idx)),
                            ]);
                            body.push(IfThenElse::make(cond, then_case, Stmt::None).into());
                            body.push(InplaceAdd::make(&idx, Int::make(Dtype::I64, 1)).into());
                            build_nested_for(iter_vars, Seq::make(body))
                        }
                        "argmax" => {
                            assert!(reduce.identity().len() == 2);
                            assert!(reduce.expr().len() == 1);
                            let idx = Variable::make(&format!("idx_{}", name));
                            main_stmt.push(LetStmt::make(&idx, Int::make(Dtype::I64, 0)).into());
                            let max_val = Variable::make(&format!("max_val_{}", name));
                            main_stmt.push(LetStmt::make(&max_val, &reduce.identity()[1]).into());
                            let name = Variable::make(&format!("{}", name));
                            main_stmt.push(
                                StoreStmt::make(&name, &indices, Int::make(Dtype::I64, 0)).into()
                            );
                            let mut body: Vec<Stmt> = vec![];
                            let cond = Gt::make(&reduce.expr()[0], &max_val);
                            let then_case = Seq::make([
                                Stmt::AssignStmt(AssignStmt::make(max_val, &reduce.expr()[0])),
                                Stmt::StoreStmt(StoreStmt::make(&name, &indices, &idx)),
                            ]);
                            body.push(IfThenElse::make(cond, then_case, Stmt::None).into());
                            body.push(InplaceAdd::make(&idx, Int::make(Dtype::I64, 1)).into());
                            build_nested_for(iter_vars, Seq::make(body))
                        }
                        _ => todo!(),
                    };
                    main_stmt.push(fors);
                }
                _ => todo!(),
            }
            let loop_stmt = build_nested_for(&shape, Stmt::Seq(Seq::make(main_stmt)));
            stmts.push(loop_stmt);
        }
        stmts
    }
}

#[cfg(test)]
mod tests {
    use tensor_types::dtype::Dtype;

    use super::*;
    use crate::{
        halide::{
            exprs::{ Float, Int },
            loop_utils::reduction::{ argmax, argmin, max, min, sum },
            prime_expr::PrimeExpr,
            printer::IRPrinter,
        },
        hlir::traits::IntoVar,
    };

    #[test]
    fn test_argmax() {
        let n = Variable::make("n");
        let m = Variable::make("m");
        let a = Tensor::placeholder([&n, &m], Dtype::BF16, "a");
        let _a = a.clone();
        let m_clone = m.clone();
        let c = compute(Dtype::I64, [&n], [&a], "c", move |[a], [i]| {
            argmax(
                [a.slice([i, Variable::make("k").into()])],
                [
                    PrimeExpr::Int(Int::make(Dtype::BF16, 0)),
                    PrimeExpr::Float(Float::make(Dtype::F64, f64::NEG_INFINITY)),
                ],
                [(0, &m_clone, 1, "k")]
            )
        });
        let schedule = Schedule::create(vec![c]);
        let lowered = schedule.lower();
        for stmt in lowered {
            IRPrinter.print_stmt(stmt);
        }
        let d = _a.argmax(0.0, 1);
        let schedule = Schedule::create(vec![d]);
        let lowered = schedule.lower();
        for stmt in lowered {
            IRPrinter.print_stmt(stmt);
        }
    }

    #[test]
    fn test_argmin() {
        let n = Variable::make("n");
        let m = Variable::make("m");
        let a = Tensor::placeholder([&n, &m], Dtype::BF16, "a");
        let m_clone = m.clone();
        let c = compute(Dtype::I64, [&n], [&a], "c", move |[a], [i]| {
            argmin(
                [a.slice([i, Variable::make("k").into()])],
                [
                    PrimeExpr::Int(Int::make(Dtype::BF16, 0)),
                    PrimeExpr::Float(Float::make(Dtype::F64, f64::INFINITY)),
                ],
                [(0, &m_clone, 1, "k")]
            )
        });
        let schedule = Schedule::create(vec![c]);
        let lowered = schedule.lower();
        for stmt in lowered {
            IRPrinter.print_stmt(stmt);
        }
        let d = a.argmin(0.0, 1);
        let schedule = Schedule::create(vec![d]);
        let lowered = schedule.lower();
        for stmt in lowered {
            IRPrinter.print_stmt(stmt);
        }
    }
    #[test]
    fn test_max() {
        let n = Variable::make("n");
        let m = Variable::make("m");
        let a = Tensor::placeholder([&n, &m], Dtype::BF16, "a");
        let c = compute(Dtype::BF16, [&n], [&a], "c", move |[a], [i]| {
            max(
                [a.slice([i, Variable::make("k").into()])],
                [dtype_neg_inf(Dtype::BF16)],
                [(0, &m, 1, "k")]
            )
        });
        let schedule = Schedule::create(vec![c]);
        let lowered = schedule.lower();
        for stmt in lowered {
            IRPrinter.print_stmt(stmt);
        }
        let d = a.max(1);
        let schedule = Schedule::create(vec![d]);
        let lowered = schedule.lower();
        for stmt in lowered {
            IRPrinter.print_stmt(stmt);
        }
    }

    #[test]
    fn test_min() {
        let n = Variable::make("n");
        let m = Variable::make("m");
        let a = Tensor::placeholder([&n, &m], Dtype::BF16, "a");
        let c = compute(Dtype::BF16, [&n], [&a], "c", move |[a], [i]| {
            min(
                [a.slice([i, Variable::make("k").into()])],
                [dtype_inf(Dtype::BF16)],
                [(0, &m, 1, "k")]
            )
        });
        let schedule = Schedule::create(vec![c]);
        let lowered = schedule.lower();
        for stmt in lowered {
            IRPrinter.print_stmt(stmt);
        }
        let d = a.min(1);
        let schedule = Schedule::create(vec![d]);
        let lowered = schedule.lower();
        for stmt in lowered {
            IRPrinter.print_stmt(stmt);
        }
    }
    #[test]
    fn test_sum() {
        let n = Variable::make("n");
        let m = Variable::make("m");
        let a = Tensor::placeholder([&n, &m], Dtype::BF16, "a");
        let c = compute(Dtype::BF16, [&n], [&a], "c", move |[a], [i]| {
            sum(
                [a.slice([i, Variable::make("k").into()])],
                [Int::make(Dtype::BF16, 0)],
                [(0, &m, 1, "k")]
            )
        });
        let schedule = Schedule::create(vec![c]);
        let lowered = schedule.lower();
        for stmt in lowered {
            IRPrinter.print_stmt(stmt);
        }
        let d = a.sum(0.0, 1);
        let schedule = Schedule::create(vec![d]);
        let lowered = schedule.lower();
        for stmt in lowered {
            IRPrinter.print_stmt(stmt);
        }
    }
}
