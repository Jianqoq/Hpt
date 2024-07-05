#![allow(unused_imports)]
#![allow(unused_variables)]
use std::{ fmt::Display, mem::MaybeUninit, sync::Arc };

use half::{ bf16, f16 };
use hashbrown::HashMap;
use tensor_common::{ axis::{ process_axes, Axis }, shape::Shape };
use tensor_types::{ dtype::Dtype, type_promote::NormalOut };

use crate::{
    halide::{
        assign_stmt::AssignStmt,
        exprs::{ Add, And, Div, Float, Gt, Int, Load, Lt, Max, Min, Mod, Mul, Or, Sub, UInt, Xor },
        if_stmt::IfThenElse,
        inplace_store_stmt::InplaceAdd,
        let_stmt::LetStmt,
        loop_utils::{ build_nested::build_nested_for, reduction::* },
        prime_expr::PrimeExpr,
        seq_stmt::Seq,
        stmt::Stmt,
        store_stmt::StoreStmt,
        traits::{ Accepter, AccepterMut },
        variable::Variable,
    },
    iter_var::{IterVar, _IterVar},
    to_prim_expr::ToPrimeExpr,
};
use tensor_types::type_promote::FloatOut;
use tensor_types::type_promote::BitWiseOut;
use tensor_types::dtype::TypeCommon;

use super::{
    schedule::lowered::FindInputs,
    tensor_slice::TensorSlice,
};

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
    strides: Arc<Vec<usize>>,
    body: PrimeExpr,
    name: Arc<String>,
    inputs: Arc<Vec<TensorSlice>>,
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

macro_rules! impl_binops {
    ($op:ident, $fn_name:ident, $type_infer:ident, $var_name:expr) => {
        pub fn $fn_name(&self, rhs: &Self) -> Self {
            let _a = self.clone();
            let lhs_shape = &self.shape;
            let rhs_shape = &rhs.shape;
            let mut res_shape = vec![];
            let mut lhs_indices = vec![];
            let mut rhs_indices = vec![];
            let (lhs_start, rhs_start) = if lhs_shape.len() < rhs_shape.len() {
                (0..rhs_shape.len() - lhs_shape.len()).for_each(|x| {
                    res_shape.push(rhs_shape[x].clone());
                    lhs_indices.push(x);
                });
                (0, lhs_shape.len())
            } else if lhs_shape.len() > rhs_shape.len() {
                (0..lhs_shape.len() - rhs_shape.len()).for_each(|x| {
                    res_shape.push(lhs_shape[x].clone());
                    rhs_indices.push(x);
                });
                (rhs_shape.len(), 0)
            } else {
                (0, 0)
            };
            let one = PrimeExpr::Int(Int::make(Dtype::I64, 1));
            lhs_shape[lhs_start..]
                .iter()
                .zip(rhs_shape[rhs_start..].iter())
                .enumerate()
                .for_each(|(idx, (x, y))| {
                    let x = x.to_iter_var().unwrap();
                    let y = y.to_iter_var().unwrap();
                    if x.end() == &one {
                        res_shape.push(y.into());
                        rhs_indices.push(idx + rhs_start);
                    } else if y.end() == &one {
                        res_shape.push(x.into());
                        lhs_indices.push(idx + lhs_start);
                    } else if x.end() - x.start() == y.end() - x.start() {
                        res_shape.push(x.into());
                        lhs_indices.push(idx + lhs_start);
                        rhs_indices.push(idx + rhs_start);
                    } else {
                        panic!("Incompatible shapes. {} and {}", x, y);
                    }
                });
            let lhs_indices = Arc::new(lhs_indices);
            let rhs_indices = Arc::new(rhs_indices);
            _compute(
                self.dtype.$type_infer(rhs.dtype),
                res_shape,
                vec![self.clone(), rhs.clone()],
                &format!($var_name, self.name, rhs.name),
                move |inputs, indices| {
                    let lhs_indices = lhs_indices.clone();
                    let rhs_indices = rhs_indices.clone();
                    let lhs_indices = lhs_indices
                        .iter()
                        .map(|x| indices[*x].clone())
                        .collect::<Vec<_>>();
                    let rhs_indices = rhs_indices
                        .iter()
                        .map(|x| indices[*x].clone())
                        .collect::<Vec<_>>();
                    $op::make(inputs[0].slice(lhs_indices), inputs[1].slice(rhs_indices)).into()
                }
            )
        }
    };
}

impl Tensor {
    pub fn axes(&self) ->&Vec<IterVar> {
        &self.shape
    }
    pub fn strides(&self) -> &Vec<usize> {
        &self.strides
    }
    pub fn shape(&self) -> &Vec<IterVar> {
        &self.shape
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn name_(&self) -> &Arc<String> {
        &self.name
    }
    pub fn inputs(&self) -> &Vec<TensorSlice> {
        &self.inputs
    }
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }
    pub fn body(&self) -> &PrimeExpr {
        &self.body
    }
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    pub fn sum<T: Into<PrimeExpr> + Clone>(&self, init: T, axes: i64) -> Self {
        let axes: Vec<usize> = process_axes([axes], self.ndim()).unwrap();
        let axis = axes[0];
        let _a = self.clone();
        let mut res_shape = self.shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !axes.contains(i))
            .map(|(_, x)| x.clone())
            .collect::<Vec<_>>();
        if res_shape.is_empty() {
            res_shape.push((0, 1, 1, "red").into());
        }
        let init: PrimeExpr = init.clone().into();
        _compute(
            self.dtype,
            res_shape,
            vec![self.clone()],
            &format!("{}_red", self.name),
            move |inputs, indices| {
                let mut indices = indices
                    .into_iter()
                    .enumerate()
                    .filter(|(i, _)| !axes.contains(&i))
                    .map(|(_, x)| x)
                    .collect::<Vec<_>>();
                let var = Variable::new(format!("red_{}", inputs[0].name));
                let mut reduce_iter_var = inputs[0].shape[axis].clone();
                reduce_iter_var.set_var(var);
                indices.push(reduce_iter_var.clone());
                sum([inputs[0].slice(indices)], &[&init], [reduce_iter_var])
            }
        )
    }
    pub fn prod(&self, init: impl ToPrimeExpr + Clone, axes: i64) -> Self {
        let axes: Vec<usize> = process_axes([axes], self.ndim()).unwrap();
        let axis = axes[0];
        let _a = self.clone();
        let res_shape = self.shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !axes.contains(i))
            .map(|(_, x)| x.clone())
            .collect::<Vec<_>>();
        let init: PrimeExpr = init.clone().to_prime_expr();
        _compute(
            self.dtype,
            res_shape,
            vec![self.clone()],
            &format!("{}_red", self.name),
            move |inputs, indices| {
                let mut indices = indices
                    .into_iter()
                    .enumerate()
                    .filter(|(i, _)| !axes.contains(&i))
                    .map(|(_, x)| x)
                    .collect::<Vec<_>>();
                let var = Variable::new(format!("red_{}", inputs[0].name));
                let mut reduce_iter_var = inputs[0].shape[axis].clone();
                reduce_iter_var.set_var(var);
                indices.push(reduce_iter_var.clone());
                prod([inputs[0].slice(indices)], &[&init], [reduce_iter_var])
            }
        )
    }
    pub fn argmax(&self, init: i64, axes: i64) -> Self {
        let axes: Vec<usize> = process_axes([axes], self.ndim()).unwrap();
        let axis = axes[0];
        let _a = self.clone();
        let res_shape = self.shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !axes.contains(i))
            .map(|(_, x)| x.clone())
            .collect::<Vec<_>>();
        _compute(
            Dtype::I64,
            res_shape,
            vec![self.clone()],
            &format!("{}_red", self.name),
            move |inputs, indices| {
                let mut indices = indices
                    .into_iter()
                    .enumerate()
                    .filter(|(i, _)| !axes.contains(&i))
                    .map(|(_, x)| x)
                    .collect::<Vec<_>>();
                let var = Variable::new(format!("red_{}", inputs[0].name));
                let mut reduce_iter_var = inputs[0].shape[axis].clone();
                reduce_iter_var.set_var(var);
                indices.push(reduce_iter_var.clone());
                argmax([inputs[0].slice(indices)], &[&init, &dtype_neg_inf(inputs[0].dtype)], [
                    reduce_iter_var,
                ])
            }
        )
    }
    pub fn argmin(&self, init: i64, axes: i64) -> Self {
        let axes: Vec<usize> = process_axes([axes], self.ndim()).unwrap();
        let axis = axes[0];
        let _a = self.clone();
        let res_shape = self.shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !axes.contains(i))
            .map(|(_, x)| x.clone())
            .collect::<Vec<_>>();
        _compute(
            Dtype::I64,
            res_shape,
            vec![self.clone()],
            &format!("{}_red", self.name),
            move |inputs, indices| {
                let mut indices = indices
                    .into_iter()
                    .enumerate()
                    .filter(|(i, _)| !axes.contains(&i))
                    .map(|(_, x)| x)
                    .collect::<Vec<_>>();
                let var = Variable::new(format!("red_{}", inputs[0].name));
                let mut reduce_iter_var = inputs[0].shape[axis].clone();
                reduce_iter_var.set_var(var);
                indices.push(reduce_iter_var.clone());
                argmin([inputs[0].slice(indices)], &[&init, &dtype_inf(inputs[0].dtype)], [
                    reduce_iter_var,
                ])
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
        _compute(
            self.dtype,
            res_shape,
            vec![self.clone()],
            &format!("{}_red", self.name),
            move |inputs, indices| {
                let mut indices = indices
                    .into_iter()
                    .enumerate()
                    .filter(|(i, _)| !axes.contains(&i))
                    .map(|(_, x)| x)
                    .collect::<Vec<_>>();
                let var = Variable::new(format!("red_{}", inputs[0].name));
                let mut reduce_iter_var = inputs[0].shape[axis].clone();
                reduce_iter_var.to_iter_var_mut().unwrap().set_var(var);
                indices.push(reduce_iter_var.clone());
                max([inputs[0].slice(indices)], &[&dtype_neg_inf(inputs[0].dtype)], [
                    reduce_iter_var,
                ])
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
        _compute(
            self.dtype,
            res_shape,
            vec![self.clone()],
            &format!("{}_red", self.name),
            move |inputs, indices| {
                let mut indices = indices
                    .into_iter()
                    .enumerate()
                    .filter(|(i, _)| !axes.contains(&i))
                    .map(|(_, x)| x)
                    .collect::<Vec<_>>();
                let var = Variable::new(format!("red_{}", inputs[0].name));
                let mut reduce_iter_var = inputs[0].shape[axis].clone();
                reduce_iter_var.set_var(var);
                indices.push(reduce_iter_var.clone());
                min([inputs[0].slice(indices)], &[&dtype_inf(inputs[0].dtype)], [reduce_iter_var])
            }
        )
    }

    impl_binops!(Add, add, _add, "{}_add_{}");
    impl_binops!(Mul, mul, _mul, "{}_mul_{}");
    impl_binops!(Div, div, _div, "{}_div_{}");
    impl_binops!(Sub, sub, _sub, "{}_sub_{}");
    impl_binops!(Mod, rem, _rem, "{}_rem_{}");
    impl_binops!(And, and, _and, "{}_and_{}");
    impl_binops!(Or, or, _or, "{}_or_{}");
    impl_binops!(Xor, xor, _xor, "{}_xor_{}");
}

impl Eq for Tensor {}

impl std::hash::Hash for Tensor {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.shape.hash(state);
        self.body.hash(state);
        self.name.hash(state);
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.body == other.body && self.name == other.name
    }
}

impl Tensor {
    pub fn placeholder(shape: &[&dyn ToPrimeExpr], dtype: Dtype, name: &str) -> Self {
        let tensor_name = name.to_string();
        let iter_vars = shape
            .into_iter()
            .map(|x| x.to_prime_expr())
            .enumerate()
            .map(|(i, x)| {
                IterVar::IterVar(_IterVar::new(
                    Int::make(Dtype::I64, 0),
                    x,
                    Int::make(Dtype::I64, 1),
                    Variable::new(format!("ax{}", i))
                ))
            })
            .collect::<Vec<_>>();
        let body = Load::make(
            Variable::new(tensor_name.clone()),
            iter_vars
                .iter()
                .map(|x| x.to_iter_var().unwrap().var().to_prime_expr())
                .reduce(|acc, x| acc + x)
                .unwrap()
        ).into();
        Self {
            shape: Arc::new(iter_vars),
            strides: Arc::new((0..shape.len()).collect::<Vec<_>>()),
            body,
            name: name.to_string().into(),
            inputs: vec![].into(),
            dtype,
        }
    }
    pub fn slice<T: IntoIterator<Item: Into<IterVar>>>(&self, indices: T) -> TensorSlice {
        let indices = indices
            .into_iter()
            .map(|x| x.into())
            .collect::<Vec<IterVar>>();
        assert!(indices.len() == self.ndim());
        let indices = indices
            .iter()
            .map(|x| x.to_iter_var().unwrap().var().into())
            .collect::<Vec<_>>();
        TensorSlice::make(self.name.clone(), indices)
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
    A: Into<Tensor> + Clone,
    C: Into<PrimeExpr>
    >(dtype: Dtype, res_shape: [T; N], inputs: [A; M], name: &str, op: F) -> Tensor
    where F: Fn([Tensor; M], [IterVar; N]) -> C
{
    let iter_vars = res_shape
        .iter()
        .enumerate()
        .map(|(i, x)| {
            IterVar::IterVar(_IterVar::new(
                Int::make(Dtype::I64, 0),
                x.clone(),
                Int::make(Dtype::I64, 1),
                Variable::new(format!("ax{}", i))
            ))
        })
        .collect::<Vec<_>>();
    let inputs = inputs
        .into_iter()
        .map(|x| x.into())
        .collect::<[Tensor; M]>();
    let body = op(
        inputs,
        iter_vars
            .iter()
            .map(|x| x.clone())
            .collect::<[IterVar; N]>()
    );
    let mut input_visitor = FindInputs::new();
    let body: PrimeExpr = body.into();
    body.accept_mut(&mut input_visitor);
    Tensor {
        shape: Arc::new(iter_vars),
        strides: Arc::new((0..res_shape.len()).collect::<Vec<_>>()),
        body: body.into(),
        name: name.to_string().into(),
        inputs: input_visitor.to_vec().into(),
        dtype,
    }
}

pub fn _compute<F>(
    dtype: Dtype,
    res_shape: Vec<IterVar>,
    inputs: Vec<Tensor>,
    name: &str,
    op: F
) -> Tensor
    where F: Fn(Vec<Tensor>, Vec<IterVar>) -> PrimeExpr + 'static
{
    let inputs_cloned = inputs.clone();
    let body = op(inputs_cloned, res_shape.clone());
    let strides = (0..res_shape.len()).collect::<Vec<_>>();
    let mut input_visitor = FindInputs::new();
    let body: PrimeExpr = body.into();
    body.accept_mut(&mut input_visitor);
    Tensor {
        shape: Arc::new(res_shape),
        strides: Arc::new(strides),
        body,
        name: name.to_string().into(),
        inputs: input_visitor.to_vec().into(),
        dtype,
    }
}

impl<const N: usize> FromIterator<Tensor> for [Tensor; N] {
    fn from_iter<T: IntoIterator<Item = Tensor>>(iter: T) -> Self {
        let mut arr: [MaybeUninit<Tensor>; N] = unsafe { MaybeUninit::uninit().assume_init() };
        let mut iter = iter.into_iter();
        for i in 0..N {
            arr[i] = MaybeUninit::new(iter.next().unwrap());
        }
        unsafe { std::mem::transmute_copy(&arr) }
    }
}

impl<const N: usize> FromIterator<PrimeExpr> for [PrimeExpr; N] {
    fn from_iter<T: IntoIterator<Item = PrimeExpr>>(iter: T) -> Self {
        let mut arr: [MaybeUninit<PrimeExpr>; N] = unsafe { MaybeUninit::uninit().assume_init() };
        let mut iter = iter.into_iter();
        for i in 0..N {
            arr[i] = MaybeUninit::new(iter.next().unwrap());
        }
        unsafe { std::mem::transmute_copy(&arr) }
    }
}

impl<const N: usize> FromIterator<IterVar> for [IterVar; N] {
    fn from_iter<T: IntoIterator<Item = IterVar>>(iter: T) -> Self {
        let mut arr: [MaybeUninit<IterVar>; N] = unsafe { MaybeUninit::uninit().assume_init() };
        let mut iter = iter.into_iter();
        for i in 0..N {
            arr[i] = MaybeUninit::new(iter.next().unwrap());
        }
        unsafe { std::mem::transmute_copy(&arr) }
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
    use crate::to_prim_expr::ToPrimeExpr;

    #[test]
    fn test_argmax() {
        let n = Variable::make("n");
        let m = Variable::make("m");
        let a = Tensor::placeholder(&[&n, &m], Dtype::BF16, "a");
        let g = compute(Dtype::BF16, [&n, &m], [&a], "a", |[a], [i, j]| { 2 + a.slice([i, j]) });
        let d = a.argmax(0, 1);
        println!("d body: {}", g.body());
    }

    #[test]
    fn test_argmin() {
        let n = Variable::make("n");
        let m = Variable::make("m");
        let a = Tensor::placeholder(&[&n, &m], Dtype::BF16, "a");
        let d = a.argmin(0, 1);
    }
    #[test]
    fn test_max() {
        let n = Variable::make("n");
        let m = Variable::make("m");
        let a = Tensor::placeholder(&[&n, &m], Dtype::BF16, "a");
        let d = a.max(1);
    }

    #[test]
    fn test_min() {
        let n = Variable::make("n");
        let m = Variable::make("m");
        let a = Tensor::placeholder(&[&n, &m], Dtype::BF16, "a");
        let d = a.min(1);
    }
    #[test]
    fn test_sum() {
        let n = Variable::make("n");
        let m = Variable::make("m");
        let a = Tensor::placeholder(&[&n, &m], Dtype::BF16, "a");
        let d = a.sum(0.0, 1);
        let q = d.sum(0.0, 0);
    }
    #[test]
    fn test_add() {
        let n = Variable::make("n");
        let m = Variable::make("m");
        let a = Tensor::placeholder(&[&n, &1i64], Dtype::BF16, "a");
        let b = Tensor::placeholder(&[&1i64, &m], Dtype::BF16, "b");
        let c = a.add(&b);
    }
    #[test]
    fn test_mul() {
        let n = Variable::make("n");
        let m = Variable::make("m");
        let a = Tensor::placeholder(&[&n, &1i64], Dtype::BF16, "a");
        let b = Tensor::placeholder(&[&1i64, &m], Dtype::BF16, "b");
        let c = a.mul(&b);
    }
    #[test]
    fn test_div() {
        let n = Variable::make("n");
        let m = Variable::make("m");
        let a = Tensor::placeholder(&[&n, &1i64], Dtype::BF16, "a");
        let b = Tensor::placeholder(&[&1i64, &m], Dtype::BF16, "b");
        let c = a.div(&b);
    }
    #[test]
    fn test_sub() {
        let n = Variable::make("n");
        let m = Variable::make("m");
        let a = Tensor::placeholder(&[&n, &1i64], Dtype::BF16, "a");
        let b = Tensor::placeholder(&[&1i64, &m], Dtype::BF16, "b");
        let c = a.sub(&b);
    }
    #[test]
    fn test_mod() {
        let n = Variable::make("n");
        let m = Variable::make("m");
        let a = Tensor::placeholder(&[&n, &1i64], Dtype::BF16, "a");
        let b = Tensor::placeholder(&[&1i64, &m], Dtype::BF16, "b");
        let c = a.rem(&b);
    }
    #[test]
    fn test_and() {
        let n = Variable::make("n");
        let m = Variable::make("m");
        let a = Tensor::placeholder(&[&n, &1i64], Dtype::BF16, "a");
        let b = Tensor::placeholder(&[&1i64, &m], Dtype::BF16, "b");
        let c = a.and(&b);
    }
    #[test]
    fn test_or() {
        let n = Variable::make("n");
        let m = Variable::make("m");
        let a = Tensor::placeholder(&[&n, &1i64], Dtype::BF16, "a");
        let b = Tensor::placeholder(&[&1i64, &m], Dtype::BF16, "b");
        let c = a.or(&b);
    }
    #[test]
    fn test_xor() {
        let n = Variable::make("n");
        let m = Variable::make("m");
        let a = Tensor::placeholder(&[&n, &1i64], Dtype::BF16, "a");
        let b = Tensor::placeholder(&[&1i64, &m], Dtype::BF16, "b");
        let c = a.xor(&b);
    }
}
