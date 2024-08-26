use tensor_macros::{ impl_bitwise_out, impl_cmp, impl_eval, impl_float_out, impl_normal_out };
use half::f16;
use crate::convertion::Convertor;
use num_traits::float::Float;
use crate::dtype::TypeCommon;
use statrs::function::erf::erf;
use crate::dtype::FloatConst;
use tensor_macros::impl_normal_out_simd;
use crate::vectors::*;
use half::bf16;
use crate::convertion::VecConvertor;
use std::simd::num::SimdInt;
use std::simd::num::SimdFloat;
use std::simd::StdFloat;
use std::simd::cmp::SimdOrd;

/// this trait is used to perform type promotion in dynamic graph
pub trait FloatOut<RHS = Self> {
    type Output;
    fn _div(self, rhs: RHS) -> Self::Output;
    fn _exp(self) -> Self::Output;
    fn _exp2(self) -> Self::Output;
    fn _ln(self) -> Self::Output;
    fn _log(self, base: RHS) -> Self::Output;
    fn _celu(self, alpha: Self::Output) -> Self::Output;
    fn _log2(self) -> Self::Output;
    fn _log10(self) -> Self::Output;
    fn _sqrt(self) -> Self::Output;
    fn _sin(self) -> Self::Output;
    fn _cos(self) -> Self::Output;
    fn _tan(self) -> Self::Output;
    fn _asin(self) -> Self::Output;
    fn _acos(self) -> Self::Output;
    fn _atan(self) -> Self::Output;
    fn _sinh(self) -> Self::Output;
    fn _cosh(self) -> Self::Output;
    fn _tanh(self) -> Self::Output;
    fn _asinh(self) -> Self::Output;
    fn _acosh(self) -> Self::Output;
    fn _atanh(self) -> Self::Output;
    fn _recip(self) -> Self::Output;
    fn _erf(self) -> Self::Output;
    fn _sigmoid(self) -> Self::Output;
    fn _elu(self, alpha: Self::Output) -> Self::Output;
    fn _leaky_relu(self, alpha: Self::Output) -> Self::Output;
    fn _relu(self) -> Self::Output;
    fn _gelu(self) -> Self::Output;
    fn _selu(self, alpha: Self::Output, scale: Self::Output) -> Self::Output;
    fn _hard_sigmoid(self, alpha: Self::Output, beta: Self::Output) -> Self::Output;
    fn _relu6(self) -> Self::Output;
    fn _hard_swish(self) -> Self::Output;
    fn _softplus(self) -> Self::Output;
    fn _softsign(self) -> Self::Output;
    fn _mish(self) -> Self::Output;
    fn _cbrt(self) -> Self::Output;
}

impl_float_out!();

pub trait NormalOut<RHS = Self> {
    type Output;
    fn _add(self, rhs: RHS) -> Self::Output;
    fn _sub(self, rhs: RHS) -> Self::Output;
    fn _mul(self, rhs: RHS) -> Self::Output;
    fn _pow(self, rhs: RHS) -> Self::Output;
    fn _rem(self, rhs: RHS) -> Self::Output;
    fn _square(self) -> Self;
    fn _abs(self) -> Self;
    fn _ceil(self) -> Self;
    fn _floor(self) -> Self;
    fn _sign(self) -> Self::Output;
    fn _max(self, rhs: RHS) -> Self::Output;
    fn _min(self, rhs: RHS) -> Self::Output;
    fn _clip(self, min: Self::Output, max: Self::Output) -> Self::Output;
    fn _round(self) -> Self;
}

impl_normal_out!();

impl_normal_out_simd!();

pub trait BitWiseOut<RHS = Self> {
    type Output;
    fn _bitand(self, rhs: RHS) -> Self::Output;
    fn _bitor(self, rhs: RHS) -> Self::Output;
    fn _bitxor(self, rhs: RHS) -> Self::Output;
    fn _not(self) -> Self::Output;
    fn _shl(self, rhs: RHS) -> Self::Output;
    fn _shr(self, rhs: RHS) -> Self::Output;
}

impl_bitwise_out!();

pub trait Cmp<RHS = Self> {
    fn _eq(self, rhs: RHS) -> bool;
    fn _ne(self, rhs: RHS) -> bool;
    fn _lt(self, rhs: RHS) -> bool;
    fn _le(self, rhs: RHS) -> bool;
    fn _gt(self, rhs: RHS) -> bool;
    fn _ge(self, rhs: RHS) -> bool;
}

impl_cmp!();

pub trait Eval {
    type Output;
    fn _is_nan(&self) -> Self::Output;
    fn _is_true(&self) -> Self::Output;
    fn _is_inf(&self) -> Self::Output;
}

impl_eval!();
