use crate::convertion::Convertor;
use crate::convertion::VecConvertor;
use crate::dtype::FloatConst;
use crate::dtype::TypeCommon;
#[cfg(all(
    any(target_feature = "sse2", target_feature = "neon"),
    not(target_feature = "avx2")
))]
use crate::vectors::_128bit::*;
#[cfg(target_feature = "avx2")]
use crate::vectors::_256bit::*;
#[cfg(target_feature = "avx512f")]
use crate::vectors::_512bit::*;
use crate::vectors::traits::Init;
use half::bf16;
use half::f16;
use num_traits::float::Float;
use sleef::Sleef;
use std::ops::Neg;
use std::simd::cmp::SimdOrd;
use std::simd::cmp::SimdPartialEq;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdFloat;
use std::simd::num::SimdInt;
use std::simd::num::SimdUint;
use std::simd::Simd;
use tensor_macros::float_out_unary;
use tensor_macros::impl_normal_out_simd;
use tensor_macros::{
    float_out_binary, impl_bitwise_out, impl_cmp, impl_eval, impl_normal_out, simd_cmp, simd_eval,
    simd_float_out_unary,
};
use tensor_macros::{float_out_binary_simd, simd_bitwise};
/// this trait is used to perform type promotion in dynamic graph
pub trait FloatOutBinary<RHS = Self> {
    type Output;
    fn _div(self, rhs: RHS) -> Self::Output;
    fn _log(self, base: RHS) -> Self::Output;
}

float_out_binary!();
float_out_binary_simd!();

pub trait NormalOut<RHS = Self> {
    type Output;
    fn _add(self, rhs: RHS) -> Self::Output;
    fn _sub(self, rhs: RHS) -> Self::Output;
    fn _mul_add(self, a: RHS, b: RHS) -> Self::Output;
    fn _mul(self, rhs: RHS) -> Self::Output;
    fn _pow(self, rhs: RHS) -> Self::Output;
    fn _rem(self, rhs: RHS) -> Self::Output;
    fn _square(self) -> Self;
    fn _abs(self) -> Self;
    fn _ceil(self) -> Self;
    fn _floor(self) -> Self;
    fn _neg(self) -> Self;
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

simd_bitwise!();

pub trait Cmp<RHS = Self> {
    fn _eq(self, rhs: RHS) -> bool;
    fn _ne(self, rhs: RHS) -> bool;
    fn _lt(self, rhs: RHS) -> bool;
    fn _le(self, rhs: RHS) -> bool;
    fn _gt(self, rhs: RHS) -> bool;
    fn _ge(self, rhs: RHS) -> bool;
}

impl_cmp!();

pub trait SimdCmp<RHS = Self> {
    type Output;
    fn _eq(self, rhs: RHS) -> Self::Output;
    fn _ne(self, rhs: RHS) -> Self::Output;
    fn _lt(self, rhs: RHS) -> Self::Output;
    fn _le(self, rhs: RHS) -> Self::Output;
    fn _gt(self, rhs: RHS) -> Self::Output;
    fn _ge(self, rhs: RHS) -> Self::Output;
}

simd_cmp!();

pub trait Eval {
    type Output;
    fn _is_nan(&self) -> Self::Output;
    fn _is_true(&self) -> Self::Output;
    fn _is_inf(&self) -> Self::Output;
}

impl_eval!();
simd_eval!();

pub trait FloatOutUnary {
    type Output;
    type Base;
    fn _exp(self) -> Self::Output;
    fn _exp2(self) -> Self::Output;
    fn _ln(self) -> Self::Output;
    fn _celu(self, alpha: Self::Base) -> Self::Output;
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
    fn _elu(self, alpha: Self::Base) -> Self::Output;
    fn _leaky_relu(self, alpha: Self::Base) -> Self::Output;
    fn _relu(self) -> Self::Output;
    fn _gelu(self) -> Self::Output;
    fn _selu(self, alpha: Self::Base, scale: Self::Base) -> Self::Output;
    fn _hard_sigmoid(self) -> Self::Output;
    fn _relu6(self) -> Self::Output;
    fn _hard_swish(self) -> Self::Output;
    fn _softplus(self) -> Self::Output;
    fn _softsign(self) -> Self::Output;
    fn _mish(self) -> Self::Output;
    fn _cbrt(self) -> Self::Output;
}

float_out_unary!();

simd_float_out_unary!();
