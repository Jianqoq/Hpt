use tensor_macros::{ impl_bitwise_out, impl_cmp, impl_eval, impl_float_out, impl_normal_out };
use half::f16;
use crate::convertion::Convertor;
use num_traits::float::Float;
use crate::dtype::TypeCommon;

/// this trait is used to perform type promotion in dynamic graph
pub trait FloatOut<RHS = Self> {
    type Output;
    fn _div(self, rhs: RHS) -> Self::Output;
    fn _exp(self) -> Self::Output;
    fn _exp2(self) -> Self::Output;
    fn _ln(self) -> Self::Output;
    fn _log(self, base: RHS) -> Self::Output;
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
}

impl_float_out!();

pub trait NormalOut<RHS = Self> {
    type Output;
    fn _add(self, rhs: RHS) -> Self::Output;
    fn _sub(self, rhs: RHS) -> Self::Output;
    fn _mul(self, rhs: RHS) -> Self::Output;
    fn _pow(self, rhs: RHS) -> Self::Output;
    fn _rem(self, rhs: RHS) -> Self::Output;
    fn _square(self) -> Self::Output;
    fn _abs(self) -> Self::Output;
}

impl_normal_out!();

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
