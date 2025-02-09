#![allow(unused)]

use std::ops::{Index, IndexMut};

use hpt_core::{
    Cast, FloatOutBinary, FloatOutUnary, IntoVec, NormalOut, NormalOutUnary, ShapeManipulate,
    Tensor, TensorCreator, TypeCommon, VecTrait,
};

#[derive(Debug, Clone, Copy)]
struct CustomType {
    value: i64,
}

impl std::fmt::Display for CustomType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl NormalOutUnary for CustomType {
    fn _square(self) -> Self {
        todo!()
    }

    fn _abs(self) -> Self {
        todo!()
    }

    fn _ceil(self) -> Self {
        todo!()
    }

    fn _floor(self) -> Self {
        todo!()
    }

    fn _neg(self) -> Self {
        todo!()
    }

    fn _round(self) -> Self {
        todo!()
    }

    fn _signum(self) -> Self {
        todo!()
    }

    fn _trunc(self) -> Self {
        todo!()
    }

    fn _leaky_relu(self, alpha: Self) -> Self {
        todo!()
    }

    fn _relu(self) -> Self {
        todo!()
    }

    fn _relu6(self) -> Self {
        todo!()
    }
}

impl FloatOutUnary for CustomType {
    type Output = Self;

    fn _exp(self) -> Self::Output {
        todo!()
    }

    fn _expm1(self) -> Self::Output {
        todo!()
    }

    fn _exp2(self) -> Self::Output {
        todo!()
    }

    fn _ln(self) -> Self::Output {
        todo!()
    }

    fn _log1p(self) -> Self::Output {
        todo!()
    }

    fn _celu(self, alpha: Self::Output) -> Self::Output {
        todo!()
    }

    fn _log2(self) -> Self::Output {
        todo!()
    }

    fn _log10(self) -> Self::Output {
        todo!()
    }

    fn _sqrt(self) -> Self::Output {
        todo!()
    }

    fn _sin(self) -> Self::Output {
        todo!()
    }

    fn _cos(self) -> Self::Output {
        todo!()
    }

    fn _tan(self) -> Self::Output {
        todo!()
    }

    fn _asin(self) -> Self::Output {
        todo!()
    }

    fn _acos(self) -> Self::Output {
        todo!()
    }

    fn _atan(self) -> Self::Output {
        todo!()
    }

    fn _sinh(self) -> Self::Output {
        todo!()
    }

    fn _cosh(self) -> Self::Output {
        todo!()
    }

    fn _tanh(self) -> Self::Output {
        todo!()
    }

    fn _asinh(self) -> Self::Output {
        todo!()
    }

    fn _acosh(self) -> Self::Output {
        todo!()
    }

    fn _atanh(self) -> Self::Output {
        todo!()
    }

    fn _recip(self) -> Self::Output {
        todo!()
    }

    fn _erf(self) -> Self::Output {
        todo!()
    }

    fn _sigmoid(self) -> Self::Output {
        todo!()
    }

    fn _elu(self, alpha: Self::Output) -> Self::Output {
        todo!()
    }

    fn _gelu(self) -> Self::Output {
        todo!()
    }

    fn _selu(self, alpha: Self::Output, scale: Self::Output) -> Self::Output {
        todo!()
    }

    fn _hard_sigmoid(self) -> Self::Output {
        todo!()
    }

    fn _hard_swish(self) -> Self::Output {
        todo!()
    }

    fn _softplus(self) -> Self::Output {
        todo!()
    }

    fn _softsign(self) -> Self::Output {
        todo!()
    }

    fn _mish(self) -> Self::Output {
        todo!()
    }

    fn _cbrt(self) -> Self::Output {
        todo!()
    }
}

impl FloatOutBinary for CustomType {
    type Output = Self;

    fn _div(self, rhs: Self) -> Self::Output {
        todo!()
    }

    fn _log(self, base: Self) -> Self::Output {
        todo!()
    }
}

impl NormalOut for CustomType {
    type Output = Self;

    fn _add(self, rhs: Self) -> Self::Output {
        todo!()
    }

    fn _sub(self, rhs: Self) -> Self::Output {
        todo!()
    }

    fn _mul_add(self, a: Self, b: Self) -> Self::Output {
        todo!()
    }

    fn _mul(self, rhs: Self) -> Self::Output {
        todo!()
    }

    fn _pow(self, rhs: Self) -> Self::Output {
        todo!()
    }

    fn _rem(self, rhs: Self) -> Self::Output {
        todo!()
    }

    fn _max(self, rhs: Self) -> Self::Output {
        todo!()
    }

    fn _min(self, rhs: Self) -> Self::Output {
        todo!()
    }

    fn _clamp(self, min: Self, max: Self) -> Self::Output {
        todo!()
    }
}

impl Cast<CustomType> for CustomType {
    fn cast(self) -> CustomType {
        self
    }
}

impl TypeCommon for CustomType {
    const MAX: Self = CustomType { value: i64::MAX };

    const MIN: Self = CustomType { value: i64::MIN };

    const ZERO: Self = CustomType { value: 0 };

    const ONE: Self = CustomType { value: 1 };

    const INF: Self = CustomType { value: i64::MAX };

    const NEG_INF: Self = CustomType { value: i64::MIN };

    const TWO: Self = CustomType { value: 2 };

    const SIX: Self = CustomType { value: 6 };

    const TEN: Self = CustomType { value: 10 };

    const STR: &'static str = "CustomType";

    const BIT_SIZE: usize = 8 * std::mem::size_of::<CustomType>();

    type Vec = CustomTypeVec;
}

#[derive(Debug, Clone, Copy)]
struct CustomTypeVec([CustomType; 10]);

impl NormalOutUnary for CustomTypeVec {
    fn _square(self) -> Self {
        todo!()
    }

    fn _abs(self) -> Self {
        todo!()
    }

    fn _ceil(self) -> Self {
        todo!()
    }

    fn _floor(self) -> Self {
        todo!()
    }

    fn _neg(self) -> Self {
        todo!()
    }

    fn _round(self) -> Self {
        todo!()
    }

    fn _signum(self) -> Self {
        todo!()
    }

    fn _trunc(self) -> Self {
        todo!()
    }

    fn _leaky_relu(self, alpha: Self) -> Self {
        todo!()
    }

    fn _relu(self) -> Self {
        todo!()
    }

    fn _relu6(self) -> Self {
        todo!()
    }
}

impl FloatOutUnary for CustomTypeVec {
    type Output = Self;

    fn _exp(self) -> Self::Output {
        todo!()
    }

    fn _expm1(self) -> Self::Output {
        todo!()
    }

    fn _exp2(self) -> Self::Output {
        todo!()
    }

    fn _ln(self) -> Self::Output {
        todo!()
    }

    fn _log1p(self) -> Self::Output {
        todo!()
    }

    fn _celu(self, alpha: Self::Output) -> Self::Output {
        todo!()
    }

    fn _log2(self) -> Self::Output {
        todo!()
    }

    fn _log10(self) -> Self::Output {
        todo!()
    }

    fn _sqrt(self) -> Self::Output {
        todo!()
    }

    fn _sin(self) -> Self::Output {
        todo!()
    }

    fn _cos(self) -> Self::Output {
        todo!()
    }

    fn _tan(self) -> Self::Output {
        todo!()
    }

    fn _asin(self) -> Self::Output {
        todo!()
    }

    fn _acos(self) -> Self::Output {
        todo!()
    }

    fn _atan(self) -> Self::Output {
        todo!()
    }

    fn _sinh(self) -> Self::Output {
        todo!()
    }

    fn _cosh(self) -> Self::Output {
        todo!()
    }

    fn _tanh(self) -> Self::Output {
        todo!()
    }

    fn _asinh(self) -> Self::Output {
        todo!()
    }

    fn _acosh(self) -> Self::Output {
        todo!()
    }

    fn _atanh(self) -> Self::Output {
        todo!()
    }

    fn _recip(self) -> Self::Output {
        todo!()
    }

    fn _erf(self) -> Self::Output {
        todo!()
    }

    fn _sigmoid(self) -> Self::Output {
        todo!()
    }

    fn _elu(self, alpha: Self::Output) -> Self::Output {
        todo!()
    }

    fn _gelu(self) -> Self::Output {
        todo!()
    }

    fn _selu(self, alpha: Self::Output, scale: Self::Output) -> Self::Output {
        todo!()
    }

    fn _hard_sigmoid(self) -> Self::Output {
        todo!()
    }

    fn _hard_swish(self) -> Self::Output {
        todo!()
    }

    fn _softplus(self) -> Self::Output {
        todo!()
    }

    fn _softsign(self) -> Self::Output {
        todo!()
    }

    fn _mish(self) -> Self::Output {
        todo!()
    }

    fn _cbrt(self) -> Self::Output {
        todo!()
    }
}

impl FloatOutBinary for CustomTypeVec {
    type Output = Self;

    fn _div(self, rhs: Self) -> Self::Output {
        todo!()
    }

    fn _log(self, base: Self) -> Self::Output {
        todo!()
    }
}

impl NormalOut for CustomTypeVec {
    type Output = Self;

    fn _add(self, rhs: Self) -> Self::Output {
        todo!()
    }

    fn _sub(self, rhs: Self) -> Self::Output {
        todo!()
    }

    fn _mul_add(self, a: Self, b: Self) -> Self::Output {
        todo!()
    }

    fn _mul(self, rhs: Self) -> Self::Output {
        todo!()
    }

    fn _pow(self, rhs: Self) -> Self::Output {
        todo!()
    }

    fn _rem(self, rhs: Self) -> Self::Output {
        todo!()
    }

    fn _max(self, rhs: Self) -> Self::Output {
        todo!()
    }

    fn _min(self, rhs: Self) -> Self::Output {
        todo!()
    }

    fn _clamp(self, min: Self, max: Self) -> Self::Output {
        todo!()
    }
}

impl Index<usize> for CustomTypeVec {
    type Output = CustomType;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for CustomTypeVec {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl IntoVec<CustomTypeVec> for CustomTypeVec {
    fn into_vec(self) -> CustomTypeVec {
        self
    }
}

impl VecTrait<CustomType> for CustomTypeVec {
    const SIZE: usize = 10;

    type Base = CustomType;

    fn mul_add(self, a: Self, b: Self) -> Self {
        todo!()
    }

    fn copy_from_slice(&mut self, slice: &[CustomType]) {
        todo!()
    }

    fn sum(&self) -> CustomType {
        todo!()
    }

    fn splat(val: CustomType) -> Self {
        todo!()
    }

    unsafe fn from_ptr(ptr: *const CustomType) -> Self {
        todo!()
    }
}

impl Cast<CustomType> for usize {
    fn cast(self) -> CustomType {
        CustomType { value: self as i64 }
    }
}

impl Cast<f64> for CustomType {
    fn cast(self) -> f64 {
        self.value as f64
    }
}


fn main() -> anyhow::Result<()> {
    let a = Tensor::<CustomType>::arange(0, 16)?.reshape(&[4, 4])?;
    println!("{}", a);
    Ok(())
}

