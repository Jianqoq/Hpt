
use crate::{
    arch_simd::_128bit::i64x2::i64x2,
    convertion::VecConvertor,
    traits::{SimdCompare, SimdMath, VecTrait}, type_promote::{Eval2, FloatOutBinary2, NormalOut2, NormalOutUnary2},
};

use super::usizex2::usizex2;

/// a vector of 2 isize values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
#[cfg(target_pointer_width = "64")]
pub struct isizex2(pub(crate) i64x2);
#[cfg(target_pointer_width = "32")]
pub struct isizex4(pub(crate) i32x4);

#[cfg(target_pointer_width = "64")]
#[allow(non_camel_case_types)]
pub(crate) type isize_promote = isizex2;
#[cfg(target_pointer_width = "32")]
#[allow(non_camel_case_types)]
pub(crate) type isize_promote = isizex4;

#[cfg(target_pointer_width = "32")]
type ISizeVEC = isizex4;
#[cfg(target_pointer_width = "64")]
type ISizeVEC = isizex2;

#[cfg(target_pointer_width = "32")]
type ISizeBase = i32x4;
#[cfg(target_pointer_width = "64")]
type ISizeBase = i64x2;

impl Default for ISizeVEC {
    #[inline(always)]
    fn default() -> Self {
        Self(ISizeBase::default())
    }
}

impl PartialEq for ISizeVEC {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl VecTrait<isize> for ISizeVEC {
    #[cfg(target_pointer_width = "64")]
    const SIZE: usize = 2;
    #[cfg(target_pointer_width = "32")]
    const SIZE: usize = 4;
    type Base = isize;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[isize]) {
        ISizeBase::copy_from_slice(&mut self.0, unsafe { std::mem::transmute(slice) });
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0.mul_add(a.0, b.0))
    }
    #[inline(always)]
    fn sum(&self) -> isize {
        self.0.sum() as isize
    }
    #[inline(always)]
    fn splat(val: isize) -> ISizeVEC {
        Self(ISizeBase::splat(val as i64))
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const isize) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            Self(unsafe { ISizeBase::from_ptr(ptr as *const i64) })
        }
        #[cfg(target_pointer_width = "32")]
        {
            Self(unsafe { ISizeBase::from_ptr(ptr as *const i32) })
        }
    }
}

impl isizex2 {
    #[allow(unused)]
    #[cfg(target_pointer_width = "64")]
    #[inline(always)]
    fn as_array(&self) -> [isize; 2] {
        unsafe { std::mem::transmute(self.0) }
    }
    #[allow(unused)]
    #[cfg(target_pointer_width = "32")]
    #[inline(always)]
    fn as_array(&self) -> [isize; 4] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl SimdCompare for isizex2 {
    type SimdMask = isizex2;
    #[inline(always)]
    fn simd_eq(self, other: Self) -> isizex2 {
        Self(self.0.simd_eq(other.0))
    }
    #[inline(always)]
    fn simd_ne(self, other: Self) -> isizex2 {
        Self(self.0.simd_ne(other.0))
    }
    #[inline(always)]
    fn simd_lt(self, other: Self) -> isizex2 {
        Self(self.0.simd_lt(other.0))
    }
    #[inline(always)]
    fn simd_le(self, other: Self) -> isizex2 {
        Self(self.0.simd_le(other.0))
    }
    #[inline(always)]
    fn simd_gt(self, other: Self) -> isizex2 {
        Self(self.0.simd_gt(other.0))
    }
    #[inline(always)]
    fn simd_ge(self, other: Self) -> isizex2 {
        Self(self.0.simd_ge(other.0))
    }
}

impl std::ops::Add for isizex2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0.add(rhs.0))
    }
}
impl std::ops::Sub for isizex2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.sub(rhs.0))
    }
}
impl std::ops::Mul for isizex2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(self.0.mul(rhs.0))
    }
}
impl std::ops::Div for isizex2 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self(self.0.div(rhs.0))
    }
}
impl std::ops::Rem for isizex2 {
    type Output = Self;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self {
        Self(self.0.rem(rhs.0))
    }
}
impl std::ops::Neg for isizex2 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(SimdMath::neg(self.0))
    }
}
impl std::ops::BitAnd for isizex2 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0.bitand(rhs.0))
    }
}
impl std::ops::BitOr for isizex2 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0.bitor(rhs.0))
    }
}
impl std::ops::BitXor for isizex2 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0.bitxor(rhs.0))
    }
}
impl std::ops::Not for isizex2 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self::Output {
        Self(self.0.not())
    }
}
impl std::ops::Shl for isizex2 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self::Output {
        Self(self.0.shl(rhs.0))
    }
}
impl std::ops::Shr for isizex2 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self::Output {
        Self(self.0.shr(rhs.0))
    }
}
impl SimdMath<isize> for isizex2 {
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Self(self.0.max(other.0))
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }
    #[inline(always)]
    fn relu(self) -> Self {
        Self(self.0.relu())
    }
    #[inline(always)]
    fn relu6(self) -> Self {
        Self(self.0.relu6())
    }
    #[inline(always)]
    fn trunc(self) -> Self {
        self
    }
    #[inline(always)]
    fn floor(self) -> Self {
        self
    }
    #[inline(always)]
    fn ceil(self) -> Self {
        self
    }
    #[inline(always)]
    fn round(self) -> Self {
        self
    }
    #[inline(always)]
    fn square(self) -> Self {
        self * self
    }
    #[inline(always)]
    fn abs(self) -> Self {
        Self(self.0.abs())
    }
    #[inline(always)]
    fn neg(self) -> Self {
        -self
    }
    #[inline(always)]
    fn signum(self) -> Self {
        Self(self.0.signum())
    }
    #[inline(always)]
    fn pow(self, rhs: Self) -> Self {
        Self(self.0.pow(rhs.0))
    }
    #[inline(always)]
    fn leaky_relu(self, alpha: Self) -> Self {
        self.max(Self::splat(0)) + alpha * self.min(Self::splat(0))
    }
}

impl VecConvertor for isizex2 {
    #[inline(always)]
    fn to_isize(self) -> isizex2 {
        self
    }
    #[inline(always)]
    fn to_usize(self) -> usizex2 {
        unsafe { std::mem::transmute(self) }
    }
    #[cfg(target_pointer_width = "64")]
    #[inline(always)]
    fn to_i64(self) -> i64x2 {
        unsafe { std::mem::transmute(self) }
    }
    #[cfg(target_pointer_width = "32")]
    #[inline(always)]
    fn to_i32(self) -> i32x4 {
        unsafe { std::mem::transmute(self) }
    }
    #[cfg(target_pointer_width = "32")]
    #[inline(always)]
    fn to_u32(self) -> u32x4 {
        unsafe { std::mem::transmute(self) }
    }
    #[cfg(target_pointer_width = "32")]
    #[inline(always)]
    fn to_f32(self) -> super::f32x4::f32x4 {
        self.to_i32().to_f32()
    }
    #[cfg(target_pointer_width = "64")]
    #[inline(always)]
    fn to_f64(self) -> super::f64x2::f64x2 {
        self.to_i64().to_f64()
    }
}

impl FloatOutBinary2 for ISizeVEC {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn __log(self, _: Self) -> Self {
        panic!("Logarithm operation is not supported for i32")
    }
}

impl NormalOut2 for ISizeVEC {
    #[inline(always)]
    fn __add(self, rhs: Self) -> Self {
        self + rhs
    }

    #[inline(always)]
    fn __sub(self, rhs: Self) -> Self {
        self - rhs
    }

    #[inline(always)]
    fn __mul_add(self, a: Self, b: Self) -> Self {
        self.mul_add(a, b)
    }

    #[inline(always)]
    fn __mul(self, rhs: Self) -> Self {
        self * rhs
    }

    #[inline(always)]
    fn __pow(self, rhs: Self) -> Self {
        self.pow(rhs)
    }

    #[inline(always)]
    fn __rem(self, rhs: Self) -> Self {
        self % rhs
    }

    #[inline(always)]
    fn __max(self, rhs: Self) -> Self {
        self.max(rhs)
    }

    #[inline(always)]
    fn __min(self, rhs: Self) -> Self {
        self.min(rhs)
    }

    #[inline(always)]
    fn __clamp(self, min: Self, max: Self) -> Self {
        self.max(min).min(max)
    }
}

impl NormalOutUnary2 for ISizeVEC {
    #[inline(always)]
    fn __square(self) -> Self {
        self * self
    }

    #[inline(always)]
    fn __abs(self) -> Self {
        self.abs()
    }

    #[inline(always)]
    fn __ceil(self) -> Self {
        self
    }

    #[inline(always)]
    fn __floor(self) -> Self {
        self
    }

    #[inline(always)]
    fn __neg(self) -> Self {
        -self
    }

    #[inline(always)]
    fn __round(self) -> Self {
        self
    }

    #[inline(always)]
    fn __signum(self) -> Self {
        self.signum()
    }

    #[inline(always)]
    fn __leaky_relu(self, alpha: Self) -> Self {
        self.max(Self::splat(0)) + alpha * self.min(Self::splat(0))
    }

    #[inline(always)]
    fn __relu(self) -> Self {
        self.relu()
    }

    #[inline(always)]
    fn __relu6(self) -> Self {
        self.relu6()
    }
}

impl Eval2 for ISizeVEC {
    type Output = ISizeVEC;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        ISizeVEC::default()
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        Self(self.0.__is_true())
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        ISizeVEC::default()
    }
}
