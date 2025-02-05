use crate::{
    convertion::VecConvertor,
    traits::{SimdCompare, SimdMath, SimdSelect, VecTrait},
    type_promote::{Eval2, FloatOutBinary2, NormalOut2, NormalOutUnary2},
};

#[cfg(target_pointer_width = "32")]
use crate::arch_simd::_256bit::u32x8::u32x8;
#[cfg(target_pointer_width = "64")]
use crate::arch_simd::_256bit::u64x4::u64x4;

use super::isizex4::isizex4;

#[cfg(target_pointer_width = "32")]
/// a vector of 4 usize values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct usizex8(pub(crate) u32x8);

#[cfg(target_pointer_width = "32")]
/// helper to impl the promote trait
#[allow(non_camel_case_types)]
pub(crate) type usize_promote = usizex8;

#[cfg(target_pointer_width = "64")]
/// a vector of 4 usize values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct usizex4(pub(crate) u64x4);

#[cfg(target_pointer_width = "64")]
/// helper to impl the promote trait
#[allow(non_camel_case_types)]
pub(crate) type usize_promote = usizex4;

#[cfg(target_pointer_width = "32")]
type USizeVEC = usizex8;
#[cfg(target_pointer_width = "64")]
type USizeVEC = usizex4;

#[cfg(target_pointer_width = "32")]
type USizeBase = u32x8;
#[cfg(target_pointer_width = "64")]
type USizeBase = u64x4;

impl PartialEq for USizeVEC {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl Default for USizeVEC {
    #[inline(always)]
    fn default() -> Self {
        Self(USizeBase::default())
    }
}

impl VecTrait<usize> for USizeVEC {
    #[cfg(target_pointer_width = "64")]
    const SIZE: usize = 4;
    #[cfg(target_pointer_width = "32")]
    const SIZE: usize = 8;
    type Base = usize;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[usize]) {
        USizeBase::copy_from_slice(&mut self.0, unsafe { std::mem::transmute(slice) });
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0.mul_add(a.0, b.0))
    }
    #[inline(always)]
    fn sum(&self) -> usize {
        self.0.sum() as usize
    }
    #[inline(always)]
    fn splat(val: usize) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            Self(USizeBase::splat(val as u64))
        }
        #[cfg(target_pointer_width = "32")]
        {
            Self(USizeBase::splat(val as u32))
        }
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const usize) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            Self(USizeBase::from_ptr(ptr as *const u64))
        }
        #[cfg(target_pointer_width = "32")]
        {
            Self(USizeBase::from_ptr(ptr as *const u32))
        }
    }
}

impl USizeVEC {
    /// convert the vector to an array
    #[cfg(target_pointer_width = "64")]
    #[allow(unused)]
    #[inline(always)]
    pub fn as_array(&self) -> [usize; 4] {
        unsafe { std::mem::transmute(self.0) }
    }
    /// convert the vector to an array
    #[cfg(target_pointer_width = "32")]
    #[allow(unused)]
    #[inline(always)]
    pub fn as_array(&self) -> [usize; 8] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl SimdCompare for USizeVEC {
    #[cfg(target_pointer_width = "64")]
    type SimdMask = isizex4;
    #[cfg(target_pointer_width = "32")]
    type SimdMask = isizex8;
    #[inline(always)]
    fn simd_eq(self, other: Self) -> Self::SimdMask {
        #[cfg(target_pointer_width = "64")]
        {
            isizex4(self.0.simd_eq(other.0))
        }
        #[cfg(target_pointer_width = "32")]
        {
            isizex8(self.0.simd_eq(other.0))
        }
    }

    #[inline(always)]
    fn simd_ne(self, other: Self) -> Self::SimdMask {
        #[cfg(target_pointer_width = "64")]
        {
            isizex4(self.0.simd_ne(other.0))
        }
        #[cfg(target_pointer_width = "32")]
        {
            isizex8(self.0.simd_ne(other.0))
        }
    }

    #[inline(always)]
    fn simd_lt(self, other: Self) -> Self::SimdMask {
        #[cfg(target_pointer_width = "64")]
        {
            isizex4(self.0.simd_lt(other.0))
        }
        #[cfg(target_pointer_width = "32")]
        {
            isizex8(self.0.simd_lt(other.0))
        }
    }

    #[inline(always)]
    fn simd_le(self, other: Self) -> Self::SimdMask {
        #[cfg(target_pointer_width = "64")]
        {
            isizex4(self.0.simd_le(other.0))
        }
        #[cfg(target_pointer_width = "32")]
        {
            isizex8(self.0.simd_le(other.0))
        }
    }

    #[inline(always)]
    fn simd_gt(self, other: Self) -> Self::SimdMask {
        #[cfg(target_pointer_width = "64")]
        {
            isizex4(self.0.simd_gt(other.0))
        }
        #[cfg(target_pointer_width = "32")]
        {
            isizex8(self.0.simd_gt(other.0))
        }
    }

    #[inline(always)]
    fn simd_ge(self, other: Self) -> Self::SimdMask {
        #[cfg(target_pointer_width = "64")]
        {
            isizex4(self.0.simd_ge(other.0))
        }
        #[cfg(target_pointer_width = "32")]
        {
            isizex8(self.0.simd_ge(other.0))
        }
    }
}

#[cfg(target_pointer_width = "64")]
type Isize = isizex4;
#[cfg(target_pointer_width = "32")]
type Isize = isizex8;

impl SimdSelect<USizeVEC> for Isize {
    #[inline(always)]
    fn select(&self, true_val: USizeVEC, false_val: USizeVEC) -> USizeVEC {
        #[cfg(target_pointer_width = "64")]
        {
            usizex4(self.0.select(true_val.0, false_val.0))
        }
        #[cfg(target_pointer_width = "32")]
        {
            usizex8(self.0.select(true_val.0, false_val.0))
        }
    }
}

impl std::ops::Add for USizeVEC {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0.add(rhs.0))
    }
}
impl std::ops::Sub for usizex4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0.sub(rhs.0))
    }
}
impl std::ops::Mul for usizex4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0.mul(rhs.0))
    }
}
impl std::ops::Div for usizex4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0.div(rhs.0))
    }
}
impl std::ops::Rem for usizex4 {
    type Output = Self;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        Self(self.0.rem(rhs.0))
    }
}
impl std::ops::BitAnd for usizex4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0.bitand(rhs.0))
    }
}
impl std::ops::BitOr for usizex4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0.bitor(rhs.0))
    }
}
impl std::ops::BitXor for usizex4 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0.bitxor(rhs.0))
    }
}
impl std::ops::Not for usizex4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self(self.0.not())
    }
}
impl std::ops::Shl for usizex4 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self {
        Self(self.0.shl(rhs.0))
    }
}
impl std::ops::Shr for usizex4 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self {
        Self(self.0.shr(rhs.0))
    }
}
impl SimdMath<usize> for usizex4 {
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
        self
    }
    #[inline(always)]
    fn pow(self, rhs: Self) -> Self {
        Self(self.0.pow(rhs.0))
    }
    #[inline(always)]
    fn leaky_relu(self, alpha: Self) -> Self {
        Self(self.0.leaky_relu(alpha.0))
    }
}

impl VecConvertor for usizex4 {
    #[inline(always)]
    fn to_usize(self) -> usizex4 {
        self
    }
    #[cfg(target_pointer_width = "64")]
    #[inline(always)]
    fn to_u64(self) -> u64x4 {
        unsafe { std::mem::transmute(self) }
    }
    #[cfg(target_pointer_width = "32")]
    #[inline(always)]
    fn to_u32(self) -> u32x8 {
        unsafe { std::mem::transmute(self) }
    }
    #[cfg(target_pointer_width = "32")]
    #[inline(always)]
    fn to_f32(self) -> super::f32x8::f32x8 {
        self.to_u32().to_f32()
    }
    #[cfg(target_pointer_width = "64")]
    #[inline(always)]
    fn to_i64(self) -> crate::simd::_256bit::i64x4::i64x4 {
        unsafe { std::mem::transmute(self) }
    }
    #[cfg(target_pointer_width = "32")]
    #[inline(always)]
    fn to_i32(self) -> crate::simd::_256bit::i32x8::i32x8 {
        unsafe { std::mem::transmute(self) }
    }
}

impl FloatOutBinary2 for USizeVEC {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn __log(self, _: Self) -> Self {
        panic!("Logarithm operation is not supported for i32")
    }
}

impl NormalOut2 for USizeVEC {
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

impl NormalOutUnary2 for USizeVEC {
    #[inline(always)]
    fn __square(self) -> Self {
        self * self
    }

    #[inline(always)]
    fn __abs(self) -> Self {
        self
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
        self
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

    #[inline(always)]
    fn __trunc(self) -> Self {
        self
    }
}

impl Eval2 for USizeVEC {
    #[cfg(target_pointer_width = "64")]
    type Output = isizex4;
    #[cfg(target_pointer_width = "32")]
    type Output = isizex8;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        #[cfg(target_pointer_width = "64")]
        {
            isizex4::default()
        }
        #[cfg(target_pointer_width = "32")]
        {
            isizex8::default()
        }
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        #[cfg(target_pointer_width = "64")]
        {
            isizex4(self.0.__is_true())
        }
        #[cfg(target_pointer_width = "32")]
        {
            isizex8(self.0.__is_true())
        }
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        #[cfg(target_pointer_width = "64")]
        {
            isizex4::default()
        }
        #[cfg(target_pointer_width = "32")]
        {
            isizex8::default()
        }
    }
}
