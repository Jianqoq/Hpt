use crate::{
    convertion::VecConvertor,
    traits::{SimdCompare, SimdMath, SimdSelect, VecTrait},
};

use super::usizex4::usizex4;

#[cfg(target_pointer_width = "32")]
use crate::arch_simd::_256bit::i32x8::i32x8;
#[cfg(target_pointer_width = "64")]
use crate::arch_simd::_256bit::i64x4::i64x4;

#[cfg(target_pointer_width = "32")]
/// a vector of 4 isize values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct isizex8(pub(crate) i32x8);

#[cfg(target_pointer_width = "64")]
/// a vector of 4 isize values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct isizex4(pub(crate) i64x4);

#[cfg(target_pointer_width = "32")]
type ISizeVEC = isizex8;
#[cfg(target_pointer_width = "64")]
type ISizeVEC = isizex4;

#[cfg(target_pointer_width = "32")]
type USizeVEC = usizex8;
#[cfg(target_pointer_width = "64")]
type USizeVEC = usizex4;

#[cfg(target_pointer_width = "32")]
type ISizeBase = i32x8;
#[cfg(target_pointer_width = "64")]
type ISizeBase = i64x4;

impl Default for ISizeVEC {
    #[inline(always)]
    fn default() -> Self {
        Self(ISizeBase::default())
    }
}

impl PartialEq for ISizeVEC {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl VecTrait<isize> for ISizeVEC {
    #[cfg(target_pointer_width = "64")]
    const SIZE: usize = 4;
    #[cfg(target_pointer_width = "32")]
    const SIZE: usize = 8;
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
        #[cfg(target_pointer_width = "64")]
        {
            Self(ISizeBase::splat(val as i64))
        }
        #[cfg(target_pointer_width = "32")]
        {
            Self(ISizeBase::splat(val as i32))
        }
    }
}

impl ISizeVEC {
    /// convert the vector to an array
    #[inline(always)]
    #[cfg(target_pointer_width = "64")]
    pub fn as_array(&self) -> [isize; 4] {
        unsafe { std::mem::transmute(self.0) }
    }
    /// convert the vector to an array
    #[inline(always)]
    #[cfg(target_pointer_width = "32")]
    pub fn as_array(&self) -> [isize; 8] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl SimdCompare for ISizeVEC {
    type SimdMask = ISizeVEC;
    #[inline(always)]
    fn simd_eq(self, other: Self) -> ISizeVEC {
        Self(self.0.simd_eq(other.0))
    }
    #[inline(always)]
    fn simd_ne(self, other: Self) -> ISizeVEC {
        Self(self.0.simd_ne(other.0))
    }
    #[inline(always)]
    fn simd_lt(self, other: Self) -> ISizeVEC {
        Self(self.0.simd_lt(other.0))
    }
    #[inline(always)]
    fn simd_le(self, other: Self) -> ISizeVEC {
        Self(self.0.simd_le(other.0))
    }
    #[inline(always)]
    fn simd_gt(self, other: Self) -> ISizeVEC {
        Self(self.0.simd_gt(other.0))
    }
    #[inline(always)]
    fn simd_ge(self, other: Self) -> ISizeVEC {
        Self(self.0.simd_ge(other.0))
    }
}

impl SimdSelect<ISizeVEC> for ISizeVEC {
    #[inline(always)]
    fn select(&self, true_val: ISizeVEC, false_val: ISizeVEC) -> ISizeVEC {
        Self(self.0.select(true_val.0, false_val.0))
    }
}

impl std::ops::Add for ISizeVEC {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0.add(rhs.0))
    }
}
impl std::ops::Sub for ISizeVEC {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.sub(rhs.0))
    }
}
impl std::ops::Mul for ISizeVEC {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(self.0.mul(rhs.0))
    }
}
impl std::ops::Div for ISizeVEC {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self(self.0.div(rhs.0))
    }
}
impl std::ops::Rem for ISizeVEC {
    type Output = Self;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self {
        Self(self.0.rem(rhs.0))
    }
}
impl std::ops::Neg for ISizeVEC {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(std::ops::Neg::neg(self.0))
    }
}
impl std::ops::BitAnd for ISizeVEC {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0.bitand(rhs.0))
    }
}
impl std::ops::BitOr for ISizeVEC {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0.bitor(rhs.0))
    }
}
impl std::ops::BitXor for ISizeVEC {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0.bitxor(rhs.0))
    }
}
impl std::ops::Not for ISizeVEC {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self::Output {
        Self(self.0.not())
    }
}
impl std::ops::Shl for ISizeVEC {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self::Output {
        Self(self.0.shl(rhs.0))
    }
}
impl std::ops::Shr for ISizeVEC {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self::Output {
        Self(self.0.shr(rhs.0))
    }
}
impl SimdMath<isize> for ISizeVEC {
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
}

impl VecConvertor for ISizeVEC {
    #[inline(always)]
    fn to_isize(self) -> ISizeVEC {
        self
    }
    #[inline(always)]
    fn to_usize(self) -> USizeVEC {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    #[cfg(target_pointer_width = "64")]
    fn to_i64(self) -> i64x4 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    #[cfg(target_pointer_width = "32")]
    fn to_i32(self) -> i32x4 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    #[cfg(target_pointer_width = "32")]
    fn to_u32(self) -> u32x4 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    #[cfg(target_pointer_width = "32")]
    fn to_f32(self) -> super::f32x4::f32x4 {
        self.to_i32().to_f32()
    }
    #[inline(always)]
    #[cfg(target_pointer_width = "64")]
    fn to_f64(self) -> super::f64x4::f64x4 {
        self.to_i64().to_f64()
    }
}
