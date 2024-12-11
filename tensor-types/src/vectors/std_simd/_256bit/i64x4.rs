use std::ops::{ Deref, DerefMut, Index, IndexMut };

use crate::{traits::SimdSelect, vectors::traits::{ Init, VecTrait }};

/// a vector of 4 i64 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(32))]
pub struct i64x4(pub(crate) std::simd::i64x4);

impl Deref for i64x4 {
    type Target = std::simd::i64x4;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for i64x4 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl VecTrait<i64> for i64x4 {
    const SIZE: usize = 4;
    type Base = i64;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0 * a.0 + b.0)
    }
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[i64]) {
        self.as_mut_array().copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const i64 {
        self.as_array().as_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut i64 {
        self.as_mut_array().as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut i64 {
        self.as_array().as_ptr() as *mut _
    }
    #[inline(always)]
    fn sum(&self) -> i64 {
        self.as_array().iter().sum()
    }
}

impl SimdSelect<i64x4> for crate::vectors::std_simd::_256bit::u64x4::u64x4 {
    fn select(&self, true_val: i64x4, false_val: i64x4) -> i64x4 {
        let mask: std::simd::mask64x4 = unsafe { std::mem::transmute(*self) };
        i64x4(mask.select(true_val.0, false_val.0))
    }
}
impl Init<i64> for i64x4 {
    fn splat(val: i64) -> i64x4 {
        i64x4(std::simd::i64x4::splat(val))
    }
}
impl Index<usize> for i64x4 {
    type Output = i64;
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_array()[index]
    }
}
impl IndexMut<usize> for i64x4 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_array()[index]
    }
}
impl std::ops::Add for i64x4 {
    type Output = i64x4;
    fn add(self, rhs: Self) -> Self::Output {
        i64x4(self.0 + rhs.0)
    }
}
impl std::ops::Sub for i64x4 {
    type Output = i64x4;
    fn sub(self, rhs: Self) -> Self::Output {
        i64x4(self.0 - rhs.0)
    }
}
impl std::ops::Mul for i64x4 {
    type Output = i64x4;
    fn mul(self, rhs: Self) -> Self::Output {
        i64x4(self.0 * rhs.0)
    }
}
impl std::ops::Div for i64x4 {
    type Output = i64x4;
    fn div(self, rhs: Self) -> Self::Output {
        i64x4(self.0 / rhs.0)
    }
}
impl std::ops::Rem for i64x4 {
    type Output = i64x4;
    fn rem(self, rhs: Self) -> Self::Output {
        i64x4(self.0 % rhs.0)
    }
}
impl std::ops::Neg for i64x4 {
    type Output = i64x4;
    fn neg(self) -> Self::Output {
        i64x4(-self.0)
    }
}