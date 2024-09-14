use std::ops::{ Deref, DerefMut, Index, IndexMut };

use crate::into_vec::IntoVec;

use crate::vectors::traits::{ Init, VecCommon, VecTrait };

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
pub struct u16x16(pub(crate) std::simd::u16x16);

impl Deref for u16x16 {
    type Target = std::simd::u16x16;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for u16x16 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl VecTrait<u16> for u16x16 {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u16]) {
        self.as_mut_array().copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const u16 {
        self.as_array().as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0 * a.0 + b.0)
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut u16 {
        self.as_mut_array().as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut u16 {
        self.as_array().as_ptr() as *mut _
    }
    #[inline(always)]
    fn sum(&self) -> u16 {
        self.as_array().iter().sum()
    }
    
    fn extract(self, idx: usize) -> u16 {
        self.as_array()[idx]
    }
}
impl VecCommon for u16x16 {
    const SIZE: usize = 16;
    
    type Base = u16;
}
impl Init<u16> for u16x16 {
    fn splat(val: u16) -> u16x16 {
        u16x16(std::simd::u16x16::splat(val))
    }
}
impl IntoVec<u16x16> for u16x16 {
    fn into_vec(self) -> u16x16 {
        self
    }
}
impl Index<usize> for u16x16 {
    type Output = u16;
    fn index(&self, idx: usize) -> &Self::Output {
        &self.as_array()[idx]
    }
}
impl IndexMut<usize> for u16x16 {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.as_mut_array()[idx]
    }
}
impl std::ops::Add for u16x16 {
    type Output = u16x16;
    fn add(self, rhs: Self) -> Self::Output {
        u16x16(self.0 + rhs.0)
    }
}
impl std::ops::Sub for u16x16 {
    type Output = u16x16;
    fn sub(self, rhs: Self) -> Self::Output {
        u16x16(self.0 - rhs.0)
    }
}
impl std::ops::Mul for u16x16 {
    type Output = u16x16;
    fn mul(self, rhs: Self) -> Self::Output {
        u16x16(self.0 * rhs.0)
    }
}
impl std::ops::Div for u16x16 {
    type Output = u16x16;
    fn div(self, rhs: Self) -> Self::Output {
        u16x16(self.0 / rhs.0)
    }
}
impl std::ops::Rem for u16x16 {
    type Output = u16x16;
    fn rem(self, rhs: Self) -> Self::Output {
        u16x16(self.0 % rhs.0)
    }
}