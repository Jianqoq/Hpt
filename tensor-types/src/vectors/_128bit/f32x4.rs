use crate::traits::{ Init, SimdSelect, VecCommon, VecTrait };
use crate::vectors::_128bit::u32x4::u32x4;
use std::ops::{ Deref, DerefMut, Index, IndexMut };
use std::simd::num::SimdFloat;
use std::simd::StdFloat;

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
pub struct f32x4(pub(crate) std::simd::f32x4);

impl Deref for f32x4 {
    type Target = std::simd::f32x4;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for f32x4 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl VecTrait<f32> for f32x4 {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[f32]) {
        self.as_mut_array().copy_from_slice(slice)
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const f32 {
        self.as_array().as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, a: Self, b: Self) -> Self {
        f32x4(self.0.mul_add(a.0, b.0))
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut f32 {
        self.as_mut_array().as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut f32 {
        self.as_array().as_ptr() as *mut _
    }
    #[inline(always)]
    fn sum(&self) -> f32 {
        self.reduce_sum()
    }

    fn extract(self, idx: usize) -> f32 {
        self.as_array()[idx]
    }
}
impl VecCommon for f32x4 {
    const SIZE: usize = 4;

    type Base = f32;
}
impl Init<f32> for f32x4 {
    fn splat(val: f32) -> f32x4 {
        f32x4(std::simd::f32x4::splat(val))
    }
    unsafe fn from_ptr(ptr: *const f32) -> Self where Self: Sized {
        #[cfg(target_feature = "neon")]
        {
            use std::arch::aarch64::vld1q_f32;
            f32x4(std::mem::transmute(vld1q_f32(ptr)))
        }
        #[cfg(target_feature = "sse")]
        {
            use std::arch::x86_64::_mm_loadu_ps;
            f32x4(
                std::mem::transmute(_mm_loadu_ps(ptr))
            )
        }
    }
}
impl SimdSelect<f32x4> for u32x4 {
    fn select(&self, true_val: f32x4, false_val: f32x4) -> f32x4 {
        let mask: std::simd::mask32x4 = unsafe { std::mem::transmute(*self) };
        f32x4(mask.select(true_val.0, false_val.0))
    }
}
impl Index<usize> for f32x4 {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_array()[index]
    }
}

impl IndexMut<usize> for f32x4 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_array()[index]
    }
}
impl std::ops::Add for f32x4 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        f32x4(self.0 + rhs.0)
    }
}

impl std::ops::Sub for f32x4 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        f32x4(self.0 - rhs.0)
    }
}

impl std::ops::Mul for f32x4 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        f32x4(self.0 * rhs.0)
    }
}

impl std::ops::Div for f32x4 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        f32x4(self.0 / rhs.0)
    }
}

impl std::ops::Rem for f32x4 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        f32x4(self.0 % rhs.0)
    }
}
