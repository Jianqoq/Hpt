use crate::traits::{ Init, SimdSelect, VecTrait };
use crate::vectors::arch_simd::_128bit::u32x4::u32x4;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// a vector of 4 f32 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct f32x4(pub(crate) __m128);

impl PartialEq for f32x4 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm_cmpeq_ps(self.0, other.0);
            _mm_movemask_ps(cmp) == -1
        }
    }
}

impl Default for f32x4 {
    fn default() -> Self {
        unsafe { f32x4(_mm_setzero_ps()) }
    }
}

impl VecTrait<f32> for f32x4 {
    const SIZE: usize = 4;
    type Base = f32;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[f32]) {
        unsafe {
            _mm_storeu_ps(&mut self.0 as *mut _ as *mut f32, _mm_loadu_ps(slice.as_ptr()));
        }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(all(target_arch = "x86_64", not(target_feature = "fma")))]
        unsafe {
            f32x4(_mm_add_ps(_mm_mul_ps(self.0, a.0), b.0))
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "fma"))]
        unsafe {
            f32x4(_mm_fmadd_ps(self.0, a.0, b.0))
        }
    }
    #[inline(always)]
    fn sum(&self) -> f32 {
        unsafe {
            let sum = _mm_hadd_ps(self.0, self.0);
            _mm_cvtss_f32(_mm_hadd_ps(sum, sum))
        }
    }
}
impl Init<f32> for f32x4 {
    fn splat(val: f32) -> f32x4 {
        unsafe { f32x4(_mm_set1_ps(val)) }
    }
}
impl SimdSelect<f32x4> for u32x4 {
    fn select(&self, true_val: f32x4, false_val: f32x4) -> f32x4 {
        unsafe { f32x4(_mm_blendv_ps(false_val.0, true_val.0, std::mem::transmute(self.0))) }
    }
}
impl std::ops::Add for f32x4 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        unsafe { f32x4(_mm_add_ps(self.0, rhs.0)) }
    }
}

impl std::ops::Sub for f32x4 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { f32x4(_mm_sub_ps(self.0, rhs.0)) }
    }
}

impl std::ops::Mul for f32x4 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { f32x4(_mm_mul_ps(self.0, rhs.0)) }
    }
}

impl std::ops::Div for f32x4 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        unsafe { f32x4(_mm_div_ps(self.0, rhs.0)) }
    }
}

impl std::ops::Rem for f32x4 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [f32; 4] = std::mem::transmute(self.0);
            let b: [f32; 4] = std::mem::transmute(rhs.0);
            let c: [f32; 4] = [a[0] % b[0], a[1] % b[1], a[2] % b[2], a[3] % b[3]];
            f32x4(std::mem::transmute(c))
        }
    }
}
impl std::ops::Neg for f32x4 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        unsafe { f32x4(_mm_xor_ps(self.0, _mm_set1_ps(-0.0))) }
    }
}
