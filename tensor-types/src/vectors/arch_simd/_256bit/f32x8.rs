use crate::vectors::traits::{ SimdSelect, VecTrait };
use std::arch::x86_64::*;

/// a vector of 8 f32 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct f32x8(pub(crate) __m256);

impl Default for f32x8 {
    fn default() -> Self {
        unsafe {
            f32x8(_mm256_setzero_ps())
        }
    }
}

impl PartialEq for f32x8 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm256_cmp_ps(self.0, other.0, _CMP_EQ_OQ);
            _mm256_movemask_ps(cmp) == -1
        }
    }
}

impl VecTrait<f32> for f32x8 {
    const SIZE: usize = 8;
    type Base = f32;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f32x8(unsafe { _mm256_fmadd_ps(self.0, a.0, b.0) })
    }
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[f32]) {
        self.0 = unsafe { _mm256_loadu_ps(slice.as_ptr()) };
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const f32 {
        self as *const _ as *const f32
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut f32 {
        self as *mut _ as *mut f32
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut f32 {
        unsafe {
            std::mem::transmute(self.as_ptr())
        }
    }
    #[inline(always)]
    fn sum(&self) -> f32 {
        unsafe {
            let hadd1 = _mm256_hadd_ps(self.0, self.0);
            let hadd2 = _mm256_hadd_ps(hadd1, hadd1);
            let low = _mm256_castps256_ps128(hadd2);
            let high = _mm256_extractf128_ps(hadd2, 1);
            let sum128 = _mm_add_ps(low, high);
            _mm_cvtss_f32(sum128)
        }
    }
    fn splat(val: f32) -> f32x8 {
        f32x8(unsafe { _mm256_set1_ps(val) })
    }
}

impl SimdSelect<f32x8> for crate::vectors::arch_simd::_256bit::u32x8::u32x8 {
    fn select(&self, true_val: f32x8, false_val: f32x8) -> f32x8 {
        unsafe {
            let mask = _mm256_castsi256_ps(self.0);
            f32x8(_mm256_blendv_ps(false_val.0, true_val.0, mask))
        }
    }
}

impl std::ops::Add for f32x8 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        f32x8(unsafe { _mm256_add_ps(self.0, rhs.0) })
    }
}

impl std::ops::Sub for f32x8 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        f32x8(unsafe { _mm256_sub_ps(self.0, rhs.0) })
    }
}

impl std::ops::Mul for f32x8 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        f32x8(unsafe { _mm256_mul_ps(self.0, rhs.0) })
    }
}

impl std::ops::Div for f32x8 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        f32x8(unsafe { _mm256_div_ps(self.0, rhs.0) })
    }
}

impl std::ops::Rem for f32x8 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let div = _mm256_div_ps(self.0, rhs.0);
            let floor = _mm256_floor_ps(div);
            let mul = _mm256_mul_ps(floor, rhs.0);
            f32x8(_mm256_sub_ps(self.0, mul))
        }
    }
}
impl std::ops::Neg for f32x8 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        f32x8(unsafe { _mm256_sub_ps(_mm256_setzero_ps(), self.0) })
    }
}
