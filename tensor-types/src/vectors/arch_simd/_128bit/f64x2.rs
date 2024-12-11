use crate::traits::{ Init, VecTrait };

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// a vector of 2 f64 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct f64x2(pub(crate) __m128d);

impl PartialEq for f64x2 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm_cmpeq_pd(self.0, other.0);
            _mm_movemask_pd(cmp) == -1
        }
    }
}

impl Default for f64x2 {
    fn default() -> Self {
        unsafe { f64x2(_mm_setzero_pd()) }
    }
}

impl VecTrait<f64> for f64x2 {
    const SIZE: usize = 2;
    type Base = f64;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[f64]) {
        unsafe {
            _mm_storeu_pd(&mut self.0 as *mut _ as *mut f64, _mm_loadu_pd(slice.as_ptr()));
        }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { f64x2(_mm_fmadd_pd(self.0, a.0, b.0)) }
    }
    #[inline(always)]
    fn sum(&self) -> f64 {
        unsafe { _mm_cvtsd_f64(_mm_hadd_pd(self.0, self.0)) }
    }
}
impl Init<f64> for f64x2 {
    fn splat(val: f64) -> f64x2 {
        unsafe { f64x2(_mm_set1_pd(val)) }
    }
}
impl std::ops::Add for f64x2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        unsafe { f64x2(_mm_add_pd(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for f64x2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        unsafe { f64x2(_mm_sub_pd(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for f64x2 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        unsafe { f64x2(_mm_mul_pd(self.0, rhs.0)) }
    }
}
impl std::ops::Div for f64x2 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        unsafe { f64x2(_mm_div_pd(self.0, rhs.0)) }
    }
}
impl std::ops::Rem for f64x2 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        unsafe {
            let x: [f64; 2] = std::mem::transmute(self.0);
            let y: [f64; 2] = std::mem::transmute(rhs.0);
            let result = [x[0] % y[0], x[1] % y[1]];
            f64x2(_mm_loadu_pd(result.as_ptr()))
        }
    }
}
impl std::ops::Neg for f64x2 {
    type Output = Self;
    fn neg(self) -> Self {
        unsafe { f64x2(_mm_xor_pd(_mm_setzero_pd(), self.0)) }
    }
}
