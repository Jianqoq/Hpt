use crate::vectors::traits::VecTrait;
use std::arch::x86_64::*;
/// a vector of 4 f64 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct f64x4(pub(crate) __m256d);

impl Default for f64x4 {
    fn default() -> Self {
        unsafe { f64x4(_mm256_setzero_pd()) }
    }
}

impl PartialEq for f64x4 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm256_cmp_pd::<0>(self.0, other.0);
            _mm256_movemask_pd(cmp) == -1
        }
    }
}

impl VecTrait<f64> for f64x4 {
    const SIZE: usize = 4;
    type Base = f64;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { f64x4(_mm256_fmadd_pd(self.0, a.0, b.0)) }
    }
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[f64]) {
        unsafe {
            _mm256_storeu_pd(self.as_mut_ptr(), _mm256_loadu_pd(slice.as_ptr()));
        }
    }
    #[inline(always)]
    fn sum(&self) -> f64 {
        unsafe {
            let hadd1 = _mm256_hadd_pd(self.0, self.0);
            let low = _mm256_castpd256_pd128(hadd1);
            let high = _mm256_extractf128_pd(hadd1, 1);
            let sum128 = _mm_add_pd(low, high);
            _mm_cvtsd_f64(sum128)
        }
    }
    fn splat(val: f64) -> f64x4 {
        f64x4(unsafe { _mm256_set1_pd(val) })
    }
}

impl std::ops::Add for f64x4 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        f64x4(unsafe { _mm256_add_pd(self.0, rhs.0) })
    }
}
impl std::ops::Sub for f64x4 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        f64x4(unsafe { _mm256_sub_pd(self.0, rhs.0) })
    }
}
impl std::ops::Mul for f64x4 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        f64x4(unsafe { _mm256_mul_pd(self.0, rhs.0) })
    }
}
impl std::ops::Div for f64x4 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        f64x4(unsafe { _mm256_div_pd(self.0, rhs.0) })
    }
}
impl std::ops::Rem for f64x4 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        unsafe {
            let div = _mm256_div_pd(self.0, rhs.0);
            let floor = _mm256_floor_pd(div);
            let mul = _mm256_mul_pd(floor, rhs.0);
            f64x4(_mm256_sub_pd(self.0, mul))
        }
    }
}
impl std::ops::Neg for f64x4 {
    type Output = Self;
    fn neg(self) -> Self {
        f64x4(unsafe { _mm256_sub_pd(_mm256_setzero_pd(), self.0) })
    }
}

