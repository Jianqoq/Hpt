use std::ops::{ Index, IndexMut };

use crate::{ traits::SimdSelect, vectors::traits::{ Init, VecTrait } };
use std::arch::x86_64::*;

/// a vector of 4 i64 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct i64x4(pub(crate) __m256i);

impl PartialEq for i64x4 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm256_cmpeq_epi64(self.0, other.0);
            _mm256_movemask_epi8(cmp) == -1
        }
    }
}

impl Default for i64x4 {
    fn default() -> Self {
        i64x4(unsafe { _mm256_setzero_si256() })
    }
}

impl VecTrait<i64> for i64x4 {
    const SIZE: usize = 4;
    type Base = i64;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe {
            let x: [i64; 4] = std::mem::transmute(self.0);
            let y: [i64; 4] = std::mem::transmute(a.0);
            let z: [i64; 4] = std::mem::transmute(b.0);
            let mut result = [0i64; 4];
            for i in 0..4 {
                result[i] = x[i].wrapping_mul(y[i]).wrapping_add(z[i]);
            }
            i64x4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[i64]) {
        self.0 = unsafe { _mm256_loadu_si256(slice.as_ptr() as *const __m256i) };
    }
    #[inline(always)]
    fn sum(&self) -> i64 {
        unsafe {
            let x: [i64; 4] = std::mem::transmute(self.0);
            x.iter().sum()
        }
    }
}

impl SimdSelect<i64x4> for crate::vectors::arch_simd::_256bit::u64x4::u64x4 {
    fn select(&self, true_val: i64x4, false_val: i64x4) -> i64x4 {
        unsafe {
            let mask = _mm256_castsi256_pd(self.0);
            i64x4(
                _mm256_castpd_si256(
                    _mm256_blendv_pd(
                        _mm256_castsi256_pd(false_val.0),
                        _mm256_castsi256_pd(true_val.0),
                        mask
                    )
                )
            )
        }
    }
}

impl Init<i64> for i64x4 {
    fn splat(val: i64) -> i64x4 {
        i64x4(unsafe { _mm256_set1_epi64x(val) })
    }
}
impl Index<usize> for i64x4 {
    type Output = i64;
    fn index(&self, index: usize) -> &Self::Output {
        unsafe { &*self.as_ptr().add(index) }
    }
}
impl IndexMut<usize> for i64x4 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { &mut *self.as_mut_ptr().add(index) }
    }
}
impl std::ops::Add for i64x4 {
    type Output = i64x4;
    fn add(self, rhs: Self) -> Self::Output {
        i64x4(unsafe { _mm256_add_epi64(self.0, rhs.0) })
    }
}
impl std::ops::Sub for i64x4 {
    type Output = i64x4;
    fn sub(self, rhs: Self) -> Self::Output {
        i64x4(unsafe { _mm256_sub_epi64(self.0, rhs.0) })
    }
}
impl std::ops::Mul for i64x4 {
    type Output = i64x4;
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe {
            let x: [i64; 4] = std::mem::transmute(self.0);
            let y: [i64; 4] = std::mem::transmute(rhs.0);
            let mut result = [0i64; 4];
            for i in 0..4 {
                result[i] = x[i].wrapping_mul(y[i]);
            }
            i64x4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Div for i64x4 {
    type Output = i64x4;
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let x: [i64; 4] = std::mem::transmute(self.0);
            let y: [i64; 4] = std::mem::transmute(rhs.0);
            let mut result = [0i64; 4];
            for i in 0..4 {
                result[i] = x[i] / y[i];
            }
            i64x4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Rem for i64x4 {
    type Output = i64x4;
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let x: [i64; 4] = std::mem::transmute(self.0);
            let y: [i64; 4] = std::mem::transmute(rhs.0);
            let mut result = [0i64; 4];
            for i in 0..4 {
                result[i] = x[i] % y[i];
            }
            i64x4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Neg for i64x4 {
    type Output = i64x4;
    fn neg(self) -> Self::Output {
        i64x4(unsafe { _mm256_sub_epi64(_mm256_setzero_si256(), self.0) })
    }
}
