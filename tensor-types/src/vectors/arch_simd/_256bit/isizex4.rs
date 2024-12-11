use std::ops::{ Index, IndexMut };

use crate::vectors::traits::{ Init, VecTrait };
use std::arch::x86_64::*;

/// a vector of 4 isize values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct isizex4(pub(crate) __m256i);

impl Default for isizex4 {
    fn default() -> Self {
        isizex4(unsafe { _mm256_setzero_si256() })
    }
}

impl PartialEq for isizex4 {
    fn eq(&self, other: &Self) -> bool {
        #[cfg(target_pointer_width = "64")]
        unsafe {
            let cmp = _mm256_cmpeq_epi64(self.0, other.0);
            _mm256_movemask_epi8(cmp) == -1
        }
        #[cfg(target_pointer_width = "32")]
        unsafe {
            let cmp = _mm256_cmpeq_epi32(self.0, other.0);
            _mm256_movemask_epi8(cmp) == -1
        }
    }
}

impl VecTrait<isize> for isizex4 {
    const SIZE: usize = 4;
    type Base = isize;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        unsafe {
            let x: [isize; 4] = std::mem::transmute(self.0);
            let y: [isize; 4] = std::mem::transmute(a.0);
            let z: [isize; 4] = std::mem::transmute(b.0);
            let mut result = [0isize; 4];
            for i in 0..4 {
                result[i] = x[i].wrapping_mul(y[i]).wrapping_add(z[i]);
            }
            isizex4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
        #[cfg(target_pointer_width = "32")]
        unsafe {
            let x: [isize; 4] = std::mem::transmute(self.0);
            let y: [isize; 4] = std::mem::transmute(a.0);
            let z: [isize; 4] = std::mem::transmute(b.0);
            let mut result = [0isize; 4];
            for i in 0..4 {
                result[i] = x[i].wrapping_mul(y[i]).wrapping_add(z[i]);
            }
            isizex4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[isize]) {
        self.0 = unsafe { _mm256_loadu_si256(slice.as_ptr() as *const __m256i) };
    }
    #[inline(always)]
    fn sum(&self) -> isize {
        unsafe {
            let array: [isize; 4] = std::mem::transmute(self.0);
            array.iter().sum()
        }
    }
}

impl Init<isize> for isizex4 {
    fn splat(val: isize) -> isizex4 {
        #[cfg(target_pointer_width = "64")]
        return isizex4(unsafe { _mm256_set1_epi64x(val as i64) });
        #[cfg(target_pointer_width = "32")]
        return isizex4(unsafe { _mm256_set1_epi32(val as i32) });
    }
}

impl Index<usize> for isizex4 {
    type Output = isize;
    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < 4, "Index out of bounds for f32x8");
        unsafe { &*self.as_ptr().add(index) }
    }
}
impl IndexMut<usize> for isizex4 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < 4, "Index out of bounds for f32x8");
        unsafe { &mut *self.as_mut_ptr().add(index) }
    }
}
impl std::ops::Add for isizex4 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        return isizex4(unsafe { _mm256_add_epi64(self.0, rhs.0) });
        #[cfg(target_pointer_width = "32")]
        return isizex4(unsafe { _mm256_add_epi32(self.0, rhs.0) });
    }
}
impl std::ops::Sub for isizex4 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        return isizex4(unsafe { _mm256_sub_epi64(self.0, rhs.0) });
        #[cfg(target_pointer_width = "32")]
        return isizex4(unsafe { _mm256_sub_epi32(self.0, rhs.0) });
    }
}
impl std::ops::Mul for isizex4 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        unsafe {
            let x: [i64; 4] = std::mem::transmute(self.0);
            let y: [i64; 4] = std::mem::transmute(rhs.0);
            let mut result = [0i64; 4];
            for i in 0..4 {
                result[i] = x[i].wrapping_mul(y[i]);
            }
            isizex4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
        #[cfg(target_pointer_width = "32")]
        return isizex4(unsafe { _mm256_mullo_epi32(self.0, rhs.0) });
    }
}
impl std::ops::Div for isizex4 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        unsafe {
            let x: [i64; 4] = std::mem::transmute(self.0);
            let y: [i64; 4] = std::mem::transmute(rhs.0);
            let mut result = [0i64; 4];
            for i in 0..4 {
                result[i] = x[i] / y[i];
            }
            isizex4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
        #[cfg(target_pointer_width = "32")]
        unsafe {
            let x: [i32; 4] = std::mem::transmute(self.0);
            let y: [i32; 4] = std::mem::transmute(rhs.0);
            let mut result = [0i32; 4];
            for i in 0..4 {
                result[i] = x[i] / y[i];
            }
            isizex4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Rem for isizex4 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        unsafe {
            let x: [i64; 4] = std::mem::transmute(self.0);
            let y: [i64; 4] = std::mem::transmute(rhs.0);
            let mut result = [0i64; 4];
            for i in 0..4 {
                result[i] = x[i] % y[i];
            }
            isizex4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
        #[cfg(target_pointer_width = "32")]
        unsafe {
            let x: [i32; 4] = std::mem::transmute(self.0);
            let y: [i32; 4] = std::mem::transmute(rhs.0);
            let mut result = [0i32; 4];
            for i in 0..4 {
                result[i] = x[i] % y[i];
            }
            isizex4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Neg for isizex4 {
    type Output = Self;
    fn neg(self) -> Self {
        #[cfg(target_pointer_width = "64")]
        return isizex4(unsafe { _mm256_sub_epi64(_mm256_setzero_si256(), self.0) });
        #[cfg(target_pointer_width = "32")]
        return isizex4(unsafe { _mm256_sub_epi32(_mm256_setzero_si256(), self.0) });
    }
}
