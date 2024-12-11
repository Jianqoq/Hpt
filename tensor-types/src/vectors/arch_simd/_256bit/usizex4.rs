use crate::vectors::traits::VecTrait;
use std::arch::x86_64::*;
/// a vector of 4 usize values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct usizex4(pub(crate) __m256i);

impl PartialEq for usizex4 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm256_cmpeq_epi64(self.0, other.0);
            _mm256_movemask_epi8(cmp) == -1
        }
    }
}

impl Default for usizex4 {
    fn default() -> Self {
        unsafe { usizex4(_mm256_setzero_si256()) }
    }
}

impl VecTrait<usize> for usizex4 {
    const SIZE: usize = 4;
    type Base = usize;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(target_pointer_width = "64")]
        unsafe {
            let x: [u64; 4] = std::mem::transmute(self.0);
            let y: [u64; 4] = std::mem::transmute(a.0);
            let z: [u64; 4] = std::mem::transmute(b.0);
            let mut result = [0u64; 4];
            for i in 0..4 {
                result[i] = x[i].wrapping_mul(y[i]).wrapping_add(z[i]);
            }
            usizex4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
        #[cfg(target_pointer_width = "32")]
        unsafe {
            let x: [u32; 4] = std::mem::transmute(self.0);
            let y: [u32; 4] = std::mem::transmute(a.0);
            let z: [u32; 4] = std::mem::transmute(b.0);
            let mut result = [0u32; 4];
            for i in 0..4 {
                result[i] = x[i].wrapping_mul(y[i]).wrapping_add(z[i]);
            }
            usizex4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[usize]) {
        unsafe {
            let ptr = slice.as_ptr() as *const usize;
            self.0 = _mm256_loadu_si256(ptr as *const __m256i);
        }
    }
    #[inline(always)]
    fn sum(&self) -> usize {
        unsafe {
            let array: [usize; 4] = std::mem::transmute(self.0);
            array.iter().sum()
        }
    }
    fn splat(val: usize) -> usizex4 {
        #[cfg(target_pointer_width = "64")]
        let ret = usizex4(unsafe { _mm256_set1_epi64x(val as i64) });
        #[cfg(target_pointer_width = "32")]
        let ret = usizex4(unsafe { _mm256_set1_epi32(val as i32) });
        ret
    }
}

impl std::ops::Add for usizex4 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        #[cfg(target_pointer_width = "64")]
        return usizex4(unsafe { _mm256_add_epi64(self.0, rhs.0) });
        #[cfg(target_pointer_width = "32")]
        return usizex4(unsafe { _mm256_add_epi32(self.0, rhs.0) });
    }
}
impl std::ops::Sub for usizex4 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        #[cfg(target_pointer_width = "64")]
        return usizex4(unsafe { _mm256_sub_epi64(self.0, rhs.0) });
        #[cfg(target_pointer_width = "32")]
        return usizex4(unsafe { _mm256_sub_epi32(self.0, rhs.0) });
    }
}
impl std::ops::Mul for usizex4 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        #[cfg(target_pointer_width = "64")]
        unsafe {
            let x: [u64; 4] = std::mem::transmute(self.0);
            let y: [u64; 4] = std::mem::transmute(rhs.0);
            let mut result = [0u64; 4];
            for i in 0..4 {
                result[i] = x[i].wrapping_mul(y[i]);
            }
            usizex4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
        #[cfg(target_pointer_width = "32")]
        return usizex4(unsafe { _mm256_mullo_epi32(self.0, rhs.0) });
    }
}
impl std::ops::Div for usizex4 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        #[cfg(target_pointer_width = "64")]
        unsafe {
            let a: [u64; 4] = std::mem::transmute(self.0);
            let b: [u64; 4] = std::mem::transmute(rhs.0);
            let mut result = [0u64; 4];
            for i in 0..4 {
                result[i] = a[i] / b[i];
            }
            usizex4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
        #[cfg(target_pointer_width = "32")]
        unsafe {
            let a: [u32; 4] = std::mem::transmute(self.0);
            let b: [u32; 4] = std::mem::transmute(rhs.0);
            let mut result = [0u32; 4];
            for i in 0..4 {
                result[i] = a[i] / b[i];
            }
            usizex4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Rem for usizex4 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        #[cfg(target_pointer_width = "64")]
        unsafe {
            let a: [u64; 4] = std::mem::transmute(self.0);
            let b: [u64; 4] = std::mem::transmute(rhs.0);
            let mut result = [0u64; 4];
            for i in 0..4 {
                result[i] = a[i] % b[i];
            }
            usizex4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
        #[cfg(target_pointer_width = "32")]
        unsafe {
            let a: [u32; 4] = std::mem::transmute(self.0);
            let b: [u32; 4] = std::mem::transmute(rhs.0);
            let mut result = [0u32; 4];
            for i in 0..4 {
                result[i] = a[i] % b[i];
            }
            usizex4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
