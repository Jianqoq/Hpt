use crate::{ traits::SimdCompare, vectors::traits::VecTrait };
use std::arch::x86_64::*;

/// a vector of 16 u16 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct u16x16(pub(crate) __m256i);

impl Default for u16x16 {
    fn default() -> Self {
        u16x16(unsafe { _mm256_setzero_si256() })
    }
}

impl PartialEq for u16x16 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm256_cmpeq_epi16(self.0, other.0);
            _mm256_movemask_epi8(cmp) == -1
        }
    }
}

impl VecTrait<u16> for u16x16 {
    const SIZE: usize = 16;
    type Base = u16;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        u16x16(unsafe { _mm256_add_epi16(_mm256_mullo_epi16(self.0, a.0), b.0) })
    }
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u16]) {
        self.0 = unsafe { _mm256_loadu_si256(slice.as_ptr() as *const __m256i) };
    }
    #[inline(always)]
    fn sum(&self) -> u16 {
        unsafe {
            let array: [u16; 16] = std::mem::transmute(self.0);
            array.iter().sum()
        }
    }
    fn splat(val: u16) -> u16x16 {
        u16x16(unsafe { _mm256_set1_epi16(val as i16) })
    }
}

impl std::ops::Add for u16x16 {
    type Output = u16x16;
    fn add(self, rhs: Self) -> Self::Output {
        u16x16(unsafe { _mm256_add_epi16(self.0, rhs.0) })
    }
}
impl std::ops::Sub for u16x16 {
    type Output = u16x16;
    fn sub(self, rhs: Self) -> Self::Output {
        u16x16(unsafe { _mm256_sub_epi16(self.0, rhs.0) })
    }
}
impl std::ops::Mul for u16x16 {
    type Output = u16x16;
    fn mul(self, rhs: Self) -> Self::Output {
        u16x16(unsafe { _mm256_mullo_epi16(self.0, rhs.0) })
    }
}
impl std::ops::Div for u16x16 {
    type Output = u16x16;
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [u16; 16] = std::mem::transmute(self.0);
            let b: [u16; 16] = std::mem::transmute(rhs.0);
            let mut result = [0u16; 16];
            for i in 0..16 {
                result[i] = a[i] / b[i];
            }
            u16x16(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Rem for u16x16 {
    type Output = u16x16;
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [u16; 16] = std::mem::transmute(self.0);
            let b: [u16; 16] = std::mem::transmute(rhs.0);
            let mut result = [0u16; 16];
            for i in 0..16 {
                result[i] = a[i] % b[i];
            }
            u16x16(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::BitAnd for u16x16 {
    type Output = u16x16;
    fn bitand(self, rhs: Self) -> Self::Output {
        u16x16(unsafe { _mm256_and_si256(self.0, rhs.0) })
    }
}
impl SimdCompare for u16x16 {
    type SimdMask = u16x16;
    fn simd_eq(self, other: Self) -> Self::SimdMask {
        u16x16(unsafe { _mm256_cmpeq_epi16(self.0, other.0) })
    }

    fn simd_ne(self, other: Self) -> Self::SimdMask {
        let eq = unsafe { _mm256_cmpeq_epi16(self.0, other.0) };
        u16x16(unsafe { _mm256_xor_si256(eq, _mm256_set1_epi16(-1)) })
    }

    fn simd_lt(self, other: Self) -> Self::SimdMask {
        unsafe {
            let offset = _mm256_set1_epi16(-32768);
            let a = _mm256_add_epi16(self.0, offset);
            let b = _mm256_add_epi16(other.0, offset);
            u16x16(_mm256_cmpgt_epi16(b, a))
        }
    }

    fn simd_le(self, other: Self) -> Self::SimdMask {
        unsafe {
            let gt = _mm256_cmpgt_epi16(self.0, other.0);
            u16x16(_mm256_xor_si256(gt, _mm256_set1_epi16(-1)))
        }
    }

    fn simd_gt(self, other: Self) -> Self::SimdMask {
        u16x16(unsafe { _mm256_cmpgt_epi16(self.0, other.0) })
    }

    fn simd_ge(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lt = _mm256_cmpgt_epi16(other.0, self.0);
            u16x16(_mm256_xor_si256(lt, _mm256_set1_epi16(-1)))
        }
    }
}
