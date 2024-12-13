use crate::{convertion::VecConvertor, traits::{ SimdCompare, SimdMath, SimdSelect, VecTrait }};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::i64x4::i64x4;

/// a vector of 2 u64 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct u64x4(pub(crate) __m256i);

impl PartialEq for u64x4 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm256_cmpeq_epi64(self.0, other.0);
            _mm256_movemask_epi8(cmp) == -1
        }
    }
}

impl Default for u64x4 {
    fn default() -> Self {
        unsafe { u64x4(_mm256_setzero_si256()) }
    }
}

impl VecTrait<u64> for u64x4 {
    const SIZE: usize = 4;
    type Base = u64;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u64]) {
        unsafe { _mm256_storeu_si256(&mut self.0, _mm256_loadu_si256(slice.as_ptr() as *const __m256i)) }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe {
            let arr: [u64; 4] = std::mem::transmute(self.0);
            let arr2: [u64; 4] = std::mem::transmute(a.0);
            let arr3: [u64; 4] = std::mem::transmute(b.0);
            let mut arr4: [u64; 4] = [0; 4];
            for i in 0..4 {
                arr4[i] = arr[i] * arr2[i] + arr3[i];
            }
            u64x4(_mm256_loadu_si256(arr4.as_ptr() as *const __m256i))
        }
    }
    #[inline(always)]
    fn sum(&self) -> u64 {
        unsafe {
            let arr: [u64; 4] = std::mem::transmute(self.0);
            arr.iter().sum()
        }
    }
    fn splat(val: u64) -> u64x4 {
        unsafe { u64x4(_mm256_set1_epi64x(val as i64)) }
    }
}

impl u64x4 {
    #[allow(unused)]
    fn as_array(&self) -> [u64; 4] {
        unsafe { std::mem::transmute(self.0) }
    }
}


impl SimdCompare for u64x4 {
    type SimdMask = i64x4;

    fn simd_eq(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i64x4 = std::mem::transmute(self.0);
            let rhs: i64x4 = std::mem::transmute(other.0);
            lhs.simd_eq(rhs)
        }
    }

    fn simd_ne(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i64x4 = std::mem::transmute(self.0);
            let rhs: i64x4 = std::mem::transmute(other.0);
            lhs.simd_ne(rhs)
        }
    }

    fn simd_lt(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i64x4 = std::mem::transmute(self.0);
            let rhs: i64x4 = std::mem::transmute(other.0);
            lhs.simd_lt(rhs)
        }
    }

    fn simd_le(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i64x4 = std::mem::transmute(self.0);
            let rhs: i64x4 = std::mem::transmute(other.0);
            lhs.simd_le(rhs)
        }
    }

    fn simd_gt(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i64x4 = std::mem::transmute(self.0);
            let rhs: i64x4 = std::mem::transmute(other.0);
            lhs.simd_gt(rhs)
        }
    }

    fn simd_ge(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i64x4 = std::mem::transmute(self.0);
            let rhs: i64x4 = std::mem::transmute(other.0);
            lhs.simd_ge(rhs)
        }
    }
}

impl SimdSelect<u64x4> for u64x4 {
    fn select(&self, true_val: u64x4, false_val: u64x4) -> u64x4 {
        unsafe { u64x4(_mm256_blendv_epi8(false_val.0, true_val.0, self.0)) }
    }
}

impl std::ops::Add for u64x4 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        unsafe { u64x4(_mm256_add_epi64(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for u64x4 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        unsafe { u64x4(_mm256_sub_epi64(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for u64x4 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            let arr: [u64; 4] = std::mem::transmute(self.0);
            let arr2: [u64; 4] = std::mem::transmute(rhs.0);
            let mut arr3: [u64; 4] = [0; 4];
            for i in 0..4 {
                arr3[i] = arr[i] * arr2[i];
            }
            u64x4(_mm256_loadu_si256(arr3.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Div for u64x4 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        unsafe {
            let arr: [u64; 4] = std::mem::transmute(self.0);
            let arr2: [u64; 4] = std::mem::transmute(rhs.0);
            let mut arr3: [u64; 4] = [0; 4];
            for i in 0..4 {
                arr3[i] = arr[i] / arr2[i];
            }
            u64x4(_mm256_loadu_si256(arr3.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Rem for u64x4 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        unsafe {
            let arr: [u64; 4] = std::mem::transmute(self.0);
            let arr2: [u64; 4] = std::mem::transmute(rhs.0);
            let mut arr3: [u64; 4] = [0; 4];
            for i in 0..4 {
                arr3[i] = arr[i] % arr2[i];
            }
            u64x4(_mm256_loadu_si256(arr3.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::BitAnd for u64x4 {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        unsafe { u64x4(_mm256_and_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitOr for u64x4 {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        unsafe { u64x4(_mm256_or_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for u64x4 {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self {
        unsafe { u64x4(_mm256_xor_si256(self.0, rhs.0)) }
    }
}
impl std::ops::Not for u64x4 {
    type Output = Self;
    fn not(self) -> Self {
        unsafe { u64x4(_mm256_xor_si256(self.0, _mm256_set1_epi64x(-1))) }
    }
}
impl std::ops::Shl for u64x4 {
    type Output = Self;
    fn shl(self, rhs: Self) -> Self {
        unsafe {
            let a: [u64; 4] = std::mem::transmute(self.0);
            let b: [u64; 4] = std::mem::transmute(rhs.0);
            let mut result = [0; 4];
            for i in 0..4 {
                result[i] = a[i] << b[i];
            }
            u64x4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Shr for u64x4 {
    type Output = Self;
    fn shr(self, rhs: Self) -> Self {
        unsafe {
            let a: [u64; 4] = std::mem::transmute(self.0);
            let b: [u64; 4] = std::mem::transmute(rhs.0);
            let mut result = [0; 4];
            for i in 0..4 {
                result[i] = a[i] >> b[i];
            }
            u64x4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl SimdMath<u64> for u64x4 {
    fn max(self, other: Self) -> Self {
        unsafe {
            let arr: [u64; 4] = std::mem::transmute(self.0);
            let arr2: [u64; 4] = std::mem::transmute(other.0);
            let mut arr3: [u64; 4] = [0; 4];
            for i in 0..4 {
                arr3[i] = arr[i].max(arr2[i]);
            }
            u64x4(_mm256_loadu_si256(arr3.as_ptr() as *const __m256i))
        }
    }
    fn min(self, other: Self) -> Self {
        unsafe {
            let arr: [u64; 4] = std::mem::transmute(self.0);
            let arr2: [u64; 4] = std::mem::transmute(other.0);
            let mut arr3: [u64; 4] = [0; 4];
            for i in 0..4 {
                arr3[i] = arr[i].min(arr2[i]);
            }
            u64x4(_mm256_loadu_si256(arr3.as_ptr() as *const __m256i))
        }
    }
    fn relu(self) -> Self {
        unsafe {
            let arr: [u64; 4] = std::mem::transmute(self.0);
            let mut arr2: [u64; 4] = [0; 4];
            for i in 0..4 {
                arr2[i] = arr[i].max(0);
            }
            u64x4(_mm256_loadu_si256(arr2.as_ptr() as *const __m256i))
        }
    }
    fn relu6(self) -> Self {
        unsafe {
            let arr: [u64; 4] = std::mem::transmute(self.0);
            let mut arr2: [u64; 4] = [0; 4];
            for i in 0..4 {
                arr2[i] = arr[i].max(0).min(6);
            }
            u64x4(_mm256_loadu_si256(arr2.as_ptr() as *const __m256i))
        }
    }
}

impl VecConvertor for u64x4 {
    fn to_u64(self) -> u64x4 {
        self
    }
    fn to_i64(self) -> i64x4 {
        unsafe { std::mem::transmute(self) }
    }
    fn to_f64(self) -> super::f64x4::f64x4 {
        unsafe {
            let arr: [u64; 4] = std::mem::transmute(self.0);
            let mut result = [0.0f64; 4];
            for i in 0..4 {
                result[i] = arr[i] as f64;
            }
            super::f64x4::f64x4(_mm256_loadu_pd(result.as_ptr()))
        }
    }
    #[cfg(target_pointer_width = "64")]
    fn to_isize(self) -> super::isizex4::isizex4 {
        unsafe { std::mem::transmute(self) }
    }
    #[cfg(target_pointer_width = "64")]
    fn to_usize(self) -> super::usizex4::usizex4 {
        unsafe { std::mem::transmute(self) }
    }
}