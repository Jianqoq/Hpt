use crate::{
    convertion::VecConvertor,
    traits::{SimdCompare, SimdMath, SimdSelect, VecTrait},
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::u64x4::u64x4;

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
        unsafe { i64x4(_mm256_setzero_si256()) }
    }
}

impl VecTrait<i64> for i64x4 {
    const SIZE: usize = 4;
    type Base = i64;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[i64]) {
        unsafe {
            _mm256_storeu_si256(
                &mut self.0,
                _mm256_loadu_si256(slice.as_ptr() as *const __m256i),
            )
        }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe {
            let arr: [i64; 4] = std::mem::transmute(self.0);
            let arr2: [i64; 4] = std::mem::transmute(a.0);
            let arr3: [i64; 4] = std::mem::transmute(b.0);
            let mut arr4: [i64; 4] = [0; 4];
            for i in 0..4 {
                arr4[i] = arr[i] * arr2[i] + arr3[i];
            }
            i64x4(_mm256_loadu_si256(arr4.as_ptr() as *const __m256i))
        }
    }
    #[inline(always)]
    fn sum(&self) -> i64 {
        unsafe {
            let arr: [i64; 4] = std::mem::transmute(self.0);
            arr.iter().sum()
        }
    }
    fn splat(val: i64) -> i64x4 {
        unsafe { i64x4(_mm256_set1_epi64x(val)) }
    }
}

impl i64x4 {
    /// convert the vector to an array
    #[inline(always)]
    pub fn as_array(&self) -> [i64; 4] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl SimdCompare for i64x4 {
    type SimdMask = i64x4;
    fn simd_eq(self, other: Self) -> i64x4 {
        unsafe { i64x4(_mm256_cmpeq_epi64(self.0, other.0)) }
    }
    fn simd_ne(self, other: Self) -> i64x4 {
        unsafe { 
            let eq = _mm256_cmpeq_epi64(self.0, other.0);
            i64x4(_mm256_xor_si256(eq, _mm256_set1_epi64x(-1)))
        }
    }
    fn simd_lt(self, other: Self) -> i64x4 {
        unsafe {
            let a: [i64; 4] = std::mem::transmute(self.0);
            let b: [i64; 4] = std::mem::transmute(other.0);
            let mut result = [0; 4];
            for i in 0..4 {
                result[i] = if a[i] < b[i] { -1 } else { 0 };
            }
            i64x4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
    fn simd_le(self, other: Self) -> i64x4 {
        unsafe {
            let a: [i64; 4] = std::mem::transmute(self.0);
            let b: [i64; 4] = std::mem::transmute(other.0);
            let mut result = [0; 4];
            for i in 0..4 {
                result[i] = if a[i] <= b[i] { -1 } else { 0 };
            }
            i64x4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
    fn simd_gt(self, other: Self) -> i64x4 {
        unsafe { i64x4(_mm256_cmpgt_epi64(self.0, other.0)) }
    }
    fn simd_ge(self, other: Self) -> i64x4 {
        unsafe { 
            let gt = _mm256_cmpgt_epi64(self.0, other.0);
            let eq = _mm256_cmpeq_epi64(self.0, other.0);
            i64x4(_mm256_or_si256(gt, eq))
        }
    }
}

impl SimdSelect<i64x4> for crate::vectors::arch_simd::_256bit::i64x4::i64x4 {
    fn select(&self, true_val: i64x4, false_val: i64x4) -> i64x4 {
        unsafe { i64x4(_mm256_blendv_epi8(false_val.0, true_val.0, self.0)) }
    }
}

impl std::ops::Add for i64x4 {
    type Output = i64x4;
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { i64x4(_mm256_add_epi64(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for i64x4 {
    type Output = i64x4;
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { i64x4(_mm256_sub_epi64(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for i64x4 {
    type Output = i64x4;
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [i64; 4] = std::mem::transmute(self.0);
            let arr2: [i64; 4] = std::mem::transmute(rhs.0);
            let mut arr3: [i64; 4] = [0; 4];
            for i in 0..4 {
                arr3[i] = arr[i] * arr2[i];
            }
            i64x4(_mm256_loadu_si256(arr3.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Div for i64x4 {
    type Output = i64x4;
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [i64; 4] = std::mem::transmute(self.0);
            let arr2: [i64; 4] = std::mem::transmute(rhs.0);
            let mut arr3: [i64; 4] = [0; 4];
            for i in 0..4 {
                arr3[i] = arr[i] / arr2[i];
            }
            i64x4(_mm256_loadu_si256(arr3.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Rem for i64x4 {
    type Output = i64x4;
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [i64; 4] = std::mem::transmute(self.0);
            let arr2: [i64; 4] = std::mem::transmute(rhs.0);
            let mut arr3: [i64; 4] = [0; 4];
            for i in 0..4 {
                arr3[i] = arr[i] % arr2[i];
            }
            i64x4(_mm256_loadu_si256(arr3.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Neg for i64x4 {
    type Output = i64x4;
    fn neg(self) -> Self::Output {
        unsafe {
            i64x4(_mm256_sub_epi64(_mm256_setzero_si256(), self.0))
        }
    }
}

impl std::ops::BitAnd for i64x4 {
    type Output = i64x4;
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { i64x4(_mm256_and_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitOr for i64x4 {
    type Output = i64x4;
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { i64x4(_mm256_or_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for i64x4 {
    type Output = i64x4;
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe { i64x4(_mm256_xor_si256(self.0, rhs.0)) }
    }
}
impl std::ops::Not for i64x4 {
    type Output = Self;
    fn not(self) -> Self::Output {
        unsafe { i64x4(_mm256_xor_si256(self.0, _mm256_set1_epi64x(-1))) }
    }
}
impl std::ops::Shl for i64x4 {
    type Output = Self;
    fn shl(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i64; 4] = std::mem::transmute(self.0);
            let b: [i64; 4] = std::mem::transmute(rhs.0);
            let mut result = [0; 4];
            for i in 0..4 {
                result[i] = a[i].wrapping_shl(b[i] as u32);
            }
            i64x4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Shr for i64x4 {
    type Output = Self;
    fn shr(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i64; 4] = std::mem::transmute(self.0);
            let b: [i64; 4] = std::mem::transmute(rhs.0);
            let mut result = [0; 4];
            for i in 0..4 {
                result[i] = a[i].wrapping_shr(b[i] as u32);
            }
            i64x4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl SimdMath<i64> for i64x4 {
    fn max(self, other: Self) -> Self {
        unsafe {
            let arr: [i64; 4] = std::mem::transmute(self.0);
            let arr2: [i64; 4] = std::mem::transmute(other.0);
            let mut arr3: [i64; 4] = [0; 4];
            for i in 0..4 {
                arr3[i] = arr[i].max(arr2[i]);
            }
            i64x4(_mm256_loadu_si256(arr3.as_ptr() as *const __m256i))
        }
    }
    fn min(self, other: Self) -> Self {
        unsafe {
            let arr: [i64; 4] = std::mem::transmute(self.0);
            let arr2: [i64; 4] = std::mem::transmute(other.0);
            let mut arr3: [i64; 4] = [0; 4];
            for i in 0..4 {
                arr3[i] = arr[i].min(arr2[i]);
            }
            i64x4(_mm256_loadu_si256(arr3.as_ptr() as *const __m256i))
        }
    }
    fn relu(self) -> Self {
        unsafe {
            let arr: [i64; 4] = std::mem::transmute(self.0);
            let mut arr2: [i64; 4] = [0; 4];
            for i in 0..4 {
                arr2[i] = arr[i].max(0);
            }
            i64x4(_mm256_loadu_si256(arr2.as_ptr() as *const __m256i))
        }
    }
    fn relu6(self) -> Self {
        unsafe {
            let arr: [i64; 4] = std::mem::transmute(self.0);
            let mut arr2: [i64; 4] = [0; 4];
            for i in 0..4 {
                arr2[i] = arr[i].max(0).min(6);
            }
            i64x4(_mm256_loadu_si256(arr2.as_ptr() as *const __m256i))
        }
    }
}

impl VecConvertor for i64x4 {
    fn to_i64(self) -> i64x4 {
        self
    }
    fn to_u64(self) -> u64x4 {
        unsafe { std::mem::transmute(self) }
    }
    fn to_f64(self) -> super::f64x4::f64x4 {
        // unsafe {
        //     super::f64x4::f64x4(_mm_cvtepi64_pd(self.0))
        // }
        unsafe {
            let arr: [i64; 4] = std::mem::transmute(self.0);
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
