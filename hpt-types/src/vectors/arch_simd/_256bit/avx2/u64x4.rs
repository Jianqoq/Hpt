use crate::{
    convertion::VecConvertor,
    traits::{SimdMath, SimdSelect, VecTrait},
    type_promote::{Eval2, FloatOutBinary2},
};

use std::arch::x86_64::*;

use crate::simd::_256bit::f64x4;
use crate::simd::_256bit::i64x4;
#[cfg(target_pointer_width = "64")]
use crate::simd::_256bit::isizex4;
use crate::simd::_256bit::u64x4;
#[cfg(target_pointer_width = "64")]
use crate::simd::_256bit::usizex4;

impl PartialEq for u64x4 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm256_cmpeq_epi64(self.0, other.0);
            _mm256_movemask_epi8(cmp) == -1
        }
    }
}

impl Default for u64x4 {
    #[inline(always)]
    fn default() -> Self {
        unsafe { u64x4(_mm256_setzero_si256()) }
    }
}

impl VecTrait<u64> for u64x4 {
    const SIZE: usize = 4;
    type Base = u64;
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
    #[inline(always)]
    fn splat(val: u64) -> u64x4 {
        unsafe { u64x4(_mm256_set1_epi64x(val as i64)) }
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const u64) -> Self {
        u64x4(_mm256_loadu_si256(ptr as *const __m256i))
    }
}

impl SimdSelect<u64x4> for i64x4 {
    #[inline(always)]
    fn select(&self, true_val: u64x4, false_val: u64x4) -> u64x4 {
        unsafe { u64x4(_mm256_blendv_epi8(false_val.0, true_val.0, self.0)) }
    }
}

impl std::ops::Add for u64x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { u64x4(_mm256_add_epi64(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for u64x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe { u64x4(_mm256_sub_epi64(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for u64x4 {
    type Output = Self;
    #[inline(always)]
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
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        unsafe {
            let arr: [u64; 4] = std::mem::transmute(self.0);
            let arr2: [u64; 4] = std::mem::transmute(rhs.0);
            let mut arr3: [u64; 4] = [0; 4];
            for i in 0..4 {
                assert!(arr2[i] != 0, "division by zero");
                arr3[i] = arr[i] / arr2[i];
            }
            u64x4(_mm256_loadu_si256(arr3.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Rem for u64x4 {
    type Output = Self;
    #[inline(always)]
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
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe { u64x4(_mm256_and_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitOr for u64x4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe { u64x4(_mm256_or_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for u64x4 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        unsafe { u64x4(_mm256_xor_si256(self.0, rhs.0)) }
    }
}
impl std::ops::Not for u64x4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe { u64x4(_mm256_xor_si256(self.0, _mm256_set1_epi64x(-1))) }
    }
}
impl std::ops::Shl for u64x4 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self {
        unsafe {
            let a: [u64; 4] = std::mem::transmute(self.0);
            let b: [u64; 4] = std::mem::transmute(rhs.0);
            let mut result = [0; 4];
            for i in 0..4 {
                result[i] = a[i].wrapping_shl(b[i] as u32);
            }
            u64x4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Shr for u64x4 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self {
        unsafe {
            let a: [u64; 4] = std::mem::transmute(self.0);
            let b: [u64; 4] = std::mem::transmute(rhs.0);
            let mut result = [0; 4];
            for i in 0..4 {
                result[i] = a[i].wrapping_shr(b[i] as u32);
            }
            u64x4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl SimdMath<u64> for u64x4 {
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
    fn relu(self) -> Self {
        self.max(Self::splat(0))
    }
    #[inline(always)]
    fn relu6(self) -> Self {
        self.min(Self::splat(6)).max(Self::splat(0))
    }
    #[inline(always)]
    fn trunc(self) -> Self {
        self
    }
    #[inline(always)]
    fn floor(self) -> Self {
        self
    }
    #[inline(always)]
    fn ceil(self) -> Self {
        self
    }
    #[inline(always)]
    fn round(self) -> Self {
        self
    }
    #[inline(always)]
    fn abs(self) -> Self {
        self
    }
    #[inline(always)]
    fn pow(self, rhs: Self) -> Self {
        unsafe {
            let a: [u64; 4] = std::mem::transmute(self.0);
            let b: [u64; 4] = std::mem::transmute(rhs.0);
            let mut result = [0u64; 4];
            for i in 0..4 {
                result[i] = a[i].pow(b[i] as u32);
            }
            u64x4(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
    #[inline(always)]
    fn leaky_relu(self, alpha: Self) -> Self {
        self.max(Self::splat(0)) + alpha * self.min(Self::splat(0))
    }
}

impl VecConvertor for u64x4 {
    #[inline(always)]
    fn to_u64(self) -> u64x4 {
        self
    }
    #[inline(always)]
    fn to_i64(self) -> i64x4 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    fn to_f64(self) -> f64x4 {
        unsafe {
            let arr: [u64; 4] = std::mem::transmute(self.0);
            let mut result = [0.0f64; 4];
            for i in 0..4 {
                result[i] = arr[i] as f64;
            }
            f64x4(_mm256_loadu_pd(result.as_ptr()))
        }
    }
    #[cfg(target_pointer_width = "64")]
    #[inline(always)]
    fn to_isize(self) -> isizex4 {
        unsafe { std::mem::transmute(self) }
    }
    #[cfg(target_pointer_width = "64")]
    #[inline(always)]
    fn to_usize(self) -> usizex4 {
        unsafe { std::mem::transmute(self) }
    }
}

impl FloatOutBinary2 for u64x4 {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn __log(self, _: Self) -> Self {
        panic!("Logarithm operation is not supported for u64")
    }

    #[inline(always)]
    fn __hypot(self, _: Self) -> Self {
        panic!("Hypot operation is not supported for u64x4");
    }

    #[inline(always)]
    fn __pow(self, rhs: Self) -> Self {
        unsafe {
            let arr: [u64; 4] = std::mem::transmute(self.0);
            let arr2: [u64; 4] = std::mem::transmute(rhs.0);
            let mut arr3: [u64; 4] = [0; 4];
            for i in 0..4 {
                arr3[i] = arr[i].pow(arr2[i] as u32);
            }
            u64x4(_mm256_loadu_si256(arr3.as_ptr() as *const __m256i))
        }
    }
}

impl Eval2 for u64x4 {
    type Output = i64x4;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        i64x4::default()
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        unsafe {
            let eq = _mm256_cmpeq_epi64(self.0, _mm256_setzero_si256());
            let result = _mm256_andnot_si256(eq, _mm256_set1_epi64x(1));
            i64x4(result)
        }
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        i64x4::default()
    }
}
