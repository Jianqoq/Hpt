use crate::{
    convertion::VecConvertor,
    traits::{SimdCompare, SimdMath, VecTrait},
    type_promote::{Eval2, FloatOutBinary2, NormalOut2, NormalOutUnary2},
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::i8x32::i8x32;

/// a vector of 32 u8 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct u8x32(pub(crate) __m256i);

/// helper to impl the promote trait
#[allow(non_camel_case_types)]
pub(crate) type u8_promote = u8x32;

impl PartialEq for u8x32 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = _mm256_cmpeq_epi8(self.0, other.0);
            _mm256_movemask_epi8(cmp) == -1
        }
    }
}

impl Default for u8x32 {
    #[inline(always)]
    fn default() -> Self {
        unsafe { u8x32(_mm256_setzero_si256()) }
    }
}

impl VecTrait<u8> for u8x32 {
    const SIZE: usize = 32;
    type Base = u8;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u8]) {
        unsafe {
            _mm256_storeu_si256(
                &mut self.0,
                _mm256_loadu_si256(slice.as_ptr() as *const __m256i),
            )
        }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let mut res = [0u8; 32];
            let x: [u8; 32] = std::mem::transmute(self.0);
            let y: [u8; 32] = std::mem::transmute(a.0);
            let z: [u8; 32] = std::mem::transmute(b.0);
            for i in 0..32 {
                res[i] = x[i].wrapping_mul(y[i]).wrapping_add(z[i]);
            }
            u8x32(_mm256_loadu_si256(res.as_ptr() as *const __m256i))
        }
    }
    #[inline(always)]
    fn sum(&self) -> u8 {
        unsafe {
            let x: [u8; 32] = std::mem::transmute(self.0);
            x.iter().sum()
        }
    }
    #[inline(always)]
    fn splat(val: u8) -> u8x32 {
        unsafe { u8x32(_mm256_set1_epi8(val as i8)) }
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const u8) -> Self {
        u8x32(_mm256_loadu_si256(ptr as *const __m256i))
    }
}

impl u8x32 {
    /// convert the vector to an array
    #[inline(always)]
    pub fn as_array(&self) -> [u8; 32] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl SimdCompare for u8x32 {
    type SimdMask = i8x32;
    #[inline(always)]
    fn simd_eq(self, other: Self) -> i8x32 {
        unsafe {
            let lhs: i8x32 = std::mem::transmute(self.0);
            let rhs: i8x32 = std::mem::transmute(other.0);
            lhs.simd_eq(rhs)
        }
    }
    #[inline(always)]
    fn simd_ne(self, other: Self) -> i8x32 {
        unsafe {
            let lhs: i8x32 = std::mem::transmute(self.0);
            let rhs: i8x32 = std::mem::transmute(other.0);
            lhs.simd_ne(rhs)
        }
    }
    #[inline(always)]
    fn simd_lt(self, other: Self) -> i8x32 {
        unsafe {
            let lhs: i8x32 = std::mem::transmute(self.0);
            let rhs: i8x32 = std::mem::transmute(other.0);
            lhs.simd_lt(rhs)
        }
    }
    #[inline(always)]
    fn simd_le(self, other: Self) -> i8x32 {
        unsafe {
            let lhs: i8x32 = std::mem::transmute(self.0);
            let rhs: i8x32 = std::mem::transmute(other.0);
            lhs.simd_le(rhs)
        }
    }
    #[inline(always)]
    fn simd_gt(self, other: Self) -> i8x32 {
        unsafe {
            let lhs: i8x32 = std::mem::transmute(self.0);
            let rhs: i8x32 = std::mem::transmute(other.0);
            lhs.simd_gt(rhs)
        }
    }
    #[inline(always)]
    fn simd_ge(self, other: Self) -> i8x32 {
        unsafe {
            let lhs: i8x32 = std::mem::transmute(self.0);
            let rhs: i8x32 = std::mem::transmute(other.0);
            lhs.simd_ge(rhs)
        }
    }
}

impl std::ops::Add for u8x32 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { u8x32(_mm256_add_epi8(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for u8x32 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe { u8x32(_mm256_sub_epi8(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for u8x32 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let a: [u8; 32] = std::mem::transmute(self.0);
            let b: [u8; 32] = std::mem::transmute(rhs.0);
            let mut result = [0u8; 32];
            for i in 0..32 {
                result[i] = a[i].wrapping_mul(b[i]);
            }
            u8x32(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Div for u8x32 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        unsafe {
            let arr: [u8; 32] = std::mem::transmute(self.0);
            let arr2: [u8; 32] = std::mem::transmute(rhs.0);
            let mut arr3: [u8; 32] = [0; 32];
            for i in 0..32 {
                assert!(arr2[i] != 0, "division by zero");
                arr3[i] = arr[i] / arr2[i];
            }
            u8x32(_mm256_loadu_si256(arr3.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Rem for u8x32 {
    type Output = Self;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self {
        unsafe {
            let arr: [u8; 32] = std::mem::transmute(self.0);
            let arr2: [u8; 32] = std::mem::transmute(rhs.0);
            let mut arr3: [u8; 32] = [0; 32];
            for i in 0..32 {
                arr3[i] = arr[i] % arr2[i];
            }
            u8x32(_mm256_loadu_si256(arr3.as_ptr() as *const __m256i))
        }
    }
}

impl std::ops::BitAnd for u8x32 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe { u8x32(_mm256_and_si256(self.0, rhs.0)) }
    }
}

impl std::ops::BitOr for u8x32 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe { u8x32(_mm256_or_si256(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for u8x32 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        unsafe { u8x32(_mm256_xor_si256(self.0, rhs.0)) }
    }
}
impl std::ops::Not for u8x32 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe { u8x32(_mm256_xor_si256(self.0, _mm256_set1_epi8(-1))) }
    }
}
impl std::ops::Shl for u8x32 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self {
        unsafe {
            let a: [u8; 32] = std::mem::transmute(self.0);
            let b: [u8; 32] = std::mem::transmute(rhs.0);
            let mut result = [0; 32];
            for i in 0..32 {
                result[i] = a[i].wrapping_shl(b[i] as u32);
            }
            u8x32(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}
impl std::ops::Shr for u8x32 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self {
        unsafe {
            let a: [u8; 32] = std::mem::transmute(self.0);
            let b: [u8; 32] = std::mem::transmute(rhs.0);
            let mut result = [0; 32];
            for i in 0..32 {
                result[i] = a[i].wrapping_shr(b[i] as u32);
            }
            u8x32(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
}

impl SimdMath<u8> for u8x32 {
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        unsafe { u8x32(_mm256_max_epu8(self.0, other.0)) }
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        unsafe { u8x32(_mm256_min_epu8(self.0, other.0)) }
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
            let a: [u8; 32] = std::mem::transmute(self.0);
            let b: [u8; 32] = std::mem::transmute(rhs.0);
            let mut result = [0u8; 32];
            for i in 0..32 {
                result[i] = a[i].pow(b[i] as u32);
            }
            u8x32(_mm256_loadu_si256(result.as_ptr() as *const __m256i))
        }
    }
    #[inline(always)]
    fn leaky_relu(self, alpha: Self) -> Self {
        self.max(Self::splat(0)) + alpha * self.min(Self::splat(0))
    }
}

impl VecConvertor for u8x32 {
    #[inline(always)]
    fn to_u8(self) -> u8x32 {
        self
    }
    #[inline(always)]
    fn to_i8(self) -> i8x32 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    fn to_bool(self) -> super::boolx32::boolx32 {
        unsafe { std::mem::transmute(self) }
    }
}

impl FloatOutBinary2 for u8x32 {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn __log(self, _: Self) -> Self {
        panic!("Logarithm operation is not supported for u8")
    }

    #[inline(always)]
    fn __hypot(self, _: Self) -> Self {
        panic!("Hypot operation is not supported for u8x32");
    }

    #[inline(always)]
    fn __pow(self, rhs: Self) -> Self {
        unsafe {
            let arr: [u8; 32] = std::mem::transmute(self.0);
            let arr2: [u8; 32] = std::mem::transmute(rhs.0);
            let mut arr3: [u8; 32] = [0; 32];
            for i in 0..32 {
                arr3[i] = arr[i].pow(arr2[i] as u32);
            }
            u8x32(_mm256_loadu_si256(arr3.as_ptr() as *const __m256i))
        }
    }
}

impl NormalOut2 for u8x32 {
    #[inline(always)]
    fn __add(self, rhs: Self) -> Self {
        self + rhs
    }

    #[inline(always)]
    fn __sub(self, rhs: Self) -> Self {
        self - rhs
    }

    #[inline(always)]
    fn __mul_add(self, a: Self, b: Self) -> Self {
        self.mul_add(a, b)
    }

    #[inline(always)]
    fn __mul(self, rhs: Self) -> Self {
        self * rhs
    }

    #[inline(always)]
    fn __rem(self, rhs: Self) -> Self {
        self % rhs
    }

    #[inline(always)]
    fn __max(self, rhs: Self) -> Self {
        self.max(rhs)
    }

    #[inline(always)]
    fn __min(self, rhs: Self) -> Self {
        self.min(rhs)
    }

    #[inline(always)]
    fn __clamp(self, min: Self, max: Self) -> Self {
        self.max(min).min(max)
    }
}

impl NormalOutUnary2 for u8x32 {
    #[inline(always)]
    fn __square(self) -> Self {
        self * self
    }

    #[inline(always)]
    fn __abs(self) -> Self {
        self
    }

    #[inline(always)]
    fn __ceil(self) -> Self {
        self
    }

    #[inline(always)]
    fn __floor(self) -> Self {
        self
    }

    #[inline(always)]
    fn __neg(self) -> Self {
        self
    }

    #[inline(always)]
    fn __round(self) -> Self {
        self
    }

    #[inline(always)]
    fn __signum(self) -> Self {
        self.signum()
    }

    #[inline(always)]
    fn __leaky_relu(self, alpha: Self) -> Self {
        self.leaky_relu(alpha)
    }

    #[inline(always)]
    fn __relu(self) -> Self {
        self.relu()
    }

    #[inline(always)]
    fn __relu6(self) -> Self {
        self.relu6()
    }

    #[inline(always)]
    fn __trunc(self) -> Self {
        self
    }

    #[inline(always)]
    fn __copysign(self, rhs: Self) -> Self {
        self.abs() * rhs.signum()
    }
}

impl Eval2 for u8x32 {
    type Output = i8x32;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        i8x32::default()
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        unsafe {
            let eq = _mm256_cmpeq_epi8(self.0, _mm256_setzero_si256());
            let result = _mm256_andnot_si256(eq, _mm256_set1_epi8(1));
            i8x32(result)
        }
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        i8x32::default()
    }
}
