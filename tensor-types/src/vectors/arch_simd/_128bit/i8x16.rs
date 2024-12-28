use crate::{
    convertion::VecConvertor,
    traits::{SimdCompare, SimdMath, VecTrait},
    type_promote::{Eval2, FloatOutBinary2, NormalOut2, NormalOutUnary2},
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::u8x16::u8x16;

/// a vector of 16 i8 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct i8x16(
    #[cfg(target_arch = "x86_64")] pub(crate) __m128i,
    #[cfg(target_arch = "aarch64")] pub(crate) int8x16_t,
);

#[allow(non_camel_case_types)]
pub(crate) type i8_promote = i8x16;

impl PartialEq for i8x16 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let cmp = _mm_cmpeq_epi8(self.0, other.0);
            _mm_movemask_epi8(cmp) == 0xFFFF
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let cmp = vceqq_s8(self.0, other.0);
            vmaxvq_u8(cmp) == 0xff && vminvq_u8(cmp) == 0xff
        }
    }
}

impl Default for i8x16 {
    #[inline(always)]
    fn default() -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i8x16(_mm_setzero_si128())
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i8x16(vdupq_n_s8(0))
        }
    }
}

impl VecTrait<i8> for i8x16 {
    const SIZE: usize = 16;
    type Base = i8;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[i8]) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            _mm_storeu_si128(
                &mut self.0,
                _mm_loadu_si128(slice.as_ptr() as *const __m128i),
            );
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            self.0 = vld1q_s8(slice.as_ptr());
        }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let mut res = [0i8; 16];
            let x: [i8; 16] = std::mem::transmute(self.0);
            let y: [i8; 16] = std::mem::transmute(a.0);
            let z: [i8; 16] = std::mem::transmute(b.0);
            for i in 0..16 {
                res[i] = x[i].wrapping_mul(y[i]).wrapping_add(z[i]);
            }
            i8x16(_mm_loadu_si128(res.as_ptr() as *const __m128i))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i8x16(vaddq_s8(vmulq_s8(self.0, a.0), b.0))
        }
    }
    #[inline(always)]
    fn sum(&self) -> i8 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let sum = _mm_sad_epu8(self.0, _mm_setzero_si128());
            _mm_cvtsi128_si32(sum) as i8
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            vaddvq_s8(self.0) as i8
        }
    }
    #[inline(always)]
    fn splat(val: i8) -> i8x16 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i8x16(_mm_set1_epi8(val))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i8x16(vdupq_n_s8(val))
        }
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const i8) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i8x16(_mm_loadu_si128(ptr as *const __m128i))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i8x16(vld1q_s8(ptr))
        }
    }
}

impl i8x16 {
    #[allow(unused)]
    #[inline(always)]
    fn as_array(&self) -> [i8; 16] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl SimdCompare for i8x16 {
    type SimdMask = i8x16;
    #[inline(always)]
    fn simd_eq(self, other: Self) -> i8x16 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i8x16(_mm_cmpeq_epi8(self.0, other.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i8x16(vreinterpretq_s8_u8(vceqq_s8(self.0, other.0)))
        }
    }
    #[inline(always)]
    fn simd_ne(self, other: Self) -> i8x16 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let eq = _mm_cmpeq_epi8(self.0, other.0);
            i8x16(_mm_xor_si128(eq, _mm_set1_epi8(-1)))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i8x16(vreinterpretq_s8_u8(vmvnq_u8(vceqq_s8(self.0, other.0))))
        }
    }
    #[inline(always)]
    fn simd_lt(self, other: Self) -> i8x16 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i8x16(_mm_cmplt_epi8(self.0, other.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i8x16(vreinterpretq_s8_u8(vcltq_s8(self.0, other.0)))
        }
    }
    #[inline(always)]
    fn simd_le(self, other: Self) -> i8x16 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let lt = _mm_cmplt_epi8(self.0, other.0);
            let eq = _mm_cmpeq_epi8(self.0, other.0);
            i8x16(_mm_or_si128(lt, eq))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i8x16(vreinterpretq_s8_u8(vorrq_u8(
                vcltq_s8(self.0, other.0),
                vceqq_s8(self.0, other.0),
            )))
        }
    }
    #[inline(always)]
    fn simd_gt(self, other: Self) -> i8x16 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i8x16(_mm_cmpgt_epi8(self.0, other.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i8x16(vreinterpretq_s8_u8(vcgtq_s8(self.0, other.0)))
        }
    }
    #[inline(always)]
    fn simd_ge(self, other: Self) -> i8x16 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let gt = _mm_cmpgt_epi8(self.0, other.0);
            let eq = _mm_cmpeq_epi8(self.0, other.0);
            i8x16(_mm_or_si128(gt, eq))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i8x16(vreinterpretq_s8_u8(vorrq_u8(
                vcgtq_s8(self.0, other.0),
                vceqq_s8(self.0, other.0),
            )))
        }
    }
}

impl std::ops::Add for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i8x16(_mm_add_epi8(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i8x16(vaddq_s8(self.0, rhs.0))
        }
    }
}
impl std::ops::Sub for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i8x16(_mm_sub_epi8(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i8x16(vsubq_s8(self.0, rhs.0))
        }
    }
}
impl std::ops::Mul for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let a: [i8; 16] = std::mem::transmute(self.0);
            let b: [i8; 16] = std::mem::transmute(rhs.0);
            let mut result = [0i8; 16];
            for i in 0..16 {
                result[i] = a[i].wrapping_mul(b[i]);
            }
            i8x16(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i8x16(vmulq_s8(self.0, rhs.0))
        }
    }
}
impl std::ops::Div for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let a: [i8; 16] = std::mem::transmute(self.0);
            let b: [i8; 16] = std::mem::transmute(rhs.0);
            let mut result = [0; 16];
            for i in 0..16 {
                assert!(b[i] != 0, "division by zero");
                result[i] = a[i] / b[i];
            }
            i8x16(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let a: [i8; 16] = std::mem::transmute(self.0);
            let b: [i8; 16] = std::mem::transmute(rhs.0);
            let mut result = [0i8; 16];
            for i in 0..16 {
                result[i] = a[i] / b[i];
            }
            i8x16(vld1q_s8(result.as_ptr()))
        }
    }
}
impl std::ops::Rem for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let a: [i8; 16] = std::mem::transmute(self.0);
            let b: [i8; 16] = std::mem::transmute(rhs.0);
            let mut result = [0; 16];
            for i in 0..16 {
                result[i] = a[i] % b[i];
            }
            i8x16(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let a: [i8; 16] = std::mem::transmute(self.0);
            let b: [i8; 16] = std::mem::transmute(rhs.0);
            let mut result = [0i8; 16];
            for i in 0..16 {
                result[i] = a[i] % b[i];
            }
            i8x16(vld1q_s8(result.as_ptr()))
        }
    }
}
impl std::ops::Neg for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i8x16(_mm_sign_epi8(self.0, _mm_set1_epi8(-1)))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i8x16(vnegq_s8(self.0))
        }
    }
}
impl std::ops::BitAnd for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i8x16(_mm_and_si128(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i8x16(vandq_s8(self.0, rhs.0))
        }
    }
}
impl std::ops::BitOr for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i8x16(_mm_or_si128(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i8x16(vorrq_s8(self.0, rhs.0))
        }
    }
}
impl std::ops::BitXor for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i8x16(_mm_xor_si128(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i8x16(veorq_s8(self.0, rhs.0))
        }
    }
}
impl std::ops::Not for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i8x16(_mm_xor_si128(self.0, _mm_set1_epi8(-1)))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i8x16(vmvnq_s8(self.0))
        }
    }
}
impl std::ops::Shl for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let a: [i8; 16] = std::mem::transmute(self.0);
            let b: [i8; 16] = std::mem::transmute(rhs.0);
            let mut result = [0; 16];
            for i in 0..16 {
                result[i] = a[i].wrapping_shl(b[i] as u32);
            }
            i8x16(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i8x16(vshlq_s8(self.0, rhs.0))
        }
    }
}
impl std::ops::Shr for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let a: [i8; 16] = std::mem::transmute(self.0);
            let b: [i8; 16] = std::mem::transmute(rhs.0);
            let mut result = [0; 16];
            for i in 0..16 {
                result[i] = a[i].wrapping_shr(b[i] as u32);
            }
            i8x16(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let a: [i8; 16] = std::mem::transmute(self.0);
            let b: [i8; 16] = std::mem::transmute(rhs.0);
            let mut result = [0i8; 16];
            for i in 0..16 {
                result[i] = a[i].wrapping_shr(b[i] as u32);
            }
            i8x16(vld1q_s8(result.as_ptr()))
        }
    }
}

impl SimdMath<i8> for i8x16 {
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i8x16(_mm_max_epi8(self.0, other.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i8x16(vmaxq_s8(self.0, other.0))
        }
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i8x16(_mm_min_epi8(self.0, other.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i8x16(vminq_s8(self.0, other.0))
        }
    }
    #[inline(always)]
    fn relu(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i8x16(_mm_max_epi8(self.0, _mm_setzero_si128()))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i8x16(vmaxq_s8(self.0, vdupq_n_s8(0)))
        }
    }
    #[inline(always)]
    fn relu6(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i8x16(_mm_min_epi8(self.relu().0, _mm_set1_epi8(6)))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i8x16(vminq_s8(self.relu().0, vdupq_n_s8(6)))
        }
    }
}

impl VecConvertor for i8x16 {
    #[inline(always)]
    fn to_i8(self) -> i8x16 {
        self
    }
    #[inline(always)]
    fn to_u8(self) -> u8x16 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    fn to_bool(self) -> super::boolx16::boolx16 {
        unsafe { std::mem::transmute(self) }
    }
}

impl FloatOutBinary2 for i8x16 {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn __log(self, _: Self) -> Self {
        panic!("Logarithm operation is not supported for i8")
    }
}

impl NormalOut2 for i8x16 {
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
    fn __pow(self, rhs: Self) -> Self {
        self.pow(rhs)
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
    fn __clip(self, min: Self, max: Self) -> Self {
        self.max(min).min(max)
    }
}

impl NormalOutUnary2 for i8x16 {
    #[inline(always)]
    fn __square(self) -> Self {
        self * self
    }

    #[inline(always)]
    fn __abs(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i8x16(_mm_abs_epi8(self.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i8x16(vabsq_s8(self.0))
        }
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
        -self
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
        self.max(i8x16::splat(0)) + alpha * self.min(i8x16::splat(0))
    }

    #[inline(always)]
    fn __relu(self) -> Self {
        self.relu()
    }

    #[inline(always)]
    fn __relu6(self) -> Self {
        self.relu6()
    }
}

impl Eval2 for i8x16 {
    type Output = i8x16;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        i8x16::default()
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let eq = _mm_cmpeq_epi8(self.0, _mm_setzero_si128());
            Self(_mm_xor_si128(eq, _mm_set1_epi8(-1)))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i8x16(vmvnq_s8(vreinterpretq_s8_u8(vceqq_s8(
                self.0,
                vdupq_n_s8(0),
            ))))
        }
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        i8x16::default()
    }
}
