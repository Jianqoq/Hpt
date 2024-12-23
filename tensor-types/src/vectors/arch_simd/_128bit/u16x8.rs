use crate::{
    convertion::VecConvertor,
    traits::{SimdCompare, SimdMath, SimdSelect, VecTrait},
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::i16x8::i16x8;

/// a vector of 8 u16 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct u16x8(
    #[cfg(target_arch = "x86_64")] pub(crate) __m128i,
    #[cfg(target_arch = "aarch64")] pub(crate) uint16x8_t,
);

impl PartialEq for u16x8 {
    fn eq(&self, other: &Self) -> bool {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let cmp = _mm_cmpeq_epi16(self.0, other.0);
            _mm_movemask_epi8(cmp) == -1
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let cmp = vceqq_u16(self.0, other.0);
            vaddvq_u16(cmp) == 8
        }
    }
}

impl Default for u16x8 {
    fn default() -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { u16x8(_mm_setzero_si128()) }
        #[cfg(target_arch = "aarch64")]
        unsafe { u16x8(vdupq_n_u16(0)) }
    }
}

impl VecTrait<u16> for u16x8 {
    const SIZE: usize = 8;
    type Base = u16;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u16]) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            _mm_storeu_si128(
                &mut self.0,
                _mm_loadu_si128(slice.as_ptr() as *const __m128i),
            )
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            self.0 = vld1q_u16(slice.as_ptr());
        }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { u16x8(_mm_add_epi16(self.0, _mm_mullo_epi16(a.0, b.0))) }
        #[cfg(target_arch = "aarch64")]
        unsafe { u16x8(vaddq_u16(self.0, vmulq_u16(a.0, b.0))) }
    }
    #[inline(always)]
    fn sum(&self) -> u16 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let arr: [u16; 8] = std::mem::transmute(self.0);
            arr.iter().sum()
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            vaddvq_u16(self.0)
        }
    }
    fn splat(val: u16) -> u16x8 {
        #[cfg(target_arch = "x86_64")]
        unsafe { u16x8(_mm_set1_epi16(val as i16)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { u16x8(vdupq_n_u16(val)) }
    }
}

impl u16x8 {
    #[allow(unused)]
    fn as_array(&self) -> [u16; 8] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl SimdSelect<u16x8> for u16x8 {
    fn select(&self, true_val: u16x8, false_val: u16x8) -> u16x8 {
        #[cfg(target_arch = "x86_64")]
        unsafe { u16x8(_mm_blendv_epi8(false_val.0, true_val.0, self.0)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { u16x8(vbslq_u16(self.0, true_val.0, false_val.0)) }
    }
}

impl std::ops::Add for u16x8 {
    type Output = u16x8;
    fn add(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe { u16x8(_mm_add_epi16(self.0, rhs.0)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { u16x8(vaddq_u16(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for u16x8 {
    type Output = u16x8;
    fn sub(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe { u16x8(_mm_sub_epi16(self.0, rhs.0)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { u16x8(vsubq_u16(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for u16x8 {
    type Output = u16x8;
    fn mul(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe { u16x8(_mm_mullo_epi16(self.0, rhs.0)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { u16x8(vmulq_u16(self.0, rhs.0)) }
    }
}
impl std::ops::Div for u16x8 {
    type Output = u16x8;
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [u16; 8] = std::mem::transmute(self.0);
            let arr2: [u16; 8] = std::mem::transmute(rhs.0);
            let mut arr3: [u16; 8] = [0; 8];
            for i in 0..8 {
                arr3[i] = arr[i] / arr2[i];
            }
            #[cfg(target_arch = "x86_64")]
            return u16x8(_mm_loadu_si128(arr3.as_ptr() as *const __m128i));
            #[cfg(target_arch = "aarch64")]
            return u16x8(vld1q_u16(arr3.as_ptr()));
        }
    }
}
impl std::ops::Rem for u16x8 {
    type Output = u16x8;
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [u16; 8] = std::mem::transmute(self.0);
            let arr2: [u16; 8] = std::mem::transmute(rhs.0);
            let mut arr3: [u16; 8] = [0; 8];
            for i in 0..8 {
                arr3[i] = arr[i] % arr2[i];
            }
            #[cfg(target_arch = "x86_64")]
            return u16x8(_mm_loadu_si128(arr3.as_ptr() as *const __m128i));
            #[cfg(target_arch = "aarch64")]
            return u16x8(vld1q_u16(arr3.as_ptr()));
        }
    }
}

impl std::ops::BitAnd for u16x8 {
    type Output = u16x8;
    fn bitand(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe { u16x8(_mm_and_si128(self.0, rhs.0)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { u16x8(vandq_u16(self.0, rhs.0)) }
    }
}
impl std::ops::BitOr for u16x8 {
    type Output = u16x8;
    fn bitor(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe { u16x8(_mm_or_si128(self.0, rhs.0)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { u16x8(vorrq_u16(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for u16x8 {
    type Output = u16x8;
    fn bitxor(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe { u16x8(_mm_xor_si128(self.0, rhs.0)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { u16x8(veorq_u16(self.0, rhs.0)) }
    }
}
impl std::ops::Not for u16x8 {
    type Output = Self;
    fn not(self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe { u16x8(_mm_xor_si128(self.0, _mm_set1_epi16(-1))) }
        #[cfg(target_arch = "aarch64")]
        unsafe { u16x8(vmvnq_u16(self.0)) }
    }
}
impl std::ops::Shl for u16x8 {
    type Output = Self;
    fn shl(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let a: [u16; 8] = std::mem::transmute(self.0);
            let b: [u16; 8] = std::mem::transmute(rhs.0);
            let mut result = [0; 8];
            for i in 0..8 {
                result[i] = a[i].wrapping_shl(b[i] as u32);
            }
            u16x8(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            u16x8(vshlq_u16(self.0, vreinterpretq_s16_u16(rhs.0)))
        }
    }
}
impl std::ops::Shr for u16x8 {
    type Output = Self;
    fn shr(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [u16; 8] = std::mem::transmute(self.0);
            let b: [u16; 8] = std::mem::transmute(rhs.0);
            let mut result = [0; 8];
            for i in 0..8 {
                result[i] = a[i].wrapping_shr(b[i] as u32);
            }
            #[cfg(target_arch = "x86_64")]
            return u16x8(_mm_loadu_si128(result.as_ptr() as *const __m128i));
            #[cfg(target_arch = "aarch64")]
            return u16x8(vld1q_u16(result.as_ptr()));
        }
    }
}
impl SimdCompare for u16x8 {
    type SimdMask = i16x8;

    fn simd_eq(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i16x8 = std::mem::transmute(self.0);
            let rhs: i16x8 = std::mem::transmute(other.0);
            lhs.simd_eq(rhs)
        }
    }

    fn simd_ne(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i16x8 = std::mem::transmute(self.0);
            let rhs: i16x8 = std::mem::transmute(other.0);
            lhs.simd_ne(rhs)
        }
    }

    fn simd_lt(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i16x8 = std::mem::transmute(self.0);
            let rhs: i16x8 = std::mem::transmute(other.0);
            lhs.simd_lt(rhs)
        }
    }

    fn simd_le(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i16x8 = std::mem::transmute(self.0);
            let rhs: i16x8 = std::mem::transmute(other.0);
            lhs.simd_le(rhs)
        }
    }

    fn simd_gt(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i16x8 = std::mem::transmute(self.0);
            let rhs: i16x8 = std::mem::transmute(other.0);
            lhs.simd_gt(rhs)
        }
    }

    fn simd_ge(self, other: Self) -> Self::SimdMask {
        unsafe {
            let lhs: i16x8 = std::mem::transmute(self.0);
            let rhs: i16x8 = std::mem::transmute(other.0);
            lhs.simd_ge(rhs)
        }
    }
}

impl SimdMath<u16> for u16x8 {
    fn max(self, other: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { u16x8(_mm_max_epi16(self.0, other.0)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { u16x8(vmaxq_u16(self.0, other.0)) }
    }
    fn min(self, other: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { u16x8(_mm_min_epi16(self.0, other.0)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { u16x8(vminq_u16(self.0, other.0)) }
    }
    fn relu(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { u16x8(_mm_max_epi16(self.0, _mm_setzero_si128())) }
        #[cfg(target_arch = "aarch64")]
        unsafe { u16x8(vmaxq_u16(self.0, vdupq_n_u16(0))) }
    }
    fn relu6(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { u16x8(_mm_min_epi16(self.relu().0, _mm_set1_epi16(6))) }
        #[cfg(target_arch = "aarch64")]
        unsafe { u16x8(vminq_u16(self.relu().0, vdupq_n_u16(6))) }
    }
}

impl VecConvertor for u16x8 {
    fn to_u16(self) -> u16x8 {
        self
    }
    fn to_i16(self) -> i16x8 {
        unsafe { std::mem::transmute(self) }
    }
    fn to_f16(self) -> super::f16x8::f16x8 {
        unsafe {
            let arr: [u16; 8] = std::mem::transmute(self.0);
            let mut result = [half::f16::ZERO; 8];
            for i in 0..8 {
                result[i] = half::f16::from_f32(arr[i] as f32);
            }
            super::f16x8::f16x8(result)
        }
    }
    fn to_bf16(self) -> super::bf16x8::bf16x8 {
        unsafe {
            let arr: [u16; 8] = std::mem::transmute(self.0);
            let mut result = [half::bf16::ZERO; 8];
            for i in 0..8 {
                result[i] = half::bf16::from_f32(arr[i] as f32);
            }
            super::bf16x8::bf16x8(result)
        }
    }
}
