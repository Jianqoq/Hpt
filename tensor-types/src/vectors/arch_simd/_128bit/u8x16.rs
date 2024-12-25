use crate::{convertion::VecConvertor, traits::{ SimdCompare, SimdMath, VecTrait }};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::i8x16::i8x16;

/// a vector of 16 u8 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct u8x16(
    #[cfg(target_arch = "x86_64")]
    pub(crate) __m128i,
    #[cfg(target_arch = "aarch64")]
    pub(crate) uint8x16_t,
);

impl PartialEq for u8x16 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let cmp = _mm_cmpeq_epi8(self.0, other.0);
            _mm_movemask_epi8(cmp) == -1
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let cmp = vceqq_u8(self.0, other.0);
            vmaxvq_u8(cmp) == 0xFF && vminvq_u8(cmp) == 0xFF
        }
    }
}

impl Default for u8x16 {
    #[inline(always)]
    fn default() -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { u8x16(_mm_setzero_si128()) }
        #[cfg(target_arch = "aarch64")]
        unsafe { u8x16(vdupq_n_u8(0)) }
    }
}

impl VecTrait<u8> for u8x16 {
    const SIZE: usize = 16;
    type Base = u8;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[u8]) {
        #[cfg(target_arch = "x86_64")]
        unsafe { _mm_storeu_si128(&mut self.0, _mm_loadu_si128(slice.as_ptr() as *const __m128i)) }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            self.0 = vld1q_u8(slice.as_ptr());
        }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { u8x16(_mm_add_epi8(self.0, _mm_mullo_epi16(a.0, b.0))) }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let mul = vmulq_u8(a.0, b.0);
            u8x16(vaddq_u8(self.0, mul))
        }
    }
    #[inline(always)]
    fn sum(&self) -> u8 {
        unsafe {
            let x: [u8; 16] = std::mem::transmute(self.0);
            x.iter().sum()
        }
    }
    #[inline(always)]
    fn splat(val: u8) -> u8x16 {
        #[cfg(target_arch = "x86_64")]
        unsafe { u8x16(_mm_set1_epi8(val as i8)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { u8x16(vdupq_n_u8(val)) }
    }
}

impl u8x16 {
    #[allow(unused)]
    #[inline(always)]
    fn as_array(&self) -> [u8; 16] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl SimdCompare for u8x16 {
    type SimdMask = i8x16;
    #[inline(always)]
    fn simd_eq(self, other: Self) -> i8x16 {
        unsafe {
            let lhs: i8x16 = std::mem::transmute(self.0);
            let rhs: i8x16 = std::mem::transmute(other.0);
            lhs.simd_eq(rhs)
        }
    }
    #[inline(always)]
    fn simd_ne(self, other: Self) -> i8x16 {
        unsafe { 
            let lhs: i8x16 = std::mem::transmute(self.0);
            let rhs: i8x16 = std::mem::transmute(other.0);
            lhs.simd_ne(rhs)
        }
    }
    #[inline(always)]
    fn simd_lt(self, other: Self) -> i8x16 {
        unsafe {
            let lhs: i8x16 = std::mem::transmute(self.0);
            let rhs: i8x16 = std::mem::transmute(other.0);
            lhs.simd_lt(rhs)
        }
    }
    #[inline(always)]
    fn simd_le(self, other: Self) -> i8x16 {
        unsafe { 
            let lhs: i8x16 = std::mem::transmute(self.0);
            let rhs: i8x16 = std::mem::transmute(other.0);
            lhs.simd_le(rhs)
        }
    }
    #[inline(always)]
    fn simd_gt(self, other: Self) -> i8x16 {
        unsafe {
            let lhs: i8x16 = std::mem::transmute(self.0);
            let rhs: i8x16 = std::mem::transmute(other.0);
            lhs.simd_gt(rhs)
        }
    }
    #[inline(always)]
    fn simd_ge(self, other: Self) -> i8x16 {
        unsafe { 
            let lhs: i8x16 = std::mem::transmute(self.0);
            let rhs: i8x16 = std::mem::transmute(other.0);
            lhs.simd_ge(rhs)
        }
    }
}

impl std::ops::Add for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { u8x16(_mm_add_epi8(self.0, rhs.0)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { u8x16(vaddq_u8(self.0, rhs.0)) }
    }
}
impl std::ops::Sub for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { u8x16(_mm_sub_epi8(self.0, rhs.0)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { u8x16(vsubq_u8(self.0, rhs.0)) }
    }
}
impl std::ops::Mul for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { u8x16(_mm_mullo_epi16(self.0, rhs.0)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { u8x16(vmulq_u8(self.0, rhs.0)) }
    }
}
impl std::ops::Div for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        unsafe {
            let arr: [u8; 16] = std::mem::transmute(self.0);
            let arr2: [u8; 16] = std::mem::transmute(rhs.0);
            let mut arr3: [u8; 16] = [0; 16];
            for i in 0..16 {
                arr3[i] = arr[i] / arr2[i];
            }
            #[cfg(target_arch = "x86_64")]
            return u8x16(_mm_loadu_si128(arr3.as_ptr() as *const __m128i));
            #[cfg(target_arch = "aarch64")]
            return u8x16(vld1q_u8(arr3.as_ptr()));
        }
    }
}
impl std::ops::Rem for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self {
        unsafe {
            let arr: [u8; 16] = std::mem::transmute(self.0);
            let arr2: [u8; 16] = std::mem::transmute(rhs.0);
            let mut arr3: [u8; 16] = [0; 16];
            for i in 0..16 {
                arr3[i] = arr[i] % arr2[i];
            }
            #[cfg(target_arch = "x86_64")]
            return u8x16(_mm_loadu_si128(arr3.as_ptr() as *const __m128i));
            #[cfg(target_arch = "aarch64")]
            return u8x16(vld1q_u8(arr3.as_ptr()));
        }
    }
}

impl std::ops::BitAnd for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { u8x16(_mm_and_si128(self.0, rhs.0)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { u8x16(vandq_u8(self.0, rhs.0)) }
    }
}

impl std::ops::BitOr for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { u8x16(_mm_or_si128(self.0, rhs.0)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { u8x16(vorrq_u8(self.0, rhs.0)) }
    }
}
impl std::ops::BitXor for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { u8x16(_mm_xor_si128(self.0, rhs.0)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { u8x16(veorq_u8(self.0, rhs.0)) }
    }
}
impl std::ops::Not for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { u8x16(_mm_xor_si128(self.0, _mm_set1_epi8(-1))) }
        #[cfg(target_arch = "aarch64")]
        unsafe { u8x16(vmvnq_u8(self.0)) }
    }
}
impl std::ops::Shl for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let a: [u8; 16] = std::mem::transmute(self.0);
            let b: [u8; 16] = std::mem::transmute(rhs.0);
            let mut result = [0; 16];
            for i in 0..16 {
                result[i] = a[i].wrapping_shl(b[i] as u32);
            }
            u8x16(_mm_loadu_si128(result.as_ptr() as *const __m128i))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe { u8x16(vshlq_u8(self.0, vreinterpretq_s8_u8(rhs.0))) }
    }
}
impl std::ops::Shr for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self {
        unsafe {
            let a: [u8; 16] = std::mem::transmute(self.0);
            let b: [u8; 16] = std::mem::transmute(rhs.0);
            let mut result = [0; 16];
            for i in 0..16 {
                result[i] = a[i].wrapping_shr(b[i] as u32);
            }
            #[cfg(target_arch = "x86_64")]
            return u8x16(_mm_loadu_si128(result.as_ptr() as *const __m128i));
            #[cfg(target_arch = "aarch64")]
            return u8x16(vld1q_u8(result.as_ptr()));
        }
    }
}

impl SimdMath<u8> for u8x16 {
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { u8x16(_mm_max_epi8(self.0, other.0)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { u8x16(vmaxq_u8(self.0, other.0)) }
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { u8x16(_mm_min_epi8(self.0, other.0)) }
        #[cfg(target_arch = "aarch64")]
        unsafe { u8x16(vminq_u8(self.0, other.0)) }
    }
    #[inline(always)]
    fn relu(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { u8x16(_mm_max_epi8(self.0, _mm_setzero_si128())) }
        #[cfg(target_arch = "aarch64")]
        unsafe { u8x16(vmaxq_u8(self.0, vdupq_n_u8(0))) }
    }
    #[inline(always)]
    fn relu6(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe { u8x16(_mm_min_epi8(self.relu().0, _mm_set1_epi8(6))) }
        #[cfg(target_arch = "aarch64")]
        unsafe { u8x16(vminq_u8(self.relu().0, vdupq_n_u8(6))) }
    }
}

impl VecConvertor for u8x16 {
    #[inline(always)]
    fn to_u8(self) -> u8x16 {
        self
    }
    #[inline(always)]
    fn to_i8(self) -> i8x16 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    fn to_bool(self) -> super::boolx16::boolx16 {
        unsafe { std::mem::transmute(self) }
    }
}