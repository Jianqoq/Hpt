use crate::{ convertion::VecConvertor, traits::{ SimdCompare, SimdMath, SimdSelect, VecTrait } };
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::u32x4::u32x4;

/// a vector of 4 i32 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct i32x4(
    #[cfg(target_arch = "x86_64")] pub(crate) __m128i,
    #[cfg(target_arch = "aarch64")] pub(crate) int32x4_t,
);

impl PartialEq for i32x4 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let cmp = _mm_cmpeq_epi32(self.0, other.0);
            _mm_movemask_epi8(cmp) == -1
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let cmp = vceqq_s32(self.0, other.0);
            vmaxvq_u32(cmp) == 0xffffffff && vminvq_u32(cmp) == 0xffffffff
        }
    }
}

impl Default for i32x4 {
    #[inline(always)]
    fn default() -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i32x4(_mm_setzero_si128())
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i32x4(vdupq_n_s32(0))
        }
    }
}

impl VecTrait<i32> for i32x4 {
    const SIZE: usize = 4;
    type Base = i32;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[i32]) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            _mm_storeu_si128(&mut self.0, _mm_loadu_si128(slice.as_ptr() as *const __m128i))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            self.0 = vld1q_s32(slice.as_ptr());
        }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i32x4(_mm_add_epi32(self.0, _mm_mullo_epi32(a.0, b.0)))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let mul = vmulq_s32(a.0, b.0);
            i32x4(vaddq_s32(self.0, mul))
        }
    }
    #[inline(always)]
    fn sum(&self) -> i32 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let arr: [i32; 4] = std::mem::transmute(self.0);
            arr.iter().sum()
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let sum = vaddvq_s32(self.0);
            sum as i32
        }
    }
    #[inline(always)]
    fn splat(val: i32) -> i32x4 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i32x4(_mm_set1_epi32(val))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i32x4(vdupq_n_s32(val))
        }
    }
}

impl i32x4 {
    #[allow(unused)]
    #[inline(always)]
    fn as_array(&self) -> [i32; 4] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl SimdCompare for i32x4 {
    type SimdMask = i32x4;
    #[inline(always)]
    fn simd_eq(self, other: Self) -> i32x4 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i32x4(_mm_cmpeq_epi32(self.0, other.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let cmp = vceqq_s32(self.0, other.0);
            i32x4(vreinterpretq_s32_u32(cmp))
        }
    }
    #[inline(always)]
    fn simd_ne(self, other: Self) -> i32x4 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let eq = _mm_cmpeq_epi32(self.0, other.0);
            i32x4(_mm_xor_si128(eq, _mm_set1_epi32(-1)))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let cmp = vceqq_s32(self.0, other.0);
            i32x4(vmvnq_s32(vreinterpretq_s32_u32(cmp)))
        }
    }
    #[inline(always)]
    fn simd_lt(self, other: Self) -> i32x4 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i32x4(_mm_cmplt_epi32(self.0, other.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let cmp = vcltq_s32(self.0, other.0);
            i32x4(vreinterpretq_s32_u32(cmp))
        }
    }
    #[inline(always)]
    fn simd_le(self, other: Self) -> i32x4 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let lt = _mm_cmplt_epi32(self.0, other.0);
            let eq = _mm_cmpeq_epi32(self.0, other.0);
            i32x4(_mm_or_si128(lt, eq))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let cmp = vcleq_s32(self.0, other.0);
            i32x4(vreinterpretq_s32_u32(cmp))
        }
    }
    #[inline(always)]
    fn simd_gt(self, other: Self) -> i32x4 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i32x4(_mm_cmpgt_epi32(self.0, other.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let cmp = vcgtq_s32(self.0, other.0);
            i32x4(vreinterpretq_s32_u32(cmp))
        }
    }
    #[inline(always)]
    fn simd_ge(self, other: Self) -> i32x4 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let gt = _mm_cmpgt_epi32(self.0, other.0);
            let eq = _mm_cmpeq_epi32(self.0, other.0);
            i32x4(_mm_or_si128(gt, eq))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let cmp = vcgeq_s32(self.0, other.0);
            i32x4(vreinterpretq_s32_u32(cmp))
        }
    }
}

impl SimdSelect<i32x4> for crate::vectors::arch_simd::_128bit::i32x4::i32x4 {
    #[inline(always)]
    fn select(&self, true_val: i32x4, false_val: i32x4) -> i32x4 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i32x4(_mm_blendv_epi8(false_val.0, true_val.0, self.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let zero = vdupq_n_s32(0);
            let cmp = vcltq_s32(self.0, zero);
            i32x4(vbslq_s32(cmp, true_val.0, false_val.0))
        }
    }
}

impl std::ops::Add for i32x4 {
    type Output = i32x4;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i32x4(_mm_add_epi32(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i32x4(vaddq_s32(self.0, rhs.0))
        }
    }
}
impl std::ops::Sub for i32x4 {
    type Output = i32x4;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i32x4(_mm_sub_epi32(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i32x4(vsubq_s32(self.0, rhs.0))
        }
    }
}
impl std::ops::Mul for i32x4 {
    type Output = i32x4;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i32x4(_mm_mullo_epi32(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i32x4(vmulq_s32(self.0, rhs.0))
        }
    }
}
impl std::ops::Div for i32x4 {
    type Output = i32x4;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [i32; 4] = std::mem::transmute(self.0);
            let arr2: [i32; 4] = std::mem::transmute(rhs.0);
            let mut arr3: [i32; 4] = [0; 4];
            for i in 0..4 {
                arr3[i] = arr[i] / arr2[i];
            }
            #[cfg(target_arch = "x86_64")]
            return i32x4(_mm_loadu_si128(arr3.as_ptr() as *const __m128i));
            #[cfg(target_arch = "aarch64")]
            return i32x4(vld1q_s32(arr3.as_ptr()));
        }
    }
}
impl std::ops::Rem for i32x4 {
    type Output = i32x4;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let arr: [i32; 4] = std::mem::transmute(self.0);
            let arr2: [i32; 4] = std::mem::transmute(rhs.0);
            let mut arr3: [i32; 4] = [0; 4];
            for i in 0..4 {
                arr3[i] = arr[i] % arr2[i];
            }
            #[cfg(target_arch = "x86_64")]
            return i32x4(_mm_loadu_si128(arr3.as_ptr() as *const __m128i));
            #[cfg(target_arch = "aarch64")]
            return i32x4(vld1q_s32(arr3.as_ptr()));
        }
    }
}

impl std::ops::Neg for i32x4 {
    type Output = i32x4;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i32x4(_mm_sign_epi32(self.0, _mm_set1_epi32(-1)))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i32x4(vnegq_s32(self.0))
        }
    }
}
impl std::ops::BitAnd for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i32x4(_mm_and_si128(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i32x4(vandq_s32(self.0, rhs.0))
        }
    }
}
impl std::ops::BitOr for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i32x4(_mm_or_si128(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i32x4(vorrq_s32(self.0, rhs.0))
        }
    }
}
impl std::ops::BitXor for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i32x4(_mm_xor_si128(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i32x4(veorq_s32(self.0, rhs.0))
        }
    }
}
impl std::ops::Not for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i32x4(_mm_xor_si128(self.0, _mm_set1_epi32(-1)))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i32x4(vmvnq_s32(self.0))
        }
    }
}
impl std::ops::Shl for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i32; 4] = std::mem::transmute(self.0);
            let b: [i32; 4] = std::mem::transmute(rhs.0);
            let mut result = [0; 4];
            for i in 0..4 {
                result[i] = a[i].wrapping_shl(b[i] as u32);
            }
            #[cfg(target_arch = "x86_64")]
            return i32x4(_mm_loadu_si128(result.as_ptr() as *const __m128i));
            #[cfg(target_arch = "aarch64")]
            return i32x4(vld1q_s32(result.as_ptr()));
        }
    }
}
impl std::ops::Shr for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self::Output {
        unsafe {
            let a: [i32; 4] = std::mem::transmute(self.0);
            let b: [i32; 4] = std::mem::transmute(rhs.0);
            let mut result = [0; 4];
            for i in 0..4 {
                result[i] = a[i].wrapping_shr(b[i] as u32);
            }
            #[cfg(target_arch = "x86_64")]
            return i32x4(_mm_loadu_si128(result.as_ptr() as *const __m128i));
            #[cfg(target_arch = "aarch64")]
            return i32x4(vld1q_s32(result.as_ptr()));
        }
    }
}
impl SimdMath<i32> for i32x4 {
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i32x4(_mm_max_epi32(self.0, other.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i32x4(vmaxq_s32(self.0, other.0))
        }
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i32x4(_mm_min_epi32(self.0, other.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i32x4(vminq_s32(self.0, other.0))
        }
    }
    #[inline(always)]
    fn relu(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i32x4(_mm_max_epi32(self.0, _mm_setzero_si128()))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i32x4(vmaxq_s32(self.0, vdupq_n_s32(0)))
        }
    }
    #[inline(always)]
    fn relu6(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i32x4(_mm_min_epi32(self.relu().0, _mm_set1_epi32(6)))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i32x4(vminq_s32(self.relu().0, vdupq_n_s32(6)))
        }
    }
}

impl VecConvertor for i32x4 {
    #[inline(always)]
    fn to_i32(self) -> i32x4 {
        self
    }
    #[inline(always)]
    fn to_u32(self) -> u32x4 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    fn to_f32(self) -> super::f32x4::f32x4 {
        #[cfg(target_arch = "x86_64")]
        unsafe { super::f32x4::f32x4(_mm_cvtepi32_ps(self.0)) }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            super::f32x4::f32x4(vcvtq_f32_s32(self.0))
        }
    }
    #[cfg(target_pointer_width = "32")]
    #[inline(always)]
    fn to_isize(self) -> super::isizex2::isizex2 {
        unsafe { std::mem::transmute(self) }
    }
    #[cfg(target_pointer_width = "32")]
    #[inline(always)]
    fn to_usize(self) -> super::usizex2::usizex2 {
        unsafe { std::mem::transmute(self) }
    }
}
