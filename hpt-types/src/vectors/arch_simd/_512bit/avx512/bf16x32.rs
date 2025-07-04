use crate::arch_simd::_512bit::u16x32;
use crate::convertion::VecConvertor;
use crate::simd::_512bit::u32x16;
use crate::traits::{SimdCompare, SimdSelect};
use crate::{traits::VecTrait, vectors::arch_simd::_512bit::f32x16};

use crate::simd::_512bit::f16x32;
use crate::simd::_512bit::i16x32;
use crate::simd::_512bit::i32x16;

use std::arch::x86_64::*;

use crate::simd::_512bit::bf16x32;

impl bf16x32 {
    /// convert to 2 f32x16
    #[inline(always)]
    pub fn to_2_f32vec(&self) -> [f32x16; 2] {
        unsafe {
            let vec: u16x32 = std::mem::transmute(*self);
            let mask = (vec & u16x32::splat(0x7FFFu16)).simd_gt(u16x32::splat(0x7F80u16));
            let mask_low = i32x16(_mm512_unpacklo_epi16(mask.0, mask.0));
            let mask_high = i32x16(_mm512_unpackhi_epi16(mask.0, mask.0));
            let vec_low = u32x16(_mm512_unpacklo_epi16(vec.0, vec.0));
            let vec_high = u32x16(_mm512_unpackhi_epi16(vec.0, vec.0));
            let sixteen = u32x16::splat(16);
            let t = u32x16::splat(0x0040u32);
            let true_low = (vec_low | t) << sixteen;
            let true_high = (vec_high | t) << sixteen;
            let false_low = vec_low << sixteen;
            let false_high = vec_high << sixteen;
            let res_low = mask_low.select(true_low, false_low);
            let res_high = mask_high.select(true_high, false_high);
            [
                f32x16(std::mem::transmute(res_low.0)),
                f32x16(std::mem::transmute(res_high.0)),
            ]
        }
    }

    /// convert to f32x16
    #[inline(always)]
    pub fn high_to_f32vec(&self) -> f32x16 {
        unsafe {
            let vec: u16x32 = std::mem::transmute(*self);
            let mask = (vec & u16x32::splat(0x7FFFu16)).simd_gt(u16x32::splat(0x7F80u16));
            let mask_low = i32x16(_mm512_unpacklo_epi16(mask.0, mask.0));
            let vec_low = u32x16(_mm512_unpacklo_epi16(vec.0, vec.0));
            let sixteen = u32x16::splat(16);
            let t = u32x16::splat(0x0040u32);
            let true_low = (vec_low | t) << sixteen;
            let false_low = vec_low << sixteen;
            let res_low = mask_low.select(true_low, false_low);
            f32x16(std::mem::transmute(res_low.0))
        }
    }

    /// convert from 2 bf16x32
    #[inline(always)]
    pub fn from_2_f32vec(val: [f32x16; 2]) -> Self {
        unsafe {
            unsafe fn conv(vec: f32x16) -> __m512i {
                let x = u32x16(std::mem::transmute(vec.0));
                let nan_mask =
                    (x & u32x16::splat(0x7FFF_FFFFu32)).simd_gt(u32x16::splat(0x7F80_0000u32));
                let shifted = x >> u32x16::splat(16);

                let nan_result = shifted | u32x16::splat(0x0040u32);

                let round_bit = u32x16::splat(0x00008000u32);
                let rs_mask = (x & round_bit).simd_ne(u32x16::splat(0))
                    & (x & (u32x16::splat(3) * round_bit - u32x16::splat(1)))
                        .simd_ne(u32x16::splat(0));

                let round_result = shifted + rs_mask.select(u32x16::splat(1), u32x16::splat(0));

                let final_result = nan_mask.select(nan_result, round_result);
                _mm512_packus_epi32(final_result.0, _mm512_setzero_si512())
            }
            let high = conv(val[0]);
            let low = conv(val[1]);
            let result = _mm512_unpacklo_epi64(high, low);
            std::mem::transmute(result)
        }
    }

    /// check if the value is NaN and return a mask
    #[inline(always)]
    pub fn is_nan(&self) -> i16x32 {
        let res: [i16; 32] = self.0.map(|x| if x.is_nan() { 1 } else { 0 });
        unsafe { std::mem::transmute(res) }
    }
}
impl SimdCompare for bf16x32 {
    type SimdMask = i16x32;
    #[inline(always)]
    fn simd_eq(self, other: Self) -> i16x32 {
        unsafe {
            let self_ptr = &self.0 as *const _ as *const __m512i;
            let other_ptr = &other.0 as *const _ as *const __m512i;
            let a = _mm512_loadu_si512(self_ptr);
            let b = _mm512_loadu_si512(other_ptr);
            let mask = _mm512_cmp_epi16_mask::<_MM_CMPINT_EQ>(a, b);
            i16x32(_mm512_mask_mov_epi16(_mm512_set1_epi16(0), mask, _mm512_set1_epi16(-1)))
        }
    }
    #[inline(always)]
    fn simd_ne(self, other: Self) -> i16x32 {
        unsafe {
            let self_ptr = &self.0 as *const _ as *const __m512i;
            let other_ptr = &other.0 as *const _ as *const __m512i;
            let a = _mm512_loadu_si512(self_ptr);
            let b = _mm512_loadu_si512(other_ptr);
            let mask = _mm512_cmp_epi16_mask::<_MM_CMPINT_NE>(a, b);
            i16x32(_mm512_maskz_mov_epi16(mask, _mm512_set1_epi16(-1)))
        }
    }
    #[inline(always)]
    fn simd_lt(self, other: Self) -> i16x32 {
        unsafe {
            let self_ptr = &self.0 as *const _ as *const __m512i;
            let other_ptr = &other.0 as *const _ as *const __m512i;
            let a = _mm512_loadu_si512(self_ptr);
            let b = _mm512_loadu_si512(other_ptr);
            let mask = _mm512_cmp_epi16_mask::<_MM_CMPINT_LT>(a, b);
            i16x32(_mm512_maskz_mov_epi16(mask, _mm512_set1_epi16(-1)))
        }
    }
    #[inline(always)]
    fn simd_le(self, other: Self) -> i16x32 {
        unsafe {
            let self_ptr = &self.0 as *const _ as *const __m512i;
            let other_ptr = &other.0 as *const _ as *const __m512i;
            let a = _mm512_loadu_si512(self_ptr);
            let b = _mm512_loadu_si512(other_ptr);
            let mask = _mm512_cmp_epi16_mask::<_MM_CMPINT_LE>(a, b);
            i16x32(_mm512_maskz_mov_epi16(mask, _mm512_set1_epi16(-1)))
        }
    }
    #[inline(always)]
    fn simd_gt(self, other: Self) -> i16x32 {
        unsafe {
            let self_ptr = &self.0 as *const _ as *const __m512i;
            let other_ptr = &other.0 as *const _ as *const __m512i;
            let a = _mm512_loadu_si512(self_ptr);
            let b = _mm512_loadu_si512(other_ptr);
            let mask = _mm512_cmp_epi16_mask::<_MM_CMPINT_NLE>(a, b);
            i16x32(_mm512_maskz_mov_epi16(mask, _mm512_set1_epi16(-1)))
        }
    }
    #[inline(always)]
    fn simd_ge(self, other: Self) -> i16x32 {
        unsafe {
            let self_ptr = &self.0 as *const _ as *const __m512i;
            let other_ptr = &other.0 as *const _ as *const __m512i;
            let a = _mm512_loadu_si512(self_ptr);
            let b = _mm512_loadu_si512(other_ptr);
            let mask = _mm512_cmp_epi16_mask::<_MM_CMPINT_NLT>(a, b);
            i16x32(_mm512_maskz_mov_epi16(mask, _mm512_set1_epi16(-1)))
        }
    }
}

impl VecConvertor for bf16x32 {
    #[inline(always)]
    fn to_bf16(self) -> bf16x32 {
        self
    }
    #[inline(always)]
    fn to_f16(self) -> f16x32 {
        let [x0, x1] = self.to_2_f32vec();
        f16x32::from_2_f32vec([x0, x1])
    }
    #[inline(always)]
    fn to_i16(self) -> i16x32 {
        unsafe {
            let [x0, x1]: [f32x16; 2] = std::mem::transmute(self.to_2_f32vec());
            let i0 = _mm512_cvtps_epi32(x0.0);
            let i1 = _mm512_cvtps_epi32(x1.0);
            let packed = _mm512_packs_epi32(i0, i1);
            i16x32(packed)
        }
    }
    #[inline(always)]
    fn to_u16(self) -> u16x32 {
        unsafe {
            let [x0, x1]: [f32x16; 2] = std::mem::transmute(self.to_2_f32vec());
            let i0 = _mm512_cvtps_epi32(x0.0);
            let i1 = _mm512_cvtps_epi32(x1.0);
            let packed = _mm512_packus_epi32(i0, i1);
            u16x32(packed)
        }
    }
}
