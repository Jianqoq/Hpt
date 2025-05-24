use std::arch::x86_64::*;
use crate::simd::_256bit::common::bf16x16::bf16x16;
use crate::simd::_256bit::common::f32x8::f32x8;
use crate::simd::_256bit::common::i32x8::i32x8;
use crate::simd::_256bit::common::u16x16::u16x16;
use crate::simd::_256bit::common::u32x8::u32x8;

impl bf16x16 {
    /// convert to 2 f32x8
    #[inline(always)]
    pub(crate)  fn to_2_f32vec(&self) -> [f32x8; 2] {
        unsafe {
            let vec: u16x16 = std::mem::transmute(*self);
            let mask = (vec & u16x16::splat(0x7FFFu16)).simd_gt(u16x16::splat(0x7F80u16));
            let mask_low = i32x8(_mm256_unpacklo_epi16(mask.0, mask.0));
            let mask_high = i32x8(_mm256_unpackhi_epi16(mask.0, mask.0));
            let vec_low = u32x8(_mm256_unpacklo_epi16(vec.0, vec.0));
            let vec_high = u32x8(_mm256_unpackhi_epi16(vec.0, vec.0));
            let sixteen = u32x8::splat(16);
            let t = u32x8::splat(0x0040u32);
            let true_low = (vec_low | t) << sixteen;
            let true_high = (vec_high | t) << sixteen;
            let false_low = vec_low << sixteen;
            let false_high = vec_high << sixteen;
            let res_low = mask_low.select_u32(true_low, false_low);
            let res_high = mask_high.select_u32(true_high, false_high);
            [
                f32x8(std::mem::transmute(res_low.0)),
                f32x8(std::mem::transmute(res_high.0)),
            ]
        }
    }

    /// convert to f32x8
    #[inline(always)]
    pub(crate)  fn high_to_f32vec(&self) -> f32x8 {
        unsafe {
            let vec: u16x16 = std::mem::transmute(*self);
            let mask = (vec & u16x16::splat(0x7FFFu16)).simd_gt(u16x16::splat(0x7F80u16));
            let mask_low = i32x8(_mm256_unpacklo_epi16(mask.0, mask.0));
            let vec_low = u32x8(_mm256_unpacklo_epi16(vec.0, vec.0));
            let sixteen = u32x8::splat(16);
            let t = u32x8::splat(0x0040u32);
            let true_low = (vec_low | t) << sixteen;
            let false_low = vec_low << sixteen;
            let res_low = mask_low.select_u32(true_low, false_low);
            f32x8(std::mem::transmute(res_low.0))
        }
    }

    /// convert from 2 bf16x16
    #[inline(always)]
    pub(crate)  fn from_2_f32vec(val: [f32x8; 2]) -> Self {
        unsafe {
            unsafe fn conv(vec: f32x8) -> __m256i {
                let x = u32x8(std::mem::transmute(vec.0));
                let nan_mask =
                    (x & u32x8::splat(0x7FFF_FFFFu32)).simd_gt(u32x8::splat(0x7F80_0000u32));
                let shifted = x >> u32x8::splat(16);

                let nan_result = shifted | u32x8::splat(0x0040u32);

                let round_bit = u32x8::splat(0x00008000u32);
                let rs_mask = (x & round_bit).simd_ne(u32x8::splat(0))
                    & (x & (u32x8::splat(3) * round_bit - u32x8::splat(1)))
                        .simd_ne(u32x8::splat(0));

                let round_result = shifted + rs_mask.select_u32(u32x8::splat(1), u32x8::splat(0));

                let final_result = nan_mask.select_u32(nan_result, round_result);
                _mm256_packus_epi32(final_result.0, _mm256_setzero_si256())
            }
            let high = conv(val[0]);
            let low = conv(val[1]);
            let result = _mm256_unpacklo_epi64(high, low);
            std::mem::transmute(result)
        }
    }
}