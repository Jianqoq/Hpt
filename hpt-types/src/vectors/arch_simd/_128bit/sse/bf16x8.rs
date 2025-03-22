use crate::arch_simd::_128bit::u16x8;
use crate::convertion::VecConvertor;
use crate::traits::{SimdCompare, SimdSelect};
use crate::{traits::VecTrait, vectors::arch_simd::_128bit::f32x4};

use crate::arch_simd::_128bit::common::bf16x8::bf16x8;
use crate::arch_simd::_128bit::f16x8;
use crate::arch_simd::_128bit::i16x8;
use crate::arch_simd::_128bit::u32x4;
use std::arch::x86_64::*;

impl VecTrait<half::bf16> for bf16x8 {
    const SIZE: usize = 8;
    type Base = half::bf16;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[half::bf16]) {
        self.0.copy_from_slice(slice);
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        let [x0, x1] = self.to_2_f32vec();
        let [a0, a1] = a.to_2_f32vec();
        let [b0, b1] = b.to_2_f32vec();

        let res0 = x0.mul_add(a0, b0);
        let res1 = x1.mul_add(a1, b1);

        bf16x8::from_2_f32vec([res0, res1])
    }
    #[inline(always)]
    fn sum(&self) -> half::bf16 {
        self.0.iter().sum()
    }
    #[inline(always)]
    fn splat(val: half::bf16) -> bf16x8 {
        bf16x8([val; 8])
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const half::bf16) -> Self {
        let mut result = [half::bf16::ZERO; 8];
        for i in 0..8 {
            result[i] = unsafe { *ptr.add(i) };
        }
        bf16x8(result)
    }
}

impl bf16x8 {
    /// convert to 2 f32x4
    #[inline(always)]
    pub fn to_2_f32vec(&self) -> [f32x4; 2] {
        unsafe {
            use crate::simd::_128bit::i32x4;
            let vec: u16x8 = std::mem::transmute(*self);
            let mask = (vec & u16x8::splat(0x7fffu16)).simd_gt(u16x8::splat(0x7f80u16));
            let mask_low = i32x4(_mm_unpacklo_epi16(mask.0, mask.0));
            let mask_high = i32x4(_mm_unpackhi_epi16(mask.0, mask.0));
            let vec_low = u32x4(_mm_unpacklo_epi16(vec.0, vec.0));
            let vec_high = u32x4(_mm_unpackhi_epi16(vec.0, vec.0));
            let sixteen = u32x4::splat(16);
            let t = u32x4::splat(0x0040u32);
            let true_low = (vec_low | t) << sixteen;
            let true_high = (vec_high | t) << sixteen;
            let false_low = vec_low << sixteen;
            let false_high = vec_high << sixteen;
            let res_low = mask_low.select(true_low, false_low);
            let res_high = mask_high.select(true_high, false_high);
            [
                f32x4(std::mem::transmute(res_low.0)),
                f32x4(std::mem::transmute(res_high.0)),
            ]
        }
    }

    /// convert from 2 f32x4
    #[inline(always)]
    pub fn from_2_f32vec(val: [f32x4; 2]) -> Self {
        unsafe {
            let conv = |vec: f32x4| {
                let x = u32x4(std::mem::transmute(vec.0));
                let nan_mask =
                    (x & u32x4::splat(0x7fff_ffffu32)).simd_gt(u32x4::splat(0x7f80_0000u32));
                let shifted = x >> u32x4::splat(16);

                let nan_result = shifted | u32x4::splat(0x0040u32);

                let round_bit = u32x4::splat(0x00008000u32);
                let rs_mask = (x & round_bit).simd_ne(u32x4::splat(0))
                    & (x & (u32x4::splat(3) * round_bit - u32x4::splat(1)))
                        .simd_ne(u32x4::splat(0));
                let round_result = shifted + rs_mask.select(u32x4::splat(1), u32x4::splat(0));

                let final_result = nan_mask.select(nan_result, round_result);
                _mm_packus_epi32(final_result.0, _mm_setzero_si128())
            };

            let high = conv(val[0]);
            let low = conv(val[1]);
            let result = _mm_unpacklo_epi64(high, low);
            std::mem::transmute(result)
        }
    }
}

impl VecConvertor for bf16x8 {
    #[inline(always)]
    fn to_bf16(self) -> bf16x8 {
        self
    }
    #[inline(always)]
    fn to_f16(self) -> f16x8 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    fn to_i16(self) -> i16x8 {
        unsafe {
            let [x0, x1]: [f32x4; 2] = std::mem::transmute(self.to_2_f32vec());
            let i0 = _mm_cvtps_epi32(x0.0);
            let i1 = _mm_cvtps_epi32(x1.0);
            let packed = _mm_packs_epi32(i0, i1);
            i16x8(packed)
        }
    }
    #[inline(always)]
    fn to_u16(self) -> u16x8 {
        unsafe {
            let [x0, x1]: [f32x4; 2] = std::mem::transmute(self.to_2_f32vec());
            let i0 = _mm_cvtps_epi32(x0.0);
            let i1 = _mm_cvtps_epi32(x1.0);
            let packed = _mm_packus_epi32(i0, i1);
            u16x8(packed)
        }
    }
}
