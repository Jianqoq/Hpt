use std::arch::aarch64::*;

use crate::convertion::VecConvertor;
use crate::traits::{SimdCompare, SimdSelect};
use crate::{traits::VecTrait, vectors::arch_simd::_128bit::f32x4};

use crate::arch_simd::_128bit::common::bf16x8::bf16x8;
use crate::arch_simd::_128bit::f16x8;
use crate::arch_simd::_128bit::i16x8;
use crate::arch_simd::_128bit::u16x8;
use crate::arch_simd::_128bit::u32x4;

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
        let result = bf16x8::from_2_f32vec([res0, res1]);

        result
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
    #[inline(always)]
    fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self {
        let val = Self::splat(a[LANE as usize]);
        self.mul_add(val, b)
    }
}
impl bf16x8 {
    /// convert to 2 f32x4
    #[inline(always)]
    pub fn to_2_f32vec(&self) -> [f32x4; 2] {
        unsafe {
            let vec: u16x8 = std::mem::transmute(*self);

            // Split into low and high parts
            let (vec_low_16, vec_high_16) = (vget_low_u16(vec.0), vget_high_u16(vec.0));

            // Widen to 32-bit
            let vec_low = vshll_n_u16(vec_low_16, 0);
            let vec_high = vshll_n_u16(vec_high_16, 0);

            // Create masks and compare
            let mask = vdupq_n_u32(0x7fff);
            let threshold = vdupq_n_u32(0x7f80);
            let t = vdupq_n_u32(0x0040);
            let sixteen = vreinterpretq_s32_u32(vdupq_n_u32(16));

            // Compare and create masks
            let mask_low = vcgtq_u32(vandq_u32(vec_low, mask), threshold);
            let mask_high = vcgtq_u32(vandq_u32(vec_high, mask), threshold);

            // Create true and false results
            let true_low = vreinterpretq_u32_s32(vshlq_s32(
                vreinterpretq_s32_u32(vorrq_u32(vec_low, t)),
                sixteen,
            ));
            let true_high = vreinterpretq_u32_s32(vshlq_s32(
                vreinterpretq_s32_u32(vorrq_u32(vec_high, t)),
                sixteen,
            ));
            let false_low =
                vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(vec_low), sixteen));
            let false_high =
                vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(vec_high), sixteen));

            // Select based on mask
            let res_low = vbslq_u32(mask_low, true_low, false_low);
            let res_high = vbslq_u32(mask_high, true_high, false_high);

            [
                f32x4(vreinterpretq_f32_u32(res_low)),
                f32x4(vreinterpretq_f32_u32(res_high)),
            ]
        }
    }

    /// convert to f32x4
    #[inline(always)]
    pub fn high_to_f32vec(&self) -> f32x4 {
        unsafe {
            let vec: u16x8 = std::mem::transmute(*self);

            // Split into low and high parts
            let vec_low_16 = vget_low_u16(vec.0);

            // Widen to 32-bit
            let vec_low = vshll_n_u16(vec_low_16, 0);

            // Create masks and compare
            let mask = vdupq_n_u32(0x7fff);
            let threshold = vdupq_n_u32(0x7f80);
            let t = vdupq_n_u32(0x0040);
            let sixteen = vreinterpretq_s32_u32(vdupq_n_u32(16));

            // Compare and create masks
            let mask_low = vcgtq_u32(vandq_u32(vec_low, mask), threshold);

            // Create true and false results
            let true_low = vreinterpretq_u32_s32(vshlq_s32(
                vreinterpretq_s32_u32(vorrq_u32(vec_low, t)),
                sixteen,
            ));
            let false_low =
                vreinterpretq_u32_s32(vshlq_s32(vreinterpretq_s32_u32(vec_low), sixteen));

            // Select based on mask
            let res_low = vbslq_u32(mask_low, true_low, false_low);

            f32x4(vreinterpretq_f32_u32(res_low))
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
                vmovn_u32(final_result.0)
            };

            let high = conv(val[0]);
            let low = conv(val[1]);
            let result = vcombine_u16(high, low);
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
            let i0 = vcvtq_s32_f32(x0.0);
            let i1 = vcvtq_s32_f32(x1.0);
            i16x8(vqmovn_high_s32(vqmovn_s32(i0), i1))
        }
    }
    #[inline(always)]
    fn to_u16(self) -> u16x8 {
        unsafe {
            let [x0, x1]: [f32x4; 2] = std::mem::transmute(self.to_2_f32vec());
            let i0 = vcvtq_u32_f32(x0.0);
            let i1 = vcvtq_u32_f32(x1.0);
            u16x8(vqmovn_high_u32(vqmovn_u32(i0), i1))
        }
    }
}
