use std::arch::aarch64::*;
use crate::{
    simd::_128bit::common::{ bf16x8::bf16x8, f32x4::f32x4, u16x8::u16x8, u32x4::u32x4 },
    VecTrait,
};

impl VecTrait<half::bf16> for bf16x8 {
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
    fn splat(val: half::bf16) -> bf16x8 {
        bf16x8([val; 8])
    }
    #[inline(always)]
    fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self {
        let val = Self::splat(a.0[LANE as usize]);
        self.mul_add(val, b)
    }

    #[inline(always)]
    fn partial_load(ptr: *const half::bf16, num_elem: usize) -> Self {
        let mut result = Self::splat(half::bf16::default());
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, (&mut result.0) as *mut _ as *mut half::bf16, num_elem);
            result
        }
    }
    #[inline(always)]
    fn partial_store(self, ptr: *mut half::bf16, num_elem: usize) {
        unsafe {
            std::ptr::copy_nonoverlapping((&self.0) as *const _ as *const half::bf16, ptr, num_elem);
        }
    }
}
impl bf16x8 {
    /// convert to 2 f32x4
    #[inline(always)]
    pub fn to_2_f32vec(&self) -> [f32x4; 2] {
        unsafe {
            let vec: u16x8 = std::mem::transmute(*self);

            let (vec_low_16, vec_high_16) = (vget_low_u16(vec.0), vget_high_u16(vec.0));

            let vec_low = vshll_n_u16(vec_low_16, 0);
            let vec_high = vshll_n_u16(vec_high_16, 0);

            let mask = vdupq_n_u32(0x7fff);
            let threshold = vdupq_n_u32(0x7f80);
            let t = vdupq_n_u32(0x0040);
            let sixteen = vreinterpretq_s32_u32(vdupq_n_u32(16));

            let mask_low = vcgtq_u32(vandq_u32(vec_low, mask), threshold);
            let mask_high = vcgtq_u32(vandq_u32(vec_high, mask), threshold);

            let true_low = vreinterpretq_u32_s32(
                vshlq_s32(vreinterpretq_s32_u32(vorrq_u32(vec_low, t)), sixteen)
            );
            let true_high = vreinterpretq_u32_s32(
                vshlq_s32(vreinterpretq_s32_u32(vorrq_u32(vec_high, t)), sixteen)
            );
            let false_low = vreinterpretq_u32_s32(
                vshlq_s32(vreinterpretq_s32_u32(vec_low), sixteen)
            );
            let false_high = vreinterpretq_u32_s32(
                vshlq_s32(vreinterpretq_s32_u32(vec_high), sixteen)
            );

            let res_low = vbslq_u32(mask_low, true_low, false_low);
            let res_high = vbslq_u32(mask_high, true_high, false_high);

            [f32x4(vreinterpretq_f32_u32(res_low)), f32x4(vreinterpretq_f32_u32(res_high))]
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
            let true_low = vreinterpretq_u32_s32(
                vshlq_s32(vreinterpretq_s32_u32(vorrq_u32(vec_low, t)), sixteen)
            );
            let false_low = vreinterpretq_u32_s32(
                vshlq_s32(vreinterpretq_s32_u32(vec_low), sixteen)
            );

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
                let nan_mask = (x & u32x4::splat(0x7fff_ffffu32)).simd_gt(
                    u32x4::splat(0x7f80_0000u32)
                );
                let shifted = x >> u32x4::splat(16);

                let nan_result = shifted | u32x4::splat(0x0040u32);

                let round_bit = u32x4::splat(0x00008000u32);
                let rs_mask =
                    (x & round_bit).simd_ne(u32x4::splat(0)) &
                    (x & (u32x4::splat(3) * round_bit - u32x4::splat(1))).simd_ne(u32x4::splat(0));

                let round_result = shifted + rs_mask.select_u32x4(u32x4::splat(1), u32x4::splat(0));

                let final_result = nan_mask.select_u32x4(nan_result, round_result);
                vmovn_u32(final_result.0)
            };

            let high = conv(val[0]);
            let low = conv(val[1]);
            let result = vcombine_u16(high, low);
            std::mem::transmute(result)
        }
    }
}
