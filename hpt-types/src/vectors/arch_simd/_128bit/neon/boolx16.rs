use std::arch::aarch64::{ vbslq_u8, vreinterpretq_u8_s8 };

use crate::simd::_128bit::i8x16;
use crate::traits::{ SimdSelect, VecTrait };
use crate::vectors::arch_simd::_128bit::common::boolx16::boolx16;

impl VecTrait<bool> for boolx16 {
    const SIZE: usize = 16;
    type Base = bool;
    #[inline(always)]
    fn mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn sum(&self) -> bool {
        self.0
            .iter()
            .map(|&x| x as u8)
            .sum::<u8>() > 0
    }
    #[inline(always)]
    fn splat(val: bool) -> boolx16 {
        boolx16([val; 16])
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const bool) -> Self {
        let mut result = [false; 16];
        for i in 0..16 {
            result[i] = unsafe { *ptr.add(i) };
        }
        boolx16(result)
    }
    #[inline(always)]
    fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self {
        let val = Self::splat(a[LANE as usize]);
        self.mul_add(val, b)
    }
}

impl SimdSelect<boolx16> for i8x16 {
    #[inline(always)]
    fn select(&self, true_val: boolx16, false_val: boolx16) -> boolx16 {
        unsafe {
            let mask = vreinterpretq_u8_s8(self.0);
            boolx16(
                std::mem::transmute(
                    vbslq_u8(
                        mask,
                        std::mem::transmute(true_val.0),
                        std::mem::transmute(false_val.0)
                    )
                )
            )
        }
    }
}
