use std::simd::{ cmp::SimdPartialOrd, num::{ SimdFloat, SimdUint } };

use crate::into_vec::IntoVec;

use super::{ f32x8::f32x8, traits::{ Init, VecSize, VecTrait } };

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq)]
pub struct bf16x16(pub(crate) [half::bf16; 16]);

impl VecTrait<half::bf16> for bf16x16 {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[half::bf16]) {
        self.0.copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const half::bf16 {
        self.0.as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut half::bf16 {
        self.0.as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut half::bf16 {
        self.0.as_ptr() as *mut _
    }
    #[inline(always)]
    fn sum(&self) -> half::bf16 {
        self.0.iter().sum()
    }
}
impl VecSize for bf16x16 {
    const SIZE: usize = 16;
}
impl Init<half::bf16> for bf16x16 {
    fn splat(val: half::bf16) -> bf16x16 {
        bf16x16([val; 16])
    }

    unsafe fn from_ptr(ptr: *const half::bf16) -> Self {
        let mut arr = [half::bf16::default(); 16];
        arr.copy_from_slice(std::slice::from_raw_parts(ptr, 16));
        bf16x16(arr)
    }
}
impl IntoVec<bf16x16> for bf16x16 {
    fn into_vec(self) -> bf16x16 {
        self
    }
}

impl bf16x16 {
    #[cfg(target_feature = "avx2")]
    pub fn to_2_f32x8(&self) -> [f32x8; 2] {
        let [ai, bi]: [std::simd::u16x8; 2] = unsafe { std::mem::transmute(self.0) };

        let [ai, bi]: [std::simd::u32x8; 2] = [ai.cast(), bi.cast()];
        let [am, bm] = [
            (ai & std::simd::u32x8::splat(0x7fff)).simd_gt(std::simd::u32x8::splat(0x7f80)),
            (bi & std::simd::u32x8::splat(0x7fff)).simd_gt(std::simd::u32x8::splat(0x7f80)),
        ];
        let [an_adjusted, bn_adjusted] = [
            (ai | std::simd::u32x8::splat(0x0040)) << 16,
            (bi | std::simd::u32x8::splat(0x0040)) << 16,
        ];
        let [a_normal, b_normal] = [ai << 16, bi << 16];
        let [a_res, b_res] = [am.select(an_adjusted, a_normal), bm.select(bn_adjusted, b_normal)];
        unsafe { std::mem::transmute([a_res, b_res]) }
    }
    #[cfg(target_feature = "avx2")]
    pub fn from_2_f32x8(inp: [f32x8; 2]) -> Self {
        use std::simd::num::SimdInt;
        use std::simd::Simd;
        use std::simd::cmp::SimdPartialEq;
        let [af, bf]: [Simd<f32, 8>; 2] = unsafe { std::mem::transmute(inp) };
        let [au, bu]: [Simd<u32, 8>; 2] = unsafe { std::mem::transmute(inp) };
        let [am, bm] = [af.is_nan().cast::<i16>(), bf.is_nan().cast::<i16>()];
        let round_bit = std::simd::u32x8::splat(0x0000_8000);
        let one = std::simd::u32x8::splat(1);
        let [a_round_increment, b_round_increment] = [
            (au & round_bit).simd_ne(std::simd::u32x8::splat(0)) &
                (au & (round_bit - one)).simd_ne(std::simd::u32x8::splat(0)),
            (bu & round_bit).simd_ne(std::simd::u32x8::splat(0)) &
                (bu & (round_bit - one)).simd_ne(std::simd::u32x8::splat(0)),
        ];
        let [a_rounded, b_rounded] = [
            au + a_round_increment.to_int().cast(),
            bu + b_round_increment.to_int().cast(),
        ];
        let [a_bf16_values, b_bf16_values] = [
            (a_rounded >> 16).cast::<u16>(),
            (b_rounded >> 16).cast::<u16>(),
        ];
        let [a_nan_adjusted, b_nan_adjusted] = [
            a_bf16_values | std::simd::u16x8::splat(0x0040),
            b_bf16_values | std::simd::u16x8::splat(0x0040),
        ];
        let [a_res, b_res] = [
            am.select(a_nan_adjusted, a_bf16_values),
            bm.select(b_nan_adjusted, b_bf16_values),
        ];
        unsafe { std::mem::transmute([a_res, b_res]) }
    }
}

impl std::ops::Add for bf16x16 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = bf16x16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] + rhs.0[i];
        }
        ret
    }
}
impl std::ops::Sub for bf16x16 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut ret = bf16x16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] - rhs.0[i];
        }
        ret
    }
}
impl std::ops::Mul for bf16x16 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret = bf16x16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] * rhs.0[i];
        }
        ret
    }
}
impl std::ops::Div for bf16x16 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut ret = bf16x16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] / rhs.0[i];
        }
        ret
    }
}
impl std::ops::Rem for bf16x16 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        let mut ret = bf16x16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] % rhs.0[i];
        }
        ret
    }
}
