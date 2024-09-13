use std::simd::{ cmp::{ SimdPartialEq, SimdPartialOrd }, num::{ SimdFloat, SimdUint }, Simd };
use crate::{ vectors::_128bit::f32x4::f32x4, traits::{ Init, VecCommon, VecTrait } };
use crate::into_vec::IntoVec;
use crate::_128bit::u16x8::u16x8;

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
pub struct bf16x8(pub(crate) [half::bf16; 8]);

impl VecTrait<half::bf16> for bf16x8 {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[half::bf16]) {
        self.0.copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const half::bf16 {
        self.0.as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, a: Self, b: Self) -> Self {
        let [x0, x1]: [f32x4; 2] = unsafe { std::mem::transmute(self.to_2_f32x4()) };
        let [a0, a1]: [f32x4; 2] = unsafe { std::mem::transmute(a.to_2_f32x4()) };
        let [b0, b1]: [f32x4; 2] = unsafe { std::mem::transmute(b.to_2_f32x4()) };
        let res0 = x0._mul_add(a0, b0);
        let res1 = x1._mul_add(a1, b1);
        bf16x8::from_2_f32x8([res0, res1])
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

    fn extract(self, idx: usize) -> half::bf16 {
        self.0[idx]
    }
}
impl VecCommon for bf16x8 {
    const SIZE: usize = 8;
    
    type Base = half::bf16;
}
impl Init<half::bf16> for bf16x8 {
    fn splat(val: half::bf16) -> bf16x8 {
        bf16x8([val; 8])
    }
}
impl IntoVec<bf16x8> for bf16x8 {
    fn into_vec(self) -> bf16x8 {
        self
    }
}

impl bf16x8 {
    #[cfg(target_feature = "avx2")]
    pub fn to_2_f32x4(&self) -> [f32x4; 2] {
        let [ai, bi]: [std::simd::u16x4; 2] = unsafe { std::mem::transmute(self.0) };

        let [ai, bi]: [std::simd::u32x4; 2] = [ai.cast(), bi.cast()];
        let [am, bm] = [
            (ai & std::simd::u32x4::splat(0x7fff)).simd_gt(std::simd::u32x4::splat(0x7f80)),
            (bi & std::simd::u32x4::splat(0x7fff)).simd_gt(std::simd::u32x4::splat(0x7f80)),
        ];
        let [an_adjusted, bn_adjusted] = [
            (ai | std::simd::u32x4::splat(0x0040)) << 8,
            (bi | std::simd::u32x4::splat(0x0040)) << 8,
        ];
        let [a_normal, b_normal] = [ai << 8, bi << 8];
        let [a_res, b_res] = [am.select(an_adjusted, a_normal), bm.select(bn_adjusted, b_normal)];
        unsafe { std::mem::transmute([a_res, b_res]) }
    }
    #[cfg(target_feature = "avx2")]
    pub fn from_2_f32x8(inp: [f32x4; 2]) -> Self {
        use std::simd::num::SimdInt;
        use std::simd::Simd;
        use std::simd::cmp::SimdPartialEq;
        let [af, bf]: [Simd<f32, 4>; 2] = unsafe { std::mem::transmute(inp) };
        let [au, bu]: [Simd<u32, 4>; 2] = unsafe { std::mem::transmute(inp) };
        let [am, bm] = [af.is_nan().cast::<i16>(), bf.is_nan().cast::<i16>()];
        let round_bit = std::simd::u32x4::splat(0x0000_8000);
        let one = std::simd::u32x4::splat(1);
        let [a_round_increment, b_round_increment] = [
            (au & round_bit).simd_ne(std::simd::u32x4::splat(0)) &
                (au & (round_bit - one)).simd_ne(std::simd::u32x4::splat(0)),
            (bu & round_bit).simd_ne(std::simd::u32x4::splat(0)) &
                (bu & (round_bit - one)).simd_ne(std::simd::u32x4::splat(0)),
        ];
        let [a_rounded, b_rounded] = [
            au + a_round_increment.to_int().cast(),
            bu + b_round_increment.to_int().cast(),
        ];
        let [a_bf16_values, b_bf16_values] = [
            (a_rounded >> 8).cast::<u16>(),
            (b_rounded >> 8).cast::<u16>(),
        ];
        let [a_nan_adjusted, b_nan_adjusted] = [
            a_bf16_values | std::simd::u16x4::splat(0x0040),
            b_bf16_values | std::simd::u16x4::splat(0x0040),
        ];
        let [a_res, b_res] = [
            am.select(a_nan_adjusted, a_bf16_values),
            bm.select(b_nan_adjusted, b_bf16_values),
        ];
        unsafe { std::mem::transmute([a_res, b_res]) }
    }

    pub fn is_nan(&self) -> u16x8 {
        let x = std::simd::u16x8::splat(0x7f80u16);
        let y = std::simd::u16x8::splat(0x007fu16);
        let i: Simd<u16, 8> = unsafe { std::mem::transmute(self.0) };
        let and = i & x;
        let eq: Simd<u16, 8> = unsafe { std::mem::transmute(and.simd_eq(x)) };
        let and2 = i & y;
        let neq_zero: Simd<u16, 8> = unsafe {
            std::mem::transmute(and2.simd_ne(std::simd::u16x8::splat(0)))
        };
        unsafe { std::mem::transmute(eq & neq_zero) }
    }

    pub fn is_infinite(&self) -> u16x8 {
        let x = u16x8::splat(0x7f80u16);
        let y = u16x8::splat(0x007fu16);
        let i: Simd<u16, 8> = unsafe { std::mem::transmute(self.0) };

        let and = i & x.0;
        let eq = and.simd_eq(x.0);

        let and2 = i & y.0;
        let eq_zero = and2.simd_eq(u16x8::splat(0).0);

        let result = eq & eq_zero;

        unsafe { std::mem::transmute(result) }
    }

    pub fn simd_eq(&self, other: Self) -> u16x8 {
        let x: Simd<u16, 8> = unsafe { std::mem::transmute(self.0) };
        let y: Simd<u16, 8> = unsafe { std::mem::transmute(other.0) };
        let eq = x.simd_eq(y);
        unsafe { std::mem::transmute(eq) }
    }
    pub fn simd_ne(&self, other: Self) -> u16x8 {
        let x: Simd<u16, 8> = unsafe { std::mem::transmute(self.0) };
        let y: Simd<u16, 8> = unsafe { std::mem::transmute(other.0) };
        let ne = x.simd_ne(y);
        unsafe { std::mem::transmute(ne) }
    }
    pub fn simd_lt(&self, other: Self) -> u16x8 {
        let x: Simd<u16, 8> = unsafe { std::mem::transmute(self.0) };
        let y: Simd<u16, 8> = unsafe { std::mem::transmute(other.0) };
        let lt = x.simd_lt(y);
        unsafe { std::mem::transmute(lt) }
    }
    pub fn simd_le(&self, other: Self) -> u16x8 {
        let x: Simd<u16, 8> = unsafe { std::mem::transmute(self.0) };
        let y: Simd<u16, 8> = unsafe { std::mem::transmute(other.0) };
        let le = x.simd_le(y);
        unsafe { std::mem::transmute(le) }
    }
    pub fn simd_gt(&self, other: Self) -> u16x8 {
        let x: Simd<u16, 8> = unsafe { std::mem::transmute(self.0) };
        let y: Simd<u16, 8> = unsafe { std::mem::transmute(other.0) };
        let gt = x.simd_gt(y);
        unsafe { std::mem::transmute(gt) }
    }
    pub fn simd_ge(&self, other: Self) -> u16x8 {
        let x: Simd<u16, 8> = unsafe { std::mem::transmute(self.0) };
        let y: Simd<u16, 8> = unsafe { std::mem::transmute(other.0) };
        let ge = x.simd_ge(y);
        unsafe { std::mem::transmute(ge) }
    }
}

impl std::ops::Add for bf16x8 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = bf16x8::default();
        for i in 0..8 {
            ret.0[i] = self.0[i] + rhs.0[i];
        }
        ret
    }
}
impl std::ops::Sub for bf16x8 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut ret = bf16x8::default();
        for i in 0..8 {
            ret.0[i] = self.0[i] - rhs.0[i];
        }
        ret
    }
}
impl std::ops::Mul for bf16x8 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret = bf16x8::default();
        for i in 0..8 {
            ret.0[i] = self.0[i] * rhs.0[i];
        }
        ret
    }
}
impl std::ops::Div for bf16x8 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut ret = bf16x8::default();
        for i in 0..8 {
            ret.0[i] = self.0[i] / rhs.0[i];
        }
        ret
    }
}
impl std::ops::Rem for bf16x8 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        let mut ret = bf16x8::default();
        for i in 0..8 {
            ret.0[i] = self.0[i] % rhs.0[i];
        }
        ret
    }
}
