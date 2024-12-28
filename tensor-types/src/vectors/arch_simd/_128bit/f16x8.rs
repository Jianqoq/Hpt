use crate::convertion::VecConvertor;
use crate::traits::{SimdMath, VecTrait};
use crate::type_promote::{Eval2, FloatOutBinary2, NormalOut2, NormalOutUnary2};
use crate::vectors::arch_simd::_128bit::f32x4::f32x4;
use crate::vectors::arch_simd::_128bit::u16x8::u16x8;

use crate::traits::SimdCompare;

use super::i16x8::i16x8;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// a vector of 8 f16 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(16))]
pub struct f16x8(pub(crate) [half::f16; 8]);

#[allow(non_camel_case_types)]
pub(crate) type f16_promote = f16x8;

impl VecTrait<half::f16> for f16x8 {
    const SIZE: usize = 8;
    type Base = half::f16;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[half::f16]) {
        self.0.copy_from_slice(slice);
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        let [x0, x1] = self.to_2_f32x4();
        let [a0, a1] = a.to_2_f32x4();
        let [b0, b1] = b.to_2_f32x4();
        let res0 = x0.mul_add(a0, b0);
        let res1 = x1.mul_add(a1, b1);
        from_2_f32vec([res0, res1])
    }
    #[inline(always)]
    fn sum(&self) -> half::f16 {
        self.0.iter().sum()
    }
    #[inline(always)]
    fn splat(val: half::f16) -> f16x8 {
        f16x8([val; 8])
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const half::f16) -> Self {
        let mut result = [half::f16::ZERO; 8];
        for i in 0..8 {
            result[i] = unsafe { *ptr.add(i) };
        }
        f16x8(result)
    }
}

impl f16x8 {
    #[allow(unused)]
    #[inline(always)]
    fn as_array(&self) -> [half::f16; 8] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl f16x8 {
    /// check if the value is NaN, and return a mask
    #[inline(always)]
    pub fn is_nan(&self) -> i16x8 {
        let x = u16x8::splat(0x7c00u16);
        let y = u16x8::splat(0x03ffu16);
        let i: u16x8 = unsafe { std::mem::transmute(self.0) };

        let and = i & x;
        let eq = and.simd_eq(x);

        let and2 = i & y;
        let neq_zero = and2.simd_ne(u16x8::splat(0));

        let result = eq & neq_zero;

        unsafe { std::mem::transmute(result) }
    }
    /// check if the value is infinite, and return a mask
    #[inline(always)]
    pub fn is_infinite(&self) -> i16x8 {
        let x = u16x8::splat(0x7c00u16);
        let y = u16x8::splat(0x03ffu16);
        let i: u16x8 = unsafe { std::mem::transmute(self.0) };

        let and = i & x;
        let eq = and.simd_eq(x);

        let and2 = i & y;
        let eq_zero = and2.simd_eq(u16x8::splat(0));

        let result = eq & eq_zero;

        unsafe { std::mem::transmute(result) }
    }
    /// convert to f32x4
    #[inline(always)]
    pub fn to_2_f32x4(self) -> [f32x4; 2] {
        unsafe {
            #[cfg(target_feature = "f16c")]
            {
                use std::arch::x86_64::{_mm_cvtph_ps, _mm_loadu_si64};
                let raw_f16: [u16; 8] = std::mem::transmute(self.0);
                let f32x4_1 = _mm_cvtph_ps(_mm_loadu_si64(raw_f16.as_ptr() as *const _));
                let f32x4_2 = _mm_cvtph_ps(_mm_loadu_si64(raw_f16.as_ptr().add(4) as *const _));
                std::mem::transmute([f32x4_1, f32x4_2])
            }
            #[cfg(all(target_feature = "neon", target_arch = "aarch64"))]
            {
                use std::arch::aarch64::{float32x4_t, uint16x4_t};
                use std::arch::asm;
                use std::mem::MaybeUninit;
                let mut low_f32x4 = MaybeUninit::<uint16x4_t>::uninit();
                let mut high_f32x4 = MaybeUninit::<uint16x4_t>::uninit();
                std::ptr::copy_nonoverlapping(self.0.as_ptr(), low_f32x4.as_mut_ptr().cast(), 4);
                std::ptr::copy_nonoverlapping(
                    self.0.as_ptr().add(4),
                    high_f32x4.as_mut_ptr().cast(),
                    4,
                );
                let res0: float32x4_t;
                let res1: float32x4_t;
                asm!(
                    "fcvtl {0:v}.4s, {1:v}.4h",
                    out(vreg) res0,
                    in(vreg) low_f32x4.assume_init(),
                    options(pure, nomem, nostack)
                );
                asm!(
                    "fcvtl {0:v}.4s, {1:v}.4h",
                    out(vreg) res1,
                    in(vreg) high_f32x4.assume_init(),
                    options(pure, nomem, nostack)
                );

                std::mem::transmute([res0, res1])
            }
            #[cfg(not(any(
                target_feature = "f16c",
                all(target_feature = "neon", target_arch = "aarch64")
            )))]
            {
                let [high, low]: [[u16; 4]; 2] = std::mem::transmute(self.0);
                std::mem::transmute([u16_to_f32(high), u16_to_f32(low)])
            }
        }
    }

    /// convert from 2 f32x4
    #[inline(always)]
    pub fn from_2_f32x4(val: [f32x4; 2]) -> Self {
        unsafe {
            #[cfg(all(target_feature = "f16c", target_arch = "x86_64"))]
            {
                let f16_high = _mm_cvtps_ph(val[0].0, _MM_FROUND_TO_NEAREST_INT);
                let f16_low = _mm_cvtps_ph(val[1].0, _MM_FROUND_TO_NEAREST_INT);
                let result = _mm_unpacklo_epi64(f16_low, f16_high);
                std::mem::transmute(result)
            }
            #[cfg(not(all(target_feature = "f16c", target_arch = "x86_64")))]
            {
                let arr: [[f32; 4]; 2] = std::mem::transmute(val);
                let mut result = [0u16; 8];
                for i in 0..4 {
                    result[i] = half::f16::from_f32(arr[0][i]).to_bits();
                    result[i + 4] = half::f16::from_f32(arr[1][i]).to_bits();
                }
                std::mem::transmute(result)
            }
        }
    }
}
impl SimdCompare for f16x8 {
    type SimdMask = i16x8;
    #[inline(always)]
    fn simd_eq(self, other: Self) -> i16x8 {
        let x: i16x8 = unsafe { std::mem::transmute(self.0) };
        let y: i16x8 = unsafe { std::mem::transmute(other.0) };
        x.simd_eq(y)
    }
    #[inline(always)]
    fn simd_ne(self, other: Self) -> i16x8 {
        let x: i16x8 = unsafe { std::mem::transmute(self.0) };
        let y: i16x8 = unsafe { std::mem::transmute(other.0) };
        x.simd_ne(y)
    }
    #[inline(always)]
    fn simd_lt(self, other: Self) -> i16x8 {
        let x: i16x8 = unsafe { std::mem::transmute(self.0) };
        let y: i16x8 = unsafe { std::mem::transmute(other.0) };
        x.simd_lt(y)
    }
    #[inline(always)]
    fn simd_le(self, other: Self) -> i16x8 {
        let x: i16x8 = unsafe { std::mem::transmute(self.0) };
        let y: i16x8 = unsafe { std::mem::transmute(other.0) };
        x.simd_le(y)
    }
    #[inline(always)]
    fn simd_gt(self, other: Self) -> i16x8 {
        let x: i16x8 = unsafe { std::mem::transmute(self.0) };
        let y: i16x8 = unsafe { std::mem::transmute(other.0) };
        x.simd_gt(y)
    }
    #[inline(always)]
    fn simd_ge(self, other: Self) -> i16x8 {
        let x: i16x8 = unsafe { std::mem::transmute(self.0) };
        let y: i16x8 = unsafe { std::mem::transmute(other.0) };
        x.simd_ge(y)
    }
}

impl std::ops::Add for f16x8 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = f16x8::default();
        for i in 0..8 {
            ret.0[i] = self.0[i] + rhs.0[i];
        }
        ret
    }
}

impl std::ops::Sub for f16x8 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        let mut ret = f16x8::default();
        for i in 0..8 {
            ret.0[i] = self.0[i] - rhs.0[i];
        }
        ret
    }
}

impl std::ops::Mul for f16x8 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret = f16x8::default();
        for i in 0..8 {
            ret.0[i] = self.0[i] * rhs.0[i];
        }
        ret
    }
}

impl std::ops::Div for f16x8 {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        let mut ret = f16x8::default();
        for i in 0..8 {
            ret.0[i] = self.0[i] / rhs.0[i];
        }
        ret
    }
}
impl std::ops::Rem for f16x8 {
    type Output = Self;

    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        let mut ret = f16x8::default();
        for i in 0..8 {
            ret.0[i] = self.0[i] % rhs.0[i];
        }
        ret
    }
}
impl std::ops::Neg for f16x8 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        let mut ret = f16x8::default();
        for i in 0..8 {
            ret.0[i] = -self.0[i];
        }
        ret
    }
}

/// fallback to convert f16 to f32
#[inline(always)]
pub fn u16_to_f32(val: [u16; 4]) -> f32x4 {
    unsafe {
        std::mem::transmute([
            half::f16::from_bits(val[0]).to_f32(),
            half::f16::from_bits(val[1]).to_f32(),
            half::f16::from_bits(val[2]).to_f32(),
            half::f16::from_bits(val[3]).to_f32(),
        ])
    }
}

/// fallback to convert f32 to f16
#[inline(always)]
pub(crate) fn from_2_f32vec(val: [f32x4; 2]) -> f16x8 {
    #[cfg(all(target_feature = "f16c", target_arch = "x86_64"))]
    unsafe {
        let f16_high = _mm_cvtps_ph(val[0].0, _MM_FROUND_TO_NEAREST_INT);
        let f16_low = _mm_cvtps_ph(val[1].0, _MM_FROUND_TO_NEAREST_INT);
        let result = _mm_unpacklo_epi64(f16_high, f16_low);
        f16x8(std::mem::transmute(result))
    }
}

impl VecConvertor for f16x8 {
    #[inline(always)]
    fn to_i16(self) -> super::i16x8::i16x8 {
        #[cfg(target_feature = "sse2")]
        {
            unsafe {
                let [x0, x1]: [f32x4; 2] = std::mem::transmute(self.to_2_f32x4());
                let i0 = _mm_cvtps_epi32(x0.0);
                let i1 = _mm_cvtps_epi32(x1.0);
                let packed = _mm_packs_epi32(i0, i1);
                super::i16x8::i16x8(packed)
            }
        }
        #[cfg(all(target_feature = "neon", target_arch = "aarch64"))]
        {
            unimplemented!()
        }
    }
    #[inline(always)]
    fn to_u16(self) -> super::u16x8::u16x8 {
        #[cfg(target_feature = "sse2")]
        {
            unsafe {
                let [x0, x1]: [f32x4; 2] = std::mem::transmute(self.to_2_f32x4());
                let i0 = _mm_cvtps_epi32(x0.0);
                let i1 = _mm_cvtps_epi32(x1.0);
                let packed = _mm_packus_epi32(i0, i1);
                super::u16x8::u16x8(packed)
            }
        }
        #[cfg(all(target_feature = "neon", target_arch = "aarch64"))]
        {
            unimplemented!()
        }
    }
    #[inline(always)]
    fn to_f16(self) -> f16x8 {
        self
    }
}

impl SimdMath<half::f16> for f16x8 {
    #[inline(always)]
    fn sin(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_sin = high.sin();
        let low_sin = low.sin();
        Self::from_2_f32x4([high_sin, low_sin])
    }
    #[inline(always)]
    fn cos(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_cos = high.cos();
        let low_cos = low.cos();
        Self::from_2_f32x4([high_cos, low_cos])
    }
    #[inline(always)]
    fn tan(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_tan = high.tan();
        let low_tan = low.tan();
        Self::from_2_f32x4([high_tan, low_tan])
    }
    #[inline(always)]
    fn square(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_square = high.square();
        let low_square = low.square();
        Self::from_2_f32x4([high_square, low_square])
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_sqrt = high.sqrt();
        let low_sqrt = low.sqrt();
        Self::from_2_f32x4([high_sqrt, low_sqrt])
    }
    #[inline(always)]
    fn abs(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_abs = high.abs();
        let low_abs = low.abs();
        Self::from_2_f32x4([high_abs, low_abs])
    }
    #[inline(always)]
    fn floor(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_floor = high.floor();
        let low_floor = low.floor();
        Self::from_2_f32x4([high_floor, low_floor])
    }
    #[inline(always)]
    fn ceil(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_ceil = high.ceil();
        let low_ceil = low.ceil();
        Self::from_2_f32x4([high_ceil, low_ceil])
    }
    #[inline(always)]
    fn neg(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_neg = high.neg();
        let low_neg = low.neg();
        Self::from_2_f32x4([high_neg, low_neg])
    }
    #[inline(always)]
    fn round(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_round = high.round();
        let low_round = low.round();
        Self::from_2_f32x4([high_round, low_round])
    }
    #[inline(always)]
    fn sign(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_sign = high.sign();
        let low_sign = low.sign();
        Self::from_2_f32x4([high_sign, low_sign])
    }
    #[inline(always)]
    fn leaky_relu(self, alpha: half::f16) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_leaky_relu = high.leaky_relu(alpha.to_f32());
        let low_leaky_relu = low.leaky_relu(alpha.to_f32());
        Self::from_2_f32x4([high_leaky_relu, low_leaky_relu])
    }
    #[inline(always)]
    fn relu(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_relu = high.relu();
        let low_relu = low.relu();
        Self::from_2_f32x4([high_relu, low_relu])
    }
    #[inline(always)]
    fn relu6(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_relu6 = high.relu6();
        let low_relu6 = low.relu6();
        Self::from_2_f32x4([high_relu6, low_relu6])
    }
    #[inline(always)]
    fn pow(self, exp: Self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_pow = high.pow(exp.to_f32());
        let low_pow = low.pow(exp.to_f32());
        Self::from_2_f32x4([high_pow, low_pow])
    }
    #[inline(always)]
    fn asin(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_asin = high.asin();
        let low_asin = low.asin();
        Self::from_2_f32x4([high_asin, low_asin])
    }
    #[inline(always)]
    fn acos(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_acos = high.acos();
        let low_acos = low.acos();
        Self::from_2_f32x4([high_acos, low_acos])
    }
    #[inline(always)]
    fn atan(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_atan = high.atan();
        let low_atan = low.atan();
        Self::from_2_f32x4([high_atan, low_atan])
    }
    #[inline(always)]
    fn sinh(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_sinh = high.sinh();
        let low_sinh = low.sinh();
        Self::from_2_f32x4([high_sinh, low_sinh])
    }
    #[inline(always)]
    fn cosh(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_cosh = high.cosh();
        let low_cosh = low.cosh();
        Self::from_2_f32x4([high_cosh, low_cosh])
    }
    #[inline(always)]
    fn tanh(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_tanh = high.tanh();
        let low_tanh = low.tanh();
        Self::from_2_f32x4([high_tanh, low_tanh])
    }
    #[inline(always)]
    fn asinh(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_asinh = high.asinh();
        let low_asinh = low.asinh();
        Self::from_2_f32x4([high_asinh, low_asinh])
    }
    #[inline(always)]
    fn acosh(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_acosh = high.acosh();
        let low_acosh = low.acosh();
        Self::from_2_f32x4([high_acosh, low_acosh])
    }
    #[inline(always)]
    fn atanh(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_atanh = high.atanh();
        let low_atanh = low.atanh();
        Self::from_2_f32x4([high_atanh, low_atanh])
    }
    #[inline(always)]
    fn exp2(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_exp2 = high.exp2();
        let low_exp2 = low.exp2();
        Self::from_2_f32x4([high_exp2, low_exp2])
    }
    #[inline(always)]
    fn exp10(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_exp10 = high.exp10();
        let low_exp10 = low.exp10();
        Self::from_2_f32x4([high_exp10, low_exp10])
    }
    #[inline(always)]
    fn expm1(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_expm1 = high.expm1();
        let low_expm1 = low.expm1();
        Self::from_2_f32x4([high_expm1, low_expm1])
    }
    #[inline(always)]
    fn log10(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_log10 = high.log10();
        let low_log10 = low.log10();
        Self::from_2_f32x4([high_log10, low_log10])
    }
    #[inline(always)]
    fn log2(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_log2 = high.log2();
        let low_log2 = low.log2();
        Self::from_2_f32x4([high_log2, low_log2])
    }
    #[inline(always)]
    fn log1p(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_log1p = high.log1p();
        let low_log1p = low.log1p();
        Self::from_2_f32x4([high_log1p, low_log1p])
    }
    #[inline(always)]
    fn hypot(self, other: Self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let [high_other, low_other] = other.to_2_f32x4();
        let high_hypot = high.hypot(high_other);
        let low_hypot = low.hypot(low_other);
        Self::from_2_f32x4([high_hypot, low_hypot])
    }
    #[inline(always)]
    fn trunc(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_trunc = high.trunc();
        let low_trunc = low.trunc();
        Self::from_2_f32x4([high_trunc, low_trunc])
    }
    #[inline(always)]
    fn erf(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_erf = high.erf();
        let low_erf = low.erf();
        Self::from_2_f32x4([high_erf, low_erf])
    }
    #[inline(always)]
    fn cbrt(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_cbrt = high.cbrt();
        let low_cbrt = low.cbrt();
        Self::from_2_f32x4([high_cbrt, low_cbrt])
    }
    #[inline(always)]
    fn exp(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_exp = high.exp();
        let low_exp = low.exp();
        Self::from_2_f32x4([high_exp, low_exp])
    }
    #[inline(always)]
    fn ln(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_ln = high.ln();
        let low_ln = low.ln();
        Self::from_2_f32x4([high_ln, low_ln])
    }
    #[inline(always)]
    fn log(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_log = high.log();
        let low_log = low.log();
        Self::from_2_f32x4([high_log, low_log])
    }
    #[inline(always)]
    fn sincos(self) -> (Self, Self) {
        let [high, low] = self.to_2_f32x4();
        let (high_sin, high_cos) = high.sincos();
        let (low_sin, low_cos) = low.sincos();
        (
            Self::from_2_f32x4([high_sin, low_sin]),
            Self::from_2_f32x4([high_cos, low_cos]),
        )
    }
    #[inline(always)]
    fn atan2(self, other: Self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let [high_other, low_other] = other.to_2_f32x4();
        let high_atan2 = high.atan2(high_other);
        let low_atan2 = low.atan2(low_other);
        Self::from_2_f32x4([high_atan2, low_atan2])
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let [high_other, low_other] = other.to_2_f32x4();
        let high_min = high.min(high_other);
        let low_min = low.min(low_other);
        Self::from_2_f32x4([high_min, low_min])
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let [high_other, low_other] = other.to_2_f32x4();
        let high_max = high.max(high_other);
        let low_max = low.max(low_other);
        Self::from_2_f32x4([high_max, low_max])
    }
    #[inline(always)]
    fn hard_sigmoid(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_hard_sigmoid = high.hard_sigmoid();
        let low_hard_sigmoid = low.hard_sigmoid();
        Self::from_2_f32x4([high_hard_sigmoid, low_hard_sigmoid])
    }

    #[inline(always)]
    fn fast_hard_sigmoid(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_fast_hard_sigmoid = high.fast_hard_sigmoid();
        let low_fast_hard_sigmoid = low.fast_hard_sigmoid();
        Self::from_2_f32x4([high_fast_hard_sigmoid, low_fast_hard_sigmoid])
    }

    #[inline(always)]
    fn elu(self, alpha: half::f16) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_elu = high.elu(alpha.to_f32());
        let low_elu = low.elu(alpha.to_f32());
        Self::from_2_f32x4([high_elu, low_elu])
    }

    #[inline(always)]
    fn selu(self, alpha: half::f16, scale: half::f16) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_selu = high.selu(alpha.to_f32(), scale.to_f32());
        let low_selu = low.selu(alpha.to_f32(), scale.to_f32());
        Self::from_2_f32x4([high_selu, low_selu])
    }

    #[inline(always)]
    fn celu(self, alpha: half::f16) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_celu = high.celu(alpha.to_f32());
        let low_celu = low.celu(alpha.to_f32());
        Self::from_2_f32x4([high_celu, low_celu])
    }

    #[inline(always)]
    fn gelu(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_gelu = high.gelu();
        let low_gelu = low.gelu();
        Self::from_2_f32x4([high_gelu, low_gelu])
    }

    #[inline(always)]
    fn hard_swish(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_hard_swish = high.hard_swish();
        let low_hard_swish = low.hard_swish();
        Self::from_2_f32x4([high_hard_swish, low_hard_swish])
    }

    #[inline(always)]
    fn mish(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_mish = high.mish();
        let low_mish = low.mish();
        Self::from_2_f32x4([high_mish, low_mish])
    }

    #[inline(always)]
    fn softplus(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_softplus = high.softplus();
        let low_softplus = low.softplus();
        Self::from_2_f32x4([high_softplus, low_softplus])
    }

    #[inline(always)]
    fn recip(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_recip = high.recip();
        let low_recip = low.recip();
        Self::from_2_f32x4([high_recip, low_recip])
    }
    #[inline(always)]
    fn sigmoid(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_sigmoid = high.sigmoid();
        let low_sigmoid = low.sigmoid();
        Self::from_2_f32x4([high_sigmoid, low_sigmoid])
    }
    #[inline(always)]
    fn softsign(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_softsign = high.softsign();
        let low_softsign = low.softsign();
        Self::from_2_f32x4([high_softsign, low_softsign])
    }
}

impl FloatOutBinary2 for f16x8 {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn __log(self, base: Self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let [high_base, low_base] = base.to_2_f32x4();
        let high_log = high.__log(high_base);
        let low_log = low.__log(low_base);
        f16x8::from_2_f32x4([high_log, low_log])
    }
}

impl NormalOut2 for f16x8 {
    #[inline(always)]
    fn __add(self, rhs: Self) -> Self {
        self + rhs
    }

    #[inline(always)]
    fn __sub(self, rhs: Self) -> Self {
        self - rhs
    }

    #[inline(always)]
    fn __mul_add(self, a: Self, b: Self) -> Self {
        self.mul_add(a, b)
    }

    #[inline(always)]
    fn __mul(self, rhs: Self) -> Self {
        self * rhs
    }

    #[inline(always)]
    fn __pow(self, rhs: Self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let [high_rhs, low_rhs] = rhs.to_2_f32x4();
        let high_pow = high.__pow(high_rhs);
        let low_pow = low.__pow(low_rhs);
        f16x8::from_2_f32x4([high_pow, low_pow])
    }

    #[inline(always)]
    fn __rem(self, rhs: Self) -> Self {
        self % rhs
    }

    #[inline(always)]
    fn __max(self, rhs: Self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let [high_rhs, low_rhs] = rhs.to_2_f32x4();
        let high_max = high.__max(high_rhs);
        let low_max = low.__max(low_rhs);
        f16x8::from_2_f32x4([high_max, low_max])
    }

    #[inline(always)]
    fn __min(self, rhs: Self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let [high_rhs, low_rhs] = rhs.to_2_f32x4();
        let high_min = high.__min(high_rhs);
        let low_min = low.__min(low_rhs);
        f16x8::from_2_f32x4([high_min, low_min])
    }

    #[inline(always)]
    fn __clip(self, min: Self, max: Self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let [high_min, low_min] = min.to_2_f32x4();
        let [high_max, low_max] = max.to_2_f32x4();
        let high_clip = high.__clip(high_min, high_max);
        let low_clip = low.__clip(low_min, low_max);
        f16x8::from_2_f32x4([high_clip, low_clip])
    }
}

impl NormalOutUnary2 for f16x8 {
    #[inline(always)]
    fn __square(self) -> Self {
        self * self
    }

    #[inline(always)]
    fn __abs(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_abs = high.__abs();
        let low_abs = low.__abs();
        f16x8::from_2_f32x4([high_abs, low_abs])
    }

    #[inline(always)]
    fn __ceil(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_ceil = high.__ceil();
        let low_ceil = low.__ceil();
        f16x8::from_2_f32x4([high_ceil, low_ceil])
    }

    #[inline(always)]
    fn __floor(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_floor = high.__floor();
        let low_floor = low.__floor();
        f16x8::from_2_f32x4([high_floor, low_floor])
    }

    #[inline(always)]
    fn __neg(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_neg = high.__neg();
        let low_neg = low.__neg();
        f16x8::from_2_f32x4([high_neg, low_neg])
    }

    #[inline(always)]
    fn __round(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_round = high.__round();
        let low_round = low.__round();
        f16x8::from_2_f32x4([high_round, low_round])
    }

    #[inline(always)]
    fn __sign(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_sign = high.__sign();
        let low_sign = low.__sign();
        f16x8::from_2_f32x4([high_sign, low_sign])
    }

    #[inline(always)]
    fn __leaky_relu(self, alpha: Self) -> Self {
        self.max(f16x8::splat(half::f16::from_f32_const(0.0)))
            + alpha * self.min(f16x8::splat(half::f16::from_f32_const(0.0)))
    }

    #[inline(always)]
    fn __relu(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_relu = high.__relu();
        let low_relu = low.__relu();
        f16x8::from_2_f32x4([high_relu, low_relu])
    }

    #[inline(always)]
    fn __relu6(self) -> Self {
        let [high, low] = self.to_2_f32x4();
        let high_relu6 = high.__relu6();
        let low_relu6 = low.__relu6();
        f16x8::from_2_f32x4([high_relu6, low_relu6])
    }
}

impl Eval2 for f16x8 {
    type Output = i16x8;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        self.is_nan()
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        self.simd_ne(f16x8::default())
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        self.is_infinite()
    }
}
