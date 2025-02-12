use crate::arch_simd::_128bit::u16x8::u16x8;
use crate::convertion::VecConvertor;
use crate::traits::{SimdCompare, SimdMath, SimdSelect};
use crate::type_promote::{Eval2, FloatOutBinary2, NormalOut2, NormalOutUnary2};
use crate::{traits::VecTrait, vectors::arch_simd::_128bit::f32x4::f32x4};

use super::i16x8::i16x8;
use super::u32x4::u32x4;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// a vector of 8 bf16 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(16))]
pub struct bf16x8(pub(crate) [half::bf16; 8]);

#[allow(non_camel_case_types)]
pub(crate) type bf16_promote = bf16x8;

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
}

impl bf16x8 {
    #[allow(unused)]
    #[inline(always)]
    fn as_array(&self) -> [half::bf16; 8] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl bf16x8 {
    /// convert to 2 f32x4
    #[inline(always)]
    pub fn to_2_f32vec(&self) -> [f32x4; 2] {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use crate::simd::_128bit::i32x4::i32x4;
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
        #[cfg(target_arch = "aarch64")]
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

    /// convert from 2 f32x4
    #[inline(always)]
    pub fn from_2_f32vec(val: [f32x4; 2]) -> Self {
        unsafe {
            let conv = |vec: f32x4| {
                let x = u32x4(std::mem::transmute(vec.0));
                let nan_mask =
                    (x & u32x4::splat(0x7fff_ffffu32)).simd_gt(u32x4::splat(0x7f80_0000u32));
                let shifted = x >> u32x4::splat(16);

                // NaN 处理
                let nan_result = shifted | u32x4::splat(0x0040u32);

                // 舍入检查
                let round_bit = u32x4::splat(0x00008000u32);
                let rs_mask = (x & round_bit).simd_ne(u32x4::splat(0))
                    & (x & (u32x4::splat(3) * round_bit - u32x4::splat(1)))
                        .simd_ne(u32x4::splat(0));

                // 舍入处理
                let round_result = shifted + rs_mask.select(u32x4::splat(1), u32x4::splat(0));

                // 最终选择
                let final_result = nan_mask.select(nan_result, round_result);
                #[cfg(target_arch = "x86_64")]
                return _mm_packus_epi32(final_result.0, _mm_setzero_si128()); // 打包为 16 位
                #[cfg(target_arch = "aarch64")]
                {
                    vmovn_u32(final_result.0)
                }
            };

            let high = conv(val[0]);
            let low = conv(val[1]);
            #[cfg(target_arch = "x86_64")]
            let result = _mm_unpacklo_epi64(high, low);
            #[cfg(target_arch = "aarch64")]
            let result = vcombine_u16(high, low);
            std::mem::transmute(result)
        }
    }
}
impl SimdCompare for bf16x8 {
    type SimdMask = i16x8;
    #[inline(always)]
    fn simd_eq(self, other: Self) -> i16x8 {
        unsafe {
            let self_i16: i16x8 = std::mem::transmute(self);
            let other_i16: i16x8 = std::mem::transmute(other);
            self_i16.simd_eq(other_i16)
        }
    }

    #[inline(always)]
    fn simd_ne(self, other: Self) -> i16x8 {
        unsafe {
            let self_i16: i16x8 = std::mem::transmute(self);
            let other_i16: i16x8 = std::mem::transmute(other);
            self_i16.simd_ne(other_i16)
        }
    }

    #[inline(always)]
    fn simd_lt(self, other: Self) -> i16x8 {
        unsafe {
            let self_i16: i16x8 = std::mem::transmute(self);
            let other_i16: i16x8 = std::mem::transmute(other);
            self_i16.simd_lt(other_i16)
        }
    }
    #[inline(always)]
    fn simd_le(self, other: Self) -> i16x8 {
        unsafe {
            let self_i16: i16x8 = std::mem::transmute(self);
            let other_i16: i16x8 = std::mem::transmute(other);
            self_i16.simd_le(other_i16)
        }
    }
    #[inline(always)]
    fn simd_gt(self, other: Self) -> i16x8 {
        unsafe {
            let self_i16: i16x8 = std::mem::transmute(self);
            let other_i16: i16x8 = std::mem::transmute(other);
            self_i16.simd_gt(other_i16)
        }
    }
    #[inline(always)]
    fn simd_ge(self, other: Self) -> i16x8 {
        unsafe {
            let self_i16: i16x8 = std::mem::transmute(self);
            let other_i16: i16x8 = std::mem::transmute(other);
            self_i16.simd_ge(other_i16)
        }
    }
}

impl std::ops::Add for bf16x8 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let [x0, x1] = self.to_2_f32vec();
        let [y0, y1] = rhs.to_2_f32vec();
        let low_add = x0 + y0;
        let high_add = x1 + y1;
        bf16x8::from_2_f32vec([low_add, high_add])
    }
}
impl std::ops::Sub for bf16x8 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        let [x0, x1] = self.to_2_f32vec();
        let [y0, y1] = rhs.to_2_f32vec();
        let low_sub = x0 - y0;
        let high_sub = x1 - y1;
        bf16x8::from_2_f32vec([low_sub, high_sub])
    }
}
impl std::ops::Mul for bf16x8 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        let [x0, x1] = self.to_2_f32vec();
        let [y0, y1] = rhs.to_2_f32vec();
        let low_mul = x0 * y0;
        let high_mul = x1 * y1;
        bf16x8::from_2_f32vec([low_mul, high_mul])
    }
}
impl std::ops::Div for bf16x8 {
    type Output = Self;

    #[inline(always)]
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

    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        let mut ret = bf16x8::default();
        for i in 0..8 {
            ret.0[i] = self.0[i] % rhs.0[i];
        }
        ret
    }
}
impl std::ops::Neg for bf16x8 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        let mut ret = bf16x8::default();
        for i in 0..8 {
            ret.0[i] = -self.0[i];
        }
        ret
    }
}

impl VecConvertor for bf16x8 {
    #[inline(always)]
    fn to_bf16(self) -> bf16x8 {
        self
    }
    #[inline(always)]
    fn to_f16(self) -> super::f16x8::f16x8 {
        unsafe { std::mem::transmute(self) }
    }
    #[inline(always)]
    fn to_i16(self) -> super::i16x8::i16x8 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let [x0, x1]: [f32x4; 2] = std::mem::transmute(self.to_2_f32vec());
            let i0 = _mm_cvtps_epi32(x0.0);
            let i1 = _mm_cvtps_epi32(x1.0);
            let packed = _mm_packs_epi32(i0, i1);
            super::i16x8::i16x8(packed)
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let [x0, x1]: [f32x4; 2] = std::mem::transmute(self.to_2_f32vec());
            let i0 = vcvtq_s32_f32(x0.0);
            let i1 = vcvtq_s32_f32(x1.0);
            super::i16x8::i16x8(vqmovn_high_s32(vqmovn_s32(i0), i1))
        }
    }
    #[inline(always)]
    fn to_u16(self) -> super::u16x8::u16x8 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let [x0, x1]: [f32x4; 2] = std::mem::transmute(self.to_2_f32vec());
            let i0 = _mm_cvtps_epi32(x0.0);
            let i1 = _mm_cvtps_epi32(x1.0);
            let packed = _mm_packus_epi32(i0, i1);
            super::u16x8::u16x8(packed)
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let [x0, x1]: [f32x4; 2] = std::mem::transmute(self.to_2_f32vec());
            let i0 = vcvtq_u32_f32(x0.0);
            let i1 = vcvtq_u32_f32(x1.0);
            super::u16x8::u16x8(vqmovn_high_u32(vqmovn_u32(i0), i1))
        }
    }
}

impl SimdMath<half::bf16> for bf16x8 {
    #[inline(always)]
    fn sin(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_sin = high.sin();
        let low_sin = low.sin();
        Self::from_2_f32vec([high_sin, low_sin])
    }
    #[inline(always)]
    fn cos(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_cos = high.cos();
        let low_cos = low.cos();
        Self::from_2_f32vec([high_cos, low_cos])
    }
    #[inline(always)]
    fn tan(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_tan = high.tan();
        let low_tan = low.tan();
        Self::from_2_f32vec([high_tan, low_tan])
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_sqrt = high.sqrt();
        let low_sqrt = low.sqrt();
        Self::from_2_f32vec([high_sqrt, low_sqrt])
    }
    #[inline(always)]
    fn abs(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_abs = high.abs();
        let low_abs = low.abs();
        Self::from_2_f32vec([high_abs, low_abs])
    }
    #[inline(always)]
    fn floor(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_floor = high.floor();
        let low_floor = low.floor();
        Self::from_2_f32vec([high_floor, low_floor])
    }
    #[inline(always)]
    fn ceil(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_ceil = high.ceil();
        let low_ceil = low.ceil();
        Self::from_2_f32vec([high_ceil, low_ceil])
    }
    #[inline(always)]
    fn neg(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_neg = high.neg();
        let low_neg = low.neg();
        Self::from_2_f32vec([high_neg, low_neg])
    }
    #[inline(always)]
    fn round(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_round = high.round();
        let low_round = low.round();
        Self::from_2_f32vec([high_round, low_round])
    }
    #[inline(always)]
    fn signum(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_sign = high.signum();
        let low_sign = low.signum();
        Self::from_2_f32vec([high_sign, low_sign])
    }
    #[inline(always)]
    fn leaky_relu(self, alpha: Self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let [high_alpha, low_alpha] = alpha.to_2_f32vec();
        let high_leaky_relu = high.leaky_relu(high_alpha);
        let low_leaky_relu = low.leaky_relu(low_alpha);
        Self::from_2_f32vec([high_leaky_relu, low_leaky_relu])
    }
    #[inline(always)]
    fn relu(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_relu = high.relu();
        let low_relu = low.relu();
        Self::from_2_f32vec([high_relu, low_relu])
    }
    #[inline(always)]
    fn relu6(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_relu6 = high.relu6();
        let low_relu6 = low.relu6();
        Self::from_2_f32vec([high_relu6, low_relu6])
    }
    #[inline(always)]
    fn pow(self, exp: Self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let [high_exp, low_exp] = exp.to_2_f32vec();
        let high_pow = high.pow(high_exp);
        let low_pow = low.pow(low_exp);
        Self::from_2_f32vec([high_pow, low_pow])
    }
    #[inline(always)]
    fn asin(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_asin = high.asin();
        let low_asin = low.asin();
        Self::from_2_f32vec([high_asin, low_asin])
    }
    #[inline(always)]
    fn acos(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_acos = high.acos();
        let low_acos = low.acos();
        Self::from_2_f32vec([high_acos, low_acos])
    }
    #[inline(always)]
    fn atan(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_atan = high.atan();
        let low_atan = low.atan();
        Self::from_2_f32vec([high_atan, low_atan])
    }
    #[inline(always)]
    fn sinh(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_sinh = high.sinh();
        let low_sinh = low.sinh();
        Self::from_2_f32vec([high_sinh, low_sinh])
    }
    #[inline(always)]
    fn cosh(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_cosh = high.cosh();
        let low_cosh = low.cosh();
        Self::from_2_f32vec([high_cosh, low_cosh])
    }
    #[inline(always)]
    fn tanh(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_tanh = high.tanh();
        let low_tanh = low.tanh();
        Self::from_2_f32vec([high_tanh, low_tanh])
    }
    #[inline(always)]
    fn asinh(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_asinh = high.asinh();
        let low_asinh = low.asinh();
        Self::from_2_f32vec([high_asinh, low_asinh])
    }
    #[inline(always)]
    fn acosh(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_acosh = high.acosh();
        let low_acosh = low.acosh();
        Self::from_2_f32vec([high_acosh, low_acosh])
    }
    #[inline(always)]
    fn atanh(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_atanh = high.atanh();
        let low_atanh = low.atanh();
        Self::from_2_f32vec([high_atanh, low_atanh])
    }
    #[inline(always)]
    fn exp2(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_exp2 = high.exp2();
        let low_exp2 = low.exp2();
        Self::from_2_f32vec([high_exp2, low_exp2])
    }
    #[inline(always)]
    fn exp10(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_exp10 = high.exp10();
        let low_exp10 = low.exp10();
        Self::from_2_f32vec([high_exp10, low_exp10])
    }
    #[inline(always)]
    fn expm1(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_expm1 = high.expm1();
        let low_expm1 = low.expm1();
        Self::from_2_f32vec([high_expm1, low_expm1])
    }
    #[inline(always)]
    fn log10(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_log10 = high.log10();
        let low_log10 = low.log10();
        Self::from_2_f32vec([high_log10, low_log10])
    }
    #[inline(always)]
    fn log2(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_log2 = high.log2();
        let low_log2 = low.log2();
        Self::from_2_f32vec([high_log2, low_log2])
    }
    #[inline(always)]
    fn log1p(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_log1p = high.log1p();
        let low_log1p = low.log1p();
        Self::from_2_f32vec([high_log1p, low_log1p])
    }
    #[inline(always)]
    fn hypot(self, other: Self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let [high_other, low_other] = other.to_2_f32vec();
        let high_hypot = high.hypot(high_other);
        let low_hypot = low.hypot(low_other);
        Self::from_2_f32vec([high_hypot, low_hypot])
    }
    #[inline(always)]
    fn trunc(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_trunc = high.trunc();
        let low_trunc = low.trunc();
        Self::from_2_f32vec([high_trunc, low_trunc])
    }
    #[inline(always)]
    fn erf(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_erf = high.erf();
        let low_erf = low.erf();
        Self::from_2_f32vec([high_erf, low_erf])
    }
    #[inline(always)]
    fn cbrt(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_cbrt = high.cbrt();
        let low_cbrt = low.cbrt();
        Self::from_2_f32vec([high_cbrt, low_cbrt])
    }
    #[inline(always)]
    fn exp(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_exp = high.exp();
        let low_exp = low.exp();
        Self::from_2_f32vec([high_exp, low_exp])
    }
    #[inline(always)]
    fn ln(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_ln = high.ln();
        let low_ln = low.ln();
        Self::from_2_f32vec([high_ln, low_ln])
    }
    #[inline(always)]
    fn sincos(self) -> (Self, Self) {
        let [high, low] = self.to_2_f32vec();
        let (high_sin, high_cos) = high.sincos();
        let (low_sin, low_cos) = low.sincos();
        (
            Self::from_2_f32vec([high_sin, low_sin]),
            Self::from_2_f32vec([high_cos, low_cos]),
        )
    }
    #[inline(always)]
    fn atan2(self, other: Self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let [high_other, low_other] = other.to_2_f32vec();
        let high_atan2 = high.atan2(high_other);
        let low_atan2 = low.atan2(low_other);
        Self::from_2_f32vec([high_atan2, low_atan2])
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let [high_other, low_other] = other.to_2_f32vec();
        let high_min = high.min(high_other);
        let low_min = low.min(low_other);
        Self::from_2_f32vec([high_min, low_min])
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let [high_other, low_other] = other.to_2_f32vec();
        let high_max = high.max(high_other);
        let low_max = low.max(low_other);
        Self::from_2_f32vec([high_max, low_max])
    }
    #[inline(always)]
    fn hard_sigmoid(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_hard_sigmoid = high.hard_sigmoid();
        let low_hard_sigmoid = low.hard_sigmoid();
        Self::from_2_f32vec([high_hard_sigmoid, low_hard_sigmoid])
    }

    #[inline(always)]
    fn elu(self, alpha: Self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let [high_alpha, low_alpha] = alpha.to_2_f32vec();
        let high_elu = high.elu(high_alpha);
        let low_elu = low.elu(low_alpha);
        Self::from_2_f32vec([high_elu, low_elu])
    }

    #[inline(always)]
    fn selu(self, alpha: Self, scale: Self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let [high_alpha, low_alpha] = alpha.to_2_f32vec();
        let [high_scale, low_scale] = scale.to_2_f32vec();
        let high_selu = high.selu(high_alpha, high_scale);
        let low_selu = low.selu(low_alpha, low_scale);
        Self::from_2_f32vec([high_selu, low_selu])
    }

    #[inline(always)]
    fn celu(self, scale: Self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let [high_scale, low_scale] = scale.to_2_f32vec();
        let high_celu = high.celu(high_scale);
        let low_celu = low.celu(low_scale);
        Self::from_2_f32vec([high_celu, low_celu])
    }

    #[inline(always)]
    fn gelu(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_gelu = high.gelu();
        let low_gelu = low.gelu();
        Self::from_2_f32vec([high_gelu, low_gelu])
    }

    #[inline(always)]
    fn hard_swish(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_hard_swish = high.hard_swish();
        let low_hard_swish = low.hard_swish();
        Self::from_2_f32vec([high_hard_swish, low_hard_swish])
    }

    #[inline(always)]
    fn mish(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_mish = high.mish();
        let low_mish = low.mish();
        Self::from_2_f32vec([high_mish, low_mish])
    }

    #[inline(always)]
    fn softplus(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_softplus = high.softplus();
        let low_softplus = low.softplus();
        Self::from_2_f32vec([high_softplus, low_softplus])
    }

    #[inline(always)]
    fn recip(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_recip = high.recip();
        let low_recip = low.recip();
        Self::from_2_f32vec([high_recip, low_recip])
    }
    #[inline(always)]
    fn sigmoid(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_sigmoid = high.sigmoid();
        let low_sigmoid = low.sigmoid();
        Self::from_2_f32vec([high_sigmoid, low_sigmoid])
    }
    #[inline(always)]
    fn softsign(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_softsign = high.softsign();
        let low_softsign = low.softsign();
        Self::from_2_f32vec([high_softsign, low_softsign])
    }
}

impl FloatOutBinary2 for bf16x8 {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn __log(self, base: Self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let [high_base, low_base] = base.to_2_f32vec();
        let high_log = high.__log(high_base);
        let low_log = low.__log(low_base);
        bf16x8::from_2_f32vec([high_log, low_log])
    }

    #[inline(always)]
    fn __hypot(self, rhs: Self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let [high_rhs, low_rhs] = rhs.to_2_f32vec();
        let high_hypot = high.__hypot(high_rhs);
        let low_hypot = low.__hypot(low_rhs);
        bf16x8::from_2_f32vec([high_hypot, low_hypot])
    }
}

impl NormalOut2 for bf16x8 {
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
        self.pow(rhs)
    }

    #[inline(always)]
    fn __rem(self, rhs: Self) -> Self {
        self % rhs
    }

    #[inline(always)]
    fn __max(self, rhs: Self) -> Self {
        self.max(rhs)
    }

    #[inline(always)]
    fn __min(self, rhs: Self) -> Self {
        self.min(rhs)
    }

    #[inline(always)]
    fn __clamp(self, min: Self, max: Self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let [high_min, low_min] = min.to_2_f32vec();
        let [high_max, low_max] = max.to_2_f32vec();
        let high_clip = high.__clamp(high_min, high_max);
        let low_clip = low.__clamp(low_min, low_max);
        bf16x8::from_2_f32vec([high_clip, low_clip])
    }
}

impl NormalOutUnary2 for bf16x8 {
    #[inline(always)]
    fn __square(self) -> Self {
        self * self
    }

    #[inline(always)]
    fn __abs(self) -> Self {
        self.abs()
    }

    #[inline(always)]
    fn __ceil(self) -> Self {
        self.ceil()
    }

    #[inline(always)]
    fn __floor(self) -> Self {
        self.floor()
    }

    #[inline(always)]
    fn __neg(self) -> Self {
        self.neg()
    }

    #[inline(always)]
    fn __round(self) -> Self {
        self.round()
    }

    #[inline(always)]
    fn __signum(self) -> Self {
        self.signum()
    }

    #[inline(always)]
    fn __trunc(self) -> Self {
        self.trunc()
    }

    #[inline(always)]
    fn __leaky_relu(self, alpha: Self) -> Self {
        self.leaky_relu(alpha)
    }

    #[inline(always)]
    fn __relu(self) -> Self {
        self.relu()
    }

    #[inline(always)]
    fn __relu6(self) -> Self {
        self.relu6()
    }

    #[inline(always)]
    fn __copysign(self, rhs: Self) -> Self {
        self.copysign(rhs)
    }
}

impl Eval2 for bf16x8 {
    type Output = i16x8;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        let res: [i16; 8] = self.0.map(|x| if x.is_nan() { -1 } else { 0 });
        unsafe { std::mem::transmute(res) }
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        self.simd_ne(bf16x8::default())
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        let sign_mask = u16x8::splat(0x8000u16);
        let inf_mask = u16x8::splat(0x7f80u16);
        let frac_mask = u16x8::splat(0x007fu16);

        let i: u16x8 = unsafe { std::mem::transmute(self.0) };

        let exp = i & inf_mask;
        let frac = i & frac_mask;
        let is_inf = exp.simd_eq(inf_mask) & frac.simd_eq(u16x8::splat(0));
        let is_neg = (i & sign_mask).simd_ne(u16x8::splat(0));

        let result = is_inf.select(
            is_neg.select(i16x8::splat(-1), i16x8::splat(1)),
            i16x8::splat(0),
        );

        result
    }
}
