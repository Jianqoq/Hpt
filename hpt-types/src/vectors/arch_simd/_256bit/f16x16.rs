use crate::convertion::VecConvertor;
use crate::traits::{SimdMath, SimdSelect, VecTrait};
use crate::type_promote::{Eval2, FloatOutBinary2, NormalOut2, NormalOutUnary2};
use crate::vectors::arch_simd::_256bit::f32x8::f32x8;
use crate::vectors::arch_simd::_256bit::u16x16::u16x16;

use crate::traits::SimdCompare;

use super::i16x16::i16x16;

/// a vector of 16 f16 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(32))]
pub struct f16x16(pub(crate) [half::f16; 16]);

/// helper to impl the promote trait
#[allow(non_camel_case_types)]
pub(crate) type f16_promote = f16x16;

impl VecTrait<half::f16> for f16x16 {
    const SIZE: usize = 16;
    type Base = half::f16;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[half::f16]) {
        self.0.copy_from_slice(slice);
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        let [x0, x1]: [f32x8; 2] = unsafe { std::mem::transmute(self.to_2_f32vec()) };
        let [a0, a1]: [f32x8; 2] = unsafe { std::mem::transmute(a.to_2_f32vec()) };
        let [b0, b1]: [f32x8; 2] = unsafe { std::mem::transmute(b.to_2_f32vec()) };
        let res0 = x0.mul_add(a0, b0);
        let res1 = x1.mul_add(a1, b1);
        let res0 = f32x8_to_f16x8(res0);
        let res1 = f32x8_to_f16x8(res1);
        unsafe { std::mem::transmute([res0, res1]) }
    }
    #[inline(always)]
    fn sum(&self) -> half::f16 {
        self.0.iter().sum()
    }
    #[inline(always)]
    fn splat(val: half::f16) -> f16x16 {
        f16x16([val; 16])
    }

    unsafe fn from_ptr(ptr: *const half::f16) -> Self {
        f16x16([
            ptr.read_unaligned(),
            ptr.add(1).read_unaligned(),
            ptr.add(2).read_unaligned(),
            ptr.add(3).read_unaligned(),
            ptr.add(4).read_unaligned(),
            ptr.add(5).read_unaligned(),
            ptr.add(6).read_unaligned(),
            ptr.add(7).read_unaligned(),
            ptr.add(8).read_unaligned(),
            ptr.add(9).read_unaligned(),
            ptr.add(10).read_unaligned(),
            ptr.add(11).read_unaligned(),
            ptr.add(12).read_unaligned(),
            ptr.add(13).read_unaligned(),
            ptr.add(14).read_unaligned(),
            ptr.add(15).read_unaligned(),
        ])
    }
}

impl f16x16 {
    /// convert the vector to an array
    #[inline(always)]
    pub fn as_array(&self) -> [half::f16; 16] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl f16x16 {
    /// convert to Self
    #[inline(always)]
    pub fn to_2_f32vec(self) -> [f32x8; 2] {
        unsafe {
            #[cfg(all(
                target_feature = "f16c",
                target_arch = "x86_64",
                target_feature = "avx2"
            ))]
            {
                use std::arch::x86_64::*;
                let raw_f16: [u16; 16] = std::mem::transmute(self.0);
                let f32x4_1 = _mm256_cvtph_ps(_mm_loadu_si128(raw_f16.as_ptr() as *const _));
                let f32x4_2 = _mm256_cvtph_ps(_mm_loadu_si128(raw_f16.as_ptr().add(8) as *const _));
                std::mem::transmute([f32x4_1, f32x4_2])
            }
            #[cfg(not(all(
                target_feature = "f16c",
                target_arch = "x86_64",
                target_feature = "avx2"
            )))]
            {
                let mut result = [[0f32; 8]; 2];
                for i in 0..8 {
                    result[0][i] = self.0[i].to_f32();
                    result[1][i] = self.0[i + 8].to_f32();
                }
                std::mem::transmute(result)
            }
        }
    }

    /// convert from 2 f32x4
    #[inline(always)]
    pub fn from_2_f32vec(val: [f32x8; 2]) -> Self {
        unsafe {
            #[cfg(all(target_feature = "f16c", target_arch = "x86_64"))]
            {
                use std::arch::x86_64::*;
                let f16_low = _mm256_cvtps_ph(val[0].0, _MM_FROUND_TO_NEAREST_INT);
                let f16_high = _mm256_cvtps_ph(val[1].0, _MM_FROUND_TO_NEAREST_INT);
                let tmp_256 = _mm256_castsi128_si256(f16_low);
                let result = _mm256_insertf128_si256(tmp_256, f16_high, 1);

                std::mem::transmute(result)
            }
            #[cfg(not(all(target_feature = "f16c", target_arch = "x86_64")))]
            {
                let arr: [[f32; 8]; 2] = std::mem::transmute(val);
                let mut result = [0u16; 16];
                for i in 0..8 {
                    result[i] = half::f16::from_f32(arr[0][i]).to_bits();
                    result[i + 8] = half::f16::from_f32(arr[1][i]).to_bits();
                }
                std::mem::transmute(result)
            }
        }
    }
}
impl SimdCompare for f16x16 {
    type SimdMask = i16x16;
    #[inline(always)]
    fn simd_eq(self, other: Self) -> i16x16 {
        let x: i16x16 = unsafe { std::mem::transmute(self.0) };
        let y: i16x16 = unsafe { std::mem::transmute(other.0) };
        x.simd_eq(y)
    }
    #[inline(always)]
    fn simd_ne(self, other: Self) -> i16x16 {
        let x: i16x16 = unsafe { std::mem::transmute(self.0) };
        let y: i16x16 = unsafe { std::mem::transmute(other.0) };
        x.simd_ne(y)
    }
    #[inline(always)]
    fn simd_lt(self, other: Self) -> i16x16 {
        let x: i16x16 = unsafe { std::mem::transmute(self.0) };
        let y: i16x16 = unsafe { std::mem::transmute(other.0) };
        x.simd_lt(y)
    }
    #[inline(always)]
    fn simd_le(self, other: Self) -> i16x16 {
        let x: i16x16 = unsafe { std::mem::transmute(self.0) };
        let y: i16x16 = unsafe { std::mem::transmute(other.0) };
        x.simd_le(y)
    }
    #[inline(always)]
    fn simd_gt(self, other: Self) -> i16x16 {
        let x: i16x16 = unsafe { std::mem::transmute(self.0) };
        let y: i16x16 = unsafe { std::mem::transmute(other.0) };
        x.simd_gt(y)
    }
    #[inline(always)]
    fn simd_ge(self, other: Self) -> i16x16 {
        let x: i16x16 = unsafe { std::mem::transmute(self.0) };
        let y: i16x16 = unsafe { std::mem::transmute(other.0) };
        x.simd_ge(y)
    }
}

impl SimdSelect<f16x16> for i16x16 {
    #[inline(always)]
    fn select(&self, true_val: f16x16, false_val: f16x16) -> f16x16 {
        let mut ret = f16x16::default();
        let arr = self.as_array();
        for i in 0..16 {
            ret.0[i] = if arr[i] != 0 {
                true_val.0[i]
            } else {
                false_val.0[i]
            };
        }
        ret
    }
}

impl std::ops::Add for f16x16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let [x0, x1] = self.to_2_f32vec();
        let [y0, y1] = rhs.to_2_f32vec();
        let low_add = x0 + y0;
        let high_add = x1 + y1;
        f16x16::from_2_f32vec([low_add, high_add])
    }
}

impl std::ops::Sub for f16x16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        let [x0, x1] = self.to_2_f32vec();
        let [y0, y1] = rhs.to_2_f32vec();
        let low_sub = x0 - y0;
        let high_sub = x1 - y1;
        f16x16::from_2_f32vec([low_sub, high_sub])
    }
}

impl std::ops::Mul for f16x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        let [x0, x1] = self.to_2_f32vec();
        let [y0, y1] = rhs.to_2_f32vec();
        let low_mul = x0 * y0;
        let high_mul = x1 * y1;
        f16x16::from_2_f32vec([low_mul, high_mul])
    }
}

impl std::ops::Div for f16x16 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        let mut ret = f16x16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] / rhs.0[i];
        }
        ret
    }
}
impl std::ops::Rem for f16x16 {
    type Output = Self;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        let mut ret = f16x16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] % rhs.0[i];
        }
        ret
    }
}
impl std::ops::Neg for f16x16 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        let mut ret = f16x16::default();
        for i in 0..16 {
            ret.0[i] = -self.0[i];
        }
        ret
    }
}

#[inline(always)]
pub(crate) fn f32x8_to_f16x8(val: f32x8) -> [u16; 8] {
    unsafe {
        #[cfg(all(target_feature = "f16c", target_arch = "x86_64"))]
        {
            use std::arch::x86_64::*;
            let f16_bits = _mm256_cvtps_ph(val.0, _MM_FROUND_TO_NEAREST_INT);
            std::mem::transmute(f16_bits)
        }
        #[cfg(not(all(target_feature = "f16c", target_arch = "x86_64")))]
        {
            let arr: [f32; 8] = std::mem::transmute(val);
            let mut result = [0u16; 8];
            for i in 0..8 {
                result[i] = half::f16::from_f32(arr[i]).to_bits();
            }
            result
        }
    }
}

impl VecConvertor for f16x16 {
    #[inline(always)]
    fn to_i16(self) -> super::i16x16::i16x16 {
        #[cfg(all(target_feature = "avx2", target_feature = "f16c"))]
        {
            use std::arch::x86_64::*;
            unsafe {
                let [x0, x1]: [f32x8; 2] = std::mem::transmute(self.to_2_f32vec());
                let i0 = _mm256_cvtps_epi32(x0.0);
                let i1 = _mm256_cvtps_epi32(x1.0);
                let packed = _mm256_packs_epi32(i0, i1);
                return super::i16x16::i16x16(packed);
            }
        }
        #[cfg(all(target_feature = "neon", target_arch = "aarch64"))]
        {
            unimplemented!()
        }
        #[cfg(not(any(
            all(target_feature = "avx2", target_feature = "f16c"),
            all(target_feature = "neon", target_arch = "aarch64")
        )))]
        {
            let arr: [half::f16; 16] = unsafe { std::mem::transmute(self) };
            let mut result = [0i16; 16];
            for i in 0..16 {
                result[i] = arr[i].to_f32() as i16;
            }
            return unsafe { std::mem::transmute(result) };
        }
    }
    #[inline(always)]
    fn to_u16(self) -> super::u16x16::u16x16 {
        #[cfg(all(target_feature = "avx2", target_feature = "f16c"))]
        {
            use std::arch::x86_64::*;
            unsafe {
                let [x0, x1]: [f32x8; 2] = std::mem::transmute(self.to_2_f32vec());
                let i0 = _mm256_cvtps_epi32(x0.0);
                let i1 = _mm256_cvtps_epi32(x1.0);
                let packed = _mm256_packus_epi32(i0, i1);
                super::u16x16::u16x16(packed)
            }
        }
        #[cfg(all(target_feature = "neon", target_arch = "aarch64"))]
        {
            unimplemented!()
        }
        #[cfg(not(any(
            all(target_feature = "avx2", target_feature = "f16c"),
            all(target_feature = "neon", target_arch = "aarch64")
        )))]
        {
            let arr: [half::f16; 16] = unsafe { std::mem::transmute(self) };
            let mut result = [0u16; 16];
            for i in 0..16 {
                result[i] = arr[i].to_f32() as u16;
            }
            unsafe { std::mem::transmute(result) }
        }
    }
    #[inline(always)]
    fn to_f16(self) -> f16x16 {
        self
    }
}

impl SimdMath<half::f16> for f16x16 {
    #[inline(always)]
    fn sin(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_sin = high.sin();
        let low_sin = low.sin();
        f16x16::from_2_f32vec([high_sin, low_sin])
    }
    #[inline(always)]
    fn cos(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_cos = high.cos();
        let low_cos = low.cos();
        f16x16::from_2_f32vec([high_cos, low_cos])
    }
    #[inline(always)]
    fn tan(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_tan = high.tan();
        let low_tan = low.tan();
        f16x16::from_2_f32vec([high_tan, low_tan])
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_sqrt = high.sqrt();
        let low_sqrt = low.sqrt();
        f16x16::from_2_f32vec([high_sqrt, low_sqrt])
    }
    #[inline(always)]
    fn abs(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_abs = high.abs();
        let low_abs = low.abs();
        f16x16::from_2_f32vec([high_abs, low_abs])
    }
    #[inline(always)]
    fn floor(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_floor = high.floor();
        let low_floor = low.floor();
        f16x16::from_2_f32vec([high_floor, low_floor])
    }
    #[inline(always)]
    fn ceil(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_ceil = high.ceil();
        let low_ceil = low.ceil();
        f16x16::from_2_f32vec([high_ceil, low_ceil])
    }
    #[inline(always)]
    fn neg(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_neg = high.neg();
        let low_neg = low.neg();
        f16x16::from_2_f32vec([high_neg, low_neg])
    }
    #[inline(always)]
    fn round(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_round = high.round();
        let low_round = low.round();
        f16x16::from_2_f32vec([high_round, low_round])
    }
    #[inline(always)]
    fn signum(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_signum = high.signum();
        let low_signum = low.signum();
        f16x16::from_2_f32vec([high_signum, low_signum])
    }
    #[inline(always)]
    fn leaky_relu(self, alpha: Self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let [high_alpha, low_alpha] = alpha.to_2_f32vec();
        let high_leaky_relu = high.leaky_relu(high_alpha);
        let low_leaky_relu = low.leaky_relu(low_alpha);
        f16x16::from_2_f32vec([high_leaky_relu, low_leaky_relu])
    }
    #[inline(always)]
    fn relu(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_relu = high.relu();
        let low_relu = low.relu();
        f16x16::from_2_f32vec([high_relu, low_relu])
    }
    #[inline(always)]
    fn relu6(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_relu6 = high.relu6();
        let low_relu6 = low.relu6();
        f16x16::from_2_f32vec([high_relu6, low_relu6])
    }
    #[inline(always)]
    fn pow(self, exp: Self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let [high_exp, low_exp] = exp.to_2_f32vec();
        let high_pow = high.pow(high_exp);
        let low_pow = low.pow(low_exp);
        f16x16::from_2_f32vec([high_pow, low_pow])
    }
    #[inline(always)]
    fn asin(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_asin = high.asin();
        let low_asin = low.asin();
        f16x16::from_2_f32vec([high_asin, low_asin])
    }
    #[inline(always)]
    fn acos(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_acos = high.acos();
        let low_acos = low.acos();
        f16x16::from_2_f32vec([high_acos, low_acos])
    }
    #[inline(always)]
    fn atan(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_atan = high.atan();
        let low_atan = low.atan();
        f16x16::from_2_f32vec([high_atan, low_atan])
    }
    #[inline(always)]
    fn sinh(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_sinh = high.sinh();
        let low_sinh = low.sinh();
        f16x16::from_2_f32vec([high_sinh, low_sinh])
    }
    #[inline(always)]
    fn cosh(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_cosh = high.cosh();
        let low_cosh = low.cosh();
        f16x16::from_2_f32vec([high_cosh, low_cosh])
    }
    #[inline(always)]
    fn tanh(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_tanh = high.tanh();
        let low_tanh = low.tanh();
        f16x16::from_2_f32vec([high_tanh, low_tanh])
    }
    #[inline(always)]
    fn asinh(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_asinh = high.asinh();
        let low_asinh = low.asinh();
        f16x16::from_2_f32vec([high_asinh, low_asinh])
    }
    #[inline(always)]
    fn acosh(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_acosh = high.acosh();
        let low_acosh = low.acosh();
        f16x16::from_2_f32vec([high_acosh, low_acosh])
    }
    #[inline(always)]
    fn atanh(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_atanh = high.atanh();
        let low_atanh = low.atanh();
        f16x16::from_2_f32vec([high_atanh, low_atanh])
    }
    #[inline(always)]
    fn exp2(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_exp2 = high.exp2();
        let low_exp2 = low.exp2();
        f16x16::from_2_f32vec([high_exp2, low_exp2])
    }
    #[inline(always)]
    fn exp10(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_exp10 = high.exp10();
        let low_exp10 = low.exp10();
        f16x16::from_2_f32vec([high_exp10, low_exp10])
    }
    #[inline(always)]
    fn expm1(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_expm1 = high.expm1();
        let low_expm1 = low.expm1();
        f16x16::from_2_f32vec([high_expm1, low_expm1])
    }
    #[inline(always)]
    fn log10(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_log10 = high.log10();
        let low_log10 = low.log10();
        f16x16::from_2_f32vec([high_log10, low_log10])
    }
    #[inline(always)]
    fn log2(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_log2 = high.log2();
        let low_log2 = low.log2();
        f16x16::from_2_f32vec([high_log2, low_log2])
    }
    #[inline(always)]
    fn log1p(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_log1p = high.log1p();
        let low_log1p = low.log1p();
        f16x16::from_2_f32vec([high_log1p, low_log1p])
    }
    #[inline(always)]
    fn hypot(self, other: Self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let [high_other, low_other] = other.to_2_f32vec();
        let high_hypot = high.hypot(high_other);
        let low_hypot = low.hypot(low_other);
        f16x16::from_2_f32vec([high_hypot, low_hypot])
    }
    #[inline(always)]
    fn trunc(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_trunc = high.trunc();
        let low_trunc = low.trunc();
        f16x16::from_2_f32vec([high_trunc, low_trunc])
    }
    #[inline(always)]
    fn erf(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_erf = high.erf();
        let low_erf = low.erf();
        f16x16::from_2_f32vec([high_erf, low_erf])
    }
    #[inline(always)]
    fn cbrt(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_cbrt = high.cbrt();
        let low_cbrt = low.cbrt();
        f16x16::from_2_f32vec([high_cbrt, low_cbrt])
    }
    #[inline(always)]
    fn exp(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_exp = high.exp();
        let low_exp = low.exp();
        f16x16::from_2_f32vec([high_exp, low_exp])
    }
    #[inline(always)]
    fn ln(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_ln = high.ln();
        let low_ln = low.ln();
        f16x16::from_2_f32vec([high_ln, low_ln])
    }
    #[inline(always)]
    fn sincos(self) -> (Self, Self) {
        let [high, low] = self.to_2_f32vec();
        let (high_sin, high_cos) = high.sincos();
        let (low_sin, low_cos) = low.sincos();
        (
            f16x16::from_2_f32vec([high_sin, low_sin]),
            f16x16::from_2_f32vec([high_cos, low_cos]),
        )
    }
    #[inline(always)]
    fn atan2(self, other: Self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let [high_other, low_other] = other.to_2_f32vec();
        let high_atan2 = high.atan2(high_other);
        let low_atan2 = low.atan2(low_other);
        f16x16::from_2_f32vec([high_atan2, low_atan2])
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let [high_other, low_other] = other.to_2_f32vec();
        let high_min = high.min(high_other);
        let low_min = low.min(low_other);
        f16x16::from_2_f32vec([high_min, low_min])
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let [high_other, low_other] = other.to_2_f32vec();
        let high_max = high.max(high_other);
        let low_max = low.max(low_other);
        f16x16::from_2_f32vec([high_max, low_max])
    }

    #[inline(always)]
    fn hard_sigmoid(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_hard_sigmoid = high.hard_sigmoid();
        let low_hard_sigmoid = low.hard_sigmoid();
        f16x16::from_2_f32vec([high_hard_sigmoid, low_hard_sigmoid])
    }

    #[inline(always)]
    fn elu(self, alpha: Self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let [high_alpha, low_alpha] = alpha.to_2_f32vec();
        let high_elu = high.elu(high_alpha);
        let low_elu = low.elu(low_alpha);
        f16x16::from_2_f32vec([high_elu, low_elu])
    }

    #[inline(always)]
    fn selu(self, alpha: Self, scale: Self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let [high_alpha, low_alpha] = alpha.to_2_f32vec();
        let [high_scale, low_scale] = scale.to_2_f32vec();
        let high_selu = high.selu(high_alpha, high_scale);
        let low_selu = low.selu(low_alpha, low_scale);
        f16x16::from_2_f32vec([high_selu, low_selu])
    }

    #[inline(always)]
    fn celu(self, alpha: Self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let [high_alpha, low_alpha] = alpha.to_2_f32vec();
        let high_celu = high.celu(high_alpha);
        let low_celu = low.celu(low_alpha);
        f16x16::from_2_f32vec([high_celu, low_celu])
    }

    #[inline(always)]
    fn gelu(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_gelu = high.gelu();
        let low_gelu = low.gelu();
        f16x16::from_2_f32vec([high_gelu, low_gelu])
    }

    #[inline(always)]
    fn hard_swish(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_hard_swish = high.hard_swish();
        let low_hard_swish = low.hard_swish();
        f16x16::from_2_f32vec([high_hard_swish, low_hard_swish])
    }

    #[inline(always)]
    fn mish(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_mish = high.mish();
        let low_mish = low.mish();
        f16x16::from_2_f32vec([high_mish, low_mish])
    }

    #[inline(always)]
    fn softplus(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_softplus = high.softplus();
        let low_softplus = low.softplus();
        f16x16::from_2_f32vec([high_softplus, low_softplus])
    }

    #[inline(always)]
    fn recip(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_recip = high.recip();
        let low_recip = low.recip();
        f16x16::from_2_f32vec([high_recip, low_recip])
    }
    #[inline(always)]
    fn sigmoid(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_sigmoid = high.sigmoid();
        let low_sigmoid = low.sigmoid();
        f16x16::from_2_f32vec([high_sigmoid, low_sigmoid])
    }
    #[inline(always)]
    fn softsign(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_softsign = high.softsign();
        let low_softsign = low.softsign();
        f16x16::from_2_f32vec([high_softsign, low_softsign])
    }
    #[inline(always)]
    fn copysign(self, rhs: Self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let [high_rhs, low_rhs] = rhs.to_2_f32vec();
        let high_copysign = high.copysign(high_rhs);
        let low_copysign = low.copysign(low_rhs);
        f16x16::from_2_f32vec([high_copysign, low_copysign])
    }
}

impl FloatOutBinary2 for f16x16 {
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
        f16x16::from_2_f32vec([high_log, low_log])
    }

    #[inline(always)]
    fn __hypot(self, rhs: Self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let [high_rhs, low_rhs] = rhs.to_2_f32vec();
        let high_hypot = high.__hypot(high_rhs);
        let low_hypot = low.__hypot(low_rhs);
        f16x16::from_2_f32vec([high_hypot, low_hypot])
    }

    #[inline(always)]
    fn __pow(self, rhs: Self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let [high_base, low_base] = rhs.to_2_f32vec();
        let high_pow = high.__pow(high_base);
        let low_pow = low.__pow(low_base);
        f16x16::from_2_f32vec([high_pow, low_pow])
    }
}

impl NormalOut2 for f16x16 {
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
    fn __rem(self, rhs: Self) -> Self {
        self % rhs
    }

    #[inline(always)]
    fn __max(self, rhs: Self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let [high_base, low_base] = rhs.to_2_f32vec();
        let high_max = high.__max(high_base);
        let low_max = low.__max(low_base);
        f16x16::from_2_f32vec([high_max, low_max])
    }

    #[inline(always)]
    fn __min(self, rhs: Self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let [high_base, low_base] = rhs.to_2_f32vec();
        let high_min = high.__min(high_base);
        let low_min = low.__min(low_base);
        f16x16::from_2_f32vec([high_min, low_min])
    }

    #[inline(always)]
    fn __clamp(self, min: Self, max: Self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let [high_min, low_min] = min.to_2_f32vec();
        let [high_max, low_max] = max.to_2_f32vec();
        let high_clamp = high.__clamp(high_min, high_max);
        let low_clamp = low.__clamp(low_min, low_max);
        f16x16::from_2_f32vec([high_clamp, low_clamp])
    }
}

impl NormalOutUnary2 for f16x16 {
    #[inline(always)]
    fn __square(self) -> Self {
        self * self
    }

    #[inline(always)]
    fn __abs(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_abs = high.__abs();
        let low_abs = low.__abs();
        f16x16::from_2_f32vec([high_abs, low_abs])
    }

    #[inline(always)]
    fn __ceil(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_ceil = high.__ceil();
        let low_ceil = low.__ceil();
        f16x16::from_2_f32vec([high_ceil, low_ceil])
    }

    #[inline(always)]
    fn __floor(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_floor = high.__floor();
        let low_floor = low.__floor();
        f16x16::from_2_f32vec([high_floor, low_floor])
    }

    #[inline(always)]
    fn __neg(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_neg = high.__neg();
        let low_neg = low.__neg();
        f16x16::from_2_f32vec([high_neg, low_neg])
    }

    #[inline(always)]
    fn __round(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_round = high.__round();
        let low_round = low.__round();
        f16x16::from_2_f32vec([high_round, low_round])
    }

    #[inline(always)]
    fn __signum(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_signum = high.__signum();
        let low_signum = low.__signum();
        f16x16::from_2_f32vec([high_signum, low_signum])
    }

    #[inline(always)]
    fn __leaky_relu(self, alpha: Self) -> Self {
        self.max(f16x16::splat(half::f16::from_f32_const(0.0)))
            + alpha * self.min(f16x16::splat(half::f16::from_f32_const(0.0)))
    }

    #[inline(always)]
    fn __relu(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_relu = high.__relu();
        let low_relu = low.__relu();
        f16x16::from_2_f32vec([high_relu, low_relu])
    }

    #[inline(always)]
    fn __relu6(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_relu6 = high.__relu6();
        let low_relu6 = low.__relu6();
        f16x16::from_2_f32vec([high_relu6, low_relu6])
    }

    #[inline(always)]
    fn __trunc(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_trunc = high.__trunc();
        let low_trunc = low.__trunc();
        f16x16::from_2_f32vec([high_trunc, low_trunc])
    }

    #[inline(always)]
    fn __copysign(self, rhs: Self) -> Self {
        self.copysign(rhs)
    }
}

impl Eval2 for f16x16 {
    type Output = i16x16;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        let x = u16x16::splat(0x7c00u16);
        let y = u16x16::splat(0x03ffu16);
        let i: u16x16 = unsafe { std::mem::transmute(self.0) };

        let and = i & x;
        let eq = and.simd_eq(x);

        let and2 = i & y;
        let neq_zero = and2.simd_ne(u16x16::splat(0));

        let result = eq & neq_zero;

        unsafe { std::mem::transmute(result) }
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        self.simd_ne(f16x16::default())
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        let sign_mask = u16x16::splat(0x8000u16);
        let inf_mask = u16x16::splat(0x7c00u16);
        let frac_mask = u16x16::splat(0x03ffu16);

        let i: u16x16 = unsafe { std::mem::transmute(self.0) };

        let exp = i & inf_mask;
        let frac = i & frac_mask;
        let is_inf = exp.simd_eq(inf_mask) & frac.simd_eq(u16x16::splat(0));

        let is_neg = (i & sign_mask).simd_ne(u16x16::splat(0));

        let result = is_inf.select(
            is_neg.select(i16x16::splat(-1), i16x16::splat(1)),
            i16x16::splat(0),
        );

        result
    }
}
