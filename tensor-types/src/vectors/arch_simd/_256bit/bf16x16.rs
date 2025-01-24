use half::bf16;

use crate::arch_simd::_256bit::u16x16::u16x16;
use crate::convertion::VecConvertor;
use crate::simd::_256bit::u32x8::u32x8;
use crate::traits::{SimdCompare, SimdMath, SimdSelect};
use crate::type_promote::{Eval2, FloatOutBinary2, NormalOut2, NormalOutUnary2};
use crate::{traits::VecTrait, vectors::arch_simd::_256bit::f32x8::f32x8};

use super::i16x16::i16x16;
use super::i32x8::i32x8;

use std::arch::x86_64::*;

/// a vector of 16 bf16 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(32))]
pub struct bf16x16(pub(crate) [half::bf16; 16]);

/// helper to impl the promote trait
#[allow(non_camel_case_types)]
pub(crate) type bf16_promote = bf16x16;

impl VecTrait<half::bf16> for bf16x16 {
    const SIZE: usize = 16;
    type Base = half::bf16;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[half::bf16]) {
        self.0.copy_from_slice(slice);
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        let [x0, x1]: [f32x8; 2] = unsafe { std::mem::transmute(self.to_2_f32vec()) };
        let [a0, a1]: [f32x8; 2] = unsafe { std::mem::transmute(a.to_2_f32vec()) };
        let [b0, b1]: [f32x8; 2] = unsafe { std::mem::transmute(b.to_2_f32vec()) };
        let res0 = x0.mul_add(a0, b0);
        let res1 = x1.mul_add(a1, b1);
        bf16x16::from_2_f32vec([res0, res1])
    }
    #[inline(always)]
    fn sum(&self) -> half::bf16 {
        self.0.iter().sum()
    }
    #[inline(always)]
    fn splat(val: half::bf16) -> bf16x16 {
        bf16x16([val; 16])
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const half::bf16) -> Self {
        bf16x16([
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

impl bf16x16 {
    /// convert the vector to an array
    #[inline(always)]
    pub fn as_array(&self) -> [half::bf16; 16] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl bf16x16 {
    /// convert to 2 f32x8
    #[inline(always)]
    pub fn to_2_f32vec(&self) -> [f32x8; 2] {
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
            let res_low = mask_low.select(true_low, false_low);
            let res_high = mask_high.select(true_high, false_high);
            [
                f32x8(std::mem::transmute(res_low.0)),
                f32x8(std::mem::transmute(res_high.0)),
            ]
        }
    }

/// convert from 2 f32x4
#[inline(always)]
pub fn from_2_f32vec(val: [f32x8; 2]) -> Self {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        unsafe fn conv(vec: f32x8) -> __m256i {
            let x = u32x8(std::mem::transmute(vec.0));
            let nan_mask =
                (x & u32x8::splat(0x7FFF_FFFFu32)).simd_gt(u32x8::splat(0x7F80_0000u32));
            let shifted = x >> u32x8::splat(16);

            // NaN 处理
            let nan_result = shifted | u32x8::splat(0x0040u32);

            // 舍入检查
            let round_bit = u32x8::splat(0x00008000u32);
            let rs_mask = (x & round_bit).simd_ne(u32x8::splat(0))
                & (x & (u32x8::splat(3) * round_bit - u32x8::splat(1)))
                    .simd_ne(u32x8::splat(0));

            // 舍入处理
            let round_result = shifted + rs_mask.select(u32x8::splat(1), u32x8::splat(0));

            // 最终选择
            let final_result = nan_mask.select(nan_result, round_result);
            _mm256_packus_epi32(final_result.0, _mm256_setzero_si256()) // 打包为 16 位
        }
        let high = conv(val[0]);
        let low = conv(val[1]);
        let result = _mm256_unpacklo_epi64(high, low);
        std::mem::transmute(result)
    }
}

    /// check if the value is NaN and return a mask
    #[inline(always)]
    pub fn is_nan(&self) -> i16x16 {
        let res: [i16; 16] = self.0.map(|x| if x.is_nan() { 1 } else { 0 });
        unsafe { std::mem::transmute(res) }
    }
}
impl SimdCompare for bf16x16 {
    type SimdMask = i16x16;
    #[inline(always)]
    fn simd_eq(self, other: Self) -> i16x16 {
        unsafe {
            let self_ptr = &self.0 as *const _ as *const __m256i;
            let other_ptr = &other.0 as *const _ as *const __m256i;
            let a = _mm256_loadu_si256(self_ptr);
            let b = _mm256_loadu_si256(other_ptr);
            i16x16(_mm256_cmpeq_epi16(a, b))
        }
    }
    #[inline(always)]
    fn simd_ne(self, other: Self) -> i16x16 {
        unsafe {
            let self_ptr = &self.0 as *const _ as *const __m256i;
            let other_ptr = &other.0 as *const _ as *const __m256i;
            let a = _mm256_loadu_si256(self_ptr);
            let b = _mm256_loadu_si256(other_ptr);
            let eq = _mm256_cmpeq_epi16(a, b);
            i16x16(_mm256_xor_si256(eq, _mm256_set1_epi16(-1)))
        }
    }
    #[inline(always)]
    fn simd_lt(self, other: Self) -> i16x16 {
        unsafe {
            let self_ptr = &self.0 as *const _ as *const __m256i;
            let other_ptr = &other.0 as *const _ as *const __m256i;
            let a = _mm256_loadu_si256(self_ptr);
            let b = _mm256_loadu_si256(other_ptr);
            i16x16(_mm256_cmpgt_epi16(b, a))
        }
    }
    #[inline(always)]
    fn simd_le(self, other: Self) -> i16x16 {
        unsafe {
            let self_ptr = &self.0 as *const _ as *const __m256i;
            let other_ptr = &other.0 as *const _ as *const __m256i;
            let a = _mm256_loadu_si256(self_ptr);
            let b = _mm256_loadu_si256(other_ptr);
            let lt = _mm256_cmpgt_epi16(b, a); // 交换 a 和 b
            let eq = _mm256_cmpeq_epi16(a, b);
            i16x16(_mm256_or_si256(lt, eq))
        }
    }
    #[inline(always)]
    fn simd_gt(self, other: Self) -> i16x16 {
        unsafe {
            let self_ptr = &self.0 as *const _ as *const __m256i;
            let other_ptr = &other.0 as *const _ as *const __m256i;
            let a = _mm256_loadu_si256(self_ptr);
            let b = _mm256_loadu_si256(other_ptr);
            i16x16(_mm256_cmpgt_epi16(a, b))
        }
    }
    #[inline(always)]
    fn simd_ge(self, other: Self) -> i16x16 {
        unsafe {
            let self_ptr = &self.0 as *const _ as *const __m256i;
            let other_ptr = &other.0 as *const _ as *const __m256i;
            let a = _mm256_loadu_si256(self_ptr);
            let b = _mm256_loadu_si256(other_ptr);
            let gt = _mm256_cmpgt_epi16(a, b);
            let eq = _mm256_cmpeq_epi16(a, b);
            i16x16(_mm256_or_si256(gt, eq))
        }
    }
}

impl SimdSelect<bf16x16> for i16x16 {
    #[inline(always)]
    fn select(&self, true_val: bf16x16, false_val: bf16x16) -> bf16x16 {
        let mut ret = bf16x16::default();
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

impl std::ops::Add for bf16x16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let [x0, x1] = self.to_2_f32vec();
        let [y0, y1] = rhs.to_2_f32vec();
        let low_add = x0 + y0;
        let high_add = x1 + y1;
        let res = bf16x16::from_2_f32vec([low_add, high_add]);
        res
    }
}
impl std::ops::Sub for bf16x16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        let [x0, x1] = self.to_2_f32vec();
        let [y0, y1] = rhs.to_2_f32vec();
        let low_sub = x0 - y0;
        let high_sub = x1 - y1;
        bf16x16::from_2_f32vec([low_sub, high_sub])
    }
}
impl std::ops::Mul for bf16x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        let [x0, x1] = self.to_2_f32vec();
        let [y0, y1] = rhs.to_2_f32vec();
        let low_mul = x0 * y0;
        let high_mul = x1 * y1;
        bf16x16::from_2_f32vec([low_mul, high_mul])
    }
}
impl std::ops::Div for bf16x16 {
    type Output = Self;
    #[inline(always)]
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
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        let mut ret = bf16x16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] % rhs.0[i];
        }
        ret
    }
}
impl std::ops::Neg for bf16x16 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        let mut ret = bf16x16::default();
        for i in 0..16 {
            ret.0[i] = -self.0[i];
        }
        ret
    }
}

impl VecConvertor for bf16x16 {
    #[inline(always)]
    fn to_bf16(self) -> bf16x16 {
        self
    }
    #[inline(always)]
    fn to_f16(self) -> super::f16x16::f16x16 {
        let [x0, x1] = self.to_2_f32vec();
        super::f16x16::f16x16::from_2_f32vec([x0, x1])
    }
    #[inline(always)]
    fn to_i16(self) -> super::i16x16::i16x16 {
        unsafe {
            let [x0, x1]: [f32x8; 2] = std::mem::transmute(self.to_2_f32vec());
            let i0 = _mm256_cvtps_epi32(x0.0);
            let i1 = _mm256_cvtps_epi32(x1.0);
            let packed = _mm256_packs_epi32(i0, i1);
            super::i16x16::i16x16(packed)
        }
    }
    #[inline(always)]
    fn to_u16(self) -> super::u16x16::u16x16 {
        unsafe {
            let [x0, x1]: [f32x8; 2] = std::mem::transmute(self.to_2_f32vec());
            let i0 = _mm256_cvtps_epi32(x0.0);
            let i1 = _mm256_cvtps_epi32(x1.0);
            let packed = _mm256_packus_epi32(i0, i1);
            super::u16x16::u16x16(packed)
        }
    }
}

impl SimdMath<bf16> for bf16x16 {
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
    fn square(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_square = high.square();
        let low_square = low.square();
        Self::from_2_f32vec([high_square, low_square])
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
        let high_signum = high.signum();
        let low_signum = low.signum();
        Self::from_2_f32vec([high_signum, low_signum])
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
    fn log(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_log = high.log();
        let low_log = low.log();
        Self::from_2_f32vec([high_log, low_log])
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
    fn celu(self, alpha: Self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let [high_alpha, low_alpha] = alpha.to_2_f32vec();
        let high_celu = high.celu(high_alpha);
        let low_celu = low.celu(low_alpha);
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

impl FloatOutBinary2 for bf16x16 {
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
        bf16x16::from_2_f32vec([high_log, low_log])
    }
}

impl NormalOut2 for bf16x16 {
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
        let [high, low] = self.to_2_f32vec();
        let [high_rhs, low_rhs] = rhs.to_2_f32vec();
        let high_pow = high.__pow(high_rhs);
        let low_pow = low.__pow(low_rhs);
        bf16x16::from_2_f32vec([high_pow, low_pow])
    }

    #[inline(always)]
    fn __rem(self, rhs: Self) -> Self {
        self % rhs
    }

    #[inline(always)]
    fn __max(self, rhs: Self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let [high_rhs, low_rhs] = rhs.to_2_f32vec();
        let high_max = high.__max(high_rhs);
        let low_max = low.__max(low_rhs);
        bf16x16::from_2_f32vec([high_max, low_max])
    }

    #[inline(always)]
    fn __min(self, rhs: Self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let [high_rhs, low_rhs] = rhs.to_2_f32vec();
        let high_min = high.__min(high_rhs);
        let low_min = low.__min(low_rhs);
        bf16x16::from_2_f32vec([high_min, low_min])
    }

    #[inline(always)]
    fn __clamp(self, min: Self, max: Self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let [high_min, low_min] = min.to_2_f32vec();
        let [high_max, low_max] = max.to_2_f32vec();
        let high_clamp = high.__clamp(high_min, high_max);
        let low_clamp = low.__clamp(low_min, low_max);
        bf16x16::from_2_f32vec([high_clamp, low_clamp])
    }
}

impl NormalOutUnary2 for bf16x16 {
    #[inline(always)]
    fn __square(self) -> Self {
        self * self
    }

    #[inline(always)]
    fn __abs(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_abs = high.__abs();
        let low_abs = low.__abs();
        bf16x16::from_2_f32vec([high_abs, low_abs])
    }

    #[inline(always)]
    fn __ceil(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_ceil = high.__ceil();
        let low_ceil = low.__ceil();
        bf16x16::from_2_f32vec([high_ceil, low_ceil])
    }

    #[inline(always)]
    fn __floor(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_floor = high.__floor();
        let low_floor = low.__floor();
        bf16x16::from_2_f32vec([high_floor, low_floor])
    }

    #[inline(always)]
    fn __neg(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_neg = high.__neg();
        let low_neg = low.__neg();
        bf16x16::from_2_f32vec([high_neg, low_neg])
    }

    #[inline(always)]
    fn __round(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_round = high.__round();
        let low_round = low.__round();
        bf16x16::from_2_f32vec([high_round, low_round])
    }

    #[inline(always)]
    fn __signum(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_signum = high.__signum();
        let low_signum = low.__signum();
        bf16x16::from_2_f32vec([high_signum, low_signum])
    }

    #[inline(always)]
    fn __leaky_relu(self, alpha: Self) -> Self {
        self.max(bf16x16::splat(half::bf16::from_f32_const(0.0)))
            + alpha * self.min(bf16x16::splat(half::bf16::from_f32_const(0.0)))
    }

    #[inline(always)]
    fn __relu(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_relu = high.__relu();
        let low_relu = low.__relu();
        bf16x16::from_2_f32vec([high_relu, low_relu])
    }

    #[inline(always)]
    fn __relu6(self) -> Self {
        let [high, low] = self.to_2_f32vec();
        let high_relu6 = high.__relu6();
        let low_relu6 = low.__relu6();
        bf16x16::from_2_f32vec([high_relu6, low_relu6])
    }
}

impl Eval2 for bf16x16 {
    type Output = i16x16;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        let res: [i16; 16] = self.0.map(|x| if x.is_nan() { -1 } else { 0 });
        unsafe { std::mem::transmute(res) }
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        self.simd_ne(bf16x16::default())
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        let sign_mask = u16x16::splat(0x8000u16);
        let inf_mask = u16x16::splat(0x7f80u16);
        let frac_mask = u16x16::splat(0x007fu16);

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
