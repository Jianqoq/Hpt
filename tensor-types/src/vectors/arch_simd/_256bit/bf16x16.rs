use half::bf16;

use crate::arch_simd::_256bit::u16x16::u16x16;
use crate::convertion::VecConvertor;
use crate::traits::{SimdCompare, SimdMath, SimdSelect};
use crate::type_promote::{Eval2, FloatOutBinary2, NormalOut2, NormalOutUnary2};
use crate::{traits::VecTrait, vectors::arch_simd::_256bit::f32x8::f32x8};

use super::i16x16::i16x16;

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
        let [x0, x1]: [f32x8; 2] = unsafe { std::mem::transmute(self.to_2_f32x8()) };
        let [a0, a1]: [f32x8; 2] = unsafe { std::mem::transmute(a.to_2_f32x8()) };
        let [b0, b1]: [f32x8; 2] = unsafe { std::mem::transmute(b.to_2_f32x8()) };
        let res0 = x0.mul_add(a0, b0);
        let res1 = x1.mul_add(a1, b1);
        bf16x16::from_2_f32x8([res0, res1])
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
    pub fn to_2_f32x8(&self) -> [f32x8; 2] {
        unsafe {
            let bits: [u16; 16] = std::mem::transmute(self.0);

            // 转换前8个和后8个
            let mut high = [0.0f32; 8];
            let mut low = [0.0f32; 8];
            for i in 0..8 {
                high[i] = half::bf16::from_bits(bits[i]).to_f32();
                low[i] = half::bf16::from_bits(bits[i + 8]).to_f32();
            }

            [
                f32x8(std::mem::transmute(high)),
                f32x8(std::mem::transmute(low)),
            ]
        }
    }

    /// convert from 2 f32x8
    #[inline(always)]
    pub fn from_2_f32x8(val: [f32x8; 2]) -> Self {
        unsafe {
            // 转换为 f32 数组
            let high: [f32; 8] = std::mem::transmute(val[0]);
            let low: [f32; 8] = std::mem::transmute(val[1]);

            // 创建结果数组
            let mut result = [half::bf16::ZERO; 16];

            // 转换每个值
            for i in 0..8 {
                result[i] = half::bf16::from_f32(high[i]);
                result[i + 8] = half::bf16::from_f32(low[i]);
            }

            bf16x16(result)
        }
    }

    /// check if the value is NaN and return a mask
    #[inline(always)]
    pub fn is_nan(&self) -> i16x16 {
        let res: [i16; 16] = self.0.map(|x| if x.is_nan() { 1 } else { 0 });
        unsafe { std::mem::transmute(res) }
    }

    /// check if the value is infinite and return a mask
    #[inline(always)]
    pub fn is_infinite(&self) -> i16x16 {
        let x = u16x16::splat(0x7f80u16);
        let y = u16x16::splat(0x007fu16);
        let i: u16x16 = unsafe { std::mem::transmute(self.0) };

        let and = i & x;
        let eq = and.simd_eq(x);

        let and2 = i & y;
        let eq_zero = and2.simd_eq(u16x16::splat(0));

        let result = eq & eq_zero;

        unsafe { std::mem::transmute(result) }
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
        let mut ret = bf16x16::default();
        for i in 0..8 {
            ret.0[i] = self.0[i] + rhs.0[i];
        }
        ret
    }
}
impl std::ops::Sub for bf16x16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        let mut ret = bf16x16::default();
        for i in 0..8 {
            ret.0[i] = self.0[i] - rhs.0[i];
        }
        ret
    }
}
impl std::ops::Mul for bf16x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret = bf16x16::default();
        for i in 0..8 {
            ret.0[i] = self.0[i] * rhs.0[i];
        }
        ret
    }
}
impl std::ops::Div for bf16x16 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        let mut ret = bf16x16::default();
        for i in 0..8 {
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
        for i in 0..8 {
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
        for i in 0..8 {
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
        let [x0, x1] = self.to_2_f32x8();
        let high = super::f16x16::f32x8_to_f16x8(x0);
        let low = super::f16x16::f32x8_to_f16x8(x1);
        unsafe {
            std::mem::transmute([
                half::bf16::from_bits(high[0]),
                half::bf16::from_bits(high[1]),
                half::bf16::from_bits(high[2]),
                half::bf16::from_bits(high[3]),
                half::bf16::from_bits(high[4]),
                half::bf16::from_bits(high[5]),
                half::bf16::from_bits(high[6]),
                half::bf16::from_bits(high[7]),
                half::bf16::from_bits(low[0]),
                half::bf16::from_bits(low[1]),
                half::bf16::from_bits(low[2]),
                half::bf16::from_bits(low[3]),
                half::bf16::from_bits(low[4]),
                half::bf16::from_bits(low[5]),
                half::bf16::from_bits(low[6]),
                half::bf16::from_bits(low[7]),
            ])
        }
    }
    #[inline(always)]
    fn to_i16(self) -> super::i16x16::i16x16 {
        unsafe {
            let [x0, x1]: [f32x8; 2] = std::mem::transmute(self.to_2_f32x8());
            let i0 = _mm256_cvtps_epi32(x0.0);
            let i1 = _mm256_cvtps_epi32(x1.0);
            let packed = _mm256_packs_epi32(i0, i1);
            super::i16x16::i16x16(packed)
        }
    }
    #[inline(always)]
    fn to_u16(self) -> super::u16x16::u16x16 {
        unsafe {
            let [x0, x1]: [f32x8; 2] = std::mem::transmute(self.to_2_f32x8());
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
        let [high, low] = self.to_2_f32x8();
        let high_sin = high.sin();
        let low_sin = low.sin();
        Self::from_2_f32x8([high_sin, low_sin])
    }
    #[inline(always)]
    fn cos(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_cos = high.cos();
        let low_cos = low.cos();
        Self::from_2_f32x8([high_cos, low_cos])
    }
    #[inline(always)]
    fn tan(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_tan = high.tan();
        let low_tan = low.tan();
        Self::from_2_f32x8([high_tan, low_tan])
    }
    #[inline(always)]
    fn square(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_square = high.square();
        let low_square = low.square();
        Self::from_2_f32x8([high_square, low_square])
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_sqrt = high.sqrt();
        let low_sqrt = low.sqrt();
        Self::from_2_f32x8([high_sqrt, low_sqrt])
    }
    #[inline(always)]
    fn abs(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_abs = high.abs();
        let low_abs = low.abs();
        Self::from_2_f32x8([high_abs, low_abs])
    }
    #[inline(always)]
    fn floor(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_floor = high.floor();
        let low_floor = low.floor();
        Self::from_2_f32x8([high_floor, low_floor])
    }
    #[inline(always)]
    fn ceil(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_ceil = high.ceil();
        let low_ceil = low.ceil();
        Self::from_2_f32x8([high_ceil, low_ceil])
    }
    #[inline(always)]
    fn neg(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_neg = high.neg();
        let low_neg = low.neg();
        Self::from_2_f32x8([high_neg, low_neg])
    }
    #[inline(always)]
    fn round(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_round = high.round();
        let low_round = low.round();
        Self::from_2_f32x8([high_round, low_round])
    }
    #[inline(always)]
    fn sign(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_sign = high.sign();
        let low_sign = low.sign();
        Self::from_2_f32x8([high_sign, low_sign])
    }
    #[inline(always)]
    fn leaky_relu(self, alpha: half::bf16) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_leaky_relu = high.leaky_relu(alpha.to_f32());
        let low_leaky_relu = low.leaky_relu(alpha.to_f32());
        Self::from_2_f32x8([high_leaky_relu, low_leaky_relu])
    }
    #[inline(always)]
    fn relu(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_relu = high.relu();
        let low_relu = low.relu();
        Self::from_2_f32x8([high_relu, low_relu])
    }
    #[inline(always)]
    fn relu6(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_relu6 = high.relu6();
        let low_relu6 = low.relu6();
        Self::from_2_f32x8([high_relu6, low_relu6])
    }
    #[inline(always)]
    fn pow(self, exp: Self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_pow = high.pow(exp.to_f32());
        let low_pow = low.pow(exp.to_f32());
        Self::from_2_f32x8([high_pow, low_pow])
    }
    #[inline(always)]
    fn asin(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_asin = high.asin();
        let low_asin = low.asin();
        Self::from_2_f32x8([high_asin, low_asin])
    }
    #[inline(always)]
    fn acos(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_acos = high.acos();
        let low_acos = low.acos();
        Self::from_2_f32x8([high_acos, low_acos])
    }
    #[inline(always)]
    fn atan(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_atan = high.atan();
        let low_atan = low.atan();
        Self::from_2_f32x8([high_atan, low_atan])
    }
    #[inline(always)]
    fn sinh(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_sinh = high.sinh();
        let low_sinh = low.sinh();
        Self::from_2_f32x8([high_sinh, low_sinh])
    }
    #[inline(always)]
    fn cosh(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_cosh = high.cosh();
        let low_cosh = low.cosh();
        Self::from_2_f32x8([high_cosh, low_cosh])
    }
    #[inline(always)]
    fn tanh(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_tanh = high.tanh();
        let low_tanh = low.tanh();
        Self::from_2_f32x8([high_tanh, low_tanh])
    }
    #[inline(always)]
    fn asinh(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_asinh = high.asinh();
        let low_asinh = low.asinh();
        Self::from_2_f32x8([high_asinh, low_asinh])
    }
    #[inline(always)]
    fn acosh(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_acosh = high.acosh();
        let low_acosh = low.acosh();
        Self::from_2_f32x8([high_acosh, low_acosh])
    }
    #[inline(always)]
    fn atanh(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_atanh = high.atanh();
        let low_atanh = low.atanh();
        Self::from_2_f32x8([high_atanh, low_atanh])
    }
    #[inline(always)]
    fn exp2(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_exp2 = high.exp2();
        let low_exp2 = low.exp2();
        Self::from_2_f32x8([high_exp2, low_exp2])
    }
    #[inline(always)]
    fn exp10(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_exp10 = high.exp10();
        let low_exp10 = low.exp10();
        Self::from_2_f32x8([high_exp10, low_exp10])
    }
    #[inline(always)]
    fn expm1(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_expm1 = high.expm1();
        let low_expm1 = low.expm1();
        Self::from_2_f32x8([high_expm1, low_expm1])
    }
    #[inline(always)]
    fn log10(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_log10 = high.log10();
        let low_log10 = low.log10();
        Self::from_2_f32x8([high_log10, low_log10])
    }
    #[inline(always)]
    fn log2(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_log2 = high.log2();
        let low_log2 = low.log2();
        Self::from_2_f32x8([high_log2, low_log2])
    }
    #[inline(always)]
    fn log1p(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_log1p = high.log1p();
        let low_log1p = low.log1p();
        Self::from_2_f32x8([high_log1p, low_log1p])
    }
    #[inline(always)]
    fn hypot(self, other: Self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let [high_other, low_other] = other.to_2_f32x8();
        let high_hypot = high.hypot(high_other);
        let low_hypot = low.hypot(low_other);
        Self::from_2_f32x8([high_hypot, low_hypot])
    }
    #[inline(always)]
    fn trunc(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_trunc = high.trunc();
        let low_trunc = low.trunc();
        Self::from_2_f32x8([high_trunc, low_trunc])
    }
    #[inline(always)]
    fn erf(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_erf = high.erf();
        let low_erf = low.erf();
        Self::from_2_f32x8([high_erf, low_erf])
    }
    #[inline(always)]
    fn cbrt(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_cbrt = high.cbrt();
        let low_cbrt = low.cbrt();
        Self::from_2_f32x8([high_cbrt, low_cbrt])
    }
    #[inline(always)]
    fn exp(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_exp = high.exp();
        let low_exp = low.exp();
        Self::from_2_f32x8([high_exp, low_exp])
    }
    #[inline(always)]
    fn ln(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_ln = high.ln();
        let low_ln = low.ln();
        Self::from_2_f32x8([high_ln, low_ln])
    }
    #[inline(always)]
    fn log(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_log = high.log();
        let low_log = low.log();
        Self::from_2_f32x8([high_log, low_log])
    }
    #[inline(always)]
    fn sincos(self) -> (Self, Self) {
        let [high, low] = self.to_2_f32x8();
        let (high_sin, high_cos) = high.sincos();
        let (low_sin, low_cos) = low.sincos();
        (
            Self::from_2_f32x8([high_sin, low_sin]),
            Self::from_2_f32x8([high_cos, low_cos]),
        )
    }
    #[inline(always)]
    fn atan2(self, other: Self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let [high_other, low_other] = other.to_2_f32x8();
        let high_atan2 = high.atan2(high_other);
        let low_atan2 = low.atan2(low_other);
        Self::from_2_f32x8([high_atan2, low_atan2])
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let [high_other, low_other] = other.to_2_f32x8();
        let high_min = high.min(high_other);
        let low_min = low.min(low_other);
        Self::from_2_f32x8([high_min, low_min])
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let [high_other, low_other] = other.to_2_f32x8();
        let high_max = high.max(high_other);
        let low_max = low.max(low_other);
        Self::from_2_f32x8([high_max, low_max])
    }
    #[inline(always)]
    fn hard_sigmoid(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_hard_sigmoid = high.hard_sigmoid();
        let low_hard_sigmoid = low.hard_sigmoid();
        Self::from_2_f32x8([high_hard_sigmoid, low_hard_sigmoid])
    }

    #[inline(always)]
    fn fast_hard_sigmoid(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_fast_hard_sigmoid = high.fast_hard_sigmoid();
        let low_fast_hard_sigmoid = low.fast_hard_sigmoid();
        Self::from_2_f32x8([high_fast_hard_sigmoid, low_fast_hard_sigmoid])
    }

    #[inline(always)]
    fn elu(self, alpha: half::bf16) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_elu = high.elu(alpha.to_f32());
        let low_elu = low.elu(alpha.to_f32());
        Self::from_2_f32x8([high_elu, low_elu])
    }

    #[inline(always)]
    fn selu(self, alpha: half::bf16, scale: half::bf16) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_selu = high.selu(alpha.to_f32(), scale.to_f32());
        let low_selu = low.selu(alpha.to_f32(), scale.to_f32());
        Self::from_2_f32x8([high_selu, low_selu])
    }

    #[inline(always)]
    fn celu(self, alpha: half::bf16) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_celu = high.celu(alpha.to_f32());
        let low_celu = low.celu(alpha.to_f32());
        Self::from_2_f32x8([high_celu, low_celu])
    }

    #[inline(always)]
    fn gelu(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_gelu = high.gelu();
        let low_gelu = low.gelu();
        Self::from_2_f32x8([high_gelu, low_gelu])
    }

    #[inline(always)]
    fn hard_swish(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_hard_swish = high.hard_swish();
        let low_hard_swish = low.hard_swish();
        Self::from_2_f32x8([high_hard_swish, low_hard_swish])
    }

    #[inline(always)]
    fn mish(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_mish = high.mish();
        let low_mish = low.mish();
        Self::from_2_f32x8([high_mish, low_mish])
    }

    #[inline(always)]
    fn softplus(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_softplus = high.softplus();
        let low_softplus = low.softplus();
        Self::from_2_f32x8([high_softplus, low_softplus])
    }

    #[inline(always)]
    fn recip(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_recip = high.recip();
        let low_recip = low.recip();
        Self::from_2_f32x8([high_recip, low_recip])
    }
    #[inline(always)]
    fn sigmoid(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_sigmoid = high.sigmoid();
        let low_sigmoid = low.sigmoid();
        Self::from_2_f32x8([high_sigmoid, low_sigmoid])
    }
    #[inline(always)]
    fn softsign(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_softsign = high.softsign();
        let low_softsign = low.softsign();
        Self::from_2_f32x8([high_softsign, low_softsign])
    }
}

impl FloatOutBinary2 for bf16x16 {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn __log(self, base: Self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let [high_base, low_base] = base.to_2_f32x8();
        let high_log = high.__log(high_base);
        let low_log = low.__log(low_base);
        bf16x16::from_2_f32x8([high_log, low_log])
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
        let [high, low] = self.to_2_f32x8();
        let [high_rhs, low_rhs] = rhs.to_2_f32x8();
        let high_pow = high.__pow(high_rhs);
        let low_pow = low.__pow(low_rhs);
        bf16x16::from_2_f32x8([high_pow, low_pow])
    }

    #[inline(always)]
    fn __rem(self, rhs: Self) -> Self {
        self % rhs
    }

    #[inline(always)]
    fn __max(self, rhs: Self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let [high_rhs, low_rhs] = rhs.to_2_f32x8();
        let high_max = high.__max(high_rhs);
        let low_max = low.__max(low_rhs);
        bf16x16::from_2_f32x8([high_max, low_max])
    }

    #[inline(always)]
    fn __min(self, rhs: Self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let [high_rhs, low_rhs] = rhs.to_2_f32x8();
        let high_min = high.__min(high_rhs);
        let low_min = low.__min(low_rhs);
        bf16x16::from_2_f32x8([high_min, low_min])
    }

    #[inline(always)]
    fn __clip(self, min: Self, max: Self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let [high_min, low_min] = min.to_2_f32x8();
        let [high_max, low_max] = max.to_2_f32x8();
        let high_clip = high.__clip(high_min, high_max);
        let low_clip = low.__clip(low_min, low_max);
        bf16x16::from_2_f32x8([high_clip, low_clip])
    }
}

impl NormalOutUnary2 for bf16x16 {
    #[inline(always)]
    fn __square(self) -> Self {
        self * self
    }

    #[inline(always)]
    fn __abs(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_abs = high.__abs();
        let low_abs = low.__abs();
        bf16x16::from_2_f32x8([high_abs, low_abs])
    }

    #[inline(always)]
    fn __ceil(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_ceil = high.__ceil();
        let low_ceil = low.__ceil();
        bf16x16::from_2_f32x8([high_ceil, low_ceil])
    }

    #[inline(always)]
    fn __floor(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_floor = high.__floor();
        let low_floor = low.__floor();
        bf16x16::from_2_f32x8([high_floor, low_floor])
    }

    #[inline(always)]
    fn __neg(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_neg = high.__neg();
        let low_neg = low.__neg();
        bf16x16::from_2_f32x8([high_neg, low_neg])
    }

    #[inline(always)]
    fn __round(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_round = high.__round();
        let low_round = low.__round();
        bf16x16::from_2_f32x8([high_round, low_round])
    }

    #[inline(always)]
    fn __sign(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_sign = high.__sign();
        let low_sign = low.__sign();
        bf16x16::from_2_f32x8([high_sign, low_sign])
    }

    #[inline(always)]
    fn __leaky_relu(self, alpha: Self) -> Self {
        self.max(bf16x16::splat(half::bf16::from_f32_const(0.0)))
            + alpha * self.min(bf16x16::splat(half::bf16::from_f32_const(0.0)))
    }

    #[inline(always)]
    fn __relu(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_relu = high.__relu();
        let low_relu = low.__relu();
        bf16x16::from_2_f32x8([high_relu, low_relu])
    }

    #[inline(always)]
    fn __relu6(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_relu6 = high.__relu6();
        let low_relu6 = low.__relu6();
        bf16x16::from_2_f32x8([high_relu6, low_relu6])
    }
}

impl Eval2 for bf16x16 {
    type Output = i16x16;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        self.is_nan()
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        unreachable!()
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        self.is_infinite()
    }
}
