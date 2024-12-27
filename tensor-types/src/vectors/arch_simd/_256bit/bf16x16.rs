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
    fn __leaky_relu(self, _: Self) -> Self {
        unreachable!()
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
