use crate::arch_simd::_256bit::u16x16::u16x16;
use crate::convertion::VecConvertor;
use crate::{ traits::VecTrait, vectors::arch_simd::_256bit::f32x8::f32x8 };
use crate::traits::{SimdCompare, SimdSelect};

use super::i16x16::i16x16;

use std::arch::x86_64::*;

/// a vector of 16 bf16 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(32))]
pub struct bf16x16(pub(crate) [half::bf16; 16]);

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
    fn splat(val: half::bf16) -> bf16x16 {
        bf16x16([val; 16])
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
    pub fn to_2_f32x8(&self) -> [f32x8; 2] {
        todo!()
    }

    /// convert from 2 f32x8
    pub fn from_2_f32x8(_: [f32x8; 2]) -> Self {
        todo!()
    }

    /// check if the value is NaN and return a mask
    pub fn is_nan(&self) -> i16x16 {
        let res: [i16; 16] = self.0.map(|x| if x.is_nan() { 1 } else { 0 });
        unsafe { std::mem::transmute(res) }
    }

    /// check if the value is infinite and return a mask
    pub fn is_infinite(&self) -> u16x16 {
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
    fn simd_eq(self, other: Self) -> i16x16 {
        unsafe {
            let self_ptr = &self.0 as *const _ as *const __m256i;
            let other_ptr = &other.0 as *const _ as *const __m256i;
            let a = _mm256_loadu_si256(self_ptr);
            let b = _mm256_loadu_si256(other_ptr);
            i16x16(_mm256_cmpeq_epi16(a, b))
        }
    }

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

    fn simd_lt(self, other: Self) -> i16x16 {
        unsafe {
            let self_ptr = &self.0 as *const _ as *const __m256i;
            let other_ptr = &other.0 as *const _ as *const __m256i;
            let a = _mm256_loadu_si256(self_ptr);
            let b = _mm256_loadu_si256(other_ptr);
            i16x16(_mm256_cmpgt_epi16(b, a))
        }
    }
    fn simd_le(self, other: Self) -> i16x16 {
        unsafe {
            let self_ptr = &self.0 as *const _ as *const __m256i;
            let other_ptr = &other.0 as *const _ as *const __m256i;
            let a = _mm256_loadu_si256(self_ptr);
            let b = _mm256_loadu_si256(other_ptr);
            let lt = _mm256_cmpgt_epi16(b, a);  // 交换 a 和 b
            let eq = _mm256_cmpeq_epi16(a, b);
            i16x16(_mm256_or_si256(lt, eq))
        }
    }
    fn simd_gt(self, other: Self) -> i16x16 {
        unsafe {
            let self_ptr = &self.0 as *const _ as *const __m256i;
            let other_ptr = &other.0 as *const _ as *const __m256i;
            let a = _mm256_loadu_si256(self_ptr);
            let b = _mm256_loadu_si256(other_ptr);
            i16x16(_mm256_cmpgt_epi16(a, b))
        }
    }
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
    fn select(&self, true_val: bf16x16, false_val: bf16x16) -> bf16x16 {
        let mut ret = bf16x16::default();
        let arr = self.as_array();
        for i in 0..16 {
            ret.0[i] = if arr[i] != 0 { true_val.0[i] } else { false_val.0[i] };
        }
        ret
    }
}

impl std::ops::Add for bf16x16 {
    type Output = Self;

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

    fn neg(self) -> Self::Output {
        let mut ret = bf16x16::default();
        for i in 0..8 {
            ret.0[i] = -self.0[i];
        }
        ret
    }
}

impl VecConvertor for bf16x16 {
    fn to_bf16(self) -> bf16x16 {
        self
    }
    fn to_f16(self) -> super::f16x16::f16x16 {
        unsafe { std::mem::transmute(self) }
    }
    fn to_i16(self) -> super::i16x16::i16x16 {
        unsafe {
            let [x0, x1]: [f32x8; 2] = std::mem::transmute(self.to_2_f32x8());
            let i0 = _mm256_cvtps_epi32(x0.0);
            let i1 = _mm256_cvtps_epi32(x1.0);
            let packed = _mm256_packs_epi32(i0, i1);
            super::i16x16::i16x16(packed)
        }
    }
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
