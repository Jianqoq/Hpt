use crate::convertion::VecConvertor;
use crate::traits::VecTrait;
use crate::vectors::arch_simd::_256bit::f32x8::f32x8;
use crate::vectors::arch_simd::_256bit::u16x16::u16x16;

use crate::traits::SimdCompare;

use super::i16x16::i16x16;

use std::arch::x86_64::*;

/// a vector of 16 f16 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(32))]
pub struct f16x16(pub(crate) [half::f16; 16]);

impl VecTrait<half::f16> for f16x16 {
    const SIZE: usize = 16;
    type Base = half::f16;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[half::f16]) {
        self.0.copy_from_slice(slice);
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        let [x0, x1]: [f32x8; 2] = unsafe { std::mem::transmute(self.to_2_f32x8()) };
        let [a0, a1]: [f32x8; 2] = unsafe { std::mem::transmute(a.to_2_f32x8()) };
        let [b0, b1]: [f32x8; 2] = unsafe { std::mem::transmute(b.to_2_f32x8()) };
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
    fn splat(val: half::f16) -> f16x16 {
        f16x16([val; 16])
    }
}

impl f16x16 {
    #[allow(unused)]
    fn as_array(&self) -> [half::f16; 16] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl f16x16 {
    /// check if the value is NaN, and return a mask
    pub fn is_nan(&self) -> i16x16 {
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
    /// check if the value is infinite, and return a mask
    pub fn is_infinite(&self) -> u16x16 {
        let x = u16x16::splat(0x7c00u16);
        let y = u16x16::splat(0x03ffu16);
        let i: u16x16 = unsafe { std::mem::transmute(self.0) };

        let and = i & x;
        let eq = and.simd_eq(x);

        let and2 = i & y;
        let eq_zero = and2.simd_eq(u16x16::splat(0));

        let result = eq & eq_zero;

        unsafe { std::mem::transmute(result) }
    }
    /// convert to f32x8
    pub fn to_2_f32x8(self) -> [f32x8; 2] {
        unsafe {
            #[cfg(all(target_feature = "f16c", target_arch = "x86_64", target_feature = "avx2"))]
            {
                use std::arch::x86_64::_mm256_cvtph_ps;
                let raw_f16: [u16; 16] = std::mem::transmute(self.0);
                let f32x4_1 = _mm256_cvtph_ps(_mm_loadu_si128(raw_f16.as_ptr() as *const _));
                let f32x4_2 = _mm256_cvtph_ps(_mm_loadu_si128(raw_f16.as_ptr().add(8) as *const _));
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
        }
    }
}
impl SimdCompare for f16x16 {
    type SimdMask = i16x16;
    fn simd_eq(self, other: Self) -> i16x16 {
        let x: i16x16 = unsafe { std::mem::transmute(self.0) };
        let y: i16x16 = unsafe { std::mem::transmute(other.0) };
        x.simd_eq(y)
    }
    fn simd_ne(self, other: Self) -> i16x16 {
        let x: i16x16 = unsafe { std::mem::transmute(self.0) };
        let y: i16x16 = unsafe { std::mem::transmute(other.0) };
        x.simd_ne(y)
    }
    fn simd_lt(self, other: Self) -> i16x16 {
        let x: i16x16 = unsafe { std::mem::transmute(self.0) };
        let y: i16x16 = unsafe { std::mem::transmute(other.0) };
        x.simd_lt(y)
    }
    fn simd_le(self, other: Self) -> i16x16 {
        let x: i16x16 = unsafe { std::mem::transmute(self.0) };
        let y: i16x16 = unsafe { std::mem::transmute(other.0) };
        x.simd_le(y)
    }
    fn simd_gt(self, other: Self) -> i16x16 {
        let x: i16x16 = unsafe { std::mem::transmute(self.0) };
        let y: i16x16 = unsafe { std::mem::transmute(other.0) };
        x.simd_gt(y)
    }
    fn simd_ge(self, other: Self) -> i16x16 {
        let x: i16x16 = unsafe { std::mem::transmute(self.0) };
        let y: i16x16 = unsafe { std::mem::transmute(other.0) };
        x.simd_ge(y)
    }
}

impl std::ops::Add for f16x16 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = f16x16::default();
        for i in 0..8 {
            ret.0[i] = self.0[i] + rhs.0[i];
        }
        ret
    }
}

impl std::ops::Sub for f16x16 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut ret = f16x16::default();
        for i in 0..8 {
            ret.0[i] = self.0[i] - rhs.0[i];
        }
        ret
    }
}

impl std::ops::Mul for f16x16 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret = f16x16::default();
        for i in 0..8 {
            ret.0[i] = self.0[i] * rhs.0[i];
        }
        ret
    }
}

impl std::ops::Div for f16x16 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut ret = f16x16::default();
        for i in 0..8 {
            ret.0[i] = self.0[i] / rhs.0[i];
        }
        ret
    }
}
impl std::ops::Rem for f16x16 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        let mut ret = f16x16::default();
        for i in 0..8 {
            ret.0[i] = self.0[i] % rhs.0[i];
        }
        ret
    }
}
impl std::ops::Neg for f16x16 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut ret = f16x16::default();
        for i in 0..8 {
            ret.0[i] = -self.0[i];
        }
        ret
    }
}

/// fallback to convert f16 to f32
pub fn u16_to_f32(val: [u16; 8]) -> f32x8 {
    unsafe {
        std::mem::transmute([
            half::f16::from_bits(val[0]).to_f32(),
            half::f16::from_bits(val[1]).to_f32(),
            half::f16::from_bits(val[2]).to_f32(),
            half::f16::from_bits(val[3]).to_f32(),
            half::f16::from_bits(val[4]).to_f32(),
            half::f16::from_bits(val[5]).to_f32(),
            half::f16::from_bits(val[6]).to_f32(),
            half::f16::from_bits(val[7]).to_f32(),
        ])
    }
}

/// fallback to convert f32 to f16
#[inline]
pub(crate) fn f32x8_to_f16x8(_: f32x8) -> [u16; 8] {
    unimplemented!()
}

impl VecConvertor for f16x16 {
    fn to_i16(self) -> super::i16x16::i16x16 {
        #[cfg(target_feature = "avx2")]
        {
            unsafe {
                let [x0, x1]: [f32x8; 2] = std::mem::transmute(self.to_2_f32x8());
                let i0 = _mm256_cvtps_epi32(x0.0);
                let i1 = _mm256_cvtps_epi32(x1.0);
                let packed = _mm256_packs_epi32(i0, i1);
                super::i16x16::i16x16(packed)
            }
        }
        #[cfg(all(target_feature = "neon", target_arch = "aarch64"))]
        {
            unimplemented!()
        }
        #[cfg(not(any(
            target_feature = "f16c",
            all(target_feature = "neon", target_arch = "aarch64"),
            target_feature = "avx2"
        )))]
        {
            let arr: [half::f16; 8] = unsafe { std::mem::transmute(self) };
            let mut result = [0i16; 8];
            for i in 0..8 {
                result[i] = arr[i].to_f32() as i16;
            }
            unsafe { std::mem::transmute(result) }
        }
    }
    fn to_u16(self) -> super::u16x16::u16x16 {
        #[cfg(target_feature = "avx2")]
        {
            unsafe {
                let [x0, x1]: [f32x8; 2] = std::mem::transmute(self.to_2_f32x8());
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
            target_feature = "f16c",
            all(target_feature = "neon", target_arch = "aarch64"),
            target_feature = "avx2"
        )))]
        {
            let arr: [half::f16; 8] = unsafe { std::mem::transmute(self) };
            let mut result = [0u16; 8];
            for i in 0..8 {
                result[i] = arr[i].to_f32() as u16;
            }
            unsafe { std::mem::transmute(result) }
        }
    }
    fn to_f16(self) -> f16x16 {
        self
    }
}
