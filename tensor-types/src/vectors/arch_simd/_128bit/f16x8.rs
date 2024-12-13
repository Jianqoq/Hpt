use crate::convertion::VecConvertor;
use crate::traits::VecTrait;
use crate::vectors::arch_simd::_128bit::f32x4::f32x4;
use crate::vectors::arch_simd::_128bit::u16x8::u16x8;

use crate::traits::SimdCompare;

use super::i16x8::i16x8;

/// a vector of 8 f16 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(16))]
pub struct f16x8(pub(crate) [half::f16; 8]);

impl VecTrait<half::f16> for f16x8 {
    const SIZE: usize = 8;
    type Base = half::f16;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[half::f16]) {
        self.0.copy_from_slice(slice);
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        let [x0, x1]: [f32x4; 2] = unsafe { std::mem::transmute(self.to_2_f32x4()) };
        let [a0, a1]: [f32x4; 2] = unsafe { std::mem::transmute(a.to_2_f32x4()) };
        let [b0, b1]: [f32x4; 2] = unsafe { std::mem::transmute(b.to_2_f32x4()) };
        let res0 = x0.mul_add(a0, b0);
        let res1 = x1.mul_add(a1, b1);
        let res0 = f32x4_to_f16x4(res0);
        let res1 = f32x4_to_f16x4(res1);
        unsafe { std::mem::transmute([res0, res1]) }
    }
    #[inline(always)]
    fn sum(&self) -> half::f16 {
        self.0.iter().sum()
    }
    fn splat(val: half::f16) -> f16x8 {
        f16x8([val; 8])
    }
}

impl f16x8 {
    #[allow(unused)]
    fn as_array(&self) -> [half::f16; 8] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl f16x8 {
    /// check if the value is NaN, and return a mask
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
    pub fn is_infinite(&self) -> u16x8 {
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
}
impl SimdCompare for f16x8 {
    type SimdMask = i16x8;
    fn simd_eq(self, other: Self) -> i16x8 {
        let x: i16x8 = unsafe { std::mem::transmute(self.0) };
        let y: i16x8 = unsafe { std::mem::transmute(other.0) };
        x.simd_eq(y)
    }
    fn simd_ne(self, other: Self) -> i16x8 {
        let x: i16x8 = unsafe { std::mem::transmute(self.0) };
        let y: i16x8 = unsafe { std::mem::transmute(other.0) };
        x.simd_ne(y)
    }
    fn simd_lt(self, other: Self) -> i16x8 {
        let x: i16x8 = unsafe { std::mem::transmute(self.0) };
        let y: i16x8 = unsafe { std::mem::transmute(other.0) };
        x.simd_lt(y)
    }
    fn simd_le(self, other: Self) -> i16x8 {
        let x: i16x8 = unsafe { std::mem::transmute(self.0) };
        let y: i16x8 = unsafe { std::mem::transmute(other.0) };
        x.simd_le(y)
    }
    fn simd_gt(self, other: Self) -> i16x8 {
        let x: i16x8 = unsafe { std::mem::transmute(self.0) };
        let y: i16x8 = unsafe { std::mem::transmute(other.0) };
        x.simd_gt(y)
    }
    fn simd_ge(self, other: Self) -> i16x8 {
        let x: i16x8 = unsafe { std::mem::transmute(self.0) };
        let y: i16x8 = unsafe { std::mem::transmute(other.0) };
        x.simd_ge(y)
    }
}

impl std::ops::Add for f16x8 {
    type Output = Self;

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

    fn neg(self) -> Self::Output {
        let mut ret = f16x8::default();
        for i in 0..8 {
            ret.0[i] = -self.0[i];
        }
        ret
    }
}

/// fallback to convert f16 to f32
pub fn u16_to_f32(_: [u16; 4]) -> f32x4 {
    unimplemented!()
}

/// fallback to convert f32 to f16
#[inline]
pub(crate) fn f32x4_to_f16x4(_: f32x4) -> [u16; 4] {
    unimplemented!()
}

impl VecConvertor for f16x8 {
}
