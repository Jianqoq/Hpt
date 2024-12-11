use std::ops::{ Index, IndexMut };
use std::arch::x86_64::*;
use crate::traits::SimdCompare;
use crate::vectors::arch_simd::_256bit::f32x8::f32x8;
use crate::vectors::arch_simd::_256bit::u16x16::u16x16;
use crate::vectors::traits::{ Init, VecTrait };

/// a vector of 16 f16 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(32))]
pub struct f16x16(pub(crate) [half::f16; 16]);

impl VecTrait<half::f16> for f16x16 {
    const SIZE: usize = 16;
    type Base = half::f16;
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
    fn copy_from_slice(&mut self, slice: &[half::f16]) {
        self.0.copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const half::f16 {
        self.0.as_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut half::f16 {
        self.0.as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut half::f16 {
        self.0.as_ptr() as *mut _
    }

    #[inline(always)]
    fn sum(&self) -> half::f16 {
        self.0.iter().sum()
    }
}

impl Init<half::f16> for f16x16 {
    fn splat(val: half::f16) -> f16x16 {
        f16x16([val; 16])
    }
}
impl Index<usize> for f16x16 {
    type Output = half::f16;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
impl IndexMut<usize> for f16x16 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl f16x16 {
    /// check if the value is NaN, return a mask
    pub fn is_nan(&self) -> u16x16 {
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
    /// check if the value is infinite, return a mask
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
    /// convert to 2 f32x8
    #[cfg(feature = "f16c")]
    pub fn to_2_f32x8(self) -> [f32x8; 2] {
        unsafe {
            let raw_f16: [u16; 16] = std::mem::transmute(self.0);
            let f32x8_1 = _mm256_cvtph_ps(_mm_loadu_si128(raw_f16.as_ptr() as *const __m128i));
            let f32x8_2 = _mm256_cvtph_ps(
                _mm_loadu_si128(raw_f16.as_ptr().add(8) as *const __m128i)
            );

            std::mem::transmute([(f32x8_1, f32x8_2)])
        }
    }
    /// convert to 2 f32x8
    #[cfg(not(feature = "f16c"))]
    pub fn to_2_f32x8(self) -> [f32x8; 2] {
        let [a0, a1] = unsafe {
            let a: [std::simd::u16x8; 2] = std::mem::transmute(self.0);
            a
        };
        let a0 = u16_to_f16(a0);
        let a1 = u16_to_f16(a1);
        unsafe { std::mem::transmute([a0, a1]) }
    }
}

impl SimdCompare for f16x16 {
    type SimdMask = u16x16;
    fn simd_eq(self, other: Self) -> u16x16 {
        let x: u16x16 = unsafe { std::mem::transmute(self.0) };
        let y: u16x16 = unsafe { std::mem::transmute(other.0) };
        let eq = x.simd_eq(y);
        unsafe { std::mem::transmute(eq) }
    }
    fn simd_ne(self, other: Self) -> u16x16 {
        let x: u16x16 = unsafe { std::mem::transmute(self.0) };
        let y: u16x16 = unsafe { std::mem::transmute(other.0) };
        let ne = x.simd_ne(y);
        unsafe { std::mem::transmute(ne) }
    }
    fn simd_lt(self, other: Self) -> u16x16 {
        let x: u16x16 = unsafe { std::mem::transmute(self.0) };
        let y: u16x16 = unsafe { std::mem::transmute(other.0) };
        let lt = x.simd_lt(y);
        unsafe { std::mem::transmute(lt) }
    }
    fn simd_le(self, other: Self) -> u16x16 {
        let x: u16x16 = unsafe { std::mem::transmute(self.0) };
        let y: u16x16 = unsafe { std::mem::transmute(other.0) };
        let le = x.simd_le(y);
        unsafe { std::mem::transmute(le) }
    }
    fn simd_gt(self, other: Self) -> u16x16 {
        let x: u16x16 = unsafe { std::mem::transmute(self.0) };
        let y: u16x16 = unsafe { std::mem::transmute(other.0) };
        let gt = x.simd_gt(y);
        unsafe { std::mem::transmute(gt) }
    }
    fn simd_ge(self, other: Self) -> u16x16 {
        let x: u16x16 = unsafe { std::mem::transmute(self.0) };
        let y: u16x16 = unsafe { std::mem::transmute(other.0) };
        let ge = x.simd_ge(y);
        unsafe { std::mem::transmute(ge) }
    }
}

impl std::ops::Add for f16x16 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = f16x16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] + rhs.0[i];
        }
        ret
    }
}

impl std::ops::Sub for f16x16 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut ret = f16x16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] - rhs.0[i];
        }
        ret
    }
}

impl std::ops::Mul for f16x16 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret = f16x16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] * rhs.0[i];
        }
        ret
    }
}

impl std::ops::Div for f16x16 {
    type Output = Self;

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

    fn neg(self) -> Self::Output {
        let mut ret = f16x16::default();
        for i in 0..16 {
            ret.0[i] = -self.0[i];
        }
        ret
    }
}

#[inline]
pub(crate) fn f32x8_to_f16x8(values: f32x8) -> __m128i {
    unsafe { _mm256_cvtps_ph(values.0, _MM_FROUND_TO_NEAREST_INT) }
}
