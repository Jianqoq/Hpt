use crate::convertion::VecConvertor;
use crate::traits::{SimdMath, SimdSelect, VecTrait};
use crate::type_promote::{Eval2, FloatOutBinary2, NormalOut2, NormalOutUnary2};
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
    /// check if the value is NaN, and return a mask
    #[inline(always)]
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
    #[inline(always)]
    pub fn is_infinite(&self) -> i16x16 {
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
    #[inline(always)]
    pub fn to_2_f32x8(self) -> [f32x8; 2] {
        unsafe {
            #[cfg(all(
                target_feature = "f16c",
                target_arch = "x86_64",
                target_feature = "avx2"
            ))]
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
        let mut ret = f16x16::default();
        for i in 0..8 {
            ret.0[i] = self.0[i] + rhs.0[i];
        }
        ret
    }
}

impl std::ops::Sub for f16x16 {
    type Output = Self;
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
    fn neg(self) -> Self::Output {
        let mut ret = f16x16::default();
        for i in 0..8 {
            ret.0[i] = -self.0[i];
        }
        ret
    }
}

/// fallback to convert f16 to f32
#[inline(always)]
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

#[inline(always)]
pub(crate) fn f32x8_to_f16x8(val: f32x8) -> [u16; 8] {
    unsafe {
        #[cfg(all(target_feature = "f16c", target_arch = "x86_64"))]
        {
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
            unsafe {
                let [x0, x1]: [f32x8; 2] = std::mem::transmute(self.to_2_f32x8());
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
            let arr: [half::f16; 8] = unsafe { std::mem::transmute(self) };
            let mut result = [0i16; 8];
            for i in 0..8 {
                result[i] = arr[i].to_f32() as i16;
            }
            return unsafe { std::mem::transmute(result) };
        }
    }
    #[inline(always)]
    fn to_u16(self) -> super::u16x16::u16x16 {
        #[cfg(all(target_feature = "avx2", target_feature = "f16c"))]
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
            all(target_feature = "avx2", target_feature = "f16c"),
            all(target_feature = "neon", target_arch = "aarch64")
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
    #[inline(always)]
    fn to_f16(self) -> f16x16 {
        self
    }
}

impl SimdMath<half::f16> for f16x16 {
    
}

impl FloatOutBinary2 for f16x16 {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn __log(self, base: Self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let [high_base, low_base] = base.to_2_f32x8();
        let high_log = f32x8_to_f16x8(high.__log(high_base));
        let low_log = f32x8_to_f16x8(low.__log(low_base));
        f16x16([
            half::f16::from_bits(high_log[0]),
            half::f16::from_bits(high_log[1]),
            half::f16::from_bits(high_log[2]),
            half::f16::from_bits(high_log[3]),
            half::f16::from_bits(high_log[4]),
            half::f16::from_bits(high_log[5]),
            half::f16::from_bits(high_log[6]),
            half::f16::from_bits(high_log[7]),
            half::f16::from_bits(low_log[0]),
            half::f16::from_bits(low_log[1]),
            half::f16::from_bits(low_log[2]),
            half::f16::from_bits(low_log[3]),
            half::f16::from_bits(low_log[4]),
            half::f16::from_bits(low_log[5]),
            half::f16::from_bits(low_log[6]),
            half::f16::from_bits(low_log[7]),
        ])
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
    fn __pow(self, rhs: Self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let [high_base, low_base] = rhs.to_2_f32x8();
        let high_pow = f32x8_to_f16x8(high.__pow(high_base));
        let low_pow = f32x8_to_f16x8(low.__pow(low_base));
        f16x16([
            half::f16::from_bits(high_pow[0]),
            half::f16::from_bits(high_pow[1]),
            half::f16::from_bits(high_pow[2]),
            half::f16::from_bits(high_pow[3]),
            half::f16::from_bits(high_pow[4]),
            half::f16::from_bits(high_pow[5]),
            half::f16::from_bits(high_pow[6]),
            half::f16::from_bits(high_pow[7]),
            half::f16::from_bits(low_pow[0]),
            half::f16::from_bits(low_pow[1]),
            half::f16::from_bits(low_pow[2]),
            half::f16::from_bits(low_pow[3]),
            half::f16::from_bits(low_pow[4]),
            half::f16::from_bits(low_pow[5]),
            half::f16::from_bits(low_pow[6]),
            half::f16::from_bits(low_pow[7]),
        ])
    }

    #[inline(always)]
    fn __rem(self, rhs: Self) -> Self {
        self % rhs
    }

    #[inline(always)]
    fn __max(self, rhs: Self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let [high_base, low_base] = rhs.to_2_f32x8();
        let high_max = f32x8_to_f16x8(high.__max(high_base));
        let low_max = f32x8_to_f16x8(low.__max(low_base));
        f16x16([
            half::f16::from_bits(high_max[0]),
            half::f16::from_bits(high_max[1]),
            half::f16::from_bits(high_max[2]),
            half::f16::from_bits(high_max[3]),
            half::f16::from_bits(high_max[4]),
            half::f16::from_bits(high_max[5]),
            half::f16::from_bits(high_max[6]),
            half::f16::from_bits(high_max[7]),
            half::f16::from_bits(low_max[0]),
            half::f16::from_bits(low_max[1]),
            half::f16::from_bits(low_max[2]),
            half::f16::from_bits(low_max[3]),
            half::f16::from_bits(low_max[4]),
            half::f16::from_bits(low_max[5]),
            half::f16::from_bits(low_max[6]),
            half::f16::from_bits(low_max[7]),
        ])
    }

    #[inline(always)]
    fn __min(self, rhs: Self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let [high_base, low_base] = rhs.to_2_f32x8();
        let high_min = f32x8_to_f16x8(high.__min(high_base));
        let low_min = f32x8_to_f16x8(low.__min(low_base));
        f16x16([
            half::f16::from_bits(high_min[0]),
            half::f16::from_bits(high_min[1]),
            half::f16::from_bits(high_min[2]),
            half::f16::from_bits(high_min[3]),
            half::f16::from_bits(high_min[4]),
            half::f16::from_bits(high_min[5]),
            half::f16::from_bits(high_min[6]),
            half::f16::from_bits(high_min[7]),
            half::f16::from_bits(low_min[0]),
            half::f16::from_bits(low_min[1]),
            half::f16::from_bits(low_min[2]),
            half::f16::from_bits(low_min[3]),
            half::f16::from_bits(low_min[4]),
            half::f16::from_bits(low_min[5]),
            half::f16::from_bits(low_min[6]),
            half::f16::from_bits(low_min[7]),
        ])
    }

    #[inline(always)]
    fn __clip(self, min: Self, max: Self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let [high_min, low_min] = min.to_2_f32x8();
        let [high_max, low_max] = max.to_2_f32x8();
        let high_clip = f32x8_to_f16x8(high.__clip(high_min, high_max));
        let low_clip = f32x8_to_f16x8(low.__clip(low_min, low_max));
        f16x16([
            half::f16::from_bits(high_clip[0]),
            half::f16::from_bits(high_clip[1]),
            half::f16::from_bits(high_clip[2]),
            half::f16::from_bits(high_clip[3]),
            half::f16::from_bits(high_clip[4]),
            half::f16::from_bits(high_clip[5]),
            half::f16::from_bits(high_clip[6]),
            half::f16::from_bits(high_clip[7]),
            half::f16::from_bits(low_clip[0]),
            half::f16::from_bits(low_clip[1]),
            half::f16::from_bits(low_clip[2]),
            half::f16::from_bits(low_clip[3]),
            half::f16::from_bits(low_clip[4]),
            half::f16::from_bits(low_clip[5]),
            half::f16::from_bits(low_clip[6]),
            half::f16::from_bits(low_clip[7]),
        ])
    }
}

impl NormalOutUnary2 for f16x16 {
    #[inline(always)]
    fn __square(self) -> Self {
        self * self
    }

    #[inline(always)]
    fn __abs(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_abs = f32x8_to_f16x8(high.__abs());
        let low_abs = f32x8_to_f16x8(low.__abs());
        f16x16([
            half::f16::from_bits(high_abs[0]),
            half::f16::from_bits(high_abs[1]),
            half::f16::from_bits(high_abs[2]),
            half::f16::from_bits(high_abs[3]),
            half::f16::from_bits(high_abs[4]),
            half::f16::from_bits(high_abs[5]),
            half::f16::from_bits(high_abs[6]),
            half::f16::from_bits(high_abs[7]),
            half::f16::from_bits(low_abs[0]),
            half::f16::from_bits(low_abs[1]),
            half::f16::from_bits(low_abs[2]),
            half::f16::from_bits(low_abs[3]),
            half::f16::from_bits(low_abs[4]),
            half::f16::from_bits(low_abs[5]),
            half::f16::from_bits(low_abs[6]),
            half::f16::from_bits(low_abs[7]),
        ])
    }

    #[inline(always)]
    fn __ceil(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_ceil = f32x8_to_f16x8(high.__ceil());
        let low_ceil = f32x8_to_f16x8(low.__ceil());
        f16x16([
            half::f16::from_bits(high_ceil[0]),
            half::f16::from_bits(high_ceil[1]),
            half::f16::from_bits(high_ceil[2]),
            half::f16::from_bits(high_ceil[3]),
            half::f16::from_bits(high_ceil[4]),
            half::f16::from_bits(high_ceil[5]),
            half::f16::from_bits(high_ceil[6]),
            half::f16::from_bits(high_ceil[7]),
            half::f16::from_bits(low_ceil[0]),
            half::f16::from_bits(low_ceil[1]),
            half::f16::from_bits(low_ceil[2]),
            half::f16::from_bits(low_ceil[3]),
            half::f16::from_bits(low_ceil[4]),
            half::f16::from_bits(low_ceil[5]),
            half::f16::from_bits(low_ceil[6]),
            half::f16::from_bits(low_ceil[7]),
        ])
    }

    #[inline(always)]
    fn __floor(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_floor = f32x8_to_f16x8(high.__floor());
        let low_floor = f32x8_to_f16x8(low.__floor());
        f16x16([
            half::f16::from_bits(high_floor[0]),
            half::f16::from_bits(high_floor[1]),
            half::f16::from_bits(high_floor[2]),
            half::f16::from_bits(high_floor[3]),
            half::f16::from_bits(high_floor[4]),
            half::f16::from_bits(high_floor[5]),
            half::f16::from_bits(high_floor[6]),
            half::f16::from_bits(high_floor[7]),
            half::f16::from_bits(low_floor[0]),
            half::f16::from_bits(low_floor[1]),
            half::f16::from_bits(low_floor[2]),
            half::f16::from_bits(low_floor[3]),
            half::f16::from_bits(low_floor[4]),
            half::f16::from_bits(low_floor[5]),
            half::f16::from_bits(low_floor[6]),
            half::f16::from_bits(low_floor[7]),
        ])
    }

    #[inline(always)]
    fn __neg(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_neg = f32x8_to_f16x8(high.__neg());
        let low_neg = f32x8_to_f16x8(low.__neg());
        f16x16([
            half::f16::from_bits(high_neg[0]),
            half::f16::from_bits(high_neg[1]),
            half::f16::from_bits(high_neg[2]),
            half::f16::from_bits(high_neg[3]),
            half::f16::from_bits(high_neg[4]),
            half::f16::from_bits(high_neg[5]),
            half::f16::from_bits(high_neg[6]),
            half::f16::from_bits(high_neg[7]),
            half::f16::from_bits(low_neg[0]),
            half::f16::from_bits(low_neg[1]),
            half::f16::from_bits(low_neg[2]),
            half::f16::from_bits(low_neg[3]),
            half::f16::from_bits(low_neg[4]),
            half::f16::from_bits(low_neg[5]),
            half::f16::from_bits(low_neg[6]),
            half::f16::from_bits(low_neg[7]),
        ])
    }

    #[inline(always)]
    fn __round(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_round = f32x8_to_f16x8(high.__round());
        let low_round = f32x8_to_f16x8(low.__round());
        f16x16([
            half::f16::from_bits(high_round[0]),
            half::f16::from_bits(high_round[1]),
            half::f16::from_bits(high_round[2]),
            half::f16::from_bits(high_round[3]),
            half::f16::from_bits(high_round[4]),
            half::f16::from_bits(high_round[5]),
            half::f16::from_bits(high_round[6]),
            half::f16::from_bits(high_round[7]),
            half::f16::from_bits(low_round[0]),
            half::f16::from_bits(low_round[1]),
            half::f16::from_bits(low_round[2]),
            half::f16::from_bits(low_round[3]),
            half::f16::from_bits(low_round[4]),
            half::f16::from_bits(low_round[5]),
            half::f16::from_bits(low_round[6]),
            half::f16::from_bits(low_round[7]),
        ])
    }

    #[inline(always)]
    fn __sign(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_sign = f32x8_to_f16x8(high.__sign());
        let low_sign = f32x8_to_f16x8(low.__sign());
        f16x16([
            half::f16::from_bits(high_sign[0]),
            half::f16::from_bits(high_sign[1]),
            half::f16::from_bits(high_sign[2]),
            half::f16::from_bits(high_sign[3]),
            half::f16::from_bits(high_sign[4]),
            half::f16::from_bits(high_sign[5]),
            half::f16::from_bits(high_sign[6]),
            half::f16::from_bits(high_sign[7]),
            half::f16::from_bits(low_sign[0]),
            half::f16::from_bits(low_sign[1]),
            half::f16::from_bits(low_sign[2]),
            half::f16::from_bits(low_sign[3]),
            half::f16::from_bits(low_sign[4]),
            half::f16::from_bits(low_sign[5]),
            half::f16::from_bits(low_sign[6]),
            half::f16::from_bits(low_sign[7]),
        ])
    }

    #[inline(always)]
    fn __leaky_relu(self, _: Self) -> Self {
        unreachable!()
    }

    #[inline(always)]
    fn __relu(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_relu = f32x8_to_f16x8(high.__relu());
        let low_relu = f32x8_to_f16x8(low.__relu());
        f16x16([
            half::f16::from_bits(high_relu[0]),
            half::f16::from_bits(high_relu[1]),
            half::f16::from_bits(high_relu[2]),
            half::f16::from_bits(high_relu[3]),
            half::f16::from_bits(high_relu[4]),
            half::f16::from_bits(high_relu[5]),
            half::f16::from_bits(high_relu[6]),
            half::f16::from_bits(high_relu[7]),
            half::f16::from_bits(low_relu[0]),
            half::f16::from_bits(low_relu[1]),
            half::f16::from_bits(low_relu[2]),
            half::f16::from_bits(low_relu[3]),
            half::f16::from_bits(low_relu[4]),
            half::f16::from_bits(low_relu[5]),
            half::f16::from_bits(low_relu[6]),
            half::f16::from_bits(low_relu[7]),
        ])
    }

    #[inline(always)]
    fn __relu6(self) -> Self {
        let [high, low] = self.to_2_f32x8();
        let high_relu6 = f32x8_to_f16x8(high.__relu6());
        let low_relu6 = f32x8_to_f16x8(low.__relu6());
        f16x16([
            half::f16::from_bits(high_relu6[0]),
            half::f16::from_bits(high_relu6[1]),
            half::f16::from_bits(high_relu6[2]),
            half::f16::from_bits(high_relu6[3]),
            half::f16::from_bits(high_relu6[4]),
            half::f16::from_bits(high_relu6[5]),
            half::f16::from_bits(high_relu6[6]),
            half::f16::from_bits(high_relu6[7]),
            half::f16::from_bits(low_relu6[0]),
            half::f16::from_bits(low_relu6[1]),
            half::f16::from_bits(low_relu6[2]),
            half::f16::from_bits(low_relu6[3]),
            half::f16::from_bits(low_relu6[4]),
            half::f16::from_bits(low_relu6[5]),
            half::f16::from_bits(low_relu6[6]),
            half::f16::from_bits(low_relu6[7]),
        ])
    }
}

impl Eval2 for f16x16 {
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
