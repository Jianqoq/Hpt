use crate::convertion::VecConvertor;
use crate::traits::VecTrait;
use crate::vectors::arch_simd::_128bit::f32x4;
use crate::vectors::arch_simd::_128bit::u16x8;

use crate::arch_simd::_128bit::f16x8;
use crate::arch_simd::_128bit::i16x8;
use std::arch::x86_64::*;

impl VecTrait<half::f16> for f16x8 {
    const SIZE: usize = 8;
    type Base = half::f16;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            let [x0, x1] = self.to_2_f32vec();
            let [a0, a1] = a.to_2_f32vec();
            let [b0, b1] = b.to_2_f32vec();
            let res0 = x0.mul_add(a0, b0);
            let res1 = x1.mul_add(a1, b1);
            from_2_f32vec([res0, res1])
        }
    }
    #[inline(always)]
    fn sum(&self) -> half::f16 {
        self.0.iter().sum()
    }
    #[inline(always)]
    fn splat(val: half::f16) -> f16x8 {
        f16x8([val; 8])
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const half::f16) -> Self {
        let mut result = [half::f16::ZERO; 8];
        for i in 0..8 {
            result[i] = unsafe { *ptr.add(i) };
        }
        f16x8(result)
    }
}

impl f16x8 {
    /// convert to f32x4
    #[inline(always)]
    pub fn to_2_f32vec(self) -> [f32x4; 2] {
        unsafe {
            #[cfg(target_feature = "f16c")]
            {
                use std::arch::x86_64::{_mm_cvtph_ps, _mm_loadu_si64};
                let raw_f16: [u16; 8] = std::mem::transmute(self.0);
                let f32x4_1 = _mm_cvtph_ps(_mm_loadu_si64(raw_f16.as_ptr() as *const _));
                let f32x4_2 = _mm_cvtph_ps(_mm_loadu_si64(raw_f16.as_ptr().add(4) as *const _));
                std::mem::transmute([f32x4_1, f32x4_2])
            }
            #[cfg(not(target_feature = "f16c"))]
            {
                let mut res = [0f32; 4];
                for i in 0..4 {
                    res[i] = self.0[i].to_f32();
                }
                let mut res2 = [0f32; 4];
                for i in 0..4 {
                    res2[i] = self.0[i + 4].to_f32();
                }
                std::mem::transmute([res, res2])
            }
        }
    }

    /// convert to f32x4
    #[inline(always)]
    pub fn high_to_f32vec(self) -> f32x4 {
        unsafe {
            #[cfg(target_feature = "f16c")]
            {
                use std::arch::x86_64::{_mm_cvtph_ps, _mm_loadu_si64};
                let raw_f16: [u16; 8] = std::mem::transmute(self.0);
                let f32x4_1 = _mm_cvtph_ps(_mm_loadu_si64(raw_f16.as_ptr() as *const _));
                std::mem::transmute(f32x4_1)
            }
            #[cfg(not(target_feature = "f16c"))]
            {
                let mut res = [0f32; 4];
                for i in 0..4 {
                    res[i] = self.0[i].to_f32();
                }
                std::mem::transmute(res)
            }
        }
    }

    /// convert from 2 f32x4
    #[inline(always)]
    pub fn from_2_f32vec(val: [f32x4; 2]) -> Self {
        unsafe {
            #[cfg(all(target_feature = "f16c", target_arch = "x86_64"))]
            {
                let f16_high = _mm_cvtps_ph(val[0].0, _MM_FROUND_TO_NEAREST_INT);
                let f16_low = _mm_cvtps_ph(val[1].0, _MM_FROUND_TO_NEAREST_INT);
                let result = _mm_unpacklo_epi64(f16_high, f16_low);
                std::mem::transmute(result)
            }
            #[cfg(not(all(target_feature = "f16c", target_arch = "x86_64")))]
            #[cfg(not(all(target_feature = "fp16", target_arch = "aarch64")))]
            #[cfg(not(target_feature = "neon"))]
            {
                let arr: [[f32; 4]; 2] = std::mem::transmute(val);
                let mut result = [0u16; 8];
                for i in 0..4 {
                    result[i] = half::f16::from_f32(arr[0][i]).to_bits();
                    result[i + 4] = half::f16::from_f32(arr[1][i]).to_bits();
                }
                std::mem::transmute(result)
            }
        }
    }
}

impl std::ops::Add for f16x8 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let [x0, x1] = self.to_2_f32vec();
        let [y0, y1] = rhs.to_2_f32vec();
        let low_add = x0 + y0;
        let high_add = x1 + y1;
        f16x8::from_2_f32vec([low_add, high_add])
    }
}

impl std::ops::Sub for f16x8 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        let [x0, x1] = self.to_2_f32vec();
        let [y0, y1] = rhs.to_2_f32vec();
        let low_sub = x0 - y0;
        let high_sub = x1 - y1;
        f16x8::from_2_f32vec([low_sub, high_sub])
    }
}

impl std::ops::Mul for f16x8 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        let [x0, x1] = self.to_2_f32vec();
        let [y0, y1] = rhs.to_2_f32vec();
        let low_mul = x0 * y0;
        let high_mul = x1 * y1;
        f16x8::from_2_f32vec([low_mul, high_mul])
    }
}

/// fallback to convert f32 to f16
#[inline(always)]
pub(crate) fn from_2_f32vec(val: [f32x4; 2]) -> f16x8 {
    #[cfg(target_feature = "f16c")]
    unsafe {
        let f16_high = _mm_cvtps_ph(val[0].0, _MM_FROUND_TO_NEAREST_INT);
        let f16_low = _mm_cvtps_ph(val[1].0, _MM_FROUND_TO_NEAREST_INT);
        let result = _mm_unpacklo_epi64(f16_high, f16_low);
        f16x8(std::mem::transmute(result))
    }
    #[cfg(not(target_feature = "f16c"))]
    {
        use crate::convertion::Convertor;
        let mut result = [half::f16::ZERO; 8];
        for i in 0..4 {
            result[i] = val[0][i].to_f16();
            result[i + 4] = val[1][i].to_f16();
        }
        f16x8(result)
    }
}

impl VecConvertor for f16x8 {
    #[inline(always)]
    fn to_i16(self) -> i16x8 {
        unsafe {
            let [x0, x1]: [f32x4; 2] = std::mem::transmute(self.to_2_f32vec());
            let i0 = _mm_cvtps_epi32(x0.0);
            let i1 = _mm_cvtps_epi32(x1.0);
            let packed = _mm_packs_epi32(i0, i1);
            i16x8(packed)
        }
    }
    #[inline(always)]
    fn to_u16(self) -> u16x8 {
        unsafe {
            let [x0, x1]: [f32x4; 2] = std::mem::transmute(self.to_2_f32vec());
            let i0 = _mm_cvtps_epi32(x0.0);
            let i1 = _mm_cvtps_epi32(x1.0);
            let packed = _mm_packus_epi32(i0, i1);
            u16x8(packed)
        }
    }
    #[inline(always)]
    fn to_f16(self) -> f16x8 {
        self
    }
}
