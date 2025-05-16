use crate::convertion::VecConvertor;
use crate::vectors::arch_simd::_512bit::f32x16;
use crate::vectors::arch_simd::_512bit::u16x32;

use crate::simd::_512bit::f16x32;
use crate::simd::_512bit::i16x32;

impl f16x32 {
    /// convert to [f32x16; 2]
    #[inline(always)]
    pub fn to_2_f32vec(self) -> [f32x16; 2] {
        unsafe {
            #[cfg(target_feature = "f16c")]
            {
                use std::arch::x86_64::*;
                let raw_f16: [u16; 16] = std::mem::transmute(self.0);
                let f32x4_1 = _mm512_cvtph_ps(_mm256_loadu_si256(raw_f16.as_ptr() as *const _));
                let f32x4_2 =
                    _mm512_cvtph_ps(_mm256_loadu_si256(raw_f16.as_ptr().add(8) as *const _));
                std::mem::transmute([f32x4_1, f32x4_2])
            }
            #[cfg(not(target_feature = "f16c"))]
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

    /// convert to f32x16
    #[inline(always)]
    pub fn high_to_f32vec(self) -> f32x16 {
        unsafe {
            #[cfg(target_feature = "f16c")]
            {
                use std::arch::x86_64::*;
                let raw_f16: [u16; 16] = std::mem::transmute(self.0);
                let f32x4_1 = _mm512_cvtph_ps(_mm256_loadu_si256(raw_f16.as_ptr() as *const _));
                std::mem::transmute(f32x4_1)
            }
            #[cfg(not(target_feature = "f16c"))]
            {
                let mut result = [0f32; 8];
                for i in 0..8 {
                    result[i] = self.0[i].to_f32();
                }
                std::mem::transmute(result)
            }
        }
    }

    /// convert from 2 f32x4
    #[inline(always)]
    pub fn from_2_f32vec(val: [f32x16; 2]) -> Self {
        unsafe {
            #[cfg(target_feature = "f16c")]
            unsafe {
                use std::arch::x86_64::*;

                // 将两个512位f32向量转换为两个256位f16向量
                let f16_low = _mm512_cvtps_ph(val[0].0, _MM_FROUND_TO_NEAREST_INT);
                let f16_high = _mm512_cvtps_ph(val[1].0, _MM_FROUND_TO_NEAREST_INT);

                // 将两个256位向量组合成一个512位向量
                // 首先将第一个向量转换为512位向量（高256位将为0）
                let result_low = _mm512_castsi256_si512(f16_low);

                // 然后将第二个向量插入到高256位
                let result = _mm512_inserti32x8::<1>(result_low, f16_high);

                // 转换为最终类型
                std::mem::transmute(result)
            }
            #[cfg(not(target_feature = "f16c"))]
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

#[inline(always)]
pub(crate) fn f32x16_to_f16x16(val: f32x16) -> [u16; 16] {
    unsafe {
        #[cfg(target_feature = "f16c")]
        {
            use std::arch::x86_64::*;
            let f16_bits = _mm512_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(val.0);
            std::mem::transmute(f16_bits)
        }
        #[cfg(not(target_feature = "f16c"))]
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

impl VecConvertor for f16x32 {
    #[inline(always)]
    fn to_i16(self) -> i16x32 {
        #[cfg(target_feature = "f16c")]
        {
            use std::arch::x86_64::*;
            unsafe {
                let [x0, x1]: [f32x16; 2] = std::mem::transmute(self.to_2_f32vec());
                let i0 = _mm512_cvtps_epi32(x0.0);
                let i1 = _mm512_cvtps_epi32(x1.0);
                let packed = _mm512_packs_epi32(i0, i1);
                return i16x32(packed);
            }
        }
        #[cfg(not(target_feature = "f16c"))]
        {
            let arr: [half::f16; 16] = unsafe { std::mem::transmute(self) };
            let mut result = [0i16; 16];
            for i in 0..16 {
                result[i] = arr[i].to_f32() as i16;
            }
            unsafe { std::mem::transmute(result) }
        }
    }
    #[inline(always)]
    fn to_u16(self) -> u16x32 {
        #[cfg(target_feature = "f16c")]
        {
            use std::arch::x86_64::*;
            unsafe {
                let [x0, x1]: [f32x16; 2] = std::mem::transmute(self.to_2_f32vec());
                let i0 = _mm512_cvtps_epi32(x0.0);
                let i1 = _mm512_cvtps_epi32(x1.0);
                let packed = _mm512_packus_epi32(i0, i1);
                u16x32(packed)
            }
        }
        #[cfg(not(target_feature = "f16c"))]
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
    fn to_f16(self) -> f16x32 {
        self
    }
}
