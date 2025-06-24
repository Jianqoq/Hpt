use crate::simd::_512bit::common::f16x32::f16x32;
use crate::simd::_512bit::common::f32x16::f32x16;

impl f16x32 {
    /// convert to [f32x8; 2]
    #[inline(always)]
    pub fn to_2_f32vec(self) -> [f32x16; 2] {
        unsafe {
            #[cfg(target_feature = "f16c")]
            {
                use std::arch::x86_64::*;
                let raw_f16: [u16; 32] = std::mem::transmute(self.0);
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

    #[allow(unused)]
    /// convert to f32x8
    #[inline(always)]
    pub fn high_to_f32vec(self) -> f32x16 {
        unsafe {
            #[cfg(target_feature = "f16c")]
            {
                use std::arch::x86_64::*;
                let raw_f16: [u16; 32] = std::mem::transmute(self.0);
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
            {
                use std::arch::x86_64::*;
                let f16_low = _mm512_cvtps_ph::<4>(val[0].0);
                let f16_high = _mm512_cvtps_ph::<4>(val[1].0);
                let tmp_256 = _mm512_castsi256_si512(f16_low);
                let result = _mm512_inserti32x8::<1>(tmp_256, f16_high);

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
            let f16_bits = _mm512_cvtps_ph::<4>(val.0);
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
