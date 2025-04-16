use crate::convertion::VecConvertor;
use crate::traits::VecTrait;
use crate::vectors::arch_simd::_128bit::common::f16x8::f16x8;
use crate::vectors::arch_simd::_128bit::f32x4;
use crate::vectors::arch_simd::_128bit::i16x8;
use crate::vectors::arch_simd::_128bit::u16x8;
use std::arch::aarch64::uint16x8_t;

#[allow(non_camel_case_types)]
type float16x8_t = uint16x8_t;

impl VecTrait<half::f16> for f16x8 {
    const SIZE: usize = 8;
    type Base = half::f16;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[half::f16]) {
        self.0.copy_from_slice(slice);
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(target_feature = "fp16")]
        unsafe {
            let mut b: float16x8_t = std::mem::transmute(b);
            let a: float16x8_t = std::mem::transmute(a);
            let s: float16x8_t = std::mem::transmute(self);
            std::arch::asm!(
                "fmla {0:v}.8h, {1:v}.8h, {2:v}.8h",
                inout(vreg) b,
                in(vreg) a,
                in(vreg) s,
                options(pure, nomem, nostack)
            );
            std::mem::transmute(b)
        }
        #[cfg(not(target_feature = "fp16"))]
        unsafe {
            use num_traits::Float;
            let mut res = [half::f16::ZERO; 8];
            for i in 0..8 {
                res[i] = self.0[i].mul_add(a.0[i], b.0[i]);
            }
            std::mem::transmute(res)
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
    #[inline(always)]
    fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self {
        #[cfg(target_feature = "fp16")]
        unsafe {
            let a: float16x8_t = std::mem::transmute(a);
            let mut b: float16x8_t = std::mem::transmute(b);
            let c: float16x8_t = std::mem::transmute(self);
            match LANE {
                0 => std::arch::asm!(
                    "fmla {0:v}.8h, {1:v}.8h, {2:v}.h[0]",
                    inout(vreg) b,
                    in(vreg) a,
                    in(vreg_low16) c,
                    options(pure, nomem, nostack)),
                1 => std::arch::asm!(
                    "fmla {0:v}.8h, {1:v}.8h, {2:v}.h[1]",
                    inout(vreg) b,
                    in(vreg) a,
                    in(vreg_low16) c,
                    options(pure, nomem, nostack)),
                2 => std::arch::asm!(
                    "fmla {0:v}.8h, {1:v}.8h, {2:v}.h[2]",
                    inout(vreg) b,
                    in(vreg) a,
                    in(vreg_low16) c,
                    options(pure, nomem, nostack)),
                3 => std::arch::asm!(
                    "fmla {0:v}.8h, {1:v}.8h, {2:v}.h[3]",
                    inout(vreg) b,
                    in(vreg) a,
                    in(vreg_low16) c,
                    options(pure, nomem, nostack)),
                4 => std::arch::asm!(
                    "fmla {0:v}.8h, {1:v}.8h, {2:v}.h[4]",
                    inout(vreg) b,
                    in(vreg) a,
                    in(vreg_low16) c,
                    options(pure, nomem, nostack)),
                5 => std::arch::asm!(
                    "fmla {0:v}.8h, {1:v}.8h, {2:v}.h[5]",
                    inout(vreg) b,
                    in(vreg) a,
                    in(vreg_low16) c,
                    options(pure, nomem, nostack)),
                6 => std::arch::asm!(
                    "fmla {0:v}.8h, {1:v}.8h, {2:v}.h[6]",
                    inout(vreg) b,
                    in(vreg) a,
                    in(vreg_low16) c,
                    options(pure, nomem, nostack)),
                7 => std::arch::asm!(
                    "fmla {0:v}.8h, {1:v}.8h, {2:v}.h[7]",
                    inout(vreg) b,
                    in(vreg) a,
                    in(vreg_low16) c,
                    options(pure, nomem, nostack)),
                _ => unreachable!(),
            }
            std::mem::transmute(b)
        }
        #[cfg(not(target_feature = "fp16"))]
        unsafe {
            use num_traits::Float;
            let mut res = [half::f16::ZERO; 8];
            for i in 0..8 {
                res[i] = self.0[i].mul_add(a.0[LANE as usize], b.0[i]);
            }
            std::mem::transmute(res)
        }
    }
}

impl f16x8 {
    /// convert to f32x4
    #[inline(always)]
    pub fn to_2_f32vec(self) -> [f32x4; 2] {
        unsafe {
            #[cfg(target_feature = "fp16")]
            {
                use std::arch::aarch64::{float32x4_t, vld1_s16};

                let low = vld1_s16(self.0.as_ptr() as *const _);
                let high = vld1_s16(self.0.as_ptr().add(4) as *const _);

                let mut res0: float32x4_t;
                let mut res1: float32x4_t;

                std::arch::asm!(
                    "fcvtl {0:v}.4s, {1:v}.4h",
                    out(vreg) res0,
                    in(vreg) low,
                );
                std::arch::asm!(
                    "fcvtl {0:v}.4s, {1:v}.4h",
                    out(vreg) res1,
                    in(vreg) high,
                );

                std::mem::transmute([res0, res1])
            }
            #[cfg(not(target_feature = "fp16"))]
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
            #[cfg(target_feature = "fp16")]
            {
                use std::arch::aarch64::{float32x4_t, vld1_s16};

                let low = vld1_s16(self.0.as_ptr() as *const _);
                let mut res0: float32x4_t;
                std::arch::asm!(
                    "fcvtl {0:v}.4s, {1:v}.4h",
                    out(vreg) res0,
                    in(vreg) low,
                );
                std::mem::transmute(res0)
            }
            #[cfg(not(target_feature = "fp16"))]
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
            #[cfg(target_feature = "fp16")]
            {
                let mut res: float16x8_t;
                std::arch::asm!(
                    "fcvtn {0:v}.4h, {1:v}.4s",
                    "fcvtn2 {0:v}.8h, {2:v}.4s",
                    out(vreg) res,
                    in(vreg) val[0].0,
                    in(vreg) val[1].0,
                );
                std::mem::transmute(res)
            }
            #[cfg(not(target_feature = "fp16"))]
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
        #[cfg(target_feature = "fp16")]
        unsafe {
            let a: float16x8_t = std::mem::transmute(self);
            let b: float16x8_t = std::mem::transmute(rhs);
            let c: float16x8_t;
            std::arch::asm!(
                "fadd {0:v}.8h, {1:v}.8h, {2:v}.8h",
                out(vreg) c,
                in(vreg) a,
                in(vreg) b,
                options(pure, nomem, nostack)
            );
            std::mem::transmute(c)
        }
        #[cfg(not(target_feature = "fp16"))]
        unsafe {
            let mut res = [half::f16::ZERO; 8];
            for i in 0..8 {
                res[i] = self.0[i] + rhs.0[i];
            }
            std::mem::transmute(res)
        }
    }
}

impl std::ops::Sub for f16x8 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        #[cfg(target_feature = "fp16")]
        unsafe {
            let a: float16x8_t = std::mem::transmute(self);
            let b: float16x8_t = std::mem::transmute(rhs);
            let c: float16x8_t;
            std::arch::asm!(
                "fsub {0:v}.8h, {1:v}.8h, {2:v}.8h",
                out(vreg) c,
                in(vreg) a,
                in(vreg) b,
                options(pure, nomem, nostack)
            );
            std::mem::transmute(c)
        }
        #[cfg(not(target_feature = "fp16"))]
        unsafe {
            let mut res = [half::f16::ZERO; 8];
            for i in 0..8 {
                res[i] = self.0[i] - rhs.0[i];
            }
            std::mem::transmute(res)
        }
    }
}

impl std::ops::Mul for f16x8 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        #[cfg(target_feature = "fp16")]
        unsafe {
            let a: float16x8_t = std::mem::transmute(self);
            let b: float16x8_t = std::mem::transmute(rhs);
            let c: float16x8_t;
            std::arch::asm!(
                "fmul {0:v}.8h, {1:v}.8h, {2:v}.8h",
                out(vreg) c,
                in(vreg) a,
                in(vreg) b,
                options(pure, nomem, nostack)
            );
            std::mem::transmute(c)
        }
        #[cfg(not(target_feature = "fp16"))]
        unsafe {
            let mut res = [half::f16::ZERO; 8];
            for i in 0..8 {
                res[i] = self.0[i] * rhs.0[i];
            }
            std::mem::transmute(res)
        }
    }
}

impl VecConvertor for f16x8 {
    #[inline(always)]
    fn to_i16(self) -> i16x8 {
        #[cfg(target_feature = "fp16")]
        unsafe {
            use std::arch::aarch64::int16x8_t;
            let mut result: int16x8_t;
            let a: float16x8_t = std::mem::transmute(self);
            std::arch::asm!(
                "fcvtzs {0:v}.8h, {1:v}.8h",
                out(vreg) result,
                in(vreg) a,
            );
            std::mem::transmute(result)
        }
        #[cfg(not(target_feature = "fp16"))]
        unsafe {
            use crate::convertion::Convertor;
            let mut res = [0i16; 8];
            for i in 0..8 {
                res[i] = self.0[i].to_i16();
            }
            std::mem::transmute(res)
        }
    }
    #[inline(always)]
    fn to_u16(self) -> u16x8 {
        #[cfg(target_feature = "fp16")]
        unsafe {
            use std::arch::aarch64::int16x8_t;
            let mut result: int16x8_t;
            let a: float16x8_t = std::mem::transmute(self);
            std::arch::asm!(
                "fcvtzu {0:v}.8h, {1:v}.8h",
                out(vreg) result,
                in(vreg) a,
            );
            std::mem::transmute(result)
        }
        #[cfg(not(target_feature = "fp16"))]
        unsafe {
            use crate::convertion::Convertor;
            let mut res = [0u16; 8];
            for i in 0..8 {
                res[i] = self.0[i].to_u16();
            }
            std::mem::transmute(res)
        }
    }
    #[inline(always)]
    fn to_f16(self) -> f16x8 {
        self
    }
}
