#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::arch_simd::sleef::arch::helper_aarch64 as helper;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use crate::arch_simd::sleef::arch::helper_avx2 as helper;
#[cfg(all(target_arch = "x86_64", target_feature = "sse", not(target_feature = "avx2")))]
use crate::arch_simd::sleef::arch::helper_sse as helper;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::simd::sleef::arch::helper_aarch64::{ visnan_vo_vf, vneg_vf_vf };
use crate::simd::sleef::libm::sleefsimdsp::{ xceilf, xcopysignf, xfloorf };
use crate::type_promote::{ Eval2, FloatOutBinary2, NormalOut2, NormalOutUnary2 };

use crate::arch_simd::sleef::libm::sleefsimdsp::{
    xacosf_u1,
    xacoshf,
    xasinf_u1,
    xasinhf,
    xatan2f_u1,
    xatanf_u1,
    xatanhf,
    xcbrtf_u1,
    xcosf_u1,
    xcoshf,
    xerff_u1,
    xexp10f,
    xexp2f,
    xexpf,
    xexpm1f,
    xhypotf_u05,
    xlog10f,
    xlog1pf,
    xlog2f,
    xlogf_u1,
    xmaxf,
    xminf,
    xpowf,
    xroundf,
    xsincosf_u1,
    xsinf_u1,
    xsinhf,
    xsqrtf_u05,
    xtanf_u1,
    xtanhf,
    xtruncf,
};
use crate::convertion::VecConvertor;
use crate::traits::{ SimdCompare, SimdMath, SimdSelect, VecTrait };
use crate::vectors::arch_simd::_128bit::u32x4::u32x4;
use helper::vabs_vf_vf;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::i32x4::i32x4;

/// a vector of 4 f32 values
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct f32x4(
    #[cfg(target_arch = "x86_64")] pub(crate) __m128,
    #[cfg(target_arch = "aarch64")] pub(crate) float32x4_t,
);

#[allow(non_camel_case_types)]
pub(crate) type f32_promote = f32x4;

impl PartialEq for f32x4 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let cmp = _mm_cmpeq_ps(self.0, other.0);
            _mm_movemask_ps(cmp) == -1
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let cmp = vceqq_f32(self.0, other.0);
            vminvq_u32(cmp) == 0xffffffff_u32
        }
    }
}

impl Default for f32x4 {
    #[inline(always)]
    fn default() -> Self {
        #[cfg(target_arch = "x86_64")]
        return unsafe {
            f32x4(_mm_setzero_ps())
        };
        #[cfg(target_arch = "aarch64")]
        return unsafe {
            f32x4(vdupq_n_f32(0.0))
        };
    }
}

impl VecTrait<f32> for f32x4 {
    const SIZE: usize = 4;
    type Base = f32;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[f32]) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            assert_eq!(slice.len(), 4);
            _mm_storeu_ps(&mut self.0 as *mut _ as *mut f32, _mm_loadu_ps(slice.as_ptr()));
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            vst1q_f32(self.as_mut_ptr(), vld1q_f32(slice.as_ptr()));
        }
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(all(target_arch = "x86_64", not(target_feature = "fma")))]
        unsafe {
            f32x4(_mm_add_ps(_mm_mul_ps(self.0, a.0), b.0))
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "fma"))]
        unsafe {
            f32x4(_mm_fmadd_ps(self.0, a.0, b.0))
        }
        #[cfg(all(target_arch = "aarch64", not(target_feature = "fma")))]
        unsafe {
            f32x4(vmlaq_f32(b.0, self.0, a.0))
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "fma"))]
        unsafe {
            f32x4(vfmaq_f32(b.0, self.0, a.0))
        }
    }
    #[inline(always)]
    fn sum(&self) -> f32 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let sum = _mm_hadd_ps(self.0, self.0);
            _mm_cvtss_f32(_mm_hadd_ps(sum, sum))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            vaddvq_f32(self.0)
        }
    }
    #[inline(always)]
    fn splat(val: f32) -> f32x4 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            f32x4(_mm_set1_ps(val))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            f32x4(vdupq_n_f32(val))
        }
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const f32) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            f32x4(_mm_loadu_ps(ptr))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            f32x4(vld1q_f32(ptr))
        }
    }
}

impl f32x4 {
    #[allow(unused)]
    #[inline(always)]
    fn as_array(&self) -> [f32; 4] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl SimdCompare for f32x4 {
    type SimdMask = i32x4;
    #[inline(always)]
    fn simd_eq(self, rhs: Self) -> Self::SimdMask {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i32x4(_mm_castps_si128(_mm_cmpeq_ps(self.0, rhs.0)))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i32x4(vreinterpretq_s32_u32(vceqq_f32(self.0, rhs.0)))
        }
    }
    #[inline(always)]
    fn simd_ne(self, rhs: Self) -> Self::SimdMask {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i32x4(_mm_castps_si128(_mm_cmpneq_ps(self.0, rhs.0)))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i32x4(vreinterpretq_s32_u32(vmvnq_u32(vceqq_f32(self.0, rhs.0))))
        }
    }
    #[inline(always)]
    fn simd_lt(self, rhs: Self) -> Self::SimdMask {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i32x4(_mm_castps_si128(_mm_cmplt_ps(self.0, rhs.0)))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i32x4(vreinterpretq_s32_u32(vcltq_f32(self.0, rhs.0)))
        }
    }
    #[inline(always)]
    fn simd_le(self, rhs: Self) -> Self::SimdMask {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i32x4(_mm_castps_si128(_mm_cmple_ps(self.0, rhs.0)))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i32x4(vreinterpretq_s32_u32(vcleq_f32(self.0, rhs.0)))
        }
    }
    #[inline(always)]
    fn simd_gt(self, rhs: Self) -> Self::SimdMask {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i32x4(_mm_castps_si128(_mm_cmpgt_ps(self.0, rhs.0)))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i32x4(vreinterpretq_s32_u32(vcgtq_f32(self.0, rhs.0)))
        }
    }
    #[inline(always)]
    fn simd_ge(self, rhs: Self) -> Self::SimdMask {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i32x4(_mm_castps_si128(_mm_cmpge_ps(self.0, rhs.0)))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i32x4(vreinterpretq_s32_u32(vcgeq_f32(self.0, rhs.0)))
        }
    }
}

impl SimdSelect<f32x4> for i32x4 {
    #[inline(always)]
    fn select(&self, true_val: f32x4, false_val: f32x4) -> f32x4 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            f32x4(_mm_blendv_ps(false_val.0, true_val.0, std::mem::transmute(self.0)))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            f32x4(vbslq_f32(vreinterpretq_u32_s32(self.0), true_val.0, false_val.0))
        }
    }
}

impl std::ops::Add for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            f32x4(_mm_add_ps(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            f32x4(vaddq_f32(self.0, rhs.0))
        }
    }
}

impl std::ops::Sub for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            f32x4(_mm_sub_ps(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            f32x4(vsubq_f32(self.0, rhs.0))
        }
    }
}

impl std::ops::Mul for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            f32x4(_mm_mul_ps(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            f32x4(vmulq_f32(self.0, rhs.0))
        }
    }
}

impl std::ops::Div for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            f32x4(_mm_div_ps(self.0, rhs.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            f32x4(vdivq_f32(self.0, rhs.0))
        }
    }
}

impl std::ops::Rem for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let a: [f32; 4] = std::mem::transmute(self.0);
            let b: [f32; 4] = std::mem::transmute(rhs.0);
            let c: [f32; 4] = [a[0] % b[0], a[1] % b[1], a[2] % b[2], a[3] % b[3]];
            f32x4(std::mem::transmute(c))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let div = vdivq_f32(self.0, rhs.0);
            let truncated = vrndq_f32(div);
            f32x4(vsubq_f32(self.0, vmulq_f32(rhs.0, truncated)))
        }
    }
}
impl std::ops::Neg for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            f32x4(_mm_xor_ps(self.0, _mm_set1_ps(-0.0)))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            f32x4(vnegq_f32(self.0))
        }
    }
}

impl SimdMath<f32> for f32x4 {
    #[inline(always)]
    fn sin(self) -> Self {
        f32x4(unsafe { xsinf_u1(self.0) })
    }
    #[inline(always)]
    fn cos(self) -> Self {
        f32x4(unsafe { xcosf_u1(self.0) })
    }
    #[inline(always)]
    fn tan(self) -> Self {
        f32x4(unsafe { xtanf_u1(self.0) })
    }
    #[inline(always)]
    fn square(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            f32x4(_mm_mul_ps(self.0, self.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            f32x4(vmulq_f32(self.0, self.0))
        }
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        f32x4(unsafe { xsqrtf_u05(self.0) })
    }

    #[inline(always)]
    fn abs(self) -> Self {
        f32x4(unsafe { vabs_vf_vf(self.0) })
    }

    #[inline(always)]
    fn floor(self) -> Self {
        f32x4(unsafe { xfloorf(self.0) })
    }

    #[inline(always)]
    fn ceil(self) -> Self {
        f32x4(unsafe { xceilf(self.0) })
    }

    #[inline(always)]
    fn neg(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            f32x4(_mm_sub_ps(_mm_setzero_ps(), self.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            f32x4(vneg_vf_vf(self.0))
        }
    }

    #[inline(always)]
    fn round(self) -> Self {
        f32x4(unsafe { xroundf(self.0) })
    }

    #[inline(always)]
    fn signum(self) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let zero = _mm_setzero_ps();
            let one = _mm_set1_ps(1.0);
            let neg_one = _mm_set1_ps(-1.0);
            let is_positive = _mm_cmpgt_ps(self.0, zero);
            let is_negative = _mm_cmplt_ps(self.0, zero);
            let pos_result = _mm_and_ps(is_positive, one);
            let neg_result = _mm_and_ps(is_negative, neg_one);
            f32x4(_mm_or_ps(pos_result, neg_result))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let zero = vdupq_n_f32(0.0);
            let one = vdupq_n_f32(1.0);
            let neg_one = vdupq_n_f32(-1.0);
            let is_positive = vcgtq_f32(self.0, zero);
            let is_negative = vcltq_f32(self.0, zero);
            let pos_result = vreinterpretq_f32_u32(
                vandq_u32(vreinterpretq_u32_f32(one), is_positive)
            );
            let neg_result = vreinterpretq_f32_u32(
                vandq_u32(vreinterpretq_u32_f32(neg_one), is_negative)
            );
            f32x4(
                vreinterpretq_f32_u32(
                    vorrq_u32(vreinterpretq_u32_f32(pos_result), vreinterpretq_u32_f32(neg_result))
                )
            )
        }
    }

    #[inline(always)]
    fn copysign(self, rhs: Self) -> Self {
        f32x4(unsafe { xcopysignf(self.0, rhs.0) })
    }

    #[inline(always)]
    fn leaky_relu(self, alpha: Self) -> Self {
        let mask = self.simd_gt(Self::splat(0.0));
        mask.select(self, self * alpha)
    }

    #[inline(always)]
    fn relu(self) -> Self {
        self.max(Self::splat(0.0))
    }

    #[inline(always)]
    fn relu6(self) -> Self {
        self.max(Self::splat(0.0)).min(Self::splat(6.0))
    }

    #[inline(always)]
    fn pow(self, exp: Self) -> Self {
        f32x4(unsafe { xpowf(self.0, exp.0) })
    }

    #[inline(always)]
    fn asin(self) -> Self {
        f32x4(unsafe { xasinf_u1(self.0) })
    }

    #[inline(always)]
    fn acos(self) -> Self {
        f32x4(unsafe { xacosf_u1(self.0) })
    }

    #[inline(always)]
    fn atan(self) -> Self {
        f32x4(unsafe { xatanf_u1(self.0) })
    }

    #[inline(always)]
    fn sinh(self) -> Self {
        f32x4(unsafe { xsinhf(self.0) })
    }

    #[inline(always)]
    fn cosh(self) -> Self {
        f32x4(unsafe { xcoshf(self.0) })
    }

    #[inline(always)]
    fn tanh(self) -> Self {
        f32x4(unsafe { xtanhf(self.0) })
    }

    #[inline(always)]
    fn asinh(self) -> Self {
        f32x4(unsafe { xasinhf(self.0) })
    }

    #[inline(always)]
    fn acosh(self) -> Self {
        f32x4(unsafe { xacoshf(self.0) })
    }

    #[inline(always)]
    fn atanh(self) -> Self {
        f32x4(unsafe { xatanhf(self.0) })
    }

    #[inline(always)]
    fn exp2(self) -> Self {
        f32x4(unsafe { xexp2f(self.0) })
    }

    #[inline(always)]
    fn exp10(self) -> Self {
        f32x4(unsafe { xexp10f(self.0) })
    }

    #[inline(always)]
    fn expm1(self) -> Self {
        f32x4(unsafe { xexpm1f(self.0) })
    }

    #[inline(always)]
    fn log10(self) -> Self {
        f32x4(unsafe { xlog10f(self.0) })
    }

    #[inline(always)]
    fn log2(self) -> Self {
        f32x4(unsafe { xlog2f(self.0) })
    }

    #[inline(always)]
    fn log1p(self) -> Self {
        f32x4(unsafe { xlog1pf(self.0) })
    }

    #[inline(always)]
    fn hypot(self, other: Self) -> Self {
        f32x4(unsafe { xhypotf_u05(self.0, other.0) })
    }

    #[inline(always)]
    fn trunc(self) -> Self {
        f32x4(unsafe { xtruncf(self.0) })
    }

    #[inline(always)]
    fn erf(self) -> Self {
        f32x4(unsafe { xerff_u1(self.0) })
    }

    #[inline(always)]
    fn cbrt(self) -> Self {
        f32x4(unsafe { xcbrtf_u1(self.0) })
    }

    #[inline(always)]
    fn exp(self) -> Self {
        f32x4(unsafe { xexpf(self.0) })
    }

    #[inline(always)]
    fn ln(self) -> Self {
        f32x4(unsafe { xlogf_u1(self.0) })
    }

    #[inline(always)]
    fn log(self) -> Self {
        f32x4(unsafe { xlogf_u1(self.0) })
    }

    #[inline(always)]
    fn sincos(self) -> (Self, Self) {
        let ret = unsafe { xsincosf_u1(self.0) };
        (f32x4(ret.x), f32x4(ret.y))
    }

    #[inline(always)]
    fn atan2(self, other: Self) -> Self {
        f32x4(unsafe { xatan2f_u1(self.0, other.0) })
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        f32x4(unsafe { xminf(self.0, other.0) })
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        f32x4(unsafe { xmaxf(self.0, other.0) })
    }

    #[inline(always)]
    fn recip(self) -> Self {
        Self::splat(1.0) / self
    }

    #[inline(always)]
    fn hard_sigmoid(self) -> Self {
        let sixth = Self::splat(1.0 / 6.0);
        let half = Self::splat(0.5);
        let one = Self::splat(1.0);
        let zero = Self::splat(0.0);
        let result = self * sixth + half;
        result.min(one).max(zero)
    }

    #[inline(always)]
    fn elu(self, alpha: Self) -> Self {
        let mask = self.simd_gt(Self::splat(0.0));
        mask.select(self, alpha * self.expm1())
    }

    #[inline(always)]
    fn selu(self, alpha: Self, scale: Self) -> Self {
        scale * self.elu(alpha)
    }

    #[inline(always)]
    fn celu(self, scale: Self) -> Self {
        let gt_mask = self.simd_gt(Self::splat(0.0));
        gt_mask.select(self, scale * (self.exp() - Self::splat(1.0)))
    }

    #[inline(always)]
    fn gelu(self) -> Self {
        let erf = (self * Self::splat(std::f32::consts::FRAC_1_SQRT_2)).erf() + Self::splat(1.0);
        let half = Self::splat(0.5);
        half * self * erf
    }

    #[inline(always)]
    fn hard_swish(self) -> Self {
        let three = Self::splat(3.0);
        self * (self + three).relu6() * Self::splat(1.0 / 6.0)
    }

    #[inline(always)]
    fn mish(self) -> Self {
        self * self.softplus().tanh()
    }

    #[inline(always)]
    fn softplus(self) -> Self {
        let one = Self::splat(1.0);
        (one + self.exp()).ln()
    }

    #[inline(always)]
    fn sigmoid(self) -> Self {
        Self::splat(1.0) / (Self::splat(1.0) + (-self).exp())
    }
    #[inline(always)]
    fn softsign(self) -> Self {
        self / (Self::splat(1.0) + self.abs())
    }
}

impl VecConvertor for f32x4 {
    #[inline(always)]
    fn to_u32(self) -> super::u32x4::u32x4 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            u32x4(_mm_cvtps_epi32(self.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            u32x4(vcvtq_u32_f32(self.0))
        }
    }
    #[inline(always)]
    fn to_i32(self) -> super::i32x4::i32x4 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i32x4(_mm_cvtps_epi32(self.0))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i32x4(vcvtq_s32_f32(self.0))
        }
    }
    #[cfg(target_pointer_width = "32")]
    #[inline(always)]
    fn to_isize(self) -> super::isizex2::isizex2 {
        self.to_i32().to_isize()
    }
    #[cfg(target_pointer_width = "32")]
    #[inline(always)]
    fn to_usize(self) -> super::usizex2::usizex2 {
        self.to_u32().to_usize()
    }
    #[inline(always)]
    fn to_f32(self) -> f32x4 {
        self
    }
}

impl FloatOutBinary2 for f32x4 {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn __log(self, base: Self) -> Self {
        let res = [
            self[0].log(base[0]),
            self[1].log(base[1]),
            self[2].log(base[2]),
            self[3].log(base[3]),
        ];
        f32x4(unsafe { std::mem::transmute(res) })
    }
}

impl NormalOut2 for f32x4 {
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
        self.pow(rhs)
    }

    #[inline(always)]
    fn __rem(self, rhs: Self) -> Self {
        self % rhs
    }

    #[inline(always)]
    fn __max(self, rhs: Self) -> Self {
        self.max(rhs)
    }

    #[inline(always)]
    fn __min(self, rhs: Self) -> Self {
        self.min(rhs)
    }

    #[inline(always)]
    fn __clamp(self, min: Self, max: Self) -> Self {
        self.max(min).min(max)
    }
}

impl NormalOutUnary2 for f32x4 {
    #[inline(always)]
    fn __square(self) -> Self {
        self * self
    }

    #[inline(always)]
    fn __abs(self) -> Self {
        self.abs()
    }

    #[inline(always)]
    fn __ceil(self) -> Self {
        self.ceil()
    }

    #[inline(always)]
    fn __floor(self) -> Self {
        self.floor()
    }

    #[inline(always)]
    fn __neg(self) -> Self {
        -self
    }

    #[inline(always)]
    fn __round(self) -> Self {
        self.round()
    }

    #[inline(always)]
    fn __signum(self) -> Self {
        self.signum()
    }

    #[inline(always)]
    fn __leaky_relu(self, alpha: Self) -> Self {
        self.max(f32x4::splat(0.0)) + alpha * self.min(f32x4::splat(0.0))
    }

    #[inline(always)]
    fn __relu(self) -> Self {
        self.relu()
    }

    #[inline(always)]
    fn __relu6(self) -> Self {
        self.relu6()
    }
}

impl Eval2 for f32x4 {
    type Output = i32x4;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            i32x4(std::mem::transmute(_mm_cmpunord_ps(self.0, self.0)))
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            i32x4(std::mem::transmute(visnan_vo_vf(self.0)))
        }
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        self.simd_ne(f32x4::default())
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        let i: i32x4 = unsafe { std::mem::transmute(self.0) };
        let sign_mask = i32x4::splat(-0x8000_0000i32);
        let inf_mask = i32x4::splat(0x7f80_0000i32);
        let frac_mask = i32x4::splat(0x007f_ffffi32);

        let exp = i & inf_mask;
        let frac = i & frac_mask;
        let is_inf = exp.simd_eq(inf_mask) & frac.simd_eq(i32x4::splat(0));
        let is_neg = (i & sign_mask).simd_ne(i32x4::splat(0));

        is_inf.select(is_neg.select(i32x4::splat(-1), i32x4::splat(1)), i32x4::splat(0))
    }
}
