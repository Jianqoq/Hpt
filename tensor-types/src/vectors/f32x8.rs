use std::ops::{ Deref, DerefMut };
use std::simd::num::SimdFloat;
use crate::into_vec::IntoVec ;
use super::traits::{ Init, VecSize, VecTrait };

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq)]
pub struct f32x8(pub(crate) std::simd::f32x8);

impl Deref for f32x8 {
    type Target = std::simd::f32x8;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for f32x8 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl VecTrait<f32> for f32x8 {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[f32]) {
        self.as_mut_array().copy_from_slice(slice)
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const f32 {
        self.as_array().as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut f32 {
        self.as_mut_array().as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut f32 {
        self.as_array().as_ptr() as *mut _
    }
    #[inline(always)]
    fn sum(&self) -> f32 {
        self.reduce_sum()
    }
    
}
impl VecSize for f32x8 {
    const SIZE: usize = 8;
}
impl Init<f32> for f32x8 {
    fn splat(val: f32) -> f32x8 {
        f32x8(std::simd::f32x8::splat(val))
    }
    unsafe fn from_ptr(ptr: *const f32) -> Self {
        unsafe { std::mem::transmute(std::arch::x86_64::_mm256_load_ps(ptr as *const _)) }
    }
}
impl IntoVec<f32x8> for f32x8 {
    fn into_vec(self) -> f32x8 {
        self
    }
}

impl std::ops::Add for f32x8 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        f32x8(self.0 + rhs.0)
    }
}

impl std::ops::Sub for f32x8 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        f32x8(self.0 - rhs.0)
    }
}

impl std::ops::Mul for f32x8 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        f32x8(self.0 * rhs.0)
    }
}

impl std::ops::Div for f32x8 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        f32x8(self.0 / rhs.0)
    }
}

impl std::ops::Rem for f32x8 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        f32x8(self.0 % rhs.0)
    }
}
// impl FloatOut for f32x8 {
//     type Output = f32x8;

//     fn _div(self, rhs: Self) -> Self::Output {
//         f32x8(self.0 / rhs.0)
//     }

//     fn _exp(self) -> Self::Output {
//         f32x8(sleef::f32x::exp_u10(self.0))
//     }

//     fn _exp2(self) -> Self::Output {
//         f32x8(sleef::f32x::exp2_u35(self.0))
//     }

//     fn _ln(self) -> Self::Output {
//         f32x8(sleef::f32x::log_u35(self.0))
//     }

//     fn _log(self, _: Self) -> Self::Output {
//         todo!()
//     }

//     fn _celu(self, _: Self::Output) -> Self::Output {
//         todo!()
//     }

//     fn _log2(self) -> Self::Output {
//         f32x8(sleef::f32x::log2_u35(self.0))
//     }

//     fn _log10(self) -> Self::Output {
//         f32x8(sleef::f32x::log10_u10(self.0))
//     }

//     fn _sqrt(self) -> Self::Output {
//         f32x8(sleef::f32x::sqrt_u35(self.0))
//     }

//     fn _sin(self) -> Self::Output {
//         f32x8(sleef::f32x::sin_u35(self.0))
//     }

//     fn _cos(self) -> Self::Output {
//         f32x8(sleef::f32x::cos_u35(self.0))
//     }

//     fn _tan(self) -> Self::Output {
//         f32x8(sleef::f32x::tan_u35(self.0))
//     }

//     fn _asin(self) -> Self::Output {
//         f32x8(sleef::f32x::asin_u35(self.0))
//     }

//     fn _acos(self) -> Self::Output {
//         f32x8(sleef::f32x::acos_u35(self.0))
//     }

//     fn _atan(self) -> Self::Output {
//         f32x8(sleef::f32x::atan_u35(self.0))
//     }

//     fn _sinh(self) -> Self::Output {
//         f32x8(sleef::f32x::sinh_u35(self.0))
//     }

//     fn _cosh(self) -> Self::Output {
//         f32x8(sleef::f32x::cosh_u35(self.0))
//     }

//     fn _tanh(self) -> Self::Output {
//         f32x8(sleef::f32x::tanh_u35(self.0))
//     }

//     fn _asinh(self) -> Self::Output {
//         f32x8(sleef::f32x::asinh_u10(self.0))
//     }

//     fn _acosh(self) -> Self::Output {
//         f32x8(sleef::f32x::acosh_u10(self.0))
//     }

//     fn _atanh(self) -> Self::Output {
//         f32x8(sleef::f32x::atanh_u10(self.0))
//     }

//     fn _recip(self) -> Self::Output {
//         f32x8(self.recip())
//     }

//     fn _erf(self) -> Self::Output {
//         let mut res = self.clone();
//         res.0
//             .as_mut_array()
//             .iter_mut()
//             .for_each(|x| {
//                 *x = erf(*x as f64) as f32;
//             });
//         res
//     }

//     fn _sigmoid(self) -> Self::Output {
//         todo!()
//     }

//     fn _elu(self, _: Self::Output) -> Self::Output {
//         todo!()
//     }

//     fn _leaky_relu(self, _: Self::Output) -> Self::Output {
//         todo!()
//     }

//     fn _relu(self) -> Self::Output {
//         f32x8(self.0.max(std::simd::f32x8::splat(0.0)))
//     }

//     fn _gelu(self) -> Self::Output {
//         todo!()
//     }

//     fn _selu(self, alpha: Self::Output, scale: Self::Output) -> Self::Output {
//         let mask = self.0.simd_gt(std::simd::f32x8::splat(0.0));
//         f32x8(
//             mask.select(
//                 (scale * self).0,
//                 (scale * alpha).0 * (Sleef::exp(self.0) - std::simd::f32x8::splat(1.0))
//             )
//         )
//     }

//     fn _hard_sigmoid(self, _: Self::Output, _: Self::Output) -> Self::Output {
//         todo!()
//     }

//     fn _relu6(self) -> Self::Output {
//         todo!()
//     }

//     fn _hard_swish(self) -> Self::Output {
//         todo!()
//     }

//     fn _softplus(self) -> Self::Output {
//         todo!()
//     }

//     fn _softsign(self) -> Self::Output {
//         todo!()
//     }

//     fn _mish(self) -> Self::Output {
//         todo!()
//     }

//     fn _cbrt(self) -> Self::Output {
//         todo!()
//     }
// }
// impl NormalOut for f32x8 {
//     type Output = f32x8;
//     fn _add(self, rhs: Self) -> Self::Output {
//         self + rhs
//     }
//     fn _sub(self, rhs: Self) -> Self::Output {
//         self - rhs
//     }
//     fn _mul(self, rhs: Self) -> Self::Output {
//         self * rhs
//     }
//     fn _pow(self, rhs: Self) -> Self::Output {
//         f32x8(Sleef::pow(self.0, rhs.0))
//     }
//     fn _rem(self, rhs: Self) -> Self::Output {
//         f32x8(self.0 % rhs.0)
//     }
//     fn _square(self) -> Self::Output {
//         self * self
//     }
//     fn _abs(self) -> Self {
//         f32x8(Sleef::abs(self.0))
//     }
//     fn _ceil(self) -> Self::Output {
//         f32x8(Sleef::ceil(self.0))
//     }
//     fn _floor(self) -> Self::Output {
//         f32x8(Sleef::floor(self.0))
//     }
//     fn _sign(self) -> Self::Output {
//         todo!()
//     }
//     fn _max(self, rhs: Self) -> Self::Output {
//         f32x8(self.max(rhs.0))
//     }
//     fn _min(self, rhs: Self) -> Self::Output {
//         f32x8(self.min(rhs.0))
//     }
//     fn _clip(self, _: Self::Output, _: Self::Output) -> Self::Output {
//         todo!()
//     }
//     fn _round(self) -> Self::Output {
//         f32x8(Sleef::round(self.0))
//     }
// }
// impl Eval for f32x8 {
//     type Output = bool;

//     fn _is_nan(&self) -> Self::Output {
//         self.is_nan().all()
//     }

//     fn _is_true(&self) -> Self::Output {
//         self.0 != std::simd::f32x8::splat(0.0)
//     }

//     fn _is_inf(&self) -> Self::Output {
//         self.is_infinite().all()
//     }
// }
// impl Cmp for f32x8 {
//     fn _eq(self, rhs: Self) -> bool {
//         self.0 == rhs.0
//     }

//     fn _ne(self, rhs: Self) -> bool {
//         self.0 != rhs.0
//     }

//     fn _lt(self, rhs: Self) -> bool {
//         self.0 < rhs.0
//     }

//     fn _le(self, rhs: Self) -> bool {
//         self.0 <= rhs.0
//     }

//     fn _gt(self, rhs: Self) -> bool {
//         self.0 > rhs.0
//     }

//     fn _ge(self, rhs: Self) -> bool {
//         self.0 >= rhs.0
//     }
// }
