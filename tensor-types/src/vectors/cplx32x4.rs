use num_complex::Complex32;

use crate::into_vec::IntoVec;

use super::traits::{ Init, VecSize, VecTrait };

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq)]
pub struct cplx32x4(pub(crate) [Complex32; 4]);

impl VecTrait<Complex32> for cplx32x4 {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[Complex32]) {
        self.0.copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const Complex32 {
        self.0.as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut Complex32 {
        self.0.as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut Complex32 {
        self.0.as_ptr() as *mut _
    }
    #[inline(always)]
    fn sum(&self) -> Complex32 {
        self.0.iter().sum()
    }
}
impl VecSize for cplx32x4 {
    const SIZE: usize = 4;
}
impl Init<Complex32> for cplx32x4 {
    fn splat(val: Complex32) -> cplx32x4 {
        cplx32x4([val; 4])
    }

    unsafe fn from_ptr(ptr: *const Complex32) -> Self {
        unsafe { std::mem::transmute(std::arch::x86_64::_mm256_loadu_pd(ptr as *const _)) }
    }
}
impl IntoVec<cplx32x4> for cplx32x4 {
    fn into_vec(self) -> cplx32x4 {
        self
    }
}
impl std::ops::Add for cplx32x4 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = cplx32x4::default();
        for i in 0..4 {
            ret.0[i] = self.0[i] + rhs.0[i];
        }
        ret
    }
}
impl std::ops::Sub for cplx32x4 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut ret = cplx32x4::default();
        for i in 0..4 {
            ret.0[i] = self.0[i] - rhs.0[i];
        }
        ret
    }
}
impl std::ops::Mul for cplx32x4 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret = cplx32x4::default();
        for i in 0..4 {
            ret.0[i] = self.0[i] * rhs.0[i];
        }
        ret
    }
}
impl std::ops::Div for cplx32x4 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut ret = cplx32x4::default();
        for i in 0..4 {
            ret.0[i] = self.0[i] / rhs.0[i];
        }
        ret
    }
}