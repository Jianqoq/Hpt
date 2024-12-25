use num_complex::Complex32;

use crate::{convertion::VecConvertor, vectors::traits::VecTrait};

/// a vector of 2 Complex32 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(16))]
pub struct cplx32x2(pub(crate) [Complex32; 2]);

impl VecTrait<Complex32> for cplx32x2 {
    const SIZE: usize = 2;
    type Base = Complex32;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[Complex32]) {
        self.0.copy_from_slice(slice);
    }
    #[inline(always)]
    fn mul_add(mut self, a: Self, b: Self) -> Self {
        self.0[0] = self.0[0] * a.0[0] + b.0[0];
        self.0[1] = self.0[1] * a.0[1] + b.0[1];
        self
    }
    #[inline(always)]
    fn sum(&self) -> Complex32 {
        self.0.iter().sum()
    }
    #[inline(always)]
    fn splat(val: Complex32) -> cplx32x2 {
        cplx32x2([val; 2])
    }
}

impl cplx32x2 {
    #[allow(unused)]
    #[inline(always)]
    fn as_array(&self) -> [Complex32; 2] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl std::ops::Add for cplx32x2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = cplx32x2::default();
        for i in 0..2 {
            ret.0[i] = self.0[i] + rhs.0[i];
        }
        ret
    }
}
impl std::ops::Sub for cplx32x2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        let mut ret = cplx32x2::default();
        for i in 0..2 {
            ret.0[i] = self.0[i] - rhs.0[i];
        }
        ret
    }
}
impl std::ops::Mul for cplx32x2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret = cplx32x2::default();
        for i in 0..2 {
            ret.0[i] = self.0[i] * rhs.0[i];
        }
        ret
    }
}
impl std::ops::Div for cplx32x2 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        let mut ret = cplx32x2::default();
        for i in 0..2 {
            ret.0[i] = self.0[i] / rhs.0[i];
        }
        ret
    }
}
impl std::ops::Neg for cplx32x2 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        let mut ret = cplx32x2::default();
        for i in 0..2 {
            ret.0[i] = -self.0[i];
        }
        ret
    }
}
impl std::ops::Rem for cplx32x2 {
    type Output = Self;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        let mut ret = cplx32x2::default();
        for i in 0..2 {
            ret.0[i] = self.0[i] % rhs.0[i];
        }
        ret
    }
}

impl VecConvertor for cplx32x2 {
    #[inline(always)]
    fn to_complex32(self) -> cplx32x2 {
        self
    }
}

