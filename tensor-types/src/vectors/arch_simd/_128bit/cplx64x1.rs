
use num_complex::Complex64;
use crate::{convertion::VecConvertor, vectors::traits::VecTrait};

/// a vector of 1 Complex64 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(transparent)]
pub struct cplx64x1(pub(crate) [Complex64; 1]);

impl VecTrait<Complex64> for cplx64x1 {
    const SIZE: usize = 1;
    type Base = Complex64;
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[Complex64]) {
        self.0.copy_from_slice(slice);
    }
    #[inline(always)]
    fn mul_add(mut self, a: Self, b: Self) -> Self {
        self.0[0] = self.0[0] * a.0[0] + b.0[0];
        self
    }
    #[inline(always)]
    fn sum(&self) -> Complex64 {
        self.0.iter().sum()
    }
    fn splat(val: Complex64) -> cplx64x1 {
        cplx64x1([val; 1])
    }
}

impl cplx64x1 {
    #[allow(unused)]
    fn as_array(&self) -> [Complex64; 1] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl std::ops::Add for cplx64x1 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = cplx64x1::default();
        for i in 0..1 {
            ret.0[i] = self.0[i] + rhs.0[i];
        }
        ret
    }
}
impl std::ops::Sub for cplx64x1 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut ret = cplx64x1::default();
        for i in 0..1 {
            ret.0[i] = self.0[i] - rhs.0[i];
        }
        ret
    }
}
impl std::ops::Mul for cplx64x1 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret = cplx64x1::default();
        for i in 0..1 {
            ret.0[i] = self.0[i] * rhs.0[i];
        }
        ret
    }
}
impl std::ops::Div for cplx64x1 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut ret = cplx64x1::default();
        for i in 0..1 {
            ret.0[i] = self.0[i] / rhs.0[i];
        }
        ret
    }
}

impl std::ops::Neg for cplx64x1 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut ret = cplx64x1::default();
        for i in 0..1 {
            ret.0[i] = -self.0[i];
        }
        ret
    }
}
impl std::ops::Rem for cplx64x1 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        let mut ret = cplx64x1::default();
        for i in 0..1 {
            ret.0[i] = self.0[i] % rhs.0[i];
        }
        ret
    }
}

impl VecConvertor for cplx64x1 {
}
