use num_complex::Complex64;

use crate::vectors::traits::VecTrait;

/// a vector of 2 cplx64 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(32))]
pub struct cplx64x2(pub(crate) [Complex64; 2]);

impl VecTrait<Complex64> for cplx64x2 {
    const SIZE: usize = 2;
    type Base = Complex64;
    #[inline(always)]
    fn mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[Complex64]) {
        self.0.copy_from_slice(slice);
    }
    #[inline(always)]
    fn sum(&self) -> Complex64 {
        self.0.iter().sum()
    }
    fn splat(val: Complex64) -> cplx64x2 {
        cplx64x2([val; 2])
    }
}

impl std::ops::Add for cplx64x2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = cplx64x2::default();
        for i in 0..2 {
            ret.0[i] = self.0[i] + rhs.0[i];
        }
        ret
    }
}
impl std::ops::Sub for cplx64x2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut ret = cplx64x2::default();
        for i in 0..2 {
            ret.0[i] = self.0[i] - rhs.0[i];
        }
        ret
    }
}
impl std::ops::Mul for cplx64x2 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret = cplx64x2::default();
        for i in 0..2 {
            ret.0[i] = self.0[i] * rhs.0[i];
        }
        ret
    }
}
impl std::ops::Div for cplx64x2 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut ret = cplx64x2::default();
        for i in 0..2 {
            ret.0[i] = self.0[i] / rhs.0[i];
        }
        ret
    }
}

impl std::ops::Neg for cplx64x2 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut ret = cplx64x2::default();
        for i in 0..2 {
            ret.0[i] = -self.0[i];
        }
        ret
    }
}

impl std::ops::Rem for cplx64x2 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        let mut ret = cplx64x2::default();
        for i in 0..2 {
            ret.0[i] = self.0[i] % rhs.0[i];
        }
        ret
    }
}
