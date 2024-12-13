use num_complex::Complex32;

use crate::{convertion::VecConvertor, vectors::traits::VecTrait};

/// a vector of 4 cplx32 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(32))]
pub struct cplx32x4(pub(crate) [Complex32; 4]);

impl VecTrait<Complex32> for cplx32x4 {
    const SIZE: usize = 4;
    type Base = Complex32;
    #[inline(always)]
    fn mul_add(self, _: Self, _: Self) -> Self {
        todo!()
    }
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[Complex32]) {
        self.0.copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const Complex32 {
        self.0.as_ptr()
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
    fn splat(val: Complex32) -> cplx32x4 {
        cplx32x4([val; 4])
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

impl std::ops::Neg for cplx32x4 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut ret = cplx32x4::default();
        for i in 0..4 {
            ret.0[i] = -self.0[i];
        }
        ret
    }
}

impl std::ops::Rem for cplx32x4 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        let mut ret = cplx32x4::default();
        for i in 0..4 {
            ret.0[i] = self.0[i] % rhs.0[i];
        }
        ret
    }
}

impl VecConvertor for cplx32x4 {
    fn to_complex32(self) -> cplx32x4 {
        self
    }
}
