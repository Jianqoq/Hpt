use num_complex::Complex32;

use crate::{convertion::VecConvertor, traits::SimdMath, type_promote::{FloatOutBinary2, NormalOut2, NormalOutUnary2}, vectors::traits::VecTrait};

/// a vector of 2 Complex32 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(16))]
pub struct cplx32x2(pub(crate) [Complex32; 2]);

#[allow(non_camel_case_types)]
pub(crate) type Complex32_promote = cplx32x2;

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
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const Complex32) -> Self {
        let mut result = [Complex32::ZERO; 2];
        for i in 0..2 {
            result[i] = unsafe { *ptr.add(i) };
        }
        cplx32x2(result)
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

impl SimdMath<Complex32> for cplx32x2 {
}

impl FloatOutBinary2 for cplx32x2 {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn __log(self, base: Self) -> Self {
        let res = [
            self[0].__log(base[0]),
            self[1].__log(base[1]),
        ];
        cplx32x2(unsafe { std::mem::transmute(res) })
    }
}

impl NormalOut2 for cplx32x2 {
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
        let res = [
            self[0].__pow(rhs[0]),
            self[1].__pow(rhs[1]),
        ];
        cplx32x2(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __rem(self, rhs: Self) -> Self {
        self % rhs
    }

    #[inline(always)]
    fn __max(self, rhs: Self) -> Self {
        let res = [
            self[0].__max(rhs[0]),
            self[1].__max(rhs[1]),
        ];
        cplx32x2(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __min(self, rhs: Self) -> Self {
        let res = [
            self[0].__min(rhs[0]),
            self[1].__min(rhs[1]),
        ];
        cplx32x2(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __clamp(self, min: Self, max: Self) -> Self {
        let res = [
            self[0].__clamp(min[0], max[0]),
            self[1].__clamp(min[1], max[1]),
        ];
        cplx32x2(unsafe { std::mem::transmute(res) })
    }
}

impl NormalOutUnary2 for cplx32x2 {
    #[inline(always)]
    fn __square(self) -> Self {
        self * self
    }

    #[inline(always)]
    fn __abs(self) -> Self {
        let res = [self[0].__abs(), self[1].__abs()];
        cplx32x2(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __ceil(self) -> Self {
        let res = [self[0].__ceil(), self[1].__ceil()];
        cplx32x2(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __floor(self) -> Self {
        let res = [self[0].__floor(), self[1].__floor()];
        cplx32x2(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __neg(self) -> Self {
        -self
    }

    #[inline(always)]
    fn __round(self) -> Self {
        let res = [self[0].__round(), self[1].__round()];
        cplx32x2(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __signum(self) -> Self {
        let res = [self[0].__signum(), self[1].__signum()];
        cplx32x2(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __leaky_relu(self, _: Self) -> Self {
        unreachable!()
    }

    #[inline(always)]
    fn __relu(self) -> Self {
        let res = [self[0].__relu(), self[1].__relu()];
        cplx32x2(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __relu6(self) -> Self {
        let res = [self[0].__relu6(), self[1].__relu6()];
        cplx32x2(unsafe { std::mem::transmute(res) })
    }
}