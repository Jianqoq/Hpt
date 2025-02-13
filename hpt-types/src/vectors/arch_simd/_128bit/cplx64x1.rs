use crate::{
    convertion::VecConvertor,
    traits::SimdMath,
    type_promote::{FloatOutBinary2, NormalOut2, NormalOutUnary, NormalOutUnary2},
    vectors::traits::VecTrait,
};
use num_complex::Complex64;

/// a vector of 1 Complex64 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(transparent)]
pub struct cplx64x1(pub(crate) [Complex64; 1]);

#[allow(non_camel_case_types)]
pub(crate) type Complex64_promote = cplx64x1;

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
    #[inline(always)]
    fn splat(val: Complex64) -> cplx64x1 {
        cplx64x1([val; 1])
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const Complex64) -> Self {
        cplx64x1([unsafe { *ptr }])
    }
}

impl cplx64x1 {
    #[allow(unused)]
    #[inline(always)]
    fn as_array(&self) -> [Complex64; 1] {
        unsafe { std::mem::transmute(self.0) }
    }
}

impl std::ops::Add for cplx64x1 {
    type Output = Self;
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        let mut ret = cplx64x1::default();
        for i in 0..1 {
            ret.0[i] = self.0[i] % rhs.0[i];
        }
        ret
    }
}

impl VecConvertor for cplx64x1 {
    #[inline(always)]
    fn to_complex64(self) -> cplx64x1 {
        self
    }
}
impl SimdMath<Complex64> for cplx64x1 {}
impl FloatOutBinary2 for cplx64x1 {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn __log(self, base: Self) -> Self {
        let res = [self[0].__log(base[0])];
        cplx64x1(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __hypot(self, rhs: Self) -> Self {
        let res = [self[0].__hypot(rhs[0])];
        cplx64x1(unsafe { std::mem::transmute(res) })
    }
}

impl NormalOut2 for cplx64x1 {
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
        let res = [self[0].__pow(rhs[0])];
        cplx64x1(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __rem(self, rhs: Self) -> Self {
        self % rhs
    }

    #[inline(always)]
    fn __max(self, rhs: Self) -> Self {
        let res = [self[0].__max(rhs[0])];
        cplx64x1(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __min(self, rhs: Self) -> Self {
        let res = [self[0].__min(rhs[0])];
        cplx64x1(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __clamp(self, min: Self, max: Self) -> Self {
        let res = [self[0].__clamp(min[0], max[0])];
        cplx64x1(unsafe { std::mem::transmute(res) })
    }
}

impl NormalOutUnary2 for cplx64x1 {
    #[inline(always)]
    fn __square(self) -> Self {
        self * self
    }

    #[inline(always)]
    fn __abs(self) -> Self {
        let res = [self[0].__abs()];
        cplx64x1(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __ceil(self) -> Self {
        let res = [self[0].__ceil()];
        cplx64x1(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __floor(self) -> Self {
        let res = [self[0].__floor()];
        cplx64x1(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __neg(self) -> Self {
        -self
    }

    #[inline(always)]
    fn __round(self) -> Self {
        let res = [self[0].__round()];
        cplx64x1(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __signum(self) -> Self {
        let res = [self[0].__signum()];
        cplx64x1(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __trunc(self) -> Self {
        Self([self[0]._trunc()])
    }

    #[inline(always)]
    fn __leaky_relu(self, _: Self) -> Self {
        unreachable!()
    }

    #[inline(always)]
    fn __relu(self) -> Self {
        let res = [self[0].__relu()];
        cplx64x1(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __relu6(self) -> Self {
        let res = [self[0].__relu6()];
        cplx64x1(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __copysign(self, rhs: Self) -> Self {
        let res = [self[0].__copysign(rhs[0])];
        cplx64x1(unsafe { std::mem::transmute(res) })
    }
}
