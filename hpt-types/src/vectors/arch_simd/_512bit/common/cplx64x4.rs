use num_complex::Complex64;

use crate::{
    convertion::VecConvertor,
    traits::SimdMath,
    type_promote::{FloatOutBinary2, NormalOut2, NormalOutUnary2},
    vectors::traits::VecTrait,
};
use crate::type_promote::NormalOut;

/// a vector of 4 cplx64 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(64))]
pub struct cplx64x4(pub(crate) [Complex64; 4]);

/// helper to impl the promote trait
#[allow(non_camel_case_types)]
pub(crate) type Complex64_promote = cplx64x4;

impl VecTrait<Complex64> for cplx64x4 {
    const SIZE: usize = 4;
    type Base = Complex64;
    #[inline(always)]
    fn mul_add(mut self, a: Self, b: Self) -> Self {
        self.0[0] = self.0[0] * a.0[0] + b.0[0];
        self.0[1] = self.0[1] * a.0[1] + b.0[1];
        self
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const Complex64 {
        self.0.as_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut Complex64 {
        self.0.as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut Complex64 {
        self.0.as_ptr() as *mut _
    }

    #[inline(always)]
    fn sum(&self) -> Complex64 {
        self.0.iter().sum()
    }
    #[inline(always)]
    fn splat(val: Complex64) -> cplx64x4 {
        cplx64x4([val; 4])
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const Complex64) -> Self {
        cplx64x4([ptr.read_unaligned(), ptr.add(1).read_unaligned(), ptr.add(2).read_unaligned(), ptr.add(3).read_unaligned()])
    }
}

impl std::ops::Add for cplx64x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = cplx64x4::default();
        for i in 0..4 {
            ret.0[i] = self.0[i] + rhs.0[i];
        }
        ret
    }
}
impl std::ops::Sub for cplx64x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        let mut ret = cplx64x4::default();
        for i in 0..4 {
            ret.0[i] = self.0[i] - rhs.0[i];
        }
        ret
    }
}
impl std::ops::Mul for cplx64x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret = cplx64x4::default();
        for i in 0..4 {
            ret.0[i] = self.0[i] * rhs.0[i];
        }
        ret
    }
}
impl std::ops::Div for cplx64x4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        let mut ret = cplx64x4::default();
        for i in 0..4 {
            ret.0[i] = self.0[i] / rhs.0[i];
        }
        ret
    }
}

impl std::ops::Neg for cplx64x4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        let mut ret = cplx64x4::default();
        for i in 0..4 {
            ret.0[i] = -self.0[i];
        }
        ret
    }
}

impl std::ops::Rem for cplx64x4 {
    type Output = Self;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        let mut ret = cplx64x4::default();
        for i in 0..4 {
            ret.0[i] = self.0[i] % rhs.0[i];
        }
        ret
    }
}

impl VecConvertor for cplx64x4 {
    #[inline(always)]
    fn to_complex64(self) -> cplx64x4 {
        self
    }
}

impl SimdMath<Complex64> for cplx64x4 {}

impl FloatOutBinary2 for cplx64x4 {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn __log(self, base: Self) -> Self {
        let res = [self[0].__log(base[0]), self[1].__log(base[1]), self[2].__log(base[2]), self[3].__log(base[3])];
        cplx64x4(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __hypot(self, _: Self) -> Self {
        panic!("Hypot operation is not supported for cplx64x2");
    }

    #[inline(always)]
    fn __pow(self, rhs: Self) -> Self {
        let res = [self[0].__pow(rhs[0]), self[1].__pow(rhs[1]), self[2].__pow(rhs[2]), self[3].__pow(rhs[3])];
        cplx64x4(unsafe { std::mem::transmute(res) })
    }
}

impl NormalOut2 for cplx64x4 {
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
    fn __rem(self, rhs: Self) -> Self {
        self % rhs
    }

    #[inline(always)]
    fn __max(self, rhs: Self) -> Self {
        let res = [self[0].__max(rhs[0]), self[1].__max(rhs[1]), self[2].__max(rhs[2]), self[3].__max(rhs[3])];
        cplx64x4(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __min(self, rhs: Self) -> Self {
        let res = [self[0].__min(rhs[0]), self[1].__min(rhs[1]), self[2]._min(rhs[2]), self[3]._min(rhs[3])];
        cplx64x4(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __clamp(self, min: Self, max: Self) -> Self {
        let res = [
            self[0].__clamp(min[0], max[0]),
            self[1].__clamp(min[1], max[1]),
            self[2].__clamp(min[2], max[2]),
            self[3].__clamp(min[3], max[3]),
        ];
        cplx64x4(unsafe { std::mem::transmute(res) })
    }
}

impl NormalOutUnary2 for cplx64x4 {
    #[inline(always)]
    fn __square(self) -> Self {
        self * self
    }

    #[inline(always)]
    fn __abs(self) -> Self {
        let res = [self[0].__abs(), self[1].__abs(), self[2].__abs(), self[3].__abs()];
        cplx64x4(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __ceil(self) -> Self {
        let res = [self[0].__ceil(), self[1].__ceil(), self[2].__ceil(), self[3].__ceil()];
        cplx64x4(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __floor(self) -> Self {
        let res = [self[0].__floor(), self[1].__floor(), self[2].__floor(), self[3].__floor()];
        cplx64x4(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __neg(self) -> Self {
        -self
    }

    #[inline(always)]
    fn __round(self) -> Self {
        let res = [self[0].__round(), self[1].__round(), self[2].__round(), self[3].__round()];
        cplx64x4(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __signum(self) -> Self {
        let res = [self[0].__signum(), self[1].__signum(), self[2].__signum(), self[3].__signum()];
        cplx64x4(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __leaky_relu(self, _: Self) -> Self {
        unreachable!()
    }

    #[inline(always)]
    fn __relu(self) -> Self {
        let res = [self[0].__relu(), self[1].__relu(), self[2].__relu(), self[3].__relu()];
        cplx64x4(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __relu6(self) -> Self {
        let res = [self[0].__relu6(), self[1].__relu6(), self[2].__relu6(), self[3].__relu6()];
        cplx64x4(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __trunc(self) -> Self {
        let res = [self[0].__trunc(), self[1].__trunc(), self[2].__trunc(), self[3].__trunc()];
        cplx64x4(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __copysign(self, _: Self) -> Self {
        panic!("Copysign operation is not supported for complex type")
    }
}
