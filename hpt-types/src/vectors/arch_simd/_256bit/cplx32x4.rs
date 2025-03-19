use num_complex::Complex32;

use crate::{
    convertion::VecConvertor,
    traits::SimdMath,
    type_promote::{FloatOutBinary2, NormalOut2, NormalOutUnary2},
    vectors::traits::VecTrait,
};

/// a vector of 4 cplx32 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(32))]
pub struct cplx32x4(pub(crate) [Complex32; 4]);

/// helper to impl the promote trait
#[allow(non_camel_case_types)]
pub(crate) type Complex32_promote = cplx32x4;

impl VecTrait<Complex32> for cplx32x4 {
    const SIZE: usize = 4;
    type Base = Complex32;
    #[inline(always)]
    fn mul_add(mut self, a: Self, b: Self) -> Self {
        self.0[0] = self.0[0] * a.0[0] + b.0[0];
        self.0[1] = self.0[1] * a.0[1] + b.0[1];
        self.0[2] = self.0[2] * a.0[2] + b.0[2];
        self.0[3] = self.0[3] * a.0[3] + b.0[3];
        self
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
    #[inline(always)]
    fn splat(val: Complex32) -> cplx32x4 {
        cplx32x4([val; 4])
    }
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const Complex32) -> Self {
        cplx32x4([
            ptr.read_unaligned(),
            ptr.add(1).read_unaligned(),
            ptr.add(2).read_unaligned(),
            ptr.add(3).read_unaligned(),
        ])
    }
}

impl std::ops::Add for cplx32x4 {
    type Output = Self;
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        let mut ret = cplx32x4::default();
        for i in 0..4 {
            ret.0[i] = self.0[i] % rhs.0[i];
        }
        ret
    }
}

impl VecConvertor for cplx32x4 {
    #[inline(always)]
    fn to_complex32(self) -> cplx32x4 {
        self
    }
}

impl SimdMath<Complex32> for cplx32x4 {}

impl FloatOutBinary2 for cplx32x4 {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn __log(self, base: Self) -> Self {
        let res = [
            self[0].__log(base[0]),
            self[1].__log(base[1]),
            self[2].__log(base[2]),
            self[3].__log(base[3]),
        ];
        cplx32x4(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __hypot(self, _: Self) -> Self {
        panic!("Hypot operation is not supported for cplx32x4");
    }

    #[inline(always)]
    fn __pow(self, rhs: Self) -> Self {
        let res = [
            self[0].__pow(rhs[0]),
            self[1].__pow(rhs[1]),
            self[2].__pow(rhs[2]),
            self[3].__pow(rhs[3]),
        ];
        cplx32x4(unsafe { std::mem::transmute(res) })
    }
}

impl NormalOut2 for cplx32x4 {
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
        let res = [
            self[0].__max(rhs[0]),
            self[1].__max(rhs[1]),
            self[2].__max(rhs[2]),
            self[3].__max(rhs[3]),
        ];
        cplx32x4(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __min(self, rhs: Self) -> Self {
        let res = [
            self[0].__min(rhs[0]),
            self[1].__min(rhs[1]),
            self[2].__min(rhs[2]),
            self[3].__min(rhs[3]),
        ];
        cplx32x4(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __clamp(self, min: Self, max: Self) -> Self {
        let res = [
            self[0].__clamp(min[0], max[0]),
            self[1].__clamp(min[1], max[1]),
            self[2].__clamp(min[2], max[2]),
            self[3].__clamp(min[3], max[3]),
        ];
        cplx32x4(unsafe { std::mem::transmute(res) })
    }
}

impl NormalOutUnary2 for cplx32x4 {
    #[inline(always)]
    fn __square(self) -> Self {
        self * self
    }

    #[inline(always)]
    fn __abs(self) -> Self {
        let res = [
            self[0].__abs(),
            self[1].__abs(),
            self[2].__abs(),
            self[3].__abs(),
        ];
        cplx32x4(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __ceil(self) -> Self {
        let res = [
            self[0].__ceil(),
            self[1].__ceil(),
            self[2].__ceil(),
            self[3].__ceil(),
        ];
        cplx32x4(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __floor(self) -> Self {
        let res = [
            self[0].__floor(),
            self[1].__floor(),
            self[2].__floor(),
            self[3].__floor(),
        ];
        cplx32x4(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __neg(self) -> Self {
        -self
    }

    #[inline(always)]
    fn __round(self) -> Self {
        let res = [
            self[0].__round(),
            self[1].__round(),
            self[2].__round(),
            self[3].__round(),
        ];
        cplx32x4(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __signum(self) -> Self {
        let res = [
            self[0].__signum(),
            self[1].__signum(),
            self[2].__signum(),
            self[3].__signum(),
        ];
        cplx32x4(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __leaky_relu(self, _: Self) -> Self {
        unreachable!()
    }

    #[inline(always)]
    fn __relu(self) -> Self {
        let res = [
            self[0].__relu(),
            self[1].__relu(),
            self[2].__relu(),
            self[3].__relu(),
        ];
        cplx32x4(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __relu6(self) -> Self {
        let res = [
            self[0].__relu6(),
            self[1].__relu6(),
            self[2].__relu6(),
            self[3].__relu6(),
        ];
        cplx32x4(unsafe { std::mem::transmute(res) })
    }

    #[inline(always)]
    fn __trunc(self) -> Self {
        let res = [
            self[0].__trunc(),
            self[1].__trunc(),
            self[2].__trunc(),
            self[3].__trunc(),
        ];
        cplx32x4(unsafe { std::mem::transmute(res) })
    }
    #[inline(always)]
    fn __copysign(self, _: Self) -> Self {
        panic!("Copysign operation is not supported for complex type")
    }
}
