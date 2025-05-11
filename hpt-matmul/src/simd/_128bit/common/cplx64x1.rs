
use num_complex::Complex64;

/// a vector of 1 Complex64 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(transparent)]
pub struct cplx64x1(pub(crate) [Complex64; 1]);

impl cplx64x1 {
    #[inline(always)]
    pub(crate) fn mul_add(mut self, a: Self, b: Self) -> Self {
        self.0[0] = self.0[0] * a.0[0] + b.0[0];
        self
    }
    #[inline(always)]
    pub(crate) fn splat(val: Complex64) -> cplx64x1 {
        cplx64x1([val; 1])
    }
    #[inline(always)]
    #[cfg(target_feature = "neon")]
    pub(crate) fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self {
        let val = Self::splat(a.0[LANE as usize]);
        self.mul_add(val, b)
    }
    #[inline(always)]
    pub(crate) unsafe fn from_ptr(ptr: *const Complex64) -> Self {
        cplx64x1([unsafe { *ptr }])
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