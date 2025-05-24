use num_complex::Complex64;

/// a vector of 2 cplx64 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(32))]
pub struct cplx64x2(pub(crate) [Complex64; 2]);

impl cplx64x2 {
    #[inline(always)]
    pub(crate) fn mul_add(mut self, a: Self, b: Self) -> Self {
        self.0[0] = self.0[0] * a.0[0] + b.0[0];
        self.0[1] = self.0[1] * a.0[1] + b.0[1];
        self
    }
    #[inline(always)]
    pub(crate) fn as_mut_ptr(&mut self) -> *mut Complex64 {
        self.0.as_mut_ptr()
    }
    #[inline(always)]
    pub(crate) fn splat(val: Complex64) -> cplx64x2 {
        cplx64x2([val; 2])
    }
}

impl std::ops::Add for cplx64x2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = cplx64x2::default();
        for i in 0..2 {
            ret.0[i] = self.0[i] + rhs.0[i];
        }
        ret
    }
}
impl std::ops::Mul for cplx64x2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret = cplx64x2::default();
        for i in 0..2 {
            ret.0[i] = self.0[i] * rhs.0[i];
        }
        ret
    }
}