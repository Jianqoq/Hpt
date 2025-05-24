use num_complex::Complex32;

/// a vector of 4 cplx32 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(32))]
pub struct cplx32x4(pub(crate) [Complex32; 4]);

impl cplx32x4 {
    #[inline(always)]
    pub(crate) fn mul_add(mut self, a: Self, b: Self) -> Self {
        self.0[0] = self.0[0] * a.0[0] + b.0[0];
        self.0[1] = self.0[1] * a.0[1] + b.0[1];
        self.0[2] = self.0[2] * a.0[2] + b.0[2];
        self.0[3] = self.0[3] * a.0[3] + b.0[3];
        self
    }
    #[inline(always)]
    pub(crate) fn as_mut_ptr(&mut self) -> *mut Complex32 {
        self.0.as_mut_ptr()
    }
    #[inline(always)]
    pub(crate) fn splat(val: Complex32) -> cplx32x4 {
        cplx32x4([val; 4])
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