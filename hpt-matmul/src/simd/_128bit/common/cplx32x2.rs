use num_complex::Complex32;

/// a vector of 2 Complex32 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(16))]
pub struct cplx32x2(pub(crate) [Complex32; 2]);

impl cplx32x2 {
    #[inline(always)]
    pub(crate) fn mul_add(mut self, a: Self, b: Self) -> Self {
        self.0[0] = self.0[0] * a.0[0] + b.0[0];
        self.0[1] = self.0[1] * a.0[1] + b.0[1];
        self
    }
    #[inline(always)]
    pub(crate) fn splat(val: Complex32) -> cplx32x2 {
        cplx32x2([val; 2])
    }
    #[inline(always)]
    #[cfg(target_feature = "neon")]
    pub(crate) fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self {
        let val = Self::splat(a.0[LANE as usize]);
        self.mul_add(val, b)
    }
    #[inline(always)]
    pub(crate) unsafe fn from_ptr(ptr: *const Complex32) -> Self {
        let mut result = [Complex32::ZERO; 2];
        for i in 0..2 {
            result[i] = unsafe { *ptr.add(i) };
        }
        cplx32x2(result)
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