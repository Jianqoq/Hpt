
/// a vector of 16 bool values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(32))]
pub struct boolx32(pub(crate) [bool; 32]);

impl boolx32 {
    #[inline(always)]
    pub(crate) fn mul_add(self, a: Self, b: Self) -> Self {
        let mut ret = boolx32::default();
        for i in 0..32 {
            ret.0[i] = (self.0[i] && a.0[i]) || b.0[i];
        }
        ret
    }
    #[inline(always)]
    pub(crate) fn splat(val: bool) -> boolx32 {
        boolx32([val; 32])
    }
}

impl std::ops::Add for boolx32 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = boolx32::default();
        for i in 0..32 {
            ret.0[i] = self.0[i] || rhs.0[i];
        }
        ret
    }
}
impl std::ops::Mul for boolx32 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret = boolx32::default();
        for i in 0..32 {
            ret.0[i] = self.0[i] && rhs.0[i];
        }
        ret
    }
}