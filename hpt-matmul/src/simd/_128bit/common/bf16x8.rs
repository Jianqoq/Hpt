
/// a vector of 8 bf16 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(16))]
pub struct bf16x8(pub(crate) [half::bf16; 8]);

impl bf16x8 {
    #[inline(always)]
    pub(crate) unsafe fn from_ptr(ptr: *const half::bf16) -> Self {
        let mut result = [half::bf16::ZERO; 8];
        for i in 0..8 {
            result[i] = unsafe { *ptr.add(i) };
        }
        bf16x8(result)
    }
}

impl std::ops::Add for bf16x8 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let [x0, x1] = self.to_2_f32vec();
        let [y0, y1] = rhs.to_2_f32vec();
        let low_add = x0 + y0;
        let high_add = x1 + y1;
        bf16x8::from_2_f32vec([low_add, high_add])
    }
}
impl std::ops::Mul for bf16x8 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        let [x0, x1] = self.to_2_f32vec();
        let [y0, y1] = rhs.to_2_f32vec();
        let low_mul = x0 * y0;
        let high_mul = x1 * y1;
        bf16x8::from_2_f32vec([low_mul, high_mul])
    }
}