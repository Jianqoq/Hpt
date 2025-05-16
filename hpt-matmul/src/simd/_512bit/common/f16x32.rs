use crate::simd::_512bit::avx512::f16x32::f32x16_to_f16x16;
use crate::simd::_512bit::common::f32x16::f32x16;

/// a vector of 16 f16 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(64))]
pub struct f16x32(pub(crate) [half::f16; 32]);

impl f16x32 {
    #[inline(always)]
    pub(crate) fn mul_add(self, a: Self, b: Self) -> Self {
        let [x0, x1]: [f32x16; 2] = unsafe { std::mem::transmute(self.to_2_f32vec()) };
        let [a0, a1]: [f32x16; 2] = unsafe { std::mem::transmute(a.to_2_f32vec()) };
        let [b0, b1]: [f32x16; 2] = unsafe { std::mem::transmute(b.to_2_f32vec()) };
        let res0 = x0.mul_add(a0, b0);
        let res1 = x1.mul_add(a1, b1);
        let res0 = f32x16_to_f16x16(res0);
        let res1 = f32x16_to_f16x16(res1);
        unsafe { std::mem::transmute([res0, res1]) }
    }
    #[inline(always)]
    pub(crate) fn splat(val: half::f16) -> f16x32 {
        f16x32([val; 32])
    }

    pub(crate) unsafe fn from_ptr(ptr: *const half::f16) -> Self {
        f16x32([
            ptr.read_unaligned(),
            ptr.add(1).read_unaligned(),
            ptr.add(2).read_unaligned(),
            ptr.add(3).read_unaligned(),
            ptr.add(4).read_unaligned(),
            ptr.add(5).read_unaligned(),
            ptr.add(6).read_unaligned(),
            ptr.add(7).read_unaligned(),
            ptr.add(8).read_unaligned(),
            ptr.add(9).read_unaligned(),
            ptr.add(10).read_unaligned(),
            ptr.add(11).read_unaligned(),
            ptr.add(12).read_unaligned(),
            ptr.add(13).read_unaligned(),
            ptr.add(14).read_unaligned(),
            ptr.add(15).read_unaligned(),
            ptr.add(16).read_unaligned(),
            ptr.add(17).read_unaligned(),
            ptr.add(18).read_unaligned(),
            ptr.add(19).read_unaligned(),
            ptr.add(20).read_unaligned(),
            ptr.add(21).read_unaligned(),
            ptr.add(22).read_unaligned(),
            ptr.add(23).read_unaligned(),
            ptr.add(24).read_unaligned(),
            ptr.add(25).read_unaligned(),
            ptr.add(26).read_unaligned(),
            ptr.add(27).read_unaligned(),
            ptr.add(28).read_unaligned(),
            ptr.add(29).read_unaligned(),
            ptr.add(30).read_unaligned(),
            ptr.add(31).read_unaligned(),
        ])
    }
}

impl std::ops::Add for f16x32 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let [x0, x1] = self.to_2_f32vec();
        let [y0, y1] = rhs.to_2_f32vec();
        let low_add = x0 + y0;
        let high_add = x1 + y1;
        f16x32::from_2_f32vec([low_add, high_add])
    }
}
// 
// impl std::ops::Mul for f16x16 {
//     type Output = Self;
//     #[inline(always)]
//     fn mul(self, rhs: Self) -> Self::Output {
//         let [x0, x1] = self.to_2_f32vec();
//         let [y0, y1] = rhs.to_2_f32vec();
//         let low_mul = x0 * y0;
//         let high_mul = x1 * y1;
//         f16x16::from_2_f32vec([low_mul, high_mul])
//     }
// }