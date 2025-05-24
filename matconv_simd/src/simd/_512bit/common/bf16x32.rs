

/// a vector of 16 bf16 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(64))]
pub struct bf16x32(pub(crate) [half::bf16; 32]);

// impl bf16x16 {
//     #[inline(always)]
//     pub(crate) fn mul_add(self, a: Self, b: Self) -> Self {
//         let [x0, x1]: [f32x8; 2] = unsafe { std::mem::transmute(self.to_2_f32vec()) };
//         let [a0, a1]: [f32x8; 2] = unsafe { std::mem::transmute(a.to_2_f32vec()) };
//         let [b0, b1]: [f32x8; 2] = unsafe { std::mem::transmute(b.to_2_f32vec()) };
//         let res0 = x0.mul_add(a0, b0);
//         let res1 = x1.mul_add(a1, b1);
//         bf16x16::from_2_f32vec([res0, res1])
//     }
//     #[inline(always)]
//     pub(crate) fn splat(val: half::bf16) -> bf16x16 {
//         bf16x16([val; 16])
//     }
//     #[inline(always)]
//     pub(crate) unsafe fn from_ptr(ptr: *const half::bf16) -> Self {
//         bf16x16([
//             ptr.read_unaligned(),
//             ptr.add(1).read_unaligned(),
//             ptr.add(2).read_unaligned(),
//             ptr.add(3).read_unaligned(),
//             ptr.add(4).read_unaligned(),
//             ptr.add(5).read_unaligned(),
//             ptr.add(6).read_unaligned(),
//             ptr.add(7).read_unaligned(),
//             ptr.add(8).read_unaligned(),
//             ptr.add(9).read_unaligned(),
//             ptr.add(10).read_unaligned(),
//             ptr.add(11).read_unaligned(),
//             ptr.add(12).read_unaligned(),
//             ptr.add(13).read_unaligned(),
//             ptr.add(14).read_unaligned(),
//             ptr.add(15).read_unaligned(),
//         ])
//     }
// }
// 
// impl std::ops::Add for bf16x16 {
//     type Output = Self;
//     #[inline(always)]
//     fn add(self, rhs: Self) -> Self::Output {
//         let [x0, x1] = self.to_2_f32vec();
//         let [y0, y1] = rhs.to_2_f32vec();
//         let low_add = x0 + y0;
//         let high_add = x1 + y1;
// 
//         bf16x16::from_2_f32vec([low_add, high_add])
//     }
// }
// impl std::ops::Mul for bf16x16 {
//     type Output = Self;
//     #[inline(always)]
//     fn mul(self, rhs: Self) -> Self::Output {
//         let [x0, x1] = self.to_2_f32vec();
//         let [y0, y1] = rhs.to_2_f32vec();
//         let low_mul = x0 * y0;
//         let high_mul = x1 * y1;
//         bf16x16::from_2_f32vec([low_mul, high_mul])
//     }
// }