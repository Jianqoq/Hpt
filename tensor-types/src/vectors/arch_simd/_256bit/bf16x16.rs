use crate::vectors::arch_simd::_256bit::f32x8::f32x8;
use crate::vectors::arch_simd::_256bit::u16x16::u16x16;
use crate::vectors::traits::VecTrait;
use crate::traits::SimdCompare;

/// a vector of 16 bf16 values
#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
#[repr(C, align(32))]
pub struct bf16x16(pub(crate) [half::bf16; 16]);

impl VecTrait<half::bf16> for bf16x16 {
    const SIZE: usize = 16;
    type Base = half::bf16;
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        let [x0, x1]: [f32x8; 2] = unsafe { std::mem::transmute(self.to_2_f32x8()) };
        let [a0, a1]: [f32x8; 2] = unsafe { std::mem::transmute(a.to_2_f32x8()) };
        let [b0, b1]: [f32x8; 2] = unsafe { std::mem::transmute(b.to_2_f32x8()) };
        let res0 = x0.mul_add(a0, b0);
        let res1 = x1.mul_add(a1, b1);
        bf16x16::from_2_f32x8([res0, res1])
    }
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[half::bf16]) {
        self.0.copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const half::bf16 {
        self.0.as_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut half::bf16 {
        self.0.as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut half::bf16 {
        self.0.as_ptr() as *mut _
    }

    #[inline(always)]
    fn sum(&self) -> half::bf16 {
        self.0.iter().sum()
    }
    fn splat(val: half::bf16) -> bf16x16 {
        bf16x16([val; 16])
    }
    unsafe fn from_ptr(ptr: *const half::bf16) -> Self where Self: Sized {
        let mut dst = [half::bf16::ZERO; 16];
        std::ptr::copy_nonoverlapping(
            ptr as *const u8,
            std::ptr::addr_of_mut!(dst) as *mut u8,
            size_of::<Self>()
        );
        bf16x16(dst)
    }
}

impl bf16x16 {
    /// convert to 2 f32x8
    #[cfg(target_feature = "avx2")]
    pub fn to_2_f32x8(&self) -> [f32x8; 2] {
        unimplemented!()
    }

    /// convert from 2 f32x8
    #[cfg(target_feature = "avx2")]
    pub fn from_2_f32x8(_: [f32x8; 2]) -> Self {
        unimplemented!()
    }

    /// check if the value is NaN, return a mask
    pub fn is_nan(&self) -> u16x16 {
        let x = u16x16::splat(0x7f80u16);
        let y = u16x16::splat(0x007fu16);
        let i: u16x16 = unsafe { std::mem::transmute(self.0) };
        let and = i & x;
        let eq: u16x16 = unsafe { std::mem::transmute(and.simd_eq(x)) };
        let and2 = i & y;
        let neq_zero: u16x16 = unsafe { std::mem::transmute(and2.simd_ne(u16x16::splat(0))) };
        unsafe { std::mem::transmute(eq & neq_zero) }
    }

    /// check if the value is infinite, return a mask
    pub fn is_infinite(&self) -> u16x16 {
        let x = u16x16::splat(0x7f80u16);
        let y = u16x16::splat(0x007fu16);
        let i: u16x16 = unsafe { std::mem::transmute(self.0) };

        let and = i & x;
        let eq = and.simd_eq(x);

        let and2 = i & y;
        let eq_zero = and2.simd_eq(u16x16::splat(0));

        let result = eq & eq_zero;

        unsafe { std::mem::transmute(result) }
    }
}

impl SimdCompare for bf16x16 {
    type SimdMask = u16x16;
    fn simd_eq(self, other: Self) -> u16x16 {
        let x: u16x16 = unsafe { std::mem::transmute(self.0) };
        let y: u16x16 = unsafe { std::mem::transmute(other.0) };
        let eq = x.simd_eq(y);
        unsafe { std::mem::transmute(eq) }
    }
    fn simd_ne(self, other: Self) -> u16x16 {
        let x: u16x16 = unsafe { std::mem::transmute(self.0) };
        let y: u16x16 = unsafe { std::mem::transmute(other.0) };
        let ne = x.simd_ne(y);
        unsafe { std::mem::transmute(ne) }
    }
    fn simd_lt(self, other: Self) -> u16x16 {
        let x: u16x16 = unsafe { std::mem::transmute(self.0) };
        let y: u16x16 = unsafe { std::mem::transmute(other.0) };
        let lt = x.simd_lt(y);
        unsafe { std::mem::transmute(lt) }
    }
    fn simd_le(self, other: Self) -> u16x16 {
        let x: u16x16 = unsafe { std::mem::transmute(self.0) };
        let y: u16x16 = unsafe { std::mem::transmute(other.0) };
        let le = x.simd_le(y);
        unsafe { std::mem::transmute(le) }
    }
    fn simd_gt(self, other: Self) -> u16x16 {
        let x: u16x16 = unsafe { std::mem::transmute(self.0) };
        let y: u16x16 = unsafe { std::mem::transmute(other.0) };
        let gt = x.simd_gt(y);
        unsafe { std::mem::transmute(gt) }
    }
    fn simd_ge(self, other: Self) -> u16x16 {
        let x: u16x16 = unsafe { std::mem::transmute(self.0) };
        let y: u16x16 = unsafe { std::mem::transmute(other.0) };
        let ge = x.simd_ge(y);
        unsafe { std::mem::transmute(ge) }
    }
}

impl std::ops::Add for bf16x16 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = bf16x16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] + rhs.0[i];
        }
        ret
    }
}
impl std::ops::Sub for bf16x16 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut ret = bf16x16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] - rhs.0[i];
        }
        ret
    }
}
impl std::ops::Mul for bf16x16 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret = bf16x16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] * rhs.0[i];
        }
        ret
    }
}
impl std::ops::Div for bf16x16 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut ret = bf16x16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] / rhs.0[i];
        }
        ret
    }
}
impl std::ops::Rem for bf16x16 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        let mut ret = bf16x16::default();
        for i in 0..16 {
            ret.0[i] = self.0[i] % rhs.0[i];
        }
        ret
    }
}
