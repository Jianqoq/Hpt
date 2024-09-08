use std::simd::num::{ SimdFloat, SimdInt, SimdUint };
use std::simd::u16x4;
use std::simd::{ cmp::SimdPartialEq, Simd };
use std::simd::cmp::SimdPartialOrd;
use crate::into_vec::IntoVec;
use crate::traits::{ Init, VecSize, VecTrait };
use crate::_128bit::u16x8::u16x8;
use crate::_128bit::f32x4::f32x4;

#[allow(non_camel_case_types)]
#[derive(Default, Clone, Copy, PartialEq)]
pub struct f16x8(pub(crate) [half::f16; 8]);

impl VecTrait<half::f16> for f16x8 {
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[half::f16]) {
        self.0.copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const half::f16 {
        self.0.as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, a: Self, b: Self) -> Self {
        let [x0, x1]: [f32x4; 2] = unsafe { std::mem::transmute(self.to_2_f32x4()) };
        let [a0, a1]: [f32x4; 2] = unsafe { std::mem::transmute(a.to_2_f32x4()) };
        let [b0, b1]: [f32x4; 2] = unsafe { std::mem::transmute(b.to_2_f32x4()) };
        let res0 = x0._mul_add(a0, b0);
        let res1 = x1._mul_add(a1, b1);
        let res0 = f32x4_to_f16x4(res0);
        let res1 = f32x4_to_f16x4(res1);
        unsafe { std::mem::transmute([res0, res1]) }
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut half::f16 {
        self.0.as_mut_ptr()
    }
    #[inline(always)]
    fn as_mut_ptr_uncheck(&self) -> *mut half::f16 {
        self.0.as_ptr() as *mut _
    }
    #[inline(always)]
    fn sum(&self) -> half::f16 {
        self.0.iter().sum()
    }

    fn extract(self, idx: usize) -> half::f16 {
        self.0[idx]
    }
}
impl VecSize for f16x8 {
    const SIZE: usize = 8;
}
impl Init<half::f16> for f16x8 {
    fn splat(val: half::f16) -> f16x8 {
        f16x8([val; 8])
    }
}
impl IntoVec<f16x8> for f16x8 {
    fn into_vec(self) -> f16x8 {
        self
    }
}

impl f16x8 {
    pub fn is_nan(&self) -> u16x8 {
        let x = u16x8::splat(0x7c00u16);
        let y = u16x8::splat(0x03ffu16);
        let i: Simd<u16, 8> = unsafe { std::mem::transmute(self.0) };

        let and = i & x.0;
        let eq = and.simd_eq(x.0);

        let and2 = i & y.0;
        let neq_zero = and2.simd_ne(u16x8::splat(0).0);

        let result = eq & neq_zero;

        unsafe { std::mem::transmute(result) }
    }
    pub fn is_infinite(&self) -> u16x8 {
        let x = u16x8::splat(0x7c00u16);
        let y = u16x8::splat(0x03ffu16);
        let i: Simd<u16, 8> = unsafe { std::mem::transmute(self.0) };

        let and = i & x.0;
        let eq = and.simd_eq(x.0);

        let and2 = i & y.0;
        let eq_zero = and2.simd_eq(u16x8::splat(0).0);

        let result = eq & eq_zero;

        unsafe { std::mem::transmute(result) }
    }
    pub fn simd_eq(&self, other: Self) -> u16x8 {
        let x: Simd<u16, 8> = unsafe { std::mem::transmute(self.0) };
        let y: Simd<u16, 8> = unsafe { std::mem::transmute(other.0) };
        let eq = x.simd_eq(y);
        unsafe { std::mem::transmute(eq) }
    }
    pub fn simd_ne(&self, other: Self) -> u16x8 {
        let x: Simd<u16, 8> = unsafe { std::mem::transmute(self.0) };
        let y: Simd<u16, 8> = unsafe { std::mem::transmute(other.0) };
        let ne = x.simd_ne(y);
        unsafe { std::mem::transmute(ne) }
    }
    pub fn simd_lt(&self, other: Self) -> u16x8 {
        let x: Simd<u16, 8> = unsafe { std::mem::transmute(self.0) };
        let y: Simd<u16, 8> = unsafe { std::mem::transmute(other.0) };
        let lt = x.simd_lt(y);
        unsafe { std::mem::transmute(lt) }
    }
    pub fn simd_le(&self, other: Self) -> u16x8 {
        let x: Simd<u16, 8> = unsafe { std::mem::transmute(self.0) };
        let y: Simd<u16, 8> = unsafe { std::mem::transmute(other.0) };
        let le = x.simd_le(y);
        unsafe { std::mem::transmute(le) }
    }
    pub fn simd_gt(&self, other: Self) -> u16x8 {
        let x: Simd<u16, 8> = unsafe { std::mem::transmute(self.0) };
        let y: Simd<u16, 8> = unsafe { std::mem::transmute(other.0) };
        let gt = x.simd_gt(y);
        unsafe { std::mem::transmute(gt) }
    }
    pub fn simd_ge(&self, other: Self) -> u16x8 {
        let x: Simd<u16, 8> = unsafe { std::mem::transmute(self.0) };
        let y: Simd<u16, 8> = unsafe { std::mem::transmute(other.0) };
        let ge = x.simd_ge(y);
        unsafe { std::mem::transmute(ge) }
    }
    pub fn to_2_f32x4(self) -> [f32x4; 2] {
        let [a0, a1] = unsafe {
            let a: [std::simd::u16x4; 2] = std::mem::transmute(self.0);
            a
        };
        let a0 = u16_to_f16(a0);
        let a1 = u16_to_f16(a1);
        unsafe { std::mem::transmute([a0, a1]) }
    }
}

impl std::ops::Add for f16x8 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = f16x8::default();
        for i in 0..8 {
            ret.0[i] = self.0[i] + rhs.0[i];
        }
        ret
    }
}

impl std::ops::Sub for f16x8 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut ret = f16x8::default();
        for i in 0..8 {
            ret.0[i] = self.0[i] - rhs.0[i];
        }
        ret
    }
}

impl std::ops::Mul for f16x8 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut ret = f16x8::default();
        for i in 0..8 {
            ret.0[i] = self.0[i] * rhs.0[i];
        }
        ret
    }
}

impl std::ops::Div for f16x8 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut ret = f16x8::default();
        for i in 0..8 {
            ret.0[i] = self.0[i] / rhs.0[i];
        }
        ret
    }
}
impl std::ops::Rem for f16x8 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        let mut ret = f16x8::default();
        for i in 0..8 {
            ret.0[i] = self.0[i] % rhs.0[i];
        }
        ret
    }
}

pub fn u16_to_f16(val: u16x4) -> std::simd::f32x4 {
    let sign_mask = std::simd::u16x4::splat(0x8000);
    let exp_mask = u16x4::splat(0x7c00);
    let man_mask = u16x4::splat(0x03ff);
    let zero_mask = u16x4::splat(0x7fff);

    let zero_check = val & zero_mask;
    let mut result = std::simd::f32x4::from_bits(
        (val.cast::<u32>() << 8) & std::simd::u32x4::splat(0xffff_ffff)
    );

    let half_sign = (val & sign_mask).cast::<u32>();
    let half_exp = (val & exp_mask).cast::<u32>();
    let half_man = (val & man_mask).cast::<u32>();

    let infinity_or_nan_mask = half_exp.simd_eq(std::simd::u32x4::splat(0x7c00));
    let nan_mask = half_man.simd_ne(std::simd::u32x4::splat(0));
    let infinity_mask = half_man.simd_eq(std::simd::u32x4::splat(0));

    let inf_result = (half_sign << 8) | std::simd::u32x4::splat(0x7f80_0000);
    let nan_result = (half_sign << 8) | std::simd::u32x4::splat(0x7fc0_0000) | (half_man << 13);

    result = infinity_or_nan_mask.select(
        nan_mask.select(
            std::simd::f32x4::from_bits(nan_result),
            infinity_mask.select(std::simd::f32x4::from_bits(inf_result), result)
        ),
        result
    );

    let unbiased_exp = (half_exp >> 10).cast::<i32>() - std::simd::u32x4::splat(15).cast::<i32>();

    let subnormal_mask = half_exp.simd_eq(std::simd::u32x4::splat(0));

    let leading_zeros = half_man.cast::<u16>().leading_zeros();

    let e = leading_zeros - u16x4::splat(6); // Adjustment for subnormals
    let exp_subnormal = (std::simd::u32x4::splat(127 - 15) - e.cast::<u32>()) << 23;
    let man_subnormal =
        (half_man << (std::simd::u32x4::splat(14) + e.cast::<u32>())) &
        std::simd::u32x4::splat(0x7f_ff_ff);

    let exp_normal = (unbiased_exp + std::simd::i32x4::splat(127)) << 23;
    let man_normal = (half_man & std::simd::u32x4::splat(0x03ff)) << 13;

    let sign_normal = half_sign << 8;
    let normal_result = std::simd::f32x4::from_bits(
        sign_normal | exp_normal.cast::<u32>() | man_normal
    );
    let subnormal_result = std::simd::f32x4::from_bits(sign_normal | exp_subnormal | man_subnormal);

    let final_result = subnormal_mask.select(subnormal_result, normal_result);

    zero_check.cast::<f32>().simd_eq(std::simd::f32x4::splat(0.0)).select(result, final_result)
}

#[inline]
pub(crate) fn f32x4_to_f16x4(values: f32x4) -> u16x4 {
    // Convert to raw bytes
    let x: std::simd::u32x4 = values.to_bits();

    // Extract IEEE754 components
    let sign = x & std::simd::u32x4::splat(0x8000_0000);
    let exp = x & std::simd::u32x4::splat(0x7f80_0000);
    let man = x & std::simd::u32x4::splat(0x007f_ffff);

    // Check for all exponent bits being set, which is Infinity or NaN
    let infinity_or_nan_mask = exp.simd_eq(std::simd::u32x4::splat(0x7f80_0000));
    let nan_bit = man
        .simd_ne(std::simd::u32x4::splat(0))
        .select(std::simd::u32x4::splat(0x0200), std::simd::u32x4::splat(0));
    let inf_nan_result = (sign >> 8) | std::simd::u32x4::splat(0x7c00) | nan_bit | (man >> 13);

    // The number is normalized, start assembling half precision version
    let half_sign = sign >> 8;
    // Unbias the exponent, then bias for half precision
    let unbiased_exp = (
        (exp >> 23).cast::<i32>() - std::simd::u32x4::splat(127).cast::<i32>()
    ).cast::<u32>();
    let half_exp = unbiased_exp + std::simd::u32x4::splat(15);

    // Check for exponent overflow, return +infinity
    let overflow_mask = half_exp.simd_ge(std::simd::u32x4::splat(0x1f));
    let overflow_result = half_sign | std::simd::u32x4::splat(0x7c00);

    // Check for underflow
    let underflow_mask = half_exp.simd_le(std::simd::u32x4::splat(0));
    let no_rounding_possibility_mask = (std::simd::u32x4::splat(14) - half_exp).simd_gt(
        std::simd::u32x4::splat(24)
    );
    let signed_zero_result = half_sign;

    // Subnormal handling
    let man_with_hidden_bit = man | std::simd::u32x4::splat(0x0080_0000);
    let mut half_man = man_with_hidden_bit >> (std::simd::u32x4::splat(14) - half_exp);
    let round_bit_subnormal =
        std::simd::u32x4::splat(1) << (std::simd::u32x4::splat(13) - half_exp);
    let round_mask_subnormal =
        (man_with_hidden_bit & round_bit_subnormal).simd_ne(std::simd::u32x4::splat(0)) &
        (
            man_with_hidden_bit &
            (std::simd::u32x4::splat(3) * round_bit_subnormal - std::simd::u32x4::splat(1))
        ).simd_ne(std::simd::u32x4::splat(0));
    half_man += round_mask_subnormal.select(std::simd::u32x4::splat(1), std::simd::u32x4::splat(0));
    let subnormal_result = half_sign | half_man;

    // Normal result calculation
    let half_exp_normal = half_exp << 10;
    let half_man_normal = man >> 13;
    let round_bit_normal = std::simd::u32x4::splat(0x0000_1000);
    let round_mask_normal =
        (man & round_bit_normal).simd_ne(std::simd::u32x4::splat(0)) &
        (
            man &
            (std::simd::u32x4::splat(3) * round_bit_normal - std::simd::u32x4::splat(1))
        ).simd_ne(std::simd::u32x4::splat(0));
    let normal_result = half_sign | half_exp_normal | half_man_normal;
    let normal_rounded_result = round_mask_normal.select(
        normal_result + std::simd::u32x4::splat(1),
        normal_result
    );

    // Combine results for different cases
    let result = infinity_or_nan_mask.select(
        inf_nan_result,
        overflow_mask.select(
            overflow_result,
            underflow_mask.select(
                no_rounding_possibility_mask.select(signed_zero_result, subnormal_result),
                normal_rounded_result
            )
        )
    );

    // Cast to u16x8 and return the final result
    result.cast::<u16>()
}
