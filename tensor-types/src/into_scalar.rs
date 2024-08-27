use half::{f16, bf16};
use num_complex::{ Complex32, Complex64 };

pub trait IntoScalar<T> {
    fn into_scalar(self) -> T;
}

macro_rules! impl_into_scalar_same {
    ($($t:ty),*) => {
        $(
            impl IntoScalar<$t> for $t {
                #[inline(always)]
                fn into_scalar(self) -> $t {
                    self
                }
            }
        )*
    };
}

macro_rules! into_f16_from_32 {
    ($($source:ident),*) => {
        $(impl IntoScalar<f16> for $source {
            #[inline(always)]
            fn into_scalar(self) -> f16 {
                f16::from_f32(self as f32)
            }
        })*
    };
}

macro_rules! into_f16_from_64 {
    ($($source:ident),*) => {
        $(impl IntoScalar<f16> for $source {
            #[inline(always)]
            fn into_scalar(self) -> f16 {
                f16::from_f32(self as f32)
            }
        })*
    };
}

macro_rules! impl_into_scalar_for_f16 {
    ($($source:ident),*) => {
        $(impl IntoScalar<$source> for f16 {
            #[inline(always)]
            fn into_scalar(self) -> $source {
                self.to_f32() as $source
            }
        })*
    };
}

macro_rules! impl_into_scalar_not_bool {
    ($source:ident, $($target:ident),*) => {
        $(  
        impl IntoScalar<$target> for $source {
            #[inline(always)]
            fn into_scalar(self) -> $target {
                self as $target
            }
        })*
    };
}

macro_rules! impl_into_scalar_not_bool_ref {
    (&$source:ident, $($target:ident),*) => {
        $(  
        impl IntoScalar<$target> for &$source {
            #[inline(always)]
            fn into_scalar(self) -> $target {
                *self as $target
            }
        })*
    };
}

macro_rules! impl_into_scalar_not_bool_to_bool {
    ($($source:ident),*) => {
        $(  
        impl IntoScalar<bool> for $source {
            #[inline(always)]
            fn into_scalar(self) -> bool {
                if self == 0 as $source {
                    false
                } else {
                    true
                }
            }
        })*
    };
}

macro_rules! impl_into_scalar_bool {
    ($source:ident, $($target:ident),*) => {
        $(  
        impl IntoScalar<$target> for $source {
            #[inline(always)]
            fn into_scalar(self) -> $target {
                if self {
                    1 as $target
                } else {
                    0 as $target
                }
            }
        })*
    };
}

macro_rules! impl_into_scalar_bool_ref {
    (&$source:ident, $($target:ident),*) => {
        $(  
        impl IntoScalar<$target> for &$source {
            #[inline(always)]
            fn into_scalar(self) -> $target {
                if *self {
                    1 as $target
                } else {
                    0 as $target
                }
            }
        })*
    };
}

impl IntoScalar<f16> for bool {
    #[inline(always)]
    fn into_scalar(self) -> f16 {
        f16::from(self as u8)
    }
}

impl IntoScalar<f16> for i8 {
    #[inline(always)]
    fn into_scalar(self) -> f16 {
        f16::from(self)
    }
}
impl IntoScalar<f16> for u8 {
    #[inline(always)]
    fn into_scalar(self) -> f16 {
        f16::from(self)
    }
}

impl IntoScalar<f16> for f32 {
    #[inline(always)]
    fn into_scalar(self) -> f16 {
        f16::from_f32(self)
    }
}

impl IntoScalar<f16> for f64 {
    #[inline(always)]
    fn into_scalar(self) -> f16 {
        f16::from_f64(self)
    }
}

impl IntoScalar<f16> for bf16 {
    #[inline(always)]
    fn into_scalar(self) -> f16 {
        f16::from_f32(self.to_f32())
    }
}

impl IntoScalar<f32> for bf16 {
    #[inline(always)]
    fn into_scalar(self) -> f32 {
        self.to_f32()
    }
}

impl IntoScalar<f64> for bf16 {
    #[inline(always)]
    fn into_scalar(self) -> f64 {
        self.to_f64()
    }
}

impl IntoScalar<bf16> for i8 {
    #[inline(always)]
    fn into_scalar(self) -> bf16 {
        bf16::from(self)
    }
}

impl IntoScalar<bf16> for i16 {
    #[inline(always)]
    fn into_scalar(self) -> bf16 {
        bf16::from_f32(self as f32)
    }
}

impl IntoScalar<bf16> for bool {
    #[inline(always)]
    fn into_scalar(self) -> bf16 {
        bf16::from(self as u8)
    }
}

impl IntoScalar<bf16> for u64 {
    #[inline(always)]
    fn into_scalar(self) -> bf16 {
        bf16::from_f64(self as f64)
    }
}

impl IntoScalar<bf16> for i64 {
    #[inline(always)]
    fn into_scalar(self) -> bf16 {
        bf16::from_f64(self as f64)
    }
}

impl IntoScalar<bf16> for usize {
    #[inline(always)]
    fn into_scalar(self) -> bf16 {
        bf16::from_f64(self as f64)
    }
}

impl IntoScalar<bf16> for isize {
    #[inline(always)]
    fn into_scalar(self) -> bf16 {
        bf16::from_f64(self as f64)
    }
}

impl IntoScalar<bf16> for i32 {
    #[inline(always)]
    fn into_scalar(self) -> bf16 {
        bf16::from_f64(self as f64)
    }
}

impl IntoScalar<bf16> for u32 {
    #[inline(always)]
    fn into_scalar(self) -> bf16 {
        bf16::from_f64(self as f64)
    }
}

impl IntoScalar<bf16> for u16 {
    #[inline(always)]
    fn into_scalar(self) -> bf16 {
        bf16::from_f32(self as f32)
    }
}

impl IntoScalar<bf16> for u8 {
    #[inline(always)]
    fn into_scalar(self) -> bf16 {
        bf16::from(self)
    }
}

impl IntoScalar<bf16> for bf16 {
    #[inline(always)]
    fn into_scalar(self) -> bf16 {
        self
    }
}

impl IntoScalar<bf16> for f16 {
    #[inline(always)]
    fn into_scalar(self) -> bf16 {
        bf16::from_f32(self.to_f32())
    }
}

impl IntoScalar<bf16> for f32 {
    #[inline(always)]
    fn into_scalar(self) -> bf16 {
        bf16::from_f32(self)
    }
}

impl IntoScalar<bf16> for f64 {
    #[inline(always)]
    fn into_scalar(self) -> bf16 {
        bf16::from_f64(self)
    }
}

impl IntoScalar<bool> for f16 {
    #[inline(always)]
    fn into_scalar(self) -> bool {
        self.to_f32() != 0.0
    }
}

impl IntoScalar<Complex32> for bool {
    #[inline(always)]
    fn into_scalar(self) -> Complex32 {
        if self { Complex32::new(1.0, 1.0) } else { Complex32::new(0.0, 0.0) }
    }
}

impl IntoScalar<bool> for Complex32 {
    #[inline(always)]
    fn into_scalar(self) -> bool {
        self.re != 0.0
    }
}

impl IntoScalar<Complex32> for f16 {
    #[inline(always)]
    fn into_scalar(self) -> Complex32 {
        Complex32::new(self.to_f32(), self.to_f32())
    }
}

impl IntoScalar<f16> for Complex32 {
    #[inline(always)]
    fn into_scalar(self) -> f16 {
        f16::from_f32(self.re)
    }
}

impl IntoScalar<Complex32> for f32 {
    #[inline(always)]
    fn into_scalar(self) -> Complex32 {
        Complex32::new(self, self)
    }
}

impl IntoScalar<f32> for Complex32 {
    #[inline(always)]
    fn into_scalar(self) -> f32 {
        self.re
    }
}

impl IntoScalar<Complex32> for f64 {
    #[inline(always)]
    fn into_scalar(self) -> Complex32 {
        Complex32::new(self as f32, self as f32)
    }
}

impl IntoScalar<f64> for Complex32 {
    #[inline(always)]
    fn into_scalar(self) -> f64 {
        self.re as f64
    }
}

impl IntoScalar<Complex32> for i8 {
    #[inline(always)]
    fn into_scalar(self) -> Complex32 {
        Complex32::new(self as f32, self as f32)
    }
}

impl IntoScalar<i8> for Complex32 {
    #[inline(always)]
    fn into_scalar(self) -> i8 {
        self.re as i8
    }
}

impl IntoScalar<Complex32> for u8 {
    #[inline(always)]
    fn into_scalar(self) -> Complex32 {
        Complex32::new(self as f32, self as f32)
    }
}

impl IntoScalar<u8> for Complex32 {
    #[inline(always)]
    fn into_scalar(self) -> u8 {
        self.re as u8
    }
}

impl IntoScalar<Complex32> for i16 {
    #[inline(always)]
    fn into_scalar(self) -> Complex32 {
        Complex32::new(self as f32, self as f32)
    }
}

impl IntoScalar<i16> for Complex32 {
    #[inline(always)]
    fn into_scalar(self) -> i16 {
        self.re as i16
    }
}

impl IntoScalar<Complex32> for u16 {
    #[inline(always)]
    fn into_scalar(self) -> Complex32 {
        Complex32::new(self as f32, self as f32)
    }
}

impl IntoScalar<u16> for Complex32 {
    #[inline(always)]
    fn into_scalar(self) -> u16 {
        self.re as u16
    }
}

impl IntoScalar<Complex32> for i32 {
    #[inline(always)]
    fn into_scalar(self) -> Complex32 {
        Complex32::new(self as f32, self as f32)
    }
}

impl IntoScalar<i32> for Complex32 {
    #[inline(always)]
    fn into_scalar(self) -> i32 {
        self.re as i32
    }
}

impl IntoScalar<Complex32> for u32 {
    #[inline(always)]
    fn into_scalar(self) -> Complex32 {
        Complex32::new(self as f32, self as f32)
    }
}

impl IntoScalar<u32> for Complex32 {
    #[inline(always)]
    fn into_scalar(self) -> u32 {
        self.re as u32
    }
}

impl IntoScalar<Complex32> for i64 {
    #[inline(always)]
    fn into_scalar(self) -> Complex32 {
        Complex32::new(self as f32, self as f32)
    }
}

impl IntoScalar<i64> for Complex32 {
    #[inline(always)]
    fn into_scalar(self) -> i64 {
        self.re as i64
    }
}

impl IntoScalar<Complex32> for u64 {
    #[inline(always)]
    fn into_scalar(self) -> Complex32 {
        Complex32::new(self as f32, self as f32)
    }
}

impl IntoScalar<u64> for Complex32 {
    #[inline(always)]
    fn into_scalar(self) -> u64 {
        self.re as u64
    }
}

impl IntoScalar<Complex32> for i128 {
    #[inline(always)]
    fn into_scalar(self) -> Complex32 {
        Complex32::new(self as f32, self as f32)
    }
}

impl IntoScalar<i128> for Complex32 {
    #[inline(always)]
    fn into_scalar(self) -> i128 {
        self.re as i128
    }
}

impl IntoScalar<Complex32> for u128 {
    #[inline(always)]
    fn into_scalar(self) -> Complex32 {
        Complex32::new(self as f32, self as f32)
    }
}

impl IntoScalar<u128> for Complex32 {
    #[inline(always)]
    fn into_scalar(self) -> u128 {
        self.re as u128
    }
}

impl IntoScalar<Complex32> for usize {
    #[inline(always)]
    fn into_scalar(self) -> Complex32 {
        Complex32::new(self as f32, 0.0)
    }
}

impl IntoScalar<usize> for Complex32 {
    #[inline(always)]
    fn into_scalar(self) -> usize {
        self.re as usize
    }
}

impl IntoScalar<Complex32> for Complex32 {
    #[inline(always)]
    fn into_scalar(self) -> Complex32 {
        self
    }
}

impl IntoScalar<Complex64> for Complex32 {
    #[inline(always)]
    fn into_scalar(self) -> Complex64 {
        Complex64::new(self.re as f64, self.im as f64)
    }
}

impl IntoScalar<Complex64> for bool {
    #[inline(always)]
    fn into_scalar(self) -> Complex64 {
        if self { Complex64::new(1.0, 1.0) } else { Complex64::new(0.0, 0.0) }
    }
}

impl IntoScalar<bool> for Complex64 {
    #[inline(always)]
    fn into_scalar(self) -> bool {
        self.re != 0.0
    }
}

impl IntoScalar<Complex64> for f16 {
    #[inline(always)]
    fn into_scalar(self) -> Complex64 {
        Complex64::new(self.to_f64(), self.to_f64())
    }
}

impl IntoScalar<f16> for Complex64 {
    #[inline(always)]
    fn into_scalar(self) -> f16 {
        f16::from_f64(self.re)
    }
}

impl IntoScalar<Complex64> for f32 {
    #[inline(always)]
    fn into_scalar(self) -> Complex64 {
        Complex64::new(self as f64, self as f64)
    }
}

impl IntoScalar<f32> for Complex64 {
    #[inline(always)]
    fn into_scalar(self) -> f32 {
        self.re as f32
    }
}

impl IntoScalar<Complex64> for f64 {
    #[inline(always)]
    fn into_scalar(self) -> Complex64 {
        Complex64::new(self, self)
    }
}

impl IntoScalar<f64> for Complex64 {
    #[inline(always)]
    fn into_scalar(self) -> f64 {
        self.re
    }
}

impl IntoScalar<Complex64> for i8 {
    #[inline(always)]
    fn into_scalar(self) -> Complex64 {
        Complex64::new(self as f64, self as f64)
    }
}

impl IntoScalar<i8> for Complex64 {
    #[inline(always)]
    fn into_scalar(self) -> i8 {
        self.re as i8
    }
}

impl IntoScalar<Complex64> for u8 {
    #[inline(always)]
    fn into_scalar(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}

impl IntoScalar<u8> for Complex64 {
    #[inline(always)]
    fn into_scalar(self) -> u8 {
        self.re as u8
    }
}

impl IntoScalar<Complex64> for i16 {
    #[inline(always)]
    fn into_scalar(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}

impl IntoScalar<i16> for Complex64 {
    #[inline(always)]
    fn into_scalar(self) -> i16 {
        self.re as i16
    }
}

impl IntoScalar<Complex64> for u16 {
    #[inline(always)]
    fn into_scalar(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}

impl IntoScalar<u16> for Complex64 {
    #[inline(always)]
    fn into_scalar(self) -> u16 {
        self.re as u16
    }
}

impl IntoScalar<Complex64> for i32 {
    #[inline(always)]
    fn into_scalar(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}

impl IntoScalar<i32> for Complex64 {
    #[inline(always)]
    fn into_scalar(self) -> i32 {
        self.re as i32
    }
}

impl IntoScalar<Complex64> for u32 {
    #[inline(always)]
    fn into_scalar(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}

impl IntoScalar<u32> for Complex64 {
    #[inline(always)]
    fn into_scalar(self) -> u32 {
        self.re as u32
    }
}

impl IntoScalar<Complex64> for i64 {
    #[inline(always)]
    fn into_scalar(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}

impl IntoScalar<i64> for Complex64 {
    #[inline(always)]
    fn into_scalar(self) -> i64 {
        self.re as i64
    }
}

impl IntoScalar<Complex64> for u64 {
    #[inline(always)]
    fn into_scalar(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}

impl IntoScalar<u64> for Complex64 {
    #[inline(always)]
    fn into_scalar(self) -> u64 {
        self.re as u64
    }
}

impl IntoScalar<Complex64> for i128 {
    #[inline(always)]
    fn into_scalar(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}

impl IntoScalar<i128> for Complex64 {
    #[inline(always)]
    fn into_scalar(self) -> i128 {
        self.re as i128
    }
}

impl IntoScalar<Complex64> for u128 {
    #[inline(always)]
    fn into_scalar(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}

impl IntoScalar<u128> for Complex64 {
    #[inline(always)]
    fn into_scalar(self) -> u128 {
        self.re as u128
    }
}

impl IntoScalar<Complex32> for Complex64 {
    #[inline(always)]
    fn into_scalar(self) -> Complex32 {
        Complex32::new(self.re as f32, self.im as f32)
    }
}

impl IntoScalar<Complex64> for Complex64 {
    #[inline(always)]
    fn into_scalar(self) -> Complex64 {
        self
    }
}

into_f16_from_32!(i16, u16, i32, u32);
into_f16_from_64!(i64, u64, i128, u128, usize);
impl_into_scalar_for_f16!(i16, i8, u8, u16, i32, u32, i64, u64, i128, u128, usize, f32, f64);
impl_into_scalar_same!(
    bool,
    i8,
    u8,
    i16,
    u16,
    i32,
    u32,
    i64,
    u64,
    i128,
    u128,
    f16,
    f32,
    f64,
    usize,
    isize
);
impl_into_scalar_not_bool!(
    i8,
    u8,
    i16,
    u16,
    i32,
    u32,
    i64,
    u64,
    i128,
    u128,
    f32,
    f64,
    usize,
    isize
);
impl_into_scalar_not_bool_ref!(
    &i8,
    u8,
    i16,
    u16,
    i32,
    u32,
    i64,
    u64,
    i128,
    u128,
    f32,
    f64,
    usize,
    isize
);
impl_into_scalar_not_bool!(
    u8,
    i8,
    i16,
    u16,
    i32,
    u32,
    i64,
    u64,
    i128,
    u128,
    f32,
    f64,
    usize,
    isize
);
impl_into_scalar_not_bool_ref!(
    &u8,
    i8,
    i16,
    u16,
    i32,
    u32,
    i64,
    u64,
    i128,
    u128,
    f32,
    f64,
    usize,
    isize
);
impl_into_scalar_not_bool!(
    i16,
    i8,
    u8,
    u16,
    i32,
    u32,
    i64,
    u64,
    i128,
    u128,
    f32,
    f64,
    usize,
    isize
);
impl_into_scalar_not_bool_ref!(
    &i16,
    i8,
    u8,
    u16,
    i32,
    u32,
    i64,
    u64,
    i128,
    u128,
    f32,
    f64,
    usize,
    isize
);
impl_into_scalar_not_bool!(
    u16,
    i8,
    u8,
    i16,
    i32,
    u32,
    i64,
    u64,
    i128,
    u128,
    f32,
    f64,
    usize,
    isize
);
impl_into_scalar_not_bool_ref!(
    &u16,
    i8,
    u8,
    i16,
    i32,
    u32,
    i64,
    u64,
    i128,
    u128,
    f32,
    f64,
    usize,
    isize
);
impl_into_scalar_not_bool!(
    i32,
    i8,
    u8,
    i16,
    u16,
    u32,
    i64,
    u64,
    i128,
    u128,
    f32,
    f64,
    usize,
    isize
);
impl_into_scalar_not_bool_ref!(
    &i32,
    i8,
    u8,
    i16,
    u16,
    u32,
    i64,
    u64,
    i128,
    u128,
    f32,
    f64,
    usize,
    isize
);
impl_into_scalar_not_bool!(
    u32,
    i8,
    u8,
    i16,
    u16,
    i32,
    i64,
    u64,
    i128,
    u128,
    f32,
    f64,
    usize,
    isize
);
impl_into_scalar_not_bool_ref!(
    &u32,
    i8,
    u8,
    i16,
    u16,
    i32,
    i64,
    u64,
    i128,
    u128,
    f32,
    f64,
    usize,
    isize
);
impl_into_scalar_not_bool!(
    i64,
    i8,
    u8,
    i16,
    u16,
    i32,
    u32,
    u64,
    i128,
    u128,
    f32,
    f64,
    usize,
    isize
);
impl_into_scalar_not_bool_ref!(
    &i64,
    i8,
    u8,
    i16,
    u16,
    i32,
    u32,
    u64,
    i128,
    u128,
    f32,
    f64,
    usize,
    isize
);
impl_into_scalar_not_bool!(
    u64,
    i8,
    u8,
    i16,
    u16,
    i32,
    u32,
    i64,
    i128,
    u128,
    f32,
    f64,
    usize,
    isize
);
impl_into_scalar_not_bool_ref!(
    &u64,
    i8,
    u8,
    i16,
    u16,
    i32,
    u32,
    i64,
    i128,
    u128,
    f32,
    f64,
    usize,
    isize
);
impl_into_scalar_not_bool!(
    i128,
    i8,
    u8,
    i16,
    u16,
    i32,
    u32,
    i64,
    u64,
    u128,
    f32,
    f64,
    usize,
    isize
);
impl_into_scalar_not_bool_ref!(
    &i128,
    i8,
    u8,
    i16,
    u16,
    i32,
    u32,
    i64,
    u64,
    u128,
    f32,
    f64,
    usize,
    isize
);
impl_into_scalar_not_bool!(
    u128,
    i8,
    u8,
    i16,
    u16,
    i32,
    u32,
    i64,
    u64,
    i128,
    f32,
    f64,
    usize,
    isize
);
impl_into_scalar_not_bool_ref!(
    &u128,
    i8,
    u8,
    i16,
    u16,
    i32,
    u32,
    i64,
    u64,
    i128,
    f32,
    f64,
    usize,
    isize
);
impl_into_scalar_not_bool!(
    usize,
    i8,
    u8,
    i16,
    u16,
    i32,
    u32,
    i64,
    u64,
    i128,
    u128,
    f32,
    f64,
    isize
);
impl_into_scalar_not_bool_ref!(
    &usize,
    i8,
    u8,
    i16,
    u16,
    i32,
    u32,
    i64,
    u64,
    i128,
    u128,
    f32,
    f64,
    usize,
    isize
);
impl_into_scalar_not_bool!(
    isize,
    i8,
    u8,
    i16,
    u16,
    i32,
    u32,
    i64,
    u64,
    i128,
    u128,
    f32,
    f64,
    usize
);
impl_into_scalar_not_bool_ref!(
    &isize,
    i8,
    u8,
    i16,
    u16,
    i32,
    u32,
    i64,
    u64,
    i128,
    u128,
    f32,
    f64,
    usize,
    isize
);
impl_into_scalar_not_bool!(f32, i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, f64, usize);
impl_into_scalar_not_bool_ref!(&f32, i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, f64, usize);
impl_into_scalar_not_bool!(f64, i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, f32, usize);
impl_into_scalar_not_bool_ref!(&f64, i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, f32, usize);
impl_into_scalar_bool!(bool, i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, f32, f64, usize);
impl_into_scalar_bool_ref!(
    &bool,
    i8,
    u8,
    i16,
    u16,
    i32,
    u32,
    i64,
    u64,
    i128,
    u128,
    f32,
    f64,
    usize
);
impl_into_scalar_not_bool_to_bool!(
    i8,
    u8,
    i16,
    u16,
    i32,
    u32,
    i64,
    u64,
    i128,
    u128,
    f32,
    f64,
    usize
);
