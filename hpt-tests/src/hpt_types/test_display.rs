#![allow(unused)]

use half;
use half::bf16;
use half::f16;
use hpt::TypeCommon;
use num_complex::Complex32 as c32;
use num_complex::Complex64 as c64;

macro_rules! test_display {
    ($type:ty) => {
        paste::paste! {
            #[test]
            fn [<test_ $type _display>]() {
                assert_eq!(format!("{}", <$type as TypeCommon>::STR), stringify!($type));
            }
        }
    };
}

test_display!(bool);
test_display!(f32);
test_display!(f64);
test_display!(i8);
test_display!(i16);
test_display!(i32);
test_display!(i64);
test_display!(u8);
test_display!(u16);
test_display!(u32);
test_display!(u64);
test_display!(isize);
test_display!(usize);
test_display!(f16);
test_display!(bf16);
test_display!(c32);
test_display!(c64);
