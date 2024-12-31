#![allow(unused)]

use tensor_dyn::TypeCommon;
use tensor_dyn::VecTrait;
use tensor_types::half;
use half::bf16;
use half::f16;
use num_complex::Complex32 as c32;
use num_complex::Complex64 as c64;

fn test_index<T: TypeCommon>() {
    let vec = unsafe { <T as TypeCommon>::Vec::from_ptr(vec![T::ZERO; <T as TypeCommon>::Vec::SIZE].as_ptr()) };
    vec[64];
}

macro_rules! test_index {
    ($type:ty) => {
        paste::paste! {
            #[should_panic]
            #[test]
            fn [<test_ $type _vec_index>]() {
                test_index::<$type>();
            }
        }
    };
}

test_index!(bool);
test_index!(f32);
test_index!(f64);
test_index!(i8);
test_index!(i16);
test_index!(i32);
test_index!(i64);
test_index!(u8);
test_index!(u16);
test_index!(u32);
test_index!(u64);
test_index!(isize);
test_index!(usize);
test_index!(f16);
test_index!(bf16);
test_index!(c32);
test_index!(c64);
