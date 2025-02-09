#[cfg(target_feature = "avx512f")]
use crate::vectors::_512bit::*;
use crate::{
    into_vec::IntoVec,
    type_promote::{FloatOutBinary, FloatOutUnary, NormalOut, NormalOutUnary},
    vectors::traits::VecTrait,
};
use core::f32;
use half::{bf16, f16};
use num_complex::{Complex32, Complex64};
use std::fmt::Debug;

/// trait for cuda type
pub trait CudaType {
    /// the cuda type
    const CUDA_TYPE: &'static str;
}

impl CudaType for bool {
    const CUDA_TYPE: &'static str = "bool";
}

impl CudaType for i8 {
    const CUDA_TYPE: &'static str = "char";
}

impl CudaType for u8 {
    const CUDA_TYPE: &'static str = "unsigned char";
}

impl CudaType for i16 {
    const CUDA_TYPE: &'static str = "short";
}

impl CudaType for u16 {
    const CUDA_TYPE: &'static str = "unsigned short";
}

impl CudaType for i32 {
    const CUDA_TYPE: &'static str = "int";
}

impl CudaType for u32 {
    const CUDA_TYPE: &'static str = "unsigned int";
}

impl CudaType for i64 {
    const CUDA_TYPE: &'static str = "long long";
}

impl CudaType for u64 {
    const CUDA_TYPE: &'static str = "unsigned long long";
}

impl CudaType for f32 {
    const CUDA_TYPE: &'static str = "float";
}

impl CudaType for f64 {
    const CUDA_TYPE: &'static str = "double";
}

impl CudaType for Complex32 {
    const CUDA_TYPE: &'static str = "cuFloatComplex";
}

impl CudaType for Complex64 {
    const CUDA_TYPE: &'static str = "cuDoubleComplex";
}

impl CudaType for isize {
    const CUDA_TYPE: &'static str = "long long";
}

impl CudaType for usize {
    const CUDA_TYPE: &'static str = "unsigned long long";
}

impl CudaType for f16 {
    const CUDA_TYPE: &'static str = "__half";
}

impl CudaType for bf16 {
    const CUDA_TYPE: &'static str = "bfloat16";
}

/// common trait for all data types
///
/// This trait is used to define the common properties of all data types
pub trait TypeCommon
where
    Self: Sized + Copy,
{
    /// the maximum value of the data type
    const MAX: Self;
    /// the minimum value of the data type
    const MIN: Self;
    /// the zero value of the data type
    const ZERO: Self;
    /// the one value of the data type
    const ONE: Self;
    /// the infinity value of the data type, for integer types, it is the maximum value
    const INF: Self;
    /// the negative infinity value of the data type, for integer types, it is the minimum value
    const NEG_INF: Self;
    /// the two value of the data type
    const TWO: Self;
    /// the six value of the data type
    const SIX: Self;
    /// the ten value of the data type
    const TEN: Self;
    /// the string representation of the data type
    const STR: &'static str;
    /// the bit size of the data type, alias of `std::mem::size_of()`
    const BIT_SIZE: usize;
    /// the simd vector type of the data type
    type Vec: VecTrait<Self>
        + Send
        + Copy
        + IntoVec<Self::Vec>
        + std::ops::Index<usize, Output = Self>
        + std::ops::IndexMut<usize>
        + Sync
        + Debug
        + NormalOutUnary
        + NormalOut<Self::Vec, Output = Self::Vec>
        + FloatOutUnary
        + FloatOutBinary
        + FloatOutBinary<
            <Self::Vec as FloatOutUnary>::Output,
            Output = <Self::Vec as FloatOutUnary>::Output,
        >;
}

macro_rules! impl_type_common {
    (
        $type:ty,
        $max:expr,
        $min:expr,
        $zero:expr,
        $one:expr,
        $inf:expr,
        $neg_inf:expr,
        $two:expr,
        $six:expr,
        $ten:expr,
        $str:expr,
        $vec:ty,
        $mask:ty
    ) => {
        impl std::ops::Index<usize> for $vec {
            type Output = $type;
            fn index(&self, index: usize) -> &Self::Output {
                if index >= <$vec>::SIZE {
                    panic!(
                        "index out of bounds: the len is {} but the index is {}",
                        <$vec>::SIZE,
                        index
                    );
                }
                unsafe { &*self.as_ptr().add(index) }
            }
        }
        impl std::ops::IndexMut<usize> for $vec {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                if index >= <$vec>::SIZE {
                    panic!(
                        "index out of bounds: the len is {} but the index is {}",
                        <$vec>::SIZE,
                        index
                    );
                }
                unsafe { &mut *self.as_mut_ptr().add(index) }
            }
        }
        impl TypeCommon for $type {
            const MAX: Self = $max;
            const MIN: Self = $min;
            const ZERO: Self = $zero;
            const ONE: Self = $one;
            const INF: Self = $inf;
            const NEG_INF: Self = $neg_inf;
            const TWO: Self = $two;
            const SIX: Self = $six;
            const TEN: Self = $ten;
            const STR: &'static str = $str;
            const BIT_SIZE: usize = size_of::<$type>();
            type Vec = $vec;
        }
    };
}

#[cfg(target_feature = "avx2")]
mod type_impl {
    use super::TypeCommon;
    use crate::simd::_256bit::*;
    use crate::vectors::traits::VecTrait;
    use half::*;
    use num_complex::{Complex32, Complex64};
    impl_type_common!(
        bool,
        true,
        false,
        false,
        true,
        true,
        false,
        false,
        true,
        true,
        "bool",
        boolx32::boolx32,
        u8
    );
    impl_type_common!(
        i8,
        i8::MAX,
        i8::MIN,
        0,
        1,
        i8::MAX,
        i8::MIN,
        2,
        6,
        10,
        "i8",
        i8x32::i8x32,
        u8
    );
    impl_type_common!(
        u8,
        u8::MAX,
        u8::MIN,
        0,
        1,
        u8::MAX,
        u8::MIN,
        2,
        6,
        10,
        "u8",
        u8x32::u8x32,
        u8
    );
    impl_type_common!(
        i16,
        i16::MAX,
        i16::MIN,
        0,
        1,
        i16::MAX,
        i16::MIN,
        2,
        6,
        10,
        "i16",
        i16x16::i16x16,
        u16
    );
    impl_type_common!(
        u16,
        u16::MAX,
        u16::MIN,
        0,
        1,
        u16::MAX,
        u16::MIN,
        2,
        6,
        10,
        "u16",
        u16x16::u16x16,
        u16
    );
    impl_type_common!(
        i32,
        i32::MAX,
        i32::MIN,
        0,
        1,
        i32::MAX,
        i32::MIN,
        2,
        6,
        10,
        "i32",
        i32x8::i32x8,
        u32
    );
    impl_type_common!(
        u32,
        u32::MAX,
        u32::MIN,
        0,
        1,
        u32::MAX,
        u32::MIN,
        2,
        6,
        10,
        "u32",
        u32x8::u32x8,
        u32
    );
    impl_type_common!(
        i64,
        i64::MAX,
        i64::MIN,
        0,
        1,
        i64::MAX,
        i64::MIN,
        2,
        6,
        10,
        "i64",
        i64x4::i64x4,
        u64
    );
    impl_type_common!(
        u64,
        u64::MAX,
        u64::MIN,
        0,
        1,
        u64::MAX,
        u64::MIN,
        2,
        6,
        10,
        "u64",
        u64x4::u64x4,
        u64
    );
    impl_type_common!(
        f32,
        f32::MAX,
        f32::MIN,
        0.0,
        1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        2.0,
        6.0,
        10.0,
        "f32",
        f32x8::f32x8,
        u32
    );
    impl_type_common!(
        f64,
        f64::MAX,
        f64::MIN,
        0.0,
        1.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        2.0,
        6.0,
        10.0,
        "f64",
        f64x4::f64x4,
        u64
    );
    #[cfg(target_pointer_width = "64")]
    impl_type_common!(
        isize,
        isize::MAX,
        isize::MIN,
        0,
        1,
        isize::MAX,
        isize::MIN,
        2,
        6,
        10,
        "isize",
        isizex4::isizex4,
        usize
    );
    #[cfg(target_pointer_width = "32")]
    impl_type_common!(
        isize,
        isize::MAX,
        isize::MIN,
        0,
        1,
        isize::MAX,
        isize::MIN,
        2,
        6,
        10,
        "isize",
        isizex8::isizex8,
        usize
    );
    #[cfg(target_pointer_width = "64")]
    impl_type_common!(
        usize,
        usize::MAX,
        usize::MIN,
        0,
        1,
        usize::MAX,
        usize::MIN,
        2,
        6,
        10,
        "usize",
        usizex4::usizex4,
        usize
    );
    #[cfg(target_pointer_width = "32")]
    impl_type_common!(
        usize,
        usize::MAX,
        usize::MIN,
        0,
        1,
        usize::MAX,
        usize::MIN,
        2,
        6,
        10,
        "usize",
        usizex8::usizex8,
        usize
    );
    impl_type_common!(
        f16,
        f16::MAX,
        f16::MIN,
        f16::ZERO,
        f16::ONE,
        f16::INFINITY,
        f16::NEG_INFINITY,
        f16::from_f32_const(2.0),
        f16::from_f32_const(6.0),
        f16::from_f32_const(10.0),
        "f16",
        f16x16::f16x16,
        u16
    );
    impl_type_common!(
        bf16,
        bf16::MAX,
        bf16::MIN,
        bf16::ZERO,
        bf16::ONE,
        bf16::INFINITY,
        bf16::NEG_INFINITY,
        bf16::from_f32_const(2.0),
        bf16::from_f32_const(6.0),
        bf16::from_f32_const(10.0),
        "bf16",
        bf16x16::bf16x16,
        u16
    );
    impl_type_common!(
        Complex32,
        Complex32::new(f32::MAX, f32::MAX),
        Complex32::new(f32::MIN, f32::MIN),
        Complex32::new(0.0, 0.0),
        Complex32::new(1.0, 0.0),
        Complex32::new(f32::INFINITY, f32::INFINITY),
        Complex32::new(f32::NEG_INFINITY, f32::NEG_INFINITY),
        Complex32::new(2.0, 0.0),
        Complex32::new(6.0, 0.0),
        Complex32::new(10.0, 0.0),
        "c32",
        cplx32x4::cplx32x4,
        (u32, u32)
    );
    impl_type_common!(
        Complex64,
        Complex64::new(f64::MAX, f64::MAX),
        Complex64::new(f64::MIN, f64::MIN),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(f64::INFINITY, f64::INFINITY),
        Complex64::new(f64::NEG_INFINITY, f64::NEG_INFINITY),
        Complex64::new(2.0, 0.0),
        Complex64::new(6.0, 0.0),
        Complex64::new(10.0, 0.0),
        "c64",
        cplx64x2::cplx64x2,
        (u64, u64)
    );
}

#[cfg(all(
    any(target_feature = "sse", target_arch = "arm", target_arch = "aarch64"),
    not(target_feature = "avx2")
))]
mod type_impl {
    use super::{Dtype, TypeCommon};
    #[cfg(feature = "archsimd")]
    use crate::vectors::arch_simd as simd;
    #[cfg(feature = "stdsimd")]
    use crate::vectors::std_simd as simd;
    use crate::vectors::traits::VecTrait;
    use half::*;
    use num_complex::{Complex32, Complex64};
    use simd::_128bit::*;
    impl_type_common!(
        bool,
        true,
        false,
        false,
        true,
        true,
        false,
        false,
        true,
        true,
        "bool",
        boolx16::boolx16,
        u8
    );
    impl_type_common!(
        i8,
        i8::MAX,
        i8::MIN,
        0,
        1,
        i8::MAX,
        i8::MIN,
        2,
        6,
        10,
        "i8",
        i8x16::i8x16,
        u8
    );
    impl_type_common!(
        u8,
        u8::MAX,
        u8::MIN,
        0,
        1,
        u8::MAX,
        u8::MIN,
        2,
        6,
        10,
        "u8",
        u8x16::u8x16,
        u8
    );
    impl_type_common!(
        i16,
        i16::MAX,
        i16::MIN,
        0,
        1,
        i16::MAX,
        i16::MIN,
        2,
        6,
        10,
        "i16",
        i16x8::i16x8,
        u16
    );
    impl_type_common!(
        u16,
        u16::MAX,
        u16::MIN,
        0,
        1,
        u16::MAX,
        u16::MIN,
        2,
        6,
        10,
        "u16",
        u16x8::u16x8,
        u16
    );
    impl_type_common!(
        i32,
        i32::MAX,
        i32::MIN,
        0,
        1,
        i32::MAX,
        i32::MIN,
        2,
        6,
        10,
        "i32",
        i32x4::i32x4,
        u32
    );
    impl_type_common!(
        u32,
        u32::MAX,
        u32::MIN,
        0,
        1,
        u32::MAX,
        u32::MIN,
        2,
        6,
        10,
        "u32",
        u32x4::u32x4,
        u32
    );
    impl_type_common!(
        i64,
        i64::MAX,
        i64::MIN,
        0,
        1,
        i64::MAX,
        i64::MIN,
        2,
        6,
        10,
        "i64",
        i64x2::i64x2,
        u64
    );
    impl_type_common!(
        u64,
        u64::MAX,
        u64::MIN,
        0,
        1,
        u64::MAX,
        u64::MIN,
        2,
        6,
        10,
        "u64",
        u64x2::u64x2,
        u64
    );
    impl_type_common!(
        f32,
        f32::MAX,
        f32::MIN,
        0.0,
        1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        2.0,
        6.0,
        10.0,
        "f32",
        f32x4::f32x4,
        u32
    );
    impl_type_common!(
        f64,
        f64::MAX,
        f64::MIN,
        0.0,
        1.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        2.0,
        6.0,
        10.0,
        "f64",
        f64x2::f64x2,
        u64
    );
    #[cfg(target_pointer_width = "64")]
    impl_type_common!(
        isize,
        isize::MAX,
        isize::MIN,
        0,
        1,
        isize::MAX,
        isize::MIN,
        2,
        6,
        10,
        "isize",
        isizex2::isizex2,
        u64
    );
    #[cfg(target_pointer_width = "32")]
    impl_type_common!(
        isize,
        isize::MAX,
        isize::MIN,
        0,
        1,
        isize::MAX,
        isize::MIN,
        2,
        6,
        10,
        "isize",
        "int",
        isizex4::isizex4,
        u32
    );
    #[cfg(target_pointer_width = "64")]
    impl_type_common!(
        usize,
        usize::MAX,
        usize::MIN,
        0,
        1,
        usize::MAX,
        usize::MIN,
        2,
        6,
        10,
        "usize",
        usizex2::usizex2,
        usize
    );
    #[cfg(target_pointer_width = "32")]
    impl_type_common!(
        usize,
        usize::MAX,
        usize::MIN,
        0,
        1,
        usize::MAX,
        usize::MIN,
        2,
        6,
        10,
        "usize",
        "unsigned int",
        usizex4::usizex4,
        usize
    );
    impl_type_common!(
        f16,
        f16::MAX,
        f16::MIN,
        f16::ZERO,
        f16::ONE,
        f16::INFINITY,
        f16::NEG_INFINITY,
        f16::from_f32_const(2.0),
        f16::from_f32_const(6.0),
        f16::from_f32_const(10.0),
        "f16",
        f16x8::f16x8,
        u16
    );
    impl_type_common!(
        bf16,
        bf16::MAX,
        bf16::MIN,
        bf16::ZERO,
        bf16::ONE,
        bf16::INFINITY,
        bf16::NEG_INFINITY,
        bf16::from_f32_const(2.0),
        bf16::from_f32_const(6.0),
        bf16::from_f32_const(10.0),
        "bf16",
        bf16x8::bf16x8,
        u16
    );
    impl_type_common!(
        Complex32,
        Complex32::new(f32::MAX, f32::MAX),
        Complex32::new(f32::MIN, f32::MIN),
        Complex32::new(0.0, 0.0),
        Complex32::new(1.0, 0.0),
        Complex32::new(f32::INFINITY, f32::INFINITY),
        Complex32::new(f32::NEG_INFINITY, f32::NEG_INFINITY),
        Complex32::new(2.0, 0.0),
        Complex32::new(6.0, 0.0),
        Complex32::new(10.0, 0.0),
        "c32",
        cplx32x2::cplx32x2,
        (u32, u32)
    );
    impl_type_common!(
        Complex64,
        Complex64::new(f64::MAX, f64::MAX),
        Complex64::new(f64::MIN, f64::MIN),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(f64::INFINITY, f64::INFINITY),
        Complex64::new(f64::NEG_INFINITY, f64::NEG_INFINITY),
        Complex64::new(2.0, 0.0),
        Complex64::new(6.0, 0.0),
        Complex64::new(10.0, 0.0),
        "c64",
        cplx64x1::cplx64x1,
        (u64, u64)
    );
}

/// constant values for floating point data types
pub trait FloatConst {
    /// 0.5
    const HALF: Self;
    /// e
    const E: Self;
    /// π
    const PI: Self;
    /// 3.0
    const THREE: Self;
    /// 2π
    const TWOPI: Self;
    /// 4π
    const FOURPI: Self;
    /// 0.2
    const POINT_TWO: Self;
    /// 1/√2
    const FRAC_1_SQRT_2: Self;
}

impl FloatConst for f32 {
    const HALF: Self = 0.5;
    const E: Self = f32::consts::E;
    const PI: Self = f32::consts::PI;
    const THREE: Self = 3.0;
    const TWOPI: Self = f32::consts::PI * 2.0;
    const FOURPI: Self = f32::consts::PI * 4.0;
    const POINT_TWO: Self = 0.2;
    const FRAC_1_SQRT_2: Self = f32::consts::FRAC_1_SQRT_2;
}

impl FloatConst for f64 {
    const HALF: Self = 0.5;
    const E: Self = std::f64::consts::E;
    const PI: Self = std::f64::consts::PI;
    const THREE: Self = 3.0;
    const TWOPI: Self = std::f64::consts::PI * 2.0;
    const FOURPI: Self = std::f64::consts::PI * 4.0;
    const POINT_TWO: Self = 0.2;
    const FRAC_1_SQRT_2: Self = std::f64::consts::FRAC_1_SQRT_2;
}

impl FloatConst for f16 {
    const HALF: Self = f16::from_f32_const(0.5);
    const E: Self = f16::from_f32_const(f32::consts::E);
    const PI: Self = f16::from_f32_const(f32::consts::PI);
    const THREE: Self = f16::from_f32_const(3.0);
    const TWOPI: Self = f16::from_f32_const(f32::consts::PI * 2.0);
    const FOURPI: Self = f16::from_f32_const(f32::consts::PI * 4.0);
    const POINT_TWO: Self = f16::from_f32_const(0.2);
    const FRAC_1_SQRT_2: Self = f16::from_f32_const(f32::consts::FRAC_1_SQRT_2);
}

impl FloatConst for bf16 {
    const HALF: Self = bf16::from_f32_const(0.5);
    const E: Self = bf16::from_f32_const(f32::consts::E);
    const PI: Self = bf16::from_f32_const(f32::consts::PI);
    const THREE: Self = bf16::from_f32_const(3.0);
    const TWOPI: Self = bf16::from_f32_const(f32::consts::PI * 2.0);
    const FOURPI: Self = bf16::from_f32_const(f32::consts::PI * 4.0);
    const POINT_TWO: Self = bf16::from_f32_const(0.2);
    const FRAC_1_SQRT_2: Self = bf16::from_f32_const(f32::consts::FRAC_1_SQRT_2);
}
