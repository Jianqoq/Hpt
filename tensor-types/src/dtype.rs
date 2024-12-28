#[cfg(target_feature = "avx512f")]
use crate::vectors::_512bit::*;
use crate::{
    convertion::VecConvertor,
    into_vec::IntoVec,
    type_promote::{BitWiseOut, FloatOutBinary, FloatOutUnary, NormalOut, NormalOutUnary},
    vectors::traits::VecTrait,
};
use core::f32;
use half::{bf16, f16};
use std::fmt::{Debug, Display};
use tensor_macros::infer_enum_type;

/// enum for data type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Dtype {
    /// boolean
    Bool,
    /// signed 8-bit integer
    I8,
    /// unsigned 8-bit integer
    U8,
    /// signed 16-bit integer
    I16,
    /// unsigned 16-bit integer
    U16,
    /// signed 32-bit integer
    I32,
    /// unsigned 32-bit integer
    U32,
    /// signed 64-bit integer
    I64,
    /// unsigned 64-bit integer
    U64,
    /// 16-bit bfloat
    BF16,
    /// 16-bit float
    F16,
    /// 32-bit float
    F32,
    /// 64-bit float
    F64,
    /// 32-bit complex
    C32,
    /// 64-bit complex
    C64,
    /// signed isize
    Isize,
    /// unsigned usize
    Usize,
}

impl Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dtype::Bool => write!(f, "bool"),
            Dtype::I8 => write!(f, "i8"),
            Dtype::U8 => write!(f, "u8"),
            Dtype::I16 => write!(f, "i16"),
            Dtype::U16 => write!(f, "u16"),
            Dtype::I32 => write!(f, "i32"),
            Dtype::U32 => write!(f, "u32"),
            Dtype::I64 => write!(f, "i64"),
            Dtype::U64 => write!(f, "u64"),
            Dtype::BF16 => write!(f, "bf16"),
            Dtype::F16 => write!(f, "f16"),
            Dtype::F32 => write!(f, "f32"),
            Dtype::F64 => write!(f, "f64"),
            Dtype::C32 => write!(f, "c32"),
            Dtype::C64 => write!(f, "c64"),
            Dtype::Isize => write!(f, "isize"),
            Dtype::Usize => write!(f, "usize"),
        }
    }
}

impl Dtype {
    /// get the size of the data type in bytes
    pub const fn size(&self) -> usize {
        match self {
            Dtype::Bool => size_of::<bool>(),
            Dtype::I8 => size_of::<i8>(),
            Dtype::U8 => size_of::<u8>(),
            Dtype::I16 => size_of::<i16>(),
            Dtype::U16 => size_of::<u16>(),
            Dtype::I32 => size_of::<i32>(),
            Dtype::U32 => size_of::<u32>(),
            Dtype::I64 => size_of::<i64>(),
            Dtype::U64 => size_of::<u64>(),
            Dtype::BF16 => size_of::<u16>(),
            Dtype::F16 => size_of::<u16>(),
            Dtype::F32 => size_of::<f32>(),
            Dtype::F64 => size_of::<f64>(),
            Dtype::C32 => size_of::<f32>() * 2,
            Dtype::C64 => size_of::<f64>() * 2,
            Dtype::Isize => size_of::<isize>(),
            Dtype::Usize => size_of::<usize>(),
        }
    }
    /// get the number of bits in the data type
    pub const fn bits(&self) -> usize {
        match self {
            Dtype::Bool => 1,
            Dtype::I8 => 8,
            Dtype::U8 => 8,
            Dtype::I16 => 16,
            Dtype::U16 => 16,
            Dtype::I32 => 32,
            Dtype::U32 => 32,
            Dtype::I64 => 64,
            Dtype::U64 => 64,
            Dtype::BF16 => 16,
            Dtype::F16 => 16,
            Dtype::F32 => 32,
            Dtype::F64 => 64,
            Dtype::C32 => 32,
            Dtype::C64 => 64,
            Dtype::Isize => 64,
            Dtype::Usize => 64,
        }
    }
}

/// common trait for all data types
///
/// This trait is used to define the common properties of all data types
pub trait TypeCommon
where
    Self: Sized + Copy,
{
    /// the data type id
    const ID: Dtype;
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
    /// cuda type of the data type
    const CUDA_TYPE: &'static str;
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
        > + VecConvertor;
    /// the mask type of the data type
    type Mask;
    /// convert the value to the mask
    fn to_mask(self) -> Self::Mask;
    /// convert the mask to the value
    fn from_mask(mask: Self::Mask) -> Self;
}

macro_rules! impl_type_common {
    (
        $type:ty,
        $dtype:ident,
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
        $cuda_type:expr,
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
            const ID: Dtype = Dtype::$dtype;
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
            const CUDA_TYPE: &'static str = $cuda_type;
            const BIT_SIZE: usize = size_of::<$type>();
            type Vec = $vec;
            type Mask = $mask;
            fn to_mask(self) -> Self::Mask {
                unsafe { std::mem::transmute(self) }
            }
            fn from_mask(mask: Self::Mask) -> Self {
                unsafe { std::mem::transmute(mask) }
            }
        }
    };
}

#[cfg(target_feature = "avx2")]
mod type_impl {
    use super::{Dtype, TypeCommon};
    use crate::simd::_256bit::*;
    use crate::vectors::traits::VecTrait;
    use half::*;
    use num_complex::{Complex32, Complex64};
    impl_type_common!(
        bool,
        Bool,
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
        "bool",
        boolx32::boolx32,
        u8
    );
    impl_type_common!(
        i8,
        I8,
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
        "char",
        i8x32::i8x32,
        u8
    );
    impl_type_common!(
        u8,
        U8,
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
        "unsigned char",
        u8x32::u8x32,
        u8
    );
    impl_type_common!(
        i16,
        I16,
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
        "short",
        i16x16::i16x16,
        u16
    );
    impl_type_common!(
        u16,
        U16,
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
        "unsigned short",
        u16x16::u16x16,
        u16
    );
    impl_type_common!(
        i32,
        I32,
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
        "int",
        i32x8::i32x8,
        u32
    );
    impl_type_common!(
        u32,
        U32,
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
        "unsigned int",
        u32x8::u32x8,
        u32
    );
    impl_type_common!(
        i64,
        I64,
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
        "long long",
        i64x4::i64x4,
        u64
    );
    impl_type_common!(
        u64,
        U64,
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
        "unsigned long long",
        u64x4::u64x4,
        u64
    );
    impl_type_common!(
        f32,
        F32,
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
        "float",
        f32x8::f32x8,
        u32
    );
    impl_type_common!(
        f64,
        F64,
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
        "double",
        f64x4::f64x4,
        u64
    );
    #[cfg(target_pointer_width = "64")]
    impl_type_common!(
        isize,
        Isize,
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
        "long long",
        isizex4::isizex4,
        usize
    );
    #[cfg(target_pointer_width = "32")]
    impl_type_common!(
        isize,
        Isize,
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
        "long long",
        isizex8::isizex8,
        usize
    );
    #[cfg(target_pointer_width = "64")]
    impl_type_common!(
        usize,
        Usize,
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
        "unsigned long long",
        usizex4::usizex4,
        usize
    );
    #[cfg(target_pointer_width = "32")]
    impl_type_common!(
        usize,
        Usize,
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
        usizex8::usizex8,
        usize
    );
    impl_type_common!(
        f16,
        F16,
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
        "half",
        f16x16::f16x16,
        u16
    );
    impl_type_common!(
        bf16,
        BF16,
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
        "bfloat16",
        bf16x16::bf16x16,
        u16
    );
    impl_type_common!(
        Complex32,
        C32,
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
        "complex float",
        cplx32x4::cplx32x4,
        (u32, u32)
    );
    impl_type_common!(
        Complex64,
        C64,
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
        "complex double",
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
        Bool,
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
        "bool",
        boolx16::boolx16,
        u8
    );
    impl_type_common!(
        i8,
        I8,
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
        "char",
        i8x16::i8x16,
        u8
    );
    impl_type_common!(
        u8,
        U8,
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
        "unsigned char",
        u8x16::u8x16,
        u8
    );
    impl_type_common!(
        i16,
        I16,
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
        "short",
        i16x8::i16x8,
        u16
    );
    impl_type_common!(
        u16,
        U16,
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
        "unsigned short",
        u16x8::u16x8,
        u16
    );
    impl_type_common!(
        i32,
        I32,
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
        "int",
        i32x4::i32x4,
        u32
    );
    impl_type_common!(
        u32,
        U32,
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
        "unsigned int",
        u32x4::u32x4,
        u32
    );
    impl_type_common!(
        i64,
        I64,
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
        "long long",
        i64x2::i64x2,
        u64
    );
    impl_type_common!(
        u64,
        U64,
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
        "unsigned long long",
        u64x2::u64x2,
        u64
    );
    impl_type_common!(
        f32,
        F32,
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
        "float",
        f32x4::f32x4,
        u32
    );
    impl_type_common!(
        f64,
        F64,
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
        "double",
        f64x2::f64x2,
        u64
    );
    #[cfg(target_pointer_width = "64")]
    impl_type_common!(
        isize,
        Isize,
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
        "long long",
        isizex2::isizex2,
        u64
    );
    #[cfg(target_pointer_width = "32")]
    impl_type_common!(
        isize,
        Isize,
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
        Usize,
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
        "unsigned long long",
        usizex2::usizex2,
        usize
    );
    #[cfg(target_pointer_width = "32")]
    impl_type_common!(
        usize,
        Usize,
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
        F16,
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
        "half",
        f16x8::f16x8,
        u16
    );
    impl_type_common!(
        bf16,
        BF16,
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
        "bfloat16",
        bf16x8::bf16x8,
        u16
    );
    impl_type_common!(
        Complex32,
        C32,
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
        "complex32",
        cplx32x2::cplx32x2,
        (u32, u32)
    );
    impl_type_common!(
        Complex64,
        C64,
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
        "complex64",
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

impl NormalOut for Dtype {
    type Output = Dtype;

    fn _add(self, rhs: Self) -> Dtype {
        infer_enum_type!(self, rhs, normal)
    }

    fn _sub(self, rhs: Self) -> Dtype {
        infer_enum_type!(self, rhs, normal)
    }

    fn _mul_add(self, a: Self, _: Self) -> Self::Output {
        infer_enum_type!(self, a, normal)
    }

    fn _mul(self, rhs: Self) -> Dtype {
        infer_enum_type!(self, rhs, normal)
    }

    fn _pow(self, rhs: Self) -> Self::Output {
        infer_enum_type!(self, rhs, normal)
    }

    fn _rem(self, rhs: Self) -> Dtype {
        infer_enum_type!(self, rhs, normal)
    }

    fn _max(self, rhs: Self) -> Self::Output {
        infer_enum_type!(self, rhs, normal)
    }

    fn _min(self, rhs: Self) -> Self::Output {
        infer_enum_type!(self, rhs, normal)
    }

    fn _clip(self, min: Self::Output, _: Self::Output) -> Self::Output {
        infer_enum_type!(self, min, normal)
    }
}

impl BitWiseOut for Dtype {
    type Output = Dtype;

    fn _bitand(self, rhs: Self) -> Self::Output {
        infer_enum_type!(self, rhs, normal)
    }

    fn _bitor(self, rhs: Self) -> Self::Output {
        infer_enum_type!(self, rhs, normal)
    }

    fn _bitxor(self, rhs: Self) -> Self::Output {
        infer_enum_type!(self, rhs, normal)
    }

    fn _not(self) -> Self::Output {
        self
    }

    fn _shl(self, rhs: Self) -> Self::Output {
        infer_enum_type!(self, rhs, normal)
    }

    fn _shr(self, rhs: Self) -> Self::Output {
        infer_enum_type!(self, rhs, normal)
    }
}
