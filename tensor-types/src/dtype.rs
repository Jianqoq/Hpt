use half::{ bf16, f16 };
use num_complex::{ Complex32, Complex64 };
use std::fmt::Display;
use serde::{ Deserialize, Serialize };

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dtype {
    Bool,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    BF16,
    F16,
    F32,
    F64,
    C32,
    C64,
    Isize,
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
    pub const fn size(&self) -> usize {
        match self {
            Dtype::Bool => std::mem::size_of::<bool>(),
            Dtype::I8 => std::mem::size_of::<i8>(),
            Dtype::U8 => std::mem::size_of::<u8>(),
            Dtype::I16 => std::mem::size_of::<i16>(),
            Dtype::U16 => std::mem::size_of::<u16>(),
            Dtype::I32 => std::mem::size_of::<i32>(),
            Dtype::U32 => std::mem::size_of::<u32>(),
            Dtype::I64 => std::mem::size_of::<i64>(),
            Dtype::U64 => std::mem::size_of::<u64>(),
            Dtype::BF16 => std::mem::size_of::<u16>(),
            Dtype::F16 => std::mem::size_of::<u16>(),
            Dtype::F32 => std::mem::size_of::<f32>(),
            Dtype::F64 => std::mem::size_of::<f64>(),
            Dtype::C32 => std::mem::size_of::<f32>() * 2,
            Dtype::C64 => std::mem::size_of::<f64>() * 2,
            Dtype::Isize => std::mem::size_of::<isize>(),
            Dtype::Usize => std::mem::size_of::<usize>(),
        }
    }
}

pub trait TypeCommon {
    const ID: Dtype;
    const MAX: Self;
    const MIN: Self;
    const ZERO: Self;
    const ONE: Self;
    const INF: Self;
    const NEG_INF: Self;
    const TWO: Self;
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
        $two:expr
    ) => {
        impl TypeCommon for $type {
            const ID: Dtype = Dtype::$dtype;
            const MAX: Self = $max;
            const MIN: Self = $min;
            const ZERO: Self = $zero;
            const ONE: Self = $one;
            const INF: Self = $inf;
            const NEG_INF: Self = $neg_inf;
            const TWO: Self = $two;
        }
    };
}

// Implement TypeCommon for primitive types, this trait will be used when we use generic type
impl_type_common!(bool, Bool, true, false, false, true, true, false, false);
impl_type_common!(i8, I8, i8::MAX, i8::MIN, 0, 1, i8::MAX, i8::MIN, 2);
impl_type_common!(u8, U8, u8::MAX, u8::MIN, 0, 1, u8::MAX, u8::MIN, 2);
impl_type_common!(i16, I16, i16::MAX, i16::MIN, 0, 1, i16::MAX, i16::MIN, 2);
impl_type_common!(u16, U16, u16::MAX, u16::MIN, 0, 1, u16::MAX, u16::MIN, 2);
impl_type_common!(i32, I32, i32::MAX, i32::MIN, 0, 1, i32::MAX, i32::MIN, 2);
impl_type_common!(u32, U32, u32::MAX, u32::MIN, 0, 1, u32::MAX, u32::MIN, 2);
impl_type_common!(i64, I64, i64::MAX, i64::MIN, 0, 1, i64::MAX, i64::MIN, 2);
impl_type_common!(u64, U64, u64::MAX, u64::MIN, 0, 1, u64::MAX, u64::MIN, 2);
impl_type_common!(f32, F32, f32::MAX, f32::MIN, 0.0, 1.0, f32::INFINITY, f32::NEG_INFINITY, 2.0);
impl_type_common!(f64, F64, f64::MAX, f64::MIN, 0.0, 1.0, f64::INFINITY, f64::NEG_INFINITY, 2.0);
impl_type_common!(isize, Isize, isize::MAX, isize::MIN, 0, 1, isize::MAX, isize::MIN, 2);
impl_type_common!(usize, Usize, usize::MAX, usize::MIN, 0, 1, usize::MAX, usize::MIN, 2);
impl_type_common!(f16, F16, f16::MAX, f16::MIN, f16::ZERO, f16::ONE, f16::INFINITY, f16::NEG_INFINITY, f16::from_f32_const(2.0)); // prettier-ignore
impl_type_common!(bf16, BF16, bf16::MAX, bf16::MIN, bf16::ZERO, bf16::ONE, bf16::INFINITY, bf16::NEG_INFINITY, bf16::from_f32_const(2.0)); // prettier-ignore
impl_type_common!(
    Complex32,
    C32,
    Complex32::new(f32::MAX, f32::MAX),
    Complex32::new(f32::MIN, f32::MIN),
    Complex32::new(0.0, 0.0),
    Complex32::new(1.0, 0.0),
    Complex32::new(f32::INFINITY, f32::INFINITY),
    Complex32::new(f32::NEG_INFINITY, f32::NEG_INFINITY),
    Complex32::new(2.0, 0.0)
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
    Complex64::new(2.0, 0.0)
);
