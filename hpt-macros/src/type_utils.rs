use quote::quote;
use quote::ToTokens;
use std::fmt::Display;

pub fn type_simd_lanes(list: &str) -> u8 {
    #[cfg(all(
        any(target_feature = "sse", target_arch = "arm", target_arch = "aarch64"),
        not(target_feature = "avx2")
    ))]
    match list.to_lowercase().as_str() {
        "bool" => 16,
        "i8" => 16,
        "u8" => 16,
        "i16" => 8,
        "u16" => 8,
        "i32" => 4,
        "u32" => 4,
        "i64" => 2,
        "u64" => 2,
        "bf16" => 8,
        "f16" => 8,
        "f32" => 4,
        "f64" => 2,
        #[cfg(target_pointer_width = "64")]
        "isize" => 2,
        #[cfg(target_pointer_width = "64")]
        "usize" => 2,
        #[cfg(target_pointer_width = "32")]
        "isize" => 4,
        #[cfg(target_pointer_width = "32")]
        "usize" => 4,
        #[cfg(target_pointer_width = "64")]
        "complex32" => 2,
        #[cfg(target_pointer_width = "64")]
        "complex64" => 1,
        #[cfg(target_pointer_width = "32")]
        "complex32" => 4,
        #[cfg(target_pointer_width = "32")]
        "complex64" => 2,
        _ => 0,
    }
    #[cfg(target_feature = "avx2")]
    match list.to_lowercase().as_str() {
        "bool" => 32,
        "i8" => 32,
        "u8" => 32,
        "i16" => 16,
        "u16" => 16,
        "i32" => 8,
        "u32" => 8,
        "i64" => 4,
        "u64" => 4,
        "bf16" => 16,
        "f16" => 16,
        "f32" => 8,
        "f64" => 4,
        #[cfg(target_pointer_width = "64")]
        "isize" => 4,
        #[cfg(target_pointer_width = "64")]
        "usize" => 4,
        #[cfg(target_pointer_width = "32")]
        "isize" => 8,
        #[cfg(target_pointer_width = "32")]
        "usize" => 8,
        #[cfg(target_pointer_width = "64")]
        "complex32" => 4,
        #[cfg(target_pointer_width = "64")]
        "complex64" => 2,
        #[cfg(target_pointer_width = "32")]
        "complex32" => 8,
        #[cfg(target_pointer_width = "32")]
        "complex64" => 4,
        _ => 0,
    }
    #[cfg(target_feature = "avx512f")]
    match list.to_lowercase().as_str() {
        "bool" => 64,
        "i8" => 64,
        "u8" => 64,
        "i16" => 32,
        "u16" => 32,
        "i32" => 16,
        "u32" => 16,
        "i64" => 8,
        "u64" => 8,
        "bf16" => 32,
        "f16" => 32,
        "f32" => 16,
        "f64" => 8,
        #[cfg(target_pointer_width = "64")]
        "isize" => 8,
        #[cfg(target_pointer_width = "64")]
        "usize" => 8,
        #[cfg(target_pointer_width = "32")]
        "isize" => 16,
        #[cfg(target_pointer_width = "32")]
        "usize" => 16,
        #[cfg(target_pointer_width = "64")]
        "complex32" => 8,
        #[cfg(target_pointer_width = "64")]
        "complex64" => 4,
        #[cfg(target_pointer_width = "32")]
        "complex32" => 16,
        #[cfg(target_pointer_width = "32")]
        "complex64" => 8,
        _ => 0,
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) enum Type {
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
    Complex32,
    Complex64,
}

impl Type {
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            Type::BF16 | Type::F16 | Type::F32 | Type::F64 | Type::C32 | Type::C64
        )
    }
    pub fn is_bool(&self) -> bool {
        matches!(self, Type::Bool)
    }
    pub fn is_f16(&self) -> bool {
        matches!(self, Type::F16)
    }
    pub fn is_bf16(&self) -> bool {
        matches!(self, Type::BF16)
    }
    pub fn is_cplx(&self) -> bool {
        matches!(
            self,
            Type::C32 | Type::C64 | Type::Complex32 | Type::Complex64
        )
    }
    pub fn is_cplx32(&self) -> bool {
        matches!(self, Type::C32 | Type::Complex32)
    }
    pub fn is_cplx64(&self) -> bool {
        matches!(self, Type::C64 | Type::Complex64)
    }
}

impl ToTokens for Type {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let token = match self {
            Type::Bool => quote!(bool),
            Type::I8 => quote!(i8),
            Type::U8 => quote!(u8),
            Type::I16 => quote!(i16),
            Type::U16 => quote!(u16),
            Type::I32 => quote!(i32),
            Type::U32 => quote!(u32),
            Type::I64 => quote!(i64),
            Type::U64 => quote!(u64),
            Type::BF16 => quote!(bf16),
            Type::F16 => quote!(f16),
            Type::F32 => quote!(f32),
            Type::F64 => quote!(f64),
            Type::C32 => quote!(c32),
            Type::C64 => quote!(c64),
            Type::Isize => quote!(isize),
            Type::Usize => quote!(usize),
            Type::Complex32 => quote!(Complex32),
            Type::Complex64 => quote!(Complex64),
        };
        tokens.extend(token);
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            Type::Bool => "bool".to_string(),
            Type::I8 => "i8".to_string(),
            Type::U8 => "u8".to_string(),
            Type::I16 => "i16".to_string(),
            Type::U16 => "u16".to_string(),
            Type::I32 => "i32".to_string(),
            Type::U32 => "u32".to_string(),
            Type::I64 => "i64".to_string(),
            Type::U64 => "u64".to_string(),
            Type::BF16 => "bf16".to_string(),
            Type::F16 => "f16".to_string(),
            Type::F32 => "f32".to_string(),
            Type::F64 => "f64".to_string(),
            Type::C32 => "c32".to_string(),
            Type::C64 => "c64".to_string(),
            Type::Isize => "isize".to_string(),
            Type::Usize => "usize".to_string(),
            Type::Complex32 => "complex32".to_string(),
            Type::Complex64 => "complex64".to_string(),
        };
        write!(f, "{}", str)
    }
}

#[derive(Copy, Clone)]
pub(crate) enum SimdType {
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
    Complex32,
    Complex64,
}

impl From<&str> for SimdType {
    fn from(name: &str) -> Self {
        match name.to_lowercase().as_str() {
            "bool" => SimdType::Bool,
            "i8" => SimdType::I8,
            "u8" => SimdType::U8,
            "i16" => SimdType::I16,
            "u16" => SimdType::U16,
            "i32" => SimdType::I32,
            "u32" => SimdType::U32,
            "i64" => SimdType::I64,
            "u64" => SimdType::U64,
            "bf16" => SimdType::BF16,
            "f16" => SimdType::F16,
            "f32" => SimdType::F32,
            "f64" => SimdType::F64,
            "c32" => SimdType::C32,
            "c64" => SimdType::C64,
            "isize" => SimdType::Isize,
            "usize" => SimdType::Usize,
            "complex32" => SimdType::Complex32,
            "complex64" => SimdType::Complex64,
            _ => unreachable!("Invalid type"),
        }
    }
}

impl ToTokens for SimdType {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        #[cfg(target_feature = "avx2")]
        let token = match self {
            SimdType::Bool => quote!(boolx32::boolx32),
            SimdType::I8 => quote!(i8x32::i8x32),
            SimdType::U8 => quote!(u8x32::u8x32),
            SimdType::I16 => quote!(i16x16::i16x16),
            SimdType::U16 => quote!(u16x16::u16x16),
            SimdType::I32 => quote!(i32x8::i32x8),
            SimdType::U32 => quote!(u32x8::u32x8),
            SimdType::I64 => quote!(i64x4::i64x4),
            SimdType::U64 => quote!(u64x4::u64x4),
            SimdType::BF16 => quote!(bf16x16::bf16x16),
            SimdType::F16 => quote!(f16x16::f16x16),
            SimdType::F32 => quote!(f32x8::f32x8),
            SimdType::F64 => quote!(f64x4::f64x4),
            SimdType::C32 => quote!(cplx32x4::cplx32x4),
            SimdType::C64 => quote!(cplx64x2::cplx64x2),
            SimdType::Isize => quote!(isizex4::isizex4),
            SimdType::Usize => quote!(usizex4::usizex4),
            SimdType::Complex32 => quote!(cplx32x4::cplx32x4),
            SimdType::Complex64 => quote!(cplx64x2::cplx64x2),
        };
        #[cfg(all(
            any(target_feature = "sse", target_arch = "arm", target_arch = "aarch64"),
            not(target_feature = "avx2")
        ))]
        let token = match self {
            SimdType::Bool => quote!(boolx16::boolx16),
            SimdType::I8 => quote!(i8x16::i8x16),
            SimdType::U8 => quote!(u8x16::u8x16),
            SimdType::I16 => quote!(i16x8::i16x8),
            SimdType::U16 => quote!(u16x8::u16x8),
            SimdType::I32 => quote!(i32x4::i32x4),
            SimdType::U32 => quote!(u32x4::u32x4),
            SimdType::I64 => quote!(i64x2::i64x2),
            SimdType::U64 => quote!(u64x2::u64x2),
            SimdType::BF16 => quote!(bf16x8::bf16x8),
            SimdType::F16 => quote!(f16x8::f16x8),
            SimdType::F32 => quote!(f32x4::f32x4),
            SimdType::F64 => quote!(f64x2::f64x2),
            SimdType::C32 => quote!(cplx32x2::cplx32x2),
            SimdType::C64 => quote!(cplx64x1::cplx64x1),
            SimdType::Isize => quote!(isizex2::isizex2),
            SimdType::Usize => quote!(usizex2::usizex2),
            SimdType::Complex32 => quote!(cplx32x2::cplx32x2),
            SimdType::Complex64 => quote!(cplx64x1::cplx64x1),
        };
        #[cfg(target_feature = "avx512f")]
        let token = match self {
            SimdType::Bool => quote!(boolx64::boolx64),
            SimdType::I8 => quote!(i8x64::i8x64),
            SimdType::U8 => quote!(u8x64::u8x64),
            SimdType::I16 => quote!(i16x32::i16x32),
            SimdType::U16 => quote!(u16x32::u16x32),
            SimdType::I32 => quote!(i32x16::i32x16),
            SimdType::U32 => quote!(u32x16::u32x16),
            SimdType::I64 => quote!(i64x8::i64x8),
            SimdType::U64 => quote!(u64x8::u64x8),
            SimdType::BF16 => quote!(bf16x32::bf16x32),
            SimdType::F16 => quote!(f16x32::f16x32),
            SimdType::F32 => quote!(f32x16::f32x16),
            SimdType::F64 => quote!(f64x8::f64x8),
            SimdType::C32 => quote!(cplx32x8::cplx32x8),
            SimdType::C64 => quote!(cplx64x4::cplx64x4),
            SimdType::Isize => quote!(isizex8::isizex8),
            SimdType::Usize => quote!(usizex8::usizex8),
            SimdType::Complex32 => quote!(cplx32x8::cplx32x8),
            SimdType::Complex64 => quote!(cplx64x4::cplx64x4),
        };
        tokens.extend(token);
    }
}

#[derive(Copy, Clone)]
pub(crate) struct TypeInfo {
    pub(crate) dtype: Type,
}

impl TypeInfo {
    pub(crate) fn new(name: &str) -> Self {
        match name.to_lowercase().as_str() {
            "bool" => Self { dtype: Type::Bool },
            "i8" => Self { dtype: Type::I8 },
            "u8" => Self { dtype: Type::U8 },
            "i16" => Self { dtype: Type::I16 },
            "u16" => Self { dtype: Type::U16 },
            "i32" => Self { dtype: Type::I32 },
            "u32" => Self { dtype: Type::U32 },
            "i64" => Self { dtype: Type::I64 },
            "u64" => Self { dtype: Type::U64 },
            "bf16" => Self { dtype: Type::BF16 },
            "f16" => Self { dtype: Type::F16 },
            "f32" => Self { dtype: Type::F32 },
            "f64" => Self { dtype: Type::F64 },
            "c32" => Self { dtype: Type::C32 },
            "c64" => Self { dtype: Type::C64 },
            "isize" => Self { dtype: Type::Isize },
            "usize" => Self { dtype: Type::Usize },
            "complex32" => Self {
                dtype: Type::Complex32,
            },
            "complex64" => Self {
                dtype: Type::Complex64,
            },
            _ => unreachable!("Invalid type"),
        }
    }
}
