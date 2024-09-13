use quote::ToTokens;
use quote::quote;
use syn::parse_quote;

pub fn is_float(list: &str) -> bool {
    matches!(list, "BF16" | "F16" | "F32" | "F64")
}

pub fn is_signed(list: &str) -> bool {
    matches!(
        list.to_lowercase().as_str(),
        "i8" | "i16" | "i32" | "i64" | "bf16" | "f16" | "f32" | "f64"
    )
}

pub fn type_level(list: &str) -> u8 {
    match list.to_lowercase().as_str() {
        "bool" => 1,
        "i8" => 2,
        "u8" => 3,
        "i16" | "f16" | "bf16" => 4,
        "u16" => 5,
        "i32" | "f32" => 6,
        "u32" => 7,
        "i64" | "u64" | "f64" => 8,
        _ => 0,
    }
}

pub fn type_simd_lanes(list: &str) -> u8 {
    #[cfg(all(any(target_feature = "sse", target_feature = "neon"), not(target_feature = "avx2")))]
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
        _ => 0,
    }
}

pub fn type_simd_is_arr(list: &str) -> bool {
    #[cfg(all(any(target_feature = "sse", target_feature = "neon"), not(target_feature = "avx2")))]
    match list.to_lowercase().as_str() {
        "bool" => true,
        "i8" => false,
        "u8" => false,
        "i16" => false,
        "u16" => false,
        "i32" => false,
        "u32" => false,
        "i64" => false,
        "u64" => false,
        "bf16" => true,
        "f16" => true,
        "f32" => false,
        "f64" => false,
        #[cfg(target_pointer_width = "64")]
        "isize" => false,
        #[cfg(target_pointer_width = "64")]
        "usize" => false,
        #[cfg(target_pointer_width = "32")]
        "isize" => false,
        #[cfg(target_pointer_width = "32")]
        "usize" => false,
        #[cfg(target_pointer_width = "64")]
        "complex32" => true,
        #[cfg(target_pointer_width = "64")]
        "complex64" => true,
        #[cfg(target_pointer_width = "32")]
        "complex32" => true,
        #[cfg(target_pointer_width = "32")]
        "complex64" => true,
        _ => false,
    }
    #[cfg(target_feature = "avx2")]
    match list.to_lowercase().as_str() {
        "bool" => true,
        "i8" => false,
        "u8" => false,
        "i16" => false,
        "u16" => false,
        "i32" => false,
        "u32" => false,
        "i64" => false,
        "u64" => false,
        "bf16" => true,
        "f16" => true,
        "f32" => false,
        "f64" => false,
        #[cfg(target_pointer_width = "64")]
        "isize" => false,
        #[cfg(target_pointer_width = "64")]
        "usize" => false,
        #[cfg(target_pointer_width = "32")]
        "isize" => false,
        #[cfg(target_pointer_width = "32")]
        "usize" => false,
        #[cfg(target_pointer_width = "64")]
        "complex32" => true,
        #[cfg(target_pointer_width = "64")]
        "complex64" => true,
        #[cfg(target_pointer_width = "32")]
        "complex32" => true,
        #[cfg(target_pointer_width = "32")]
        "complex64" => true,
        _ => false,
    }
    #[cfg(target_feature = "avx512f")]
    match list.to_lowercase().as_str() {
        "bool" => true,
        "i8" => false,
        "u8" => false,
        "i16" => false,
        "u16" => false,
        "i32" => false,
        "u32" => false,
        "i64" => false,
        "u64" => false,
        "bf16" => true,
        "f16" => true,
        "f32" => false,
        "f64" => false,
        #[cfg(target_pointer_width = "64")]
        "isize" => false,
        #[cfg(target_pointer_width = "64")]
        "usize" => false,
        #[cfg(target_pointer_width = "32")]
        "isize" => false,
        #[cfg(target_pointer_width = "32")]
        "usize" => false,
        #[cfg(target_pointer_width = "64")]
        "Complex32" => true,
        #[cfg(target_pointer_width = "64")]
        "Complex64" => true,
        #[cfg(target_pointer_width = "32")]
        "Complex32" => true,
        #[cfg(target_pointer_width = "32")]
        "Complex64" => true,
        _ => 0,
    }
}

pub(crate) fn level_to_float(level: u8) -> Type {
    match level {
        1 => Type::F16,
        2 => Type::F16,
        3 => Type::F16,
        4 => Type::F16,
        5 => Type::F32,
        6 => Type::F32,
        7 => Type::F64,
        8 => Type::F64,
        _ => Type::F64,
    }
}

pub(crate) fn level_to_int(level: u8) -> Type {
    match level {
        1 => Type::I8,
        2 => Type::I8,
        3 => Type::I16,
        4 => Type::I16,
        5 => Type::I32,
        6 => Type::I32,
        7 => Type::I64,
        8 => Type::I64,
        _ => Type::I64,
    }
}

pub(crate) fn level_to_uint(level: u8) -> Type {
    match level {
        1 => Type::Bool,
        2 => Type::U8,
        3 => Type::U8,
        4 => Type::U16,
        5 => Type::U16,
        6 => Type::U32,
        7 => Type::U32,
        8 => Type::U64,
        _ => Type::U64,
    }
}

pub fn level_to_float_expr(level: u8) -> syn::Expr {
    match level {
        1 =>
            parse_quote! {
                Dtype::F16
            },
        2 =>
            parse_quote! {
                Dtype::F16
            },
        3 =>
            parse_quote! {
                Dtype::F16
            },
        4 =>
            parse_quote! {
                Dtype::F16
            },
        5 =>
            parse_quote! {
                Dtype::F32
            },
        6 =>
            parse_quote! {
                Dtype::F32
            },
        7 =>
            parse_quote! {
                Dtype::F64
            },
        8 =>
            parse_quote! {
                Dtype::F64
            },
        _ =>
            parse_quote! {
                Dtype::F64
            },
    }
}

pub fn level_to_int_expr(level: u8) -> syn::Expr {
    match level {
        1 =>
            parse_quote! {
                Dtype::I8
            },
        2 =>
            parse_quote! {
                Dtype::I8
            },
        3 =>
            parse_quote! {
                Dtype::I16
            },
        4 =>
            parse_quote! {
                Dtype::I16
            },
        5 =>
            parse_quote! {
                Dtype::I32
            },
        6 =>
            parse_quote! {
                Dtype::I32
            },
        7 =>
            parse_quote! {
                Dtype::I64
            },
        8 =>
            parse_quote! {
                Dtype::I64
            },
        _ =>
            parse_quote! {
                Dtype::I64
            },
    }
}

#[derive(Copy, Clone)]
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
        matches!(self, Type::BF16 | Type::F16 | Type::F32 | Type::F64 | Type::C32 | Type::C64)
    }
    pub fn is_unsigned(&self) -> bool {
        matches!(self, Type::Bool | Type::U8 | Type::U16 | Type::U32 | Type::U64 | Type::Usize)
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
    pub fn is_f32(&self) -> bool {
        matches!(self, Type::F32)
    }
    pub fn is_f64(&self) -> bool {
        matches!(self, Type::F64)
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
            Type::Complex32 => quote!(num_complex::Complex32),
            Type::Complex64 => quote!(num_complex::Complex64),
        };
        tokens.extend(token);
    }
}

impl ToString for Type {
    fn to_string(&self) -> String {
        match self {
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
        }
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
        #[cfg(all(any(target_feature = "sse", target_feature = "neon"), not(target_feature = "avx2")))]
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
    pub(crate) is_float: bool,
    pub(crate) is_signed: bool,
    pub(crate) level: u8,
    pub(crate) dtype: Type,
}

impl TypeInfo {
    pub(crate) fn new(name: &str) -> Self {
        match name.to_lowercase().as_str() {
            "bool" =>
                Self {
                    is_float: false,
                    is_signed: false,
                    level: 1,
                    dtype: Type::Bool,
                },
            "i8" =>
                Self {
                    is_float: false,
                    is_signed: true,
                    level: 2,
                    dtype: Type::I8,
                },
            "u8" =>
                Self {
                    is_float: false,
                    is_signed: false,
                    level: 3,
                    dtype: Type::U8,
                },
            "i16" =>
                Self {
                    is_float: false,
                    is_signed: true,
                    level: 4,
                    dtype: Type::I16,
                },

            "u16" =>
                Self {
                    is_float: false,
                    is_signed: false,
                    level: 5,
                    dtype: Type::U16,
                },
            "i32" =>
                Self {
                    is_float: false,
                    is_signed: true,
                    level: 6,
                    dtype: Type::I32,
                },
            "u32" =>
                Self {
                    is_float: false,
                    is_signed: false,
                    level: 7,
                    dtype: Type::U32,
                },
            "i64" =>
                Self {
                    is_float: false,
                    is_signed: true,
                    level: 8,
                    dtype: Type::I64,
                },
            "u64" =>
                Self {
                    is_float: false,
                    is_signed: false,
                    level: 8,
                    dtype: Type::U64,
                },
            "bf16" =>
                Self {
                    is_float: true,
                    is_signed: true,
                    level: 4,
                    dtype: Type::BF16,
                },
            "f16" =>
                Self {
                    is_float: true,
                    is_signed: true,
                    level: 4,
                    dtype: Type::F16,
                },
            "f32" =>
                Self {
                    is_float: true,
                    is_signed: true,
                    level: 6,
                    dtype: Type::F32,
                },
            "f64" =>
                Self {
                    is_float: true,
                    is_signed: true,
                    level: 8,
                    dtype: Type::F64,
                },
            "c32" =>
                Self {
                    is_float: true,
                    is_signed: true,
                    level: 6,
                    dtype: Type::C32,
                },
            "c64" =>
                Self {
                    is_float: true,
                    is_signed: true,
                    level: 8,
                    dtype: Type::C64,
                },
            "isize" =>
                Self {
                    is_float: false,
                    is_signed: true,
                    level: 8,
                    dtype: Type::Isize,
                },
            "usize" =>
                Self {
                    is_float: false,
                    is_signed: false,
                    level: 8,
                    dtype: Type::Usize,
                },
            "complex32" =>
                Self {
                    is_float: true,
                    is_signed: true,
                    level: 6,
                    dtype: Type::Complex32,
                },
            "complex64" =>
                Self {
                    is_float: true,
                    is_signed: true,
                    level: 8,
                    dtype: Type::Complex64,
                },
            _ => unreachable!("Invalid type"),
        }
    }
}

impl TypeInfo {
    pub(crate) fn infer_normal_res_type(&self, other: &Self) -> Type {
        match (self.is_float, other.is_float) {
            (true, true) => {
                if self.level > other.level { self.dtype } else { other.dtype }
            }
            (true, false) => {
                if self.level > other.level { self.dtype } else { level_to_float(other.level) }
            }
            (false, true) => {
                if self.level > other.level { level_to_float(self.level) } else { other.dtype }
            }
            (false, false) => {
                if self.level > other.level {
                    self.dtype
                } else {
                    if self.is_signed || other.is_signed {
                        if
                            (self.level == 8 || other.level == 8) &&
                            ((self.dtype as u8) == (Type::Isize as u8) ||
                                (other.dtype as u8) == (Type::Isize as u8))
                        {
                            Type::Isize
                        } else if
                            (self.level == 8 || other.level == 8) &&
                            ((self.dtype as u8) == (Type::Usize as u8) ||
                                (other.dtype as u8) == (Type::Usize as u8))
                        {
                            Type::Isize
                        } else {
                            level_to_int(std::cmp::max(self.level, other.level))
                        }
                    } else {
                        if
                            (self.level == 8 || other.level == 8) &&
                            ((self.dtype as u8) == (Type::Isize as u8) ||
                                (other.dtype as u8) == (Type::Isize as u8))
                        {
                            Type::Usize
                        } else if
                            (self.level == 8 || other.level == 8) &&
                            ((self.dtype as u8) == (Type::Usize as u8) ||
                                (other.dtype as u8) == (Type::Usize as u8))
                        {
                            Type::Usize
                        } else if self.level == other.level {
                            self.dtype
                        } else {
                            level_to_uint(std::cmp::max(self.level, other.level))
                        }
                    }
                }
            }
        }
    }

    pub(crate) fn infer_normal_res_type_uary(&self) -> Type {
        if self.is_float {
            self.dtype
        } else if self.is_signed {
            level_to_int(self.level)
        } else {
            level_to_uint(self.level)
        }
    }

    pub(crate) fn infer_float_res_type(&self, other: &Self) -> Type {
        match (self.is_float, other.is_float) {
            (true, true) => {
                if self.level > other.level { self.dtype } else { other.dtype }
            }
            (true, false) => {
                if self.level > other.level { self.dtype } else { level_to_float(other.level) }
            }
            (false, true) => {
                if self.level > other.level { level_to_float(self.level) } else { other.dtype }
            }
            (false, false) => { level_to_float(std::cmp::max(self.level, other.level)) }
        }
    }

    pub(crate) fn infer_float_res_type_uary(&self) -> Type {
        if self.is_float { self.dtype } else { level_to_float(self.level) }
    }
}
