use quote::ToTokens;
use quote::quote;

pub fn is_float(list: &str) -> bool {
    matches!(list, "BF16" | "F16" | "F32" | "F64")
}

pub fn is_signed(list: &str) -> bool {
    match list.to_lowercase().as_str() {
        "i8" | "i16" | "i32" | "i64" | "bf16" | "f16" | "f32" | "f64" => true,
        _ => false,
    }
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
        1 => Type::U8,
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
        match self {
            Type::BF16 | Type::F16 | Type::F32 | Type::F64 | Type::C32 | Type::C64 => true,
            _ => false,
        }
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
                if self.level > other.level { self.dtype } else { level_to_int(other.level) }
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
