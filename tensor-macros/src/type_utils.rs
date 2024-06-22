use syn::parse_quote;

pub fn is_float(list: &str) -> bool {
    matches!(list, "BF16" | "F16" | "F32" | "F64")
}

pub fn is_signed(list: &str) -> bool {
    matches!(
        list,
        "I8" | "I16" | "I32" | "I64" | "BF16" | "F16" | "F32" | "F64"
    )
}

pub fn type_level(list: &str) -> u8 {
    match list {
        "Bool" => 1,
        "I8" => 2,
        "U8" => 3,
        "I16" | "F16" | "BF16" => 4,
        "U16" => 5,
        "I32" | "F32" => 6,
        "U32" => 7,
        "I64" | "U64" | "F64" => 8,
        _ => 0,
    }
}

pub fn level_to_float(level: u8) -> syn::Expr {
    match level {
        1 => parse_quote! { Dtype::F16 },
        2 => parse_quote! { Dtype::F16 },
        3 => parse_quote! { Dtype::F16 },
        4 => parse_quote! { Dtype::F16 },
        5 => parse_quote! { Dtype::F32 },
        6 => parse_quote! { Dtype::F32 },
        7 => parse_quote! { Dtype::F64 },
        8 => parse_quote! { Dtype::F64 },
        _ => parse_quote! { Dtype::F64 },
    }
}

pub fn level_to_int(level: u8) -> syn::Expr {
    match level {
        1 => parse_quote! { Dtype::I8 },
        2 => parse_quote! { Dtype::I8 },
        3 => parse_quote! { Dtype::I16 },
        4 => parse_quote! { Dtype::I16 },
        5 => parse_quote! { Dtype::I32 },
        6 => parse_quote! { Dtype::I32 },
        7 => parse_quote! { Dtype::I64 },
        8 => parse_quote! { Dtype::I64 },
        _ => parse_quote! { Dtype::I64 },
    }
}

pub fn level_to_uint(level: u8) -> syn::Expr {
    match level {
        1 => parse_quote! { Dtype::U8 },
        2 => parse_quote! { Dtype::U8 },
        3 => parse_quote! { Dtype::U8 },
        4 => parse_quote! { Dtype::U16 },
        5 => parse_quote! { Dtype::U16 },
        6 => parse_quote! { Dtype::U32 },
        7 => parse_quote! { Dtype::U32 },
        8 => parse_quote! { Dtype::U64 },
        _ => parse_quote! { Dtype::U64 },
    }
}
