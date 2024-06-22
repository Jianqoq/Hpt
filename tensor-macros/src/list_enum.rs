use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::parse_quote;

use crate::{ match_helper, type_utils::{ is_float, is_signed, type_level } };

pub fn list_enums() -> TokenStream2 {
    let mut token_stream = TokenStream2::new();
    let enums: [syn::Expr; 13] = [
        parse_quote! {
            Dtype::Bool
        },
        parse_quote! {
            Dtype::I8
        },
        parse_quote! {
            Dtype::I16
        },
        parse_quote! {
            Dtype::I32
        },
        parse_quote! {
            Dtype::I64
        },
        parse_quote! {
            Dtype::U8
        },
        parse_quote! {
            Dtype::U16
        },
        parse_quote! {
            Dtype::U32
        },
        parse_quote! {
            Dtype::U64
        },
        parse_quote! {
            Dtype::BF16
        },
        parse_quote! {
            Dtype::F16
        },
        parse_quote! {
            Dtype::F32
        },
        parse_quote! {
            Dtype::F64
        },
    ];
    let strs = [
        "Bool",
        "I8",
        "I16",
        "I32",
        "I64",
        "U8",
        "U16",
        "U32",
        "U64",
        "BF16",
        "F16",
        "F32",
        "F64",
    ];
    for (idx, e) in enums.iter().enumerate() {
        for (idx2, e2) in enums.iter().enumerate() {
            let lhs_str = strs[idx];
            let rhs_str = strs[idx2];
            let lhs_level = type_level(lhs_str);
            let rhs_level = type_level(rhs_str);
            let lhs_signed = is_signed(lhs_str);
            let rhs_signed = is_signed(rhs_str);
            let lhs_float = is_float(lhs_str);
            let rhs_float = is_float(rhs_str);
            let ret;
            match (lhs_signed, rhs_signed) {
                (true, true) => {
                    ret = match_helper(
                        lhs_float,
                        rhs_float,
                        || {
                            if lhs_level > rhs_level { e.clone() } else { e2.clone() }
                        },
                        || {
                            if lhs_level > rhs_level {
                                e.clone()
                            } else {
                                parse_quote! {
                                    level_to_float(rhs_level)
                                }
                            }
                        },
                        || {
                            if lhs_level > rhs_level {
                                parse_quote! {
                                    level_to_float(lhs_level)
                                }
                            } else {
                                e2.clone()
                            }
                        },
                        || {
                            if lhs_level > rhs_level { e.clone() } else { e2.clone() }
                        }
                    );
                }
                (true, false) => {
                    if lhs_float {
                        ret = if lhs_level > rhs_level {
                            e.clone()
                        } else {
                            parse_quote! {
                                level_to_float(rhs_level)
                            }
                        };
                    } else {
                        ret = if lhs_level > rhs_level {
                            e.clone()
                        } else {
                            parse_quote! {
                                level_to_int(rhs_level)
                            }
                        };
                    }
                }
                (false, true) => {
                    if rhs_float {
                        ret = if lhs_level > rhs_level {
                            parse_quote! {
                                level_to_float(lhs_level)
                            }
                        } else {
                            e2.clone()
                        };
                    } else {
                        ret = if lhs_level > rhs_level {
                            parse_quote! {
                                level_to_int(lhs_level)
                            }
                        } else {
                            e2.clone()
                        };
                    }
                }
                (false, false) => {
                    ret = if lhs_level > rhs_level { e.clone() } else { e2.clone() };
                }
            }
            let tmp_stream =
                quote! {
                (#e, #e2) => {
                    #ret
                }
            };
            token_stream.extend(tmp_stream);
        }
    }
    token_stream
}

pub fn list_enums_out_float() -> TokenStream2 {
    let mut token_stream = TokenStream2::new();
    let enums: [syn::Expr; 13] = [
        parse_quote! {
            Dtype::Bool
        },
        parse_quote! {
            Dtype::I8
        },
        parse_quote! {
            Dtype::I16
        },
        parse_quote! {
            Dtype::I32
        },
        parse_quote! {
            Dtype::I64
        },
        parse_quote! {
            Dtype::U8
        },
        parse_quote! {
            Dtype::U16
        },
        parse_quote! {
            Dtype::U32
        },
        parse_quote! {
            Dtype::U64
        },
        parse_quote! {
            Dtype::BF16
        },
        parse_quote! {
            Dtype::F16
        },
        parse_quote! {
            Dtype::F32
        },
        parse_quote! {
            Dtype::F64
        },
    ];
    let strs = [
        "Bool",
        "I8",
        "I16",
        "I32",
        "I64",
        "U8",
        "U16",
        "U32",
        "U64",
        "BF16",
        "F16",
        "F32",
        "F64",
    ];
    for (idx, e) in enums.iter().enumerate() {
        for (idx2, e2) in enums.iter().enumerate() {
            let lhs_str = strs[idx];
            let rhs_str = strs[idx2];
            let lhs_level = type_level(lhs_str);
            let rhs_level = type_level(rhs_str);
            let lhs_signed = is_signed(lhs_str);
            let rhs_signed = is_signed(rhs_str);
            let lhs_float = is_float(lhs_str);
            let rhs_float = is_float(rhs_str);
            let ret = match (lhs_signed, rhs_signed) {
                (true, true) =>
                    match_helper(
                        lhs_float,
                        rhs_float,
                        || {
                            if lhs_level > rhs_level { e.clone() } else { e2.clone() }
                        },
                        || {
                            if lhs_level > rhs_level {
                                e.clone()
                            } else {
                                parse_quote! {
                                    level_to_float(rhs_level)
                                }
                            }
                        },
                        || {
                            if lhs_level > rhs_level {
                                parse_quote! {
                                    level_to_float(lhs_level)
                                }
                            } else {
                                e2.clone()
                            }
                        },
                        || {
                            if lhs_level > rhs_level {
                                parse_quote! {
                                    level_to_float(lhs_level)
                                }
                            } else {
                                parse_quote! {
                                    level_to_float(rhs_level)
                                }
                            }
                        }
                    ),
                (true, false) => {
                    if lhs_level > rhs_level {
                        e.clone()
                    } else {
                        parse_quote! {
                            level_to_float(rhs_level)
                        }
                    }
                }
                (false, true) => {
                    if lhs_level > rhs_level {
                        parse_quote! {
                            level_to_float(lhs_level)
                        }
                    } else {
                        e2.clone()
                    }
                }
                (false, false) => {
                    if lhs_level > rhs_level {
                        parse_quote! {
                            level_to_float(lhs_level)
                        }
                    } else {
                        parse_quote! {
                            level_to_float(rhs_level)
                        }
                    }
                }
            };
            let tmp_stream =
                quote! {
                (#e, #e2) => {
                    #ret
                }
            };
            token_stream.extend(tmp_stream);
        }
    }
    token_stream
}

pub fn list_enums_out_float_uary() -> TokenStream2 {
    let mut token_stream = TokenStream2::new();
    let enums: [syn::Expr; 13] = [
        parse_quote! {
            Dtype::Bool
        },
        parse_quote! {
            Dtype::I8
        },
        parse_quote! {
            Dtype::I16
        },
        parse_quote! {
            Dtype::I32
        },
        parse_quote! {
            Dtype::I64
        },
        parse_quote! {
            Dtype::U8
        },
        parse_quote! {
            Dtype::U16
        },
        parse_quote! {
            Dtype::U32
        },
        parse_quote! {
            Dtype::U64
        },
        parse_quote! {
            Dtype::BF16
        },
        parse_quote! {
            Dtype::F16
        },
        parse_quote! {
            Dtype::F32
        },
        parse_quote! {
            Dtype::F64
        },
    ];
    let strs = [
        "Bool",
        "I8",
        "I16",
        "I32",
        "I64",
        "U8",
        "U16",
        "U32",
        "U64",
        "BF16",
        "F16",
        "F32",
        "F64",
    ];
    for (idx, e) in enums.iter().enumerate() {
        let lhs_str = strs[idx];
        let lhs_signed = is_signed(lhs_str);
        let lhs_float = is_float(lhs_str);
        let ret = match lhs_signed {
            true => {
                if lhs_float {
                    e.clone()
                } else {
                    parse_quote! {
                        level_to_float(lhs_level)
                    }
                }
            }
            false => {
                parse_quote! {
                    level_to_float(lhs_level)
                }
            }
        };
        let tmp_stream = quote! {
            #e => {
                #ret
            },
        };
        token_stream.extend(tmp_stream);
    }
    token_stream
}
