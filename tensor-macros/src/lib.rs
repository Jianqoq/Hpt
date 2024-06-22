use proc_macro::TokenStream;
use syn::{ parse, parse_macro_input, Expr, Ident, Token };
mod type_utils;
mod list_enum;
use quote::quote;
use type_utils::{ is_float, is_signed, type_level, TypeInfo };

/// match (lhs, rhs), execute the corresponding function
fn match_helper(
    lhs: bool,
    rhs: bool,
    mut true_true: impl FnMut() -> Expr,
    mut true_false: impl FnMut() -> Expr,
    mut false_true: impl FnMut() -> Expr,
    mut false_false: impl FnMut() -> Expr
) -> Expr {
    match (lhs, rhs) {
        (true, true) => true_true(),
        (true, false) => true_false(),
        (false, true) => false_true(),
        (false, false) => false_false(),
    }
}

struct InferEnumType {
    lhs: Expr,
    rhs: Ident,
    mode: Ident,
}

impl syn::parse::Parse for InferEnumType {
    fn parse(input: parse::ParseStream) -> syn::Result<Self> {
        let lhs = input.parse::<Expr>().expect("lhs is not found");
        input.parse::<Token![,]>()?;
        let rhs = input.parse::<Ident>().expect("rhs is not found, use sapce when not needed");
        input.parse::<Token![,]>()?;
        let mode = input.parse::<Ident>()?;
        Ok(Self { lhs, rhs, mode })
    }
}

#[proc_macro]
pub fn infer_enum_type(input: TokenStream) -> TokenStream {
    let res: InferEnumType = parse_macro_input!(input as InferEnumType);
    let enum_name = res.mode.to_string();
    let mut ret = proc_macro2::TokenStream::new();
    let lhs = res.lhs;
    let rhs = res.rhs;

    match enum_name.as_str() {
        "normal" => {
            let tk = list_enum::list_enums();
            let tmp =
                quote!(
                match (#lhs, #rhs) {
                    #tk
                    _ => todo!(),
                }
            );
            ret.extend(tmp);
        }
        "binary_float" => {
            let tk = list_enum::list_enums_out_float();
            let tmp =
                quote!(
                match (#lhs, #rhs) {
                    #tk
                    _ => todo!(),
                }
            );
            ret.extend(tmp);
        }
        "uary_float" => {
            let tk = list_enum::list_enums_out_float_uary();
            let tmp =
                quote!(
                match #lhs {
                    #tk
                    _ => todo!(),
                }
            );
            ret.extend(tmp);
        }
        _ => {}
    }
    ret.into()
}

struct GenericCal {
    lhs: Ident,
    rhs: Ident,
    method: Ident,
}

impl syn::parse::Parse for GenericCal {
    fn parse(input: parse::ParseStream) -> syn::Result<Self> {
        let lhs = input.parse::<Ident>().expect("lhs is not found");
        input.parse::<Token![,]>()?;
        let rhs = input.parse::<Ident>().expect("rhs is not found, use sapce when not needed");
        input.parse::<Token![,]>()?;
        let method = input.parse::<Ident>()?;
        Ok(Self { lhs, rhs, method })
    }
}

#[proc_macro]
pub fn infer_cal_res_type(input: TokenStream) -> TokenStream {
    let res: GenericCal = parse_macro_input!(input as GenericCal);
    let lhs_str = res.lhs.to_string();
    let rhs_str = res.rhs.to_string();
    let method = res.method;
    let left_type = TypeInfo::new(&lhs_str);
    let right_type = TypeInfo::new(&rhs_str);
    let mut ret = proc_macro2::TokenStream::new();

    match method.to_string().as_str() {
        "add" | "sub" | "mul" => {
            let res_type = left_type.infer_normal_res_type(&right_type);
            ret.extend(quote! { #res_type });
        }
        "div" => {
            let res_type = left_type.infer_float_res_type(&right_type);
            ret.extend(quote! { #res_type });
        }
        #[rustfmt::skip]
        "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "sinh" | "cosh" | "tanh" | "asinh" | "acosh" | "atanh" |
        "exp" | "exp2" | "log" | "ln" | "log2" | "log10" | "sqrt" => {
            let res_type = left_type.infer_float_res_type_uary();
            ret.extend(quote! { #res_type });
        }

        _ => todo!(),
    }
    ret.into()
}
