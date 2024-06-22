use syn::{ parse, Expr, Ident, Token };
mod list_enum;
use quote::quote;

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
    let mut ret = TokenStream2::new();
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
fn generic_cal(input: TokenStream) -> TokenStream {
    let res: GenericCal = parse_macro_input!(input as GenericCal);
    let mut ret = TokenStream2::new();
    let lhs = res.lhs;
    let rhs = res.rhs;
    let method = res.method;

    let tmp = match method.to_string().as_str() {
        "add" => {
            let res_type = infer_enum_type(quote!(#lhs, #rhs, normal)).into();
            println!("{:?}", res_type);
        }
        "sub" => quote!(#lhs - #rhs),
        "mul" => quote!(#lhs * #rhs),
        "div" => quote!(#lhs / #rhs),
        _ => quote!(todo!()),
    };
    ret.extend(tmp);
    ret.into()
}
