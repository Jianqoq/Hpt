use proc_macro::TokenStream;
use syn::{ parse, parse_macro_input, Expr, Ident, Token };
mod type_utils;
mod list_enum;
use quote::quote;
use type_utils::TypeInfo;
use proc_macro2::{TokenStream as TokenStream2, TokenTree};

#[derive(Debug)]
struct SelectionParser {
    start: Option<isize>,
    end: Option<isize>,
    step: Option<isize>,
    start_neg: bool,
    start_ident: Option<Ident>,
    end_neg: bool,
    end_ident: Option<Ident>,
    step_neg: bool,
    step_ident: Option<Ident>,
    only_scalar: bool,
}

struct Selections {
    selections: Vec<TokenStream>,
}

impl syn::parse::Parse for SelectionParser {
    fn parse(input: parse::ParseStream) -> syn::Result<Self> {
        let mut start: Option<isize> = None;
        let mut end: Option<isize> = None;
        let mut step: Option<isize> = None;
        let mut start_ident: Option<Ident> = None;
        let mut end_ident: Option<Ident> = None;
        let mut step_ident: Option<Ident> = None;
        let mut start_neg = false;
        let mut end_neg = false;
        let mut step_neg = false;
        if input.peek(syn::LitInt) {
            start = Some(input.parse::<syn::LitInt>()?.base10_parse::<isize>()?);
        } else if input.peek(Ident) {
            start_ident = Some(input.parse::<Ident>()?);
        } else if input.peek(Token![-]) && input.peek2(syn::Ident) {
            let _ = input.parse::<Token![-]>()?;
            start_ident = Some(input.parse::<Ident>()?);
            start_neg = true;
        }
        if input.peek(Token![:]) {
            input.parse::<Token![:]>()?;
        } else if input.is_empty() {
            return Ok(Self {
                start,
                end,
                step,
                start_neg,
                start_ident,
                end_neg,
                end_ident,
                step_neg,
                step_ident,
                only_scalar: true,
            });
        } else {
            return Err(syn::Error::new(
                input.span(),
                "unexpected token, expected `:`, Int or Ident",
            ));
        }
        if input.peek(syn::LitInt) {
            end = Some(input.parse::<syn::LitInt>()?.base10_parse::<isize>()?);
        } else if input.peek(Ident) {
            end_ident = Some(input.parse::<Ident>()?);
        } else if input.peek(Token![-]) && input.peek2(syn::Ident) {
            let _ = input.parse::<Token![-]>()?;
            end_ident = Some(input.parse::<Ident>()?);
            end_neg = true;
        }
        if input.peek(Token![:]) {
            input.parse::<Token![:]>()?;
        }
        if input.peek(syn::LitInt) {
            step = Some(input.parse::<syn::LitInt>()?.base10_parse::<isize>()?);
        } else if input.peek(Ident) {
            step_ident = Some(input.parse::<Ident>()?);
        } else if input.peek(Token![-]) && input.peek2(syn::Ident) {
            let _ = input.parse::<Token![-]>()?;
            step_ident = Some(input.parse::<Ident>()?);
            step_neg = true;
        }
        Ok(Self {
            start,
            end,
            step,
            start_neg,
            start_ident,
            end_neg,
            end_ident,
            step_neg,
            step_ident,
            only_scalar: false,
        })
    }
}

impl syn::parse::Parse for Selections {
    fn parse(input: parse::ParseStream) -> syn::Result<Self> {
        let mut selections: Vec<TokenStream> = vec![];
        let mut tokenstream = TokenStream2::new();
        while !input.is_empty() {
            let lookahead = input.lookahead1();
            if lookahead.peek(Token![,]) {
                selections.push(tokenstream.into());
                tokenstream = TokenStream2::new();
                input.parse::<Token![,]>()?;
            } else {
                let t = input.parse::<TokenTree>()?;
                tokenstream.extend(quote!(#t));
            }
        }
        selections.push(tokenstream.into());
        syn::Result::Ok(Self { selections })
    }
}

#[proc_macro]
pub fn match_selection(input: TokenStream) -> TokenStream {
    let res: Selections = parse_macro_input!(input as Selections);
    let mut slices: Vec<SelectionParser> = vec![];
    for x in res.selections {
        slices.push(parse_macro_input!(x as SelectionParser));
    }
    let mut ret_stream = TokenStream2::new();
    let len = slices.len();
    for (idx, x) in slices.into_iter().enumerate() {
        if (x.start.is_some() || x.start_ident.is_some())
            && (x.end.is_some() || x.end_ident.is_some())
            && (x.step.is_some() || x.step_ident.is_some())
        {
            // start:end:step
            let start_token;
            if x.start_ident.is_some() {
                let start_ident = x.start_ident.unwrap();
                if x.start_neg {
                    start_token = quote!(-#start_ident);
                } else {
                    start_token = quote!(#start_ident);
                }
            } else {
                let start = x.start.unwrap();
                start_token = quote!(#start);
            }
            let end_token;
            if x.end_ident.is_some() {
                let end_ident = x.end_ident.unwrap();
                if x.end_neg {
                    end_token = quote!(-#end_ident);
                } else {
                    end_token = quote!(#end_ident);
                }
            } else {
                let end = x.end.unwrap();
                end_token = quote!(#end);
            }
            let step_token;
            if x.step_ident.is_some() {
                let step_ident = x.step_ident.unwrap();
                if x.step_neg {
                    step_token = quote!(-#step_ident);
                } else {
                    step_token = quote!(#step_ident);
                }
            } else {
                let step = x.step.unwrap();
                step_token = quote!(#step);
            }
            ret_stream
                .extend(quote!(Slice::StepByRangeFromTo((#start_token, #end_token, #step_token))));
        } else if x.start.is_none()
            && x.start_ident.is_none()
            && (x.end.is_some() || x.end_ident.is_some())
            && (x.step.is_some() || x.step_ident.is_some())
        {
            // :end:step
            let end_token;
            if x.end_ident.is_some() {
                let end_ident = x.end_ident.unwrap();
                if x.end_neg {
                    end_token = quote!(-#end_ident);
                } else {
                    end_token = quote!(#end_ident);
                }
            } else {
                let end = x.end.unwrap();
                end_token = quote!(#end);
            }
            let step_token;
            if x.step_ident.is_some() {
                let step_ident = x.step_ident.unwrap();
                if x.step_neg {
                    step_token = quote!(-#step_ident);
                } else {
                    step_token = quote!(#step_ident);
                }
            } else {
                let step = x.step.unwrap();
                step_token = quote!(#step);
            }
            ret_stream.extend(quote!(Slice::StepByRangeTo((#end_token, #step_token))));
        } else if (x.start.is_some() || x.start_ident.is_some())
            && x.end.is_none()
            && x.end_ident.is_none()
            && (x.step.is_some() || x.step_ident.is_some())
        {
            // start::step
            let start_token;
            if x.start_ident.is_some() {
                let start_ident = x.start_ident.unwrap();
                if x.start_neg {
                    start_token = quote!(-#start_ident);
                } else {
                    start_token = quote!(#start_ident);
                }
            } else {
                let start = x.start.unwrap();
                start_token = quote!(#start);
            }
            let step_token;
            if x.step_ident.is_some() {
                let step_ident = x.step_ident.unwrap();
                if x.step_neg {
                    step_token = quote!(-#step_ident);
                } else {
                    step_token = quote!(#step_ident);
                }
            } else {
                let step = x.step.unwrap();
                step_token = quote!(#step);
            }
            ret_stream.extend(quote!(Slice::StepByRangeFrom((#start_token, #step_token))));
        } else if (x.start.is_some() || x.start_ident.is_some())
            && (x.end.is_some() || x.end_ident.is_some())
            && x.step.is_none()
            && x.step_ident.is_none()
        {
            // start:end:
            let start_token;
            if x.start_ident.is_some() {
                let start_ident = x.start_ident.unwrap();
                if x.start_neg {
                    start_token = quote!(-#start_ident);
                } else {
                    start_token = quote!(#start_ident);
                }
            } else {
                let start = x.start.unwrap();
                start_token = quote!(#start);
            }
            let end_token;
            if x.end_ident.is_some() {
                let end_ident = x.end_ident.unwrap();
                if x.end_neg {
                    end_token = quote!(-#end_ident);
                } else {
                    end_token = quote!(#end_ident);
                }
            } else {
                let end = x.end.unwrap();
                end_token = quote!(#end);
            }
            ret_stream.extend(quote!(Slice::Range((#start_token, #end_token))));
        } else if x.start.is_none()
            && x.start_ident.is_none()
            && (x.end.is_some() || x.end_ident.is_some())
            && x.step.is_none()
            && x.step_ident.is_none()
        {
            // :end:
            let end_token;
            if x.end_ident.is_some() {
                let end_ident = x.end_ident.unwrap();
                if x.end_neg {
                    end_token = quote!(-#end_ident);
                } else {
                    end_token = quote!(#end_ident);
                }
            } else {
                let end = x.end.unwrap();
                end_token = quote!(#end);
            }
            ret_stream.extend(quote!(Slice::RangeTo(#end_token)));
        } else if (x.start.is_some() || x.start_ident.is_some())
            && x.end.is_none()
            && x.end_ident.is_none()
            && x.step.is_none()
            && x.step_ident.is_none()
        {
            // start::
            let start_token;
            if x.start_ident.is_some() {
                let start_ident = x.start_ident.unwrap();
                if x.start_neg {
                    start_token = quote!(-#start_ident);
                } else {
                    start_token = quote!(#start_ident);
                }
            } else {
                let start = x.start.unwrap();
                start_token = quote!(#start);
            }
            if x.only_scalar {
                ret_stream.extend(quote!(Slice::From(#start_token)));
            } else {
                ret_stream.extend(quote!(Slice::RangeFrom(#start_token)));
            }
        } else if x.start.is_none()
            && x.start_ident.is_none()
            && x.end.is_none()
            && x.end_ident.is_none()
            && (x.step.is_some() || x.step_ident.is_some())
        {
            // ::step
            let step_token;
            if x.step_ident.is_some() {
                let step_ident = x.step_ident.unwrap();
                if x.step_neg {
                    step_token = quote!(-#step_ident);
                } else {
                    step_token = quote!(#step_ident);
                }
            } else {
                let step = x.step.unwrap();
                step_token = quote!(#step);
            }
            ret_stream.extend(quote!(Slice::StepByFullRange(#step_token)));
        } else {
            // ::
            ret_stream.extend(quote!(Slice::Full));
        }
        if idx != len - 1 {
            ret_stream.extend(quote!(,));
        }
    }
    quote!([#ret_stream]).into()
}

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
    let mut ret = proc_macro2::TokenStream::new();

    match method.to_string().as_str() {
        "normal" => {
            if rhs_str.is_empty() {
                let res_type = left_type.infer_normal_res_type_uary();
                ret.extend(quote! { #res_type });
            } else {
                let right_type = TypeInfo::new(&rhs_str);
                let res_type = left_type.infer_normal_res_type(&right_type);
                ret.extend(quote! { #res_type });
            }
        }
        "float" => {
            if rhs_str.is_empty() {
                let res_type = left_type.infer_float_res_type_uary();
                ret.extend(quote! { #res_type });
            } else {
                let right_type = TypeInfo::new(&rhs_str);
                let res_type = left_type.infer_float_res_type(&right_type);
                ret.extend(quote! { #res_type });
            }
        }
        _ => todo!(),
    }
    ret.into()
}

#[proc_macro]
pub fn impl_float_out(_: TokenStream) -> TokenStream {
    let mut ret = proc_macro2::TokenStream::new();

    let types = [
        "bool",
        "f16",
        "f32",
        "f64",
        "i8",
        "i16",
        "i32",
        "i64",
        "u8",
        "u16",
        "u32",
        "u64",
        "isize",
        "usize",
    ];

    for lhs in types.iter() {
        for rhs in types.iter() {
            let lhs_type = TypeInfo::new(lhs);
            let rhs_type = TypeInfo::new(rhs);
            let lhs_dtype = lhs_type.dtype;
            let rhs_dtype = rhs_type.dtype;
            let res_type = lhs_type.infer_float_res_type(&rhs_type);
            let res =
                quote! {
                impl FloatOut<#rhs_dtype> for #lhs_dtype {
                    type Output = #res_type;
                
                    fn _div(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]() / rhs.[<to_ #res_type>]()
                        }
                    }
                    fn _exp(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().exp()
                        }
                    }
                    fn _exp2(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().exp2()
                        }
                    }
                    fn _ln(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().ln()
                        }
                    }
                    fn _log(self, base: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().log(base.[<to_ #res_type>]())
                        }
                    }
                    fn _log2(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().log2()
                        }
                    }
                    fn _log10(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().log10()
                        }
                    }
                    fn _sqrt(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().sqrt()
                        }
                    }
                    fn _sin(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().sin()
                        }
                    }
                    fn _cos(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().cos()
                        }
                    }
                    fn _tan(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().tan()
                        }
                    }
                    fn _asin(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().asin()
                        }
                    }
                    fn _acos(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().acos()
                        }
                    }
                    fn _atan(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().atan()
                        }
                    }
                    fn _sinh(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().sinh()
                        }
                    }
                    fn _cosh(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().cosh()
                        }
                    }
                    fn _tanh(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().tanh()
                        }
                    }
                    fn _asinh(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().asinh()
                        }
                    }
                    fn _acosh(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().acosh()
                        }
                    }
                    fn _atanh(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().atanh()
                        }
                    }
                    fn _recip(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().recip()
                        }
                    }
                }
            };
            ret.extend(res);
        }
    }

    ret.into()
}

#[proc_macro]
pub fn impl_normal_out(_: TokenStream) -> TokenStream {
    let mut ret = proc_macro2::TokenStream::new();

    let types = [
        "bool",
        "f16",
        "f32",
        "f64",
        "i8",
        "i16",
        "i32",
        "i64",
        "u8",
        "u16",
        "u32",
        "u64",
        "isize",
        "usize",
    ];

    for lhs in types.iter() {
        for rhs in types.iter() {
            let lhs_type = TypeInfo::new(lhs);
            let rhs_type = TypeInfo::new(rhs);
            let lhs_dtype = lhs_type.dtype;
            let rhs_dtype = rhs_type.dtype;
            let res_type = lhs_type.infer_normal_res_type(&rhs_type);

            let pow_method = if res_type.is_float() {
                quote! {
                    fn _pow(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().powf(rhs.[<to_ #res_type>]())
                        }
                    }
                }
            } else {
                quote! {
                    fn _pow(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().pow(rhs.to_u32())
                        }
                    }
                }
            };

            let abs_method = if res_type.is_unsigned() {
                quote! {
                    fn _abs(self) -> Self::Output {
                        paste::paste! {
                            self
                        }
                    }
                }
            } else {
                quote! {
                    fn _abs(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().abs()
                        }
                    }
                }
            };

            let res =
                quote! {
                impl NormalOut<#rhs_dtype> for #lhs_dtype {
                    type Output = #res_type;
                    fn _add(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]() + rhs.[<to_ #res_type>]()
                        }
                    }
                    fn _sub(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]() - rhs.[<to_ #res_type>]()
                        }
                    }
                    fn _mul(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]() * rhs.[<to_ #res_type>]()
                        }
                    }
                    #pow_method

                    fn _rem(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]() % rhs.[<to_ #res_type>]()
                        }
                    }
                    fn _square(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]() * self.[<to_ #res_type>]()
                        }
                    }
                    #abs_method
                }
            };
            ret.extend(res);
        }
    }

    ret.into()
}

#[proc_macro]
pub fn impl_bitwise_out(_: TokenStream) -> TokenStream {
    let mut ret = proc_macro2::TokenStream::new();

    let types = [
        "bool",
        "i8",
        "i16",
        "i32",
        "i64",
        "u8",
        "u16",
        "u32",
        "u64",
        "isize",
        "usize",
    ];

    for lhs in types.iter() {
        for rhs in types.iter() {
            let lhs_type = TypeInfo::new(lhs);
            let rhs_type = TypeInfo::new(rhs);
            let lhs_dtype = lhs_type.dtype;
            let rhs_dtype = rhs_type.dtype;
            let res_type = lhs_type.infer_normal_res_type(&rhs_type);

            let res =
                quote! {
                impl BitWiseOut<#rhs_dtype> for #lhs_dtype {
                    type Output = #res_type;
                    fn _and(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]() & rhs.[<to_ #res_type>]()
                        }
                    }
                    fn _or(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]() | rhs.[<to_ #res_type>]()
                        }
                    }
                    fn _xor(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]() ^ rhs.[<to_ #res_type>]()
                        }
                    }
                    fn _not(self) -> Self::Output {
                        paste::paste! {
                            !self.[<to_ #res_type>]()
                        }
                    }
                    fn _shl(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]() << rhs.[<to_ #res_type>]()
                        }
                    }
                    fn _shr(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]() >> rhs.[<to_ #res_type>]()
                        }
                    }
                }
            };
            ret.extend(res);
        }
    }

    ret.into()
}