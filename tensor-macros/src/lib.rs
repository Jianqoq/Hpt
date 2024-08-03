use proc_macro::TokenStream;
use syn::{ parse, parse_macro_input, Expr, Ident, Token };
mod type_utils;
mod list_enum;
use quote::quote;
use type_utils::TypeInfo;
use proc_macro2::{ TokenStream as TokenStream2, TokenTree };

#[derive(Debug)]
struct SelectionParser {
    start: Option<i64>,
    end: Option<i64>,
    step: Option<i64>,
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

impl parse::Parse for SelectionParser {
    fn parse(input: parse::ParseStream) -> syn::Result<Self> {
        let mut start: Option<i64> = None;
        let mut end: Option<i64> = None;
        let mut step: Option<i64> = None;
        let mut start_ident: Option<Ident> = None;
        let mut end_ident: Option<Ident> = None;
        let mut step_ident: Option<Ident> = None;
        let mut start_neg = false;
        let mut end_neg = false;
        let mut step_neg = false;
        if input.peek(syn::LitInt) {
            start = Some(input.parse::<syn::LitInt>()?.base10_parse::<i64>()?);
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
            return Err(
                syn::Error::new(input.span(), "unexpected token, expected `:`, Int or Ident")
            );
        }
        if input.peek(syn::LitInt) {
            end = Some(input.parse::<syn::LitInt>()?.base10_parse::<i64>()?);
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
            step = Some(input.parse::<syn::LitInt>()?.base10_parse::<i64>()?);
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

impl parse::Parse for Selections {
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
        Ok(Self { selections })
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
        if
            (x.start.is_some() || x.start_ident.is_some()) &&
            (x.end.is_some() || x.end_ident.is_some()) &&
            (x.step.is_some() || x.step_ident.is_some())
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
            ret_stream.extend(
                quote!(Slice::StepByRangeFromTo((#start_token, #end_token, #step_token)))
            );
        } else if
            x.start.is_none() &&
            x.start_ident.is_none() &&
            (x.end.is_some() || x.end_ident.is_some()) &&
            (x.step.is_some() || x.step_ident.is_some())
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
        } else if
            (x.start.is_some() || x.start_ident.is_some()) &&
            x.end.is_none() &&
            x.end_ident.is_none() &&
            (x.step.is_some() || x.step_ident.is_some())
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
        } else if
            (x.start.is_some() || x.start_ident.is_some()) &&
            (x.end.is_some() || x.end_ident.is_some()) &&
            x.step.is_none() &&
            x.step_ident.is_none()
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
        } else if
            x.start.is_none() &&
            x.start_ident.is_none() &&
            (x.end.is_some() || x.end_ident.is_some()) &&
            x.step.is_none() &&
            x.step_ident.is_none()
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
        } else if
            (x.start.is_some() || x.start_ident.is_some()) &&
            x.end.is_none() &&
            x.end_ident.is_none() &&
            x.step.is_none() &&
            x.step_ident.is_none()
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
        } else if
            x.start.is_none() &&
            x.start_ident.is_none() &&
            x.end.is_none() &&
            x.end_ident.is_none() &&
            (x.step.is_some() || x.step_ident.is_some())
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

impl parse::Parse for InferEnumType {
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

impl parse::Parse for GenericCal {
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
                    fn _erf(self) -> Self::Output {
                        paste::paste! {
                            erf(self.to_f64()).[<to_ #res_type>]()
                        }
                    }
                    fn _celu(self, alpha: Self::Output) -> Self::Output {
                        paste::paste! {
                            let x = self.[<to_ #res_type>]();
                            let alpha = alpha.[<to_ #res_type>]();
                            x.max(#res_type::ZERO) + (alpha * (x / alpha).exp() - #res_type::ONE).min(#res_type::ZERO)
                        }
                    }
                    fn _sigmoid(self) -> Self::Output {
                        paste::paste! {
                            #res_type::ONE / (#res_type::ONE + (-self.[<to_ #res_type>]()).exp())
                        }
                    }
                    fn _elu(self, alpha: Self::Output) -> Self::Output {
                        paste::paste! {
                            let x = self.[<to_ #res_type>]();
                            let alpha = alpha.[<to_ #res_type>]();
                            if x >= #res_type::ZERO {
                                x
                            } else {
                                alpha * (x.exp() - #res_type::ONE)
                            }
                        }
                    }
                    fn _leaky_relu(self, alpha: Self::Output) -> Self::Output {
                        paste::paste! {
                            let x = self.[<to_ #res_type>]();
                            let alpha = alpha.[<to_ #res_type>]();
                            if x >= #res_type::ZERO {
                                x
                            } else {
                                alpha * x
                            }
                        }
                    }
                    fn _relu(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().max(#res_type::ZERO)
                        }
                    }
                    fn _gelu(self) -> Self::Output {
                        paste::paste! {
                            let x = self.[<to_ #res_type>]();
                            let sqrt2_over_2 = std::f64::consts::FRAC_1_SQRT_2;
                            #res_type::HALF * x * (#res_type::ONE + erf(x.to_f64() * sqrt2_over_2).[<to_ #res_type>]())
                        }
                    }

                    fn _selu(self, alpha: Self::Output, scale: Self::Output) -> Self::Output {
                        paste::paste! {
                            let x = self.[<to_ #res_type>]();
                            let alpha = alpha.[<to_ #res_type>]();
                            let scale = scale.[<to_ #res_type>]();
                            if x > #res_type::ZERO {
                                scale * x
                            } else {
                                alpha * x.exp() - alpha
                            }
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

            let ceil_method = if res_type.is_float() {
                quote! {
                    fn _ceil(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().ceil()
                        }
                    }
                }
            } else {
                quote! {
                    fn _ceil(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]()
                        }
                    }
                }
            };

            let floor_method = if res_type.is_float() {
                quote! {
                    fn _floor(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().floor()
                        }
                    }
                }
            } else {
                quote! {
                    fn _floor(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]()
                        }
                    }
                }
            };

            let sign_method = if res_type.is_float() {
                quote! {
                    fn _sign(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().signum()
                        }
                    }
                }
            } else if res_type.is_unsigned() {
                quote! {
                    fn _sign(self) -> Self::Output {
                        paste::paste! {
                            #res_type::ZERO
                        }
                    }
                }
            } else {
                quote! {
                    fn _sign(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().signum()
                        }
                    }
                }
            };

            let cmp_method =
                quote! {
                    fn _max(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().max(rhs.[<to_ #res_type>]())
                        }
                    }
                    fn _min(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().min(rhs.[<to_ #res_type>]())
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
                    #ceil_method
                    #floor_method
                    #sign_method
                    #cmp_method
                    fn _clip(self, min: Self::Output, max: Self::Output) -> Self::Output {
                        paste::paste! {
                            let a = self.[<to_ #res_type>]();
                            let min = min.[<to_ #res_type>]();
                            let max = max.[<to_ #res_type>]();
                            if a < min { min } else if a > max { max } else { a }
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
pub fn impl_bitwise_out(_: TokenStream) -> TokenStream {
    let mut ret = proc_macro2::TokenStream::new();

    let types = ["bool", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "isize", "usize"];

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
                    fn _bitand(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]() & rhs.[<to_ #res_type>]()
                        }
                    }
                    fn _bitor(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]() | rhs.[<to_ #res_type>]()
                        }
                    }
                    fn _bitxor(self, rhs: #rhs_dtype) -> Self::Output {
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

#[proc_macro]
pub fn impl_cmp(_: TokenStream) -> TokenStream {
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

            let res =
                quote! {
                impl Cmp<#rhs_dtype> for #lhs_dtype {
                    fn _eq(self, rhs: #rhs_dtype) -> bool {
                        paste::paste! {
                            self.[<to_ #res_type>]() == rhs.[<to_ #res_type>]()
                        }
                    }
                    fn _ne(self, rhs: #rhs_dtype) -> bool {
                        paste::paste! {
                            self.[<to_ #res_type>]() != rhs.[<to_ #res_type>]()
                        }
                    }
                    fn _lt(self, rhs: #rhs_dtype) -> bool {
                        paste::paste! {
                            self.[<to_ #res_type>]() < rhs.[<to_ #res_type>]()
                        }
                    }

                    fn _le(self, rhs: #rhs_dtype) -> bool {
                        paste::paste! {
                            self.[<to_ #res_type>]() <= rhs.[<to_ #res_type>]()
                        }
                    }
                    fn _gt(self, rhs: #rhs_dtype) -> bool {
                        paste::paste! {
                            self.[<to_ #res_type>]() > rhs.[<to_ #res_type>]()
                        }
                    }
                    fn _ge(self, rhs: #rhs_dtype) -> bool {
                        paste::paste! {
                            self.[<to_ #res_type>]() >= rhs.[<to_ #res_type>]()
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
pub fn impl_eval(_: TokenStream) -> TokenStream {
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
        let lhs_type = TypeInfo::new(lhs);
        let lhs_dtype = lhs_type.dtype;

        let is_nan = if lhs_dtype.is_float() {
            quote! {
                fn _is_nan(&self) -> bool {
                    self.is_nan()
                }
            }
        } else {
            quote! {
                fn _is_nan(&self) -> bool {
                    false
                }
            }
        };

        let is_true = if lhs_dtype.is_bool() {
            quote! {
                fn _is_true(&self) -> bool {
                    *self
                }
            }
        } else {
            quote! {
                fn _is_true(&self) -> bool {
                    self == &#lhs_dtype::ZERO
                }
            }
        };

        let is_inf = if lhs_dtype.is_float() {
            quote! {
                fn _is_inf(&self) -> bool {
                    self.is_infinite()
                }
            }
        } else {
            quote! {
                fn _is_inf(&self) -> bool {
                    false
                }
            }
        };

        let res =
            quote! {
                impl Eval for #lhs_dtype {
                    type Output = bool;
                    #is_nan
                    #is_true
                    #is_inf
                }
            };
        ret.extend(res);
    }

    ret.into()
}

#[proc_macro]
pub fn impl_static_tensor_scalar_std_ops(_: TokenStream) -> TokenStream {
    let mut ret = proc_macro2::TokenStream::new();

    let types = [
        quote!(bool),
        quote!(f16),
        quote!(bf16),
        quote!(f32),
        quote!(f64),
        quote!(i8),
        quote!(i16),
        quote!(i32),
        quote!(i64),
        quote!(u8),
        quote!(u16),
        quote!(u32),
        quote!(u64),
        quote!(isize),
        quote!(usize),
    ];

    let std_ops = [quote!(Add), quote!(Div), quote!(Mul), quote!(Sub)];

    let std_ops_method = [quote!(add), quote!(div), quote!(mul), quote!(sub)];

    for lhs in types.iter() {
        for (op, method) in std_ops.iter().zip(std_ops_method.iter()) {
            ret.extend(
                quote!(
                impl_scalar_op_lhs!(#op, [], #lhs, [], Tensor, #method);
            )
            );
            ret.extend(
                quote!(
                impl_scalar_op_lhs!(#op, [&], #lhs, [], Tensor, #method);
            )
            );
            ret.extend(
                quote!(
                impl_scalar_op_lhs!(#op, [&], #lhs, [&], Tensor, #method);
            )
            );
            ret.extend(
                quote!(
                impl_scalar_op_lhs!(#op, [], #lhs, [&], Tensor, #method);
            )
            );
            ret.extend(
                quote!(
                impl_scalar_op_rhs!(#op, [], Tensor, [], #lhs, #method);
            )
            );
            ret.extend(
                quote!(
                impl_scalar_op_rhs!(#op, [], Tensor, [&], #lhs, #method);
            )
            );
            ret.extend(
                quote!(
                impl_scalar_op_rhs!(#op, [&], Tensor, [], #lhs, #method);
            )
            );
            ret.extend(
                quote!(
                impl_scalar_op_rhs!(#op, [&], Tensor, [&], #lhs, #method);
            )
            );
        }
    }

    ret.into()
}

#[proc_macro]
pub fn impl_tensor_slice_std_ops(_: TokenStream) -> TokenStream {
    let mut ret = proc_macro2::TokenStream::new();
    let types = [
        quote!(bool),
        quote!(f16),
        quote!(bf16),
        quote!(f32),
        quote!(f64),
        quote!(i8),
        quote!(i16),
        quote!(i32),
        quote!(i64),
        quote!(u8),
        quote!(u16),
        quote!(u32),
        quote!(u64),
        quote!(isize),
        quote!(usize),
        quote!(&bool),
        quote!(&f16),
        quote!(&bf16),
        quote!(&f32),
        quote!(&f64),
        quote!(&i8),
        quote!(&i16),
        quote!(&i32),
        quote!(&i64),
        quote!(&u8),
        quote!(&u16),
        quote!(&u32),
        quote!(&u64),
        quote!(&isize),
        quote!(&usize),
    ];

    let target = [quote!(TensorSlice), quote!(&TensorSlice)];

    let std_ops = [quote!(Add), quote!(Div), quote!(Mul), quote!(Sub), quote!(Rem)];

    let std_ops_method = [quote!(add), quote!(div), quote!(mul), quote!(sub), quote!(rem)];

    let prim_expr_methods = [quote!(Add), quote!(Div), quote!(Mul), quote!(Sub), quote!(Rem)];

    for lhs in types.iter() {
        for rhs in target.iter() {
            for ((op, method), prim_method) in std_ops
                .iter()
                .zip(std_ops_method.iter())
                .zip(prim_expr_methods.iter()) {
                ret.extend(
                    quote!(
                        impl std::ops::#op<#rhs> for #lhs {
                            type Output = PrimeExpr;
                            fn #method(self, rhs: #rhs) -> Self::Output {
                                PrimeExpr::#prim_method(#prim_method::make(self, rhs))
                            }
                        }
                    )
                );
            }
        }
    }

    ret.into()
}
