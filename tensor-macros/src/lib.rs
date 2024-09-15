use binary_float_out::impl_float_out_binary;
use float_unary::impl_float_out_unary;
use kernel_gen_helper::{ __gen_fast_reduce_simd_helper, __gen_reduce_dim_not_include_simd_helper };
use proc_macro::TokenStream;
use simd_bitwise::impl_simd_bitwise_out;
use simd_convert::__impl_simd_convert;
use simd_float_out_binary::impl_simd_binary_out_float;
use syn::{ parse, parse_macro_input, Expr, Ident, Token };
mod type_utils;
mod list_enum;
mod simd_normal_out;
mod simd_convert;
mod simd_float_out_unary;
mod binary_float_out;
mod float_unary;
mod simd_eval;
mod simd_cmp;
mod simd_bitwise;
mod kernel_gen_helper;
mod simd_float_out_binary;
mod into_vec;
use crate::simd_cmp::impl_simd_cmp;
use quote::quote;
use type_utils::TypeInfo;
use proc_macro2::{ TokenStream as TokenStream2, TokenTree };
use crate::simd_normal_out::impl_simd_normal_out;

struct SelectionParser {
    start: Option<Expr>,
    end: Option<Expr>,
    step: Option<Expr>,
}

struct Selections {
    selections: Vec<TokenStream>,
}

impl parse::Parse for SelectionParser {
    fn parse(input: parse::ParseStream) -> syn::Result<Self> {
        let mut start: Option<Expr> = None;
        let mut end: Option<Expr> = None;
        let mut step: Option<Expr> = None;
        if
            input.peek(syn::Lit) ||
            input.peek(syn::Ident) ||
            input.peek(syn::token::Paren) ||
            input.peek(Token![-])
        {
            start = Some(input.parse::<syn::Expr>()?);
        }
        if input.peek(Token![:]) {
            input.parse::<Token![:]>()?;
        } else if input.is_empty() {
            return Ok(Self {
                start,
                end,
                step,
            });
        } else {
            return Err(
                syn::Error::new(input.span(), "unexpected token, expected `:`, Int or Ident")
            );
        }
        if
            input.peek(syn::Lit) ||
            input.peek(syn::Ident) ||
            input.peek(syn::token::Paren) ||
            input.peek(Token![-])
        {
            end = Some(input.parse::<syn::Expr>()?);
        }
        if input.peek(Token![:]) {
            input.parse::<Token![:]>()?;
        }
        if
            input.peek(syn::Lit) ||
            input.peek(syn::Ident) ||
            input.peek(syn::token::Paren) ||
            input.peek(Token![-])
        {
            step = Some(input.parse::<syn::Expr>()?);
        }
        Ok(Self {
            start,
            end,
            step,
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
        match (x.start, x.end, x.step) {
            (None, None, None) => {
                ret_stream.extend(quote!(Slice::Full));
            }
            (None, None, Some(step)) => {
                ret_stream.extend(quote!(Slice::StepByFullRange(#step)));
            }
            (None, Some(end), None) => {
                ret_stream.extend(quote!(Slice::RangeTo(#end)));
            }
            (None, Some(end), Some(step)) => {
                ret_stream.extend(quote!(Slice::StepByRangeTo((#end, #step))));
            }
            (Some(start), None, None) => {
                ret_stream.extend(quote!(Slice::From(#start)));
            }
            (Some(start), None, Some(step)) => {
                ret_stream.extend(quote!(Slice::StepByRangeFrom((#start, #step))));
            }
            (Some(start), Some(end), None) =>
                ret_stream.extend(quote!(Slice::Range((#start, #end)))),
            (Some(start), Some(end), Some(step)) =>
                ret_stream.extend(quote!(Slice::StepByRangeFromTo((#start, #end, #step)))),
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
pub fn float_out_binary(_: TokenStream) -> TokenStream {
    impl_float_out_binary()
}

#[proc_macro]
pub fn float_out_binary_simd(_: TokenStream) -> TokenStream {
    impl_simd_binary_out_float()
}

#[proc_macro]
pub fn float_out_unary(_: TokenStream) -> TokenStream {
    impl_float_out_unary()
}

#[proc_macro]
pub fn simd_float_out_unary(_: TokenStream) -> TokenStream {
    crate::simd_float_out_unary::impl_float_out_unary()
}

#[proc_macro]
pub fn simd_eval(_: TokenStream) -> TokenStream {
    crate::simd_eval::impl_simd_eval()
}

#[proc_macro]
pub fn simd_bitwise(_: TokenStream) -> TokenStream {
    impl_simd_bitwise_out()
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
        "bf16",
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

            let mul_add_method = if res_type.is_float() {
                quote! {
                    #[inline(always)]
                    fn _mul_add(self, a: #rhs_dtype, b: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]() * a.[<to_ #res_type>]() + b.[<to_ #res_type>]()
                        }
                    }
                }
            } else if res_type.is_bool() {
                quote! {
                    #[inline(always)]
                    fn _mul_add(self, a: #rhs_dtype, b: #rhs_dtype) -> Self::Output {
                        self || a && b
                    }
                }
            } else {
                quote! {
                    #[inline(always)]
                    fn _mul_add(self, a: #rhs_dtype, b: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().wrapping_mul(a.[<to_ #res_type>]()) + b.[<to_ #res_type>]()
                        }
                    }
                }
            };

            let neg_method = if lhs_dtype.is_float() {
                quote! {
                    #[inline(always)]
                    fn _neg(self) -> Self {
                        -self
                    }
                }
            } else if lhs_dtype.is_bool() {
                quote! {
                    #[inline(always)]
                    fn _neg(self) -> Self {
                        !self
                    }
                }
            } else if lhs_dtype.is_unsigned() {
                quote! {
                    #[inline(always)]
                    fn _neg(self) -> Self {
                        !self + 1
                    }
                }
            } else {
                quote! {
                    #[inline(always)]
                    fn _neg(self) -> Self {
                        -self
                    }
                }
            };

            let pow_method = if res_type.is_float() {
                quote! {
                    #[inline(always)]
                    fn _pow(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().powf(rhs.[<to_ #res_type>]())
                        }
                    }
                }
            } else {
                if res_type.is_bool() {
                    quote! {
                        #[inline(always)]
                        fn _pow(self, rhs: #rhs_dtype) -> Self::Output {
                            self || rhs
                        }
                    }
                } else {
                    quote! {
                        #[inline(always)]
                        fn _pow(self, rhs: #rhs_dtype) -> Self::Output {
                            paste::paste! {
                                self.[<to_ #res_type>]().pow(rhs.to_u32())
                            }
                        }
                    }
                }
            };

            let abs_method = if lhs_dtype.is_unsigned() {
                quote! {
                    #[inline(always)]
                    fn _abs(self) -> Self {
                        self
                    }
                }
            } else {
                quote! {
                    #[inline(always)]
                    fn _abs(self) -> Self {
                        self.abs()
                    }
                }
            };

            let ceil_method = if lhs_dtype.is_float() {
                quote! {
                    #[inline(always)]
                    fn _ceil(self) -> Self {
                        self.ceil()
                    }
                }
            } else {
                quote! {
                    #[inline(always)]
                    fn _ceil(self) -> Self {
                        self
                    }
                }
            };

            let floor_method = if lhs_dtype.is_float() {
                quote! {
                    #[inline(always)]
                    fn _floor(self) -> Self {
                        self.floor()
                    }
                }
            } else {
                quote! {
                    #[inline(always)]
                    fn _floor(self) -> Self {
                        self
                    }
                }
            };

            let sign_method = if res_type.is_float() {
                quote! {
                    #[inline(always)]
                    fn _sign(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().signum()
                        }
                    }
                }
            } else if res_type.is_unsigned() {
                quote! {
                    #[inline(always)]
                    fn _sign(self) -> Self::Output {
                        #res_type::ZERO
                    }
                }
            } else {
                quote! {
                    #[inline(always)]
                    fn _sign(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().signum()
                        }
                    }
                }
            };

            let cmp_method = if res_type.is_bool() {
                quote! {
                    #[inline(always)]
                    fn _max(self, rhs: #rhs_dtype) -> Self::Output {
                        self || rhs
                    }
                    #[inline(always)]
                    fn _min(self, rhs: #rhs_dtype) -> Self::Output {
                        self && rhs
                    }
                }
            } else {
                quote! {
                    #[inline(always)]
                    fn _max(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().max(rhs.[<to_ #res_type>]())
                        }
                    }
                    #[inline(always)]
                    fn _min(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().min(rhs.[<to_ #res_type>]())
                        }
                    }
                }
            };

            let round_method = if lhs_dtype.is_float() {
                quote! {
                    #[inline(always)]
                    fn _round(self) -> Self {
                        self.round()
                    }
                }
            } else {
                quote! {
                    #[inline(always)]
                    fn _round(self) -> Self {
                        self
                    }
                }
            };
            let std_ops = if res_type.is_bool() {
                quote! {
                    #[inline(always)]
                    fn _add(self, rhs: #rhs_dtype) -> Self::Output {
                        self || rhs
                    }
                    #[inline(always)]
                    fn _sub(self, rhs: #rhs_dtype) -> Self::Output {
                        self && !rhs
                    }
                    #[inline(always)]
                    fn _mul(self, rhs: #rhs_dtype) -> Self::Output {
                        self && rhs
                    }
                    #[inline(always)]
                    fn _rem(self, rhs: #rhs_dtype) -> Self::Output {
                        self && !rhs
                    }
                }
            } else {
                let op = |
                    method: &str,
                    op: TokenStream2,
                    std_op: &str
                | {
                    let method = Ident::new(method, proc_macro2::Span::call_site());
                    let std_op = Ident::new(std_op, proc_macro2::Span::call_site());
                    if res_type.is_float() {
                    quote! {
                        #[inline(always)]
                        fn #method(self, rhs: #rhs_dtype) -> Self::Output {
                            paste::paste! {
                                self.[<to_ #res_type>]() #op rhs.[<to_ #res_type>]()
                            }
                        }
                    }
                } else {
                    quote! {
                        #[inline(always)]
                        fn #method(self, rhs: #rhs_dtype) -> Self::Output {
                            paste::paste! {
                                self.[<to_ #res_type>]().#std_op(rhs.[<to_ #res_type>]())
                            }
                        }
                    }
                }};
                let add = op("_add", quote!(+), "wrapping_add");
                let mul = op("_mul", quote!(*), "wrapping_mul");
                let sub = op("_sub", quote!(-), "wrapping_sub");
                let rem = op("_rem", quote!(%), "wrapping_rem");
                quote! {
                #add
                #sub
                #mul
                #rem
            }
            };

            let res =
                quote! {
                impl NormalOut<#rhs_dtype> for #lhs_dtype {
                    type Output = #res_type;
                    #pow_method
                    #[inline(always)]
                    fn _square(self) -> Self {
                        self._mul(self)
                    }
                    #[inline(always)]
                    fn _clip(self, min: Self::Output, max: Self::Output) -> Self::Output {
                        paste::paste! {
                            let a = self.[<to_ #res_type>]();
                            let min = min.[<to_ #res_type>]();
                            let max = max.[<to_ #res_type>]();
                            if a < min { min } else if a > max { max } else { a }
                        }
                    }
                    #mul_add_method
                    #neg_method
                    #std_ops
                    #abs_method
                    #ceil_method
                    #floor_method
                    #sign_method
                    #cmp_method
                    #round_method
                }
            };
            ret.extend(res);
        }
    }

    ret.into()
}

#[proc_macro]
pub fn impl_normal_out_simd(_: TokenStream) -> TokenStream {
    impl_simd_normal_out()
}

#[proc_macro]
pub fn impl_simd_convert(_: TokenStream) -> TokenStream {
    __impl_simd_convert()
}

#[proc_macro]
pub fn simd_cmp(_: TokenStream) -> TokenStream {
    impl_simd_cmp()
}

#[proc_macro]
pub fn impl_into_vec(_: TokenStream) -> TokenStream {
    into_vec::into_vec()
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

            let shift = if res_type.is_bool() {
                quote! {
                    fn _shl(self, rhs: #rhs_dtype) -> Self::Output {
                        self || rhs
                    }
                    fn _shr(self, rhs: #rhs_dtype) -> Self::Output {
                        self && !rhs
                    }
                }
            } else {
                quote! {
                    #[inline(always)]
                    fn _shl(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().wrapping_shl(rhs.[<to_ #res_type>]())
                        }
                    }
                    #[inline(always)]
                    fn _shr(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().wrapping_shr(rhs.[<to_ #res_type>]())
                        }
                    }
                }
            };

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
                    #shift
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
                #[inline(always)]
                fn _is_true(&self) -> bool {
                    *self
                }
            }
        } else {
            if lhs_dtype.is_f32() {
                quote! {
                    #[inline(always)]
                    fn _is_true(&self) -> bool {
                        self.to_bits() & 0x7FFFFFFF != 0
                    }
                }
            } else {
                quote! {
                    #[inline(always)]
                    fn _is_true(&self) -> bool {
                        self == &#lhs_dtype::ZERO
                    }
                }
            }
        };

        let is_inf = if lhs_dtype.is_float() {
            quote! {
                #[inline(always)]
                fn _is_inf(&self) -> bool {
                    self.is_infinite()
                }
            }
        } else {
            quote! {
                #[inline(always)]
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

#[proc_macro]
pub fn gen_fast_reduce_simd_helper(input: TokenStream) -> TokenStream {
    __gen_fast_reduce_simd_helper(input)
}

#[proc_macro]
pub fn gen_reduce_dim_not_include_simd_helper(input: TokenStream) -> TokenStream {
    __gen_reduce_dim_not_include_simd_helper(input)
}
