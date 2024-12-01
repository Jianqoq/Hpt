//! # Tensor Macros
//!
//! This crate provides a set of macros to generate code for tensor operations.
//! These macros are used to simplify and automate common tasks such as defining
//! tensor operations, reducing dimensionality, and optimizing numerical computations.
//!
//! ## Examples
//!
//! Here's an example of using a macro from this crate:
//!
//! ```rust
//! // Example code using a macro from this crate
//! ```

#![deny(missing_docs)]

use binary_float_out::impl_float_out_binary;
use float_unary::impl_float_out_unary;
use from_scalar::__impl_from_scalar;
use kernel_gen_helper::{ __gen_fast_reduce_simd_helper, __gen_reduce_dim_not_include_simd_helper };
use normal_out::__impl_normal_out_binary;
use proc_macro::TokenStream;
use scalar_convert::__impl_scalar_convert;
use simd_bitwise::impl_simd_bitwise_out;
use simd_convert::__impl_simd_convert;
use simd_float_out_binary::impl_simd_binary_out_float;
use syn::{ parse, parse_macro_input, Expr, Ident, Token };
mod binary_float_out;
mod float_unary;
mod into_vec;
mod kernel_gen_helper;
mod list_enum;
mod simd_bitwise;
mod simd_cmp;
mod simd_convert;
mod simd_eval;
mod normal_out_unary;
mod simd_normal_unary;
mod simd_float_out_binary;
mod simd_float_out_unary;
mod simd_normal_out;
mod type_utils;
mod normal_out;
mod scalar_convert;
mod from_scalar;
mod conv2d;

mod fuse {
    pub(crate) mod start;
    pub(crate) mod node;
    pub(crate) mod fuse;
    pub(crate) mod kernel_type;
    pub(crate) mod gen_fuse;
    pub(crate) mod codegen;
    pub(crate) mod cfg;
    pub(crate) mod ty_infer;
    pub(crate) mod expr_ty;
    pub(crate) mod build_graph;
    pub(crate) mod use_define_visitor;
    pub(crate) mod variable_collector;
    pub(crate) mod phi_function;
    pub(crate) mod var_recover;
    pub(crate) mod expr_call_use_visitor;
}

use crate::simd_cmp::impl_simd_cmp;
use crate::simd_normal_out::impl_simd_normal_out;
use proc_macro2::{ TokenStream as TokenStream2, TokenTree };
use quote::quote;
use type_utils::TypeInfo;

/// number of registers available for the target architecture
#[cfg(target_feature = "avx2")]
const NUM_REG: usize = 16;
#[cfg(all(any(target_feature = "sse", target_arch = "arm"), not(target_feature = "avx2")))]
const NUM_REG: usize = 8;
#[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
const NUM_REG: usize = 32;

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
            start = Some(input.parse::<Expr>()?);
        }
        if input.peek(Token![:]) {
            input.parse::<Token![:]>()?;
        } else if input.is_empty() {
            return Ok(Self { start, end, step });
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
            end = Some(input.parse::<Expr>()?);
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
            step = Some(input.parse::<Expr>()?);
        }
        Ok(Self { start, end, step })
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

/// parse the input and generate the corresponding slice
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
            (Some(start), Some(end), None) => {
                ret_stream.extend(quote!(Slice::Range((#start, #end))));
            }
            (Some(start), Some(end), Some(step)) => {
                ret_stream.extend(quote!(Slice::StepByRangeFromTo((#start, #end, #step))));
            }
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

/// infer the type of the enum
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

/// infer the type of the result
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

/// implement float out binary trait
#[proc_macro]
pub fn float_out_binary(_: TokenStream) -> TokenStream {
    impl_float_out_binary()
}

/// implement simd float out binary trait
#[proc_macro]
pub fn float_out_binary_simd(_: TokenStream) -> TokenStream {
    impl_simd_binary_out_float()
}

/// implement float out unary trait
#[proc_macro]
pub fn float_out_unary(_: TokenStream) -> TokenStream {
    impl_float_out_unary()
}

/// implement simd float out unary trait
#[proc_macro]
pub fn simd_float_out_unary(_: TokenStream) -> TokenStream {
    simd_float_out_unary::impl_float_out_unary()
}

/// implement simd eval trait
#[proc_macro]
pub fn simd_eval(_: TokenStream) -> TokenStream {
    simd_eval::impl_simd_eval()
}

/// implement simd bitwise trait
#[proc_macro]
pub fn simd_bitwise(_: TokenStream) -> TokenStream {
    impl_simd_bitwise_out()
}

/// generate notmal out trait
#[proc_macro]
pub fn impl_normal_out_binary(_: TokenStream) -> TokenStream {
    __impl_normal_out_binary()
}

/// gemerate normal out unary trait
#[proc_macro]
pub fn impl_normal_out_unary(_: TokenStream) -> TokenStream {
    normal_out_unary::__impl_normal_out_unary()
}

/// gemerate normal out unary trait
#[proc_macro]
pub fn impl_normal_out_unary_simd(_: TokenStream) -> TokenStream {
    simd_normal_unary::impl_simd_normal_out_unary()
}

/// implement simd normal out trait
#[proc_macro]
pub fn impl_normal_out_simd(_: TokenStream) -> TokenStream {
    impl_simd_normal_out()
}

/// implement simd convert trait
#[proc_macro]
pub fn impl_simd_convert(_: TokenStream) -> TokenStream {
    __impl_simd_convert()
}

/// implement scalar convert trait
#[proc_macro]
pub fn impl_scalar_convert(_: TokenStream) -> TokenStream {
    __impl_scalar_convert()
}

/// implement from scalar trait
#[proc_macro]
pub fn impl_from_scalar(_: TokenStream) -> TokenStream {
    __impl_from_scalar()
}

/// implement simd cmp trait
#[proc_macro]
pub fn simd_cmp(_: TokenStream) -> TokenStream {
    impl_simd_cmp()
}

/// implment into vec trait
#[proc_macro]
pub fn impl_into_vec(_: TokenStream) -> TokenStream {
    into_vec::into_vec()
}

/// implement bitwise out trait
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
                    #[inline(always)]
                    fn _shl(self, rhs: #rhs_dtype) -> Self::Output {
                        self || rhs
                    }
                    #[inline(always)]
                    fn _shr(self, rhs: #rhs_dtype) -> Self::Output {
                        self && !rhs
                    }
                }
            } else {
                quote! {
                    #[inline(always)]
                    fn _shl(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().wrapping_shl(rhs.to_u32())
                        }
                    }
                    #[inline(always)]
                    fn _shr(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().wrapping_shr(rhs.to_u32())
                        }
                    }
                }
            };

            let res =
                quote! {
                impl BitWiseOut<#rhs_dtype> for #lhs_dtype {
                    type Output = #res_type;
                    #[inline(always)]
                    fn _bitand(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]() & rhs.[<to_ #res_type>]()
                        }
                    }
                    #[inline(always)]
                    fn _bitor(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]() | rhs.[<to_ #res_type>]()
                        }
                    }
                    #[inline(always)]
                    fn _bitxor(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]() ^ rhs.[<to_ #res_type>]()
                        }
                    }
                    #[inline(always)]
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

/// implement compare trait
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

/// implement eval trait
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
                        self != &0.0
                    }
                }
            } else {
                quote! {
                    #[inline(always)]
                    fn _is_true(&self) -> bool {
                        self != &#lhs_dtype::ZERO
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

/// generate fast reduce simd helper
#[proc_macro]
pub fn gen_fast_reduce_simd_helper(input: TokenStream) -> TokenStream {
    __gen_fast_reduce_simd_helper(input)
}

/// generate reduce dim not include simd helper
#[proc_macro]
pub fn gen_reduce_dim_not_include_simd_helper(input: TokenStream) -> TokenStream {
    __gen_reduce_dim_not_include_simd_helper(input)
}

/// declare const values
/// 
/// const OW_BLOCK: usize = ?;
/// 
/// const OC_BLOCK: usize = ?;
#[proc_macro]
pub fn conv2d_microkernel_declare_const(input: TokenStream) -> TokenStream {
    conv2d::conv2d_microkernel_declare_const(input)
}

/// generate conv2d inps
#[proc_macro]
pub fn conv2d_microkernel_gen_inps(input: TokenStream) -> TokenStream {
    conv2d::conv2d_microkernel_gen_inps(input)
}

/// generate conv2d inps
#[proc_macro]
pub fn transpose_conv2d_microkernel_gen_inps(input: TokenStream) -> TokenStream {
    conv2d::transpose_conv2d_microkernel_gen_inps(input)
}

/// generate conv2d inps
#[proc_macro]
pub fn conv2d_microkernel_gen_pad_inps(input: TokenStream) -> TokenStream {
    conv2d::conv2d_microkernel_gen_pad_inps(input)
}

/// generate transpose conv2d inps
#[proc_macro]
pub fn transpose_conv2d_microkernel_gen_masks(input: TokenStream) -> TokenStream {
    conv2d::transpose_conv2d_microkernel_gen_masks(input)
}

/// generate transpose conv2d inps
#[proc_macro]
pub fn transpose_conv2d_microkernel_gen_outs(input: TokenStream) -> TokenStream {
    conv2d::transpose_conv2d_microkernel_gen_outs(input)
}

/// generate pwconv2d inps
#[proc_macro]
pub fn pwconv2d_microkernel_gen_pad_inps(input: TokenStream) -> TokenStream {
    conv2d::pwconv2d_microkernel_gen_pad_inps(input)
}

/// generate conv2d inps
#[proc_macro]
pub fn dwconv2d_microkernel_gen_pad_inps(input: TokenStream) -> TokenStream {
    conv2d::dwconv2d_microkernel_gen_pad_inps(input)
}

/// generate conv2d kernels
#[proc_macro]
pub fn conv2d_microkernel_gen_kernels(input: TokenStream) -> TokenStream {
    conv2d::conv2d_microkernel_gen_kernels(input)
}

/// generate conv2d repeat results
#[proc_macro]
pub fn conv2d_microkernel_gen_results(input: TokenStream) -> TokenStream {
    conv2d::conv2d_microkernel_gen_results(input)
}

/// generate transpose conv2d repeat results
#[proc_macro]
pub fn transpose_conv2d_microkernel_gen_results(input: TokenStream) -> TokenStream {
    conv2d::transpose_conv2d_microkernel_gen_results(input)
}

/// generate transpose conv2d repeat results
#[proc_macro]
pub fn transpose_conv2d_microkernel_flush_results(input: TokenStream) -> TokenStream {
    conv2d::transpose_conv2d_microkernel_flush_results(input)
}

/// generate transpose conv2d repeat results
#[proc_macro]
pub fn transpose_conv2d_microkernel_pad_flush_results(input: TokenStream) -> TokenStream {
    conv2d::transpose_conv2d_microkernel_pad_flush_results(input)
}

/// generate conv2d repeat results
#[proc_macro]
pub fn dwconv2d_microkernel_gen_results(input: TokenStream) -> TokenStream {
    conv2d::dwconv2d_microkernel_gen_results(input)
}

/// generate maxpool2d kernels
/// generate conv2d repeat results
#[proc_macro]
pub fn maxpool2d_microkernel_gen_results(input: TokenStream) -> TokenStream {
    conv2d::maxpool2d_microkernel_gen_results(input)
}

/// perform fuse optimization
#[proc_macro_attribute]
pub fn fuse(_: TokenStream, item: TokenStream) -> TokenStream
{
    fuse::start::fuse_impl(item)
}

/// fuse proc macro
#[proc_macro]
pub fn fuse_proc_macro(item: TokenStream) -> TokenStream {
    fuse::start::fuse_proc_macro(item)
}
