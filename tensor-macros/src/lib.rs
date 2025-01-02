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
#[cfg(feature = "cuda")]
use crate::binary_float_out::impl_cuda_float_out_binary;
use binary_float_out::impl_float_out_binary;
use float_unary::impl_float_out_unary;
use from_scalar::__impl_from_scalar;
use kernel_gen_helper::{__gen_fast_reduce_simd_helper, __gen_reduce_dim_not_include_simd_helper};
use normal_out::__impl_normal_out_binary;
use proc_macro::TokenStream;
use scalar_convert::__impl_scalar_convert;
use simd_bitwise::impl_simd_bitwise_out;
use simd_convert::__impl_simd_convert;
use simd_float_out_binary::{
    impl_simd_binary_out_float, impl_simd_binary_out_float_lhs_scalar,
    impl_simd_binary_out_float_rhs_scalar,
};
use simd_normal_out::{impl_simd_normal_out_with_lhs_scalar, impl_simd_normal_out_with_rhs_scalar};
use syn::{parse, parse_macro_input, Expr, Token};
mod binary_float_out;
mod conv2d;
mod float_unary;
mod from_scalar;
mod into_cuda_scalar;
mod into_scalar;
mod into_vec;
mod kernel_gen_helper;
mod normal_out;
mod normal_out_unary;
mod scalar_convert;
mod simd_bitwise;
mod simd_cmp;
mod simd_convert;
mod simd_eval;
mod simd_float_out_binary;
mod simd_float_out_unary;
mod simd_normal_out;
mod simd_normal_unary;
mod type_utils;

use crate::simd_cmp::impl_simd_cmp;
use crate::simd_normal_out::impl_simd_normal_out;
use proc_macro2::{TokenStream as TokenStream2, TokenTree};
use quote::{format_ident, quote};
use type_utils::TypeInfo;

/// number of registers available for the target architecture
#[cfg(target_feature = "avx2")]
const NUM_REG: usize = 16;
#[cfg(all(
    any(target_feature = "sse", target_arch = "arm"),
    not(target_feature = "avx2")
))]
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
        if input.peek(syn::Lit)
            || input.peek(syn::Ident)
            || input.peek(syn::token::Paren)
            || input.peek(Token![-])
        {
            start = Some(input.parse::<Expr>()?);
        }
        if input.peek(Token![:]) {
            input.parse::<Token![:]>()?;
        } else if input.is_empty() {
            return Ok(Self { start, end, step });
        } else {
            return Err(syn::Error::new(
                input.span(),
                "unexpected token, expected `:`, Int or Ident",
            ));
        }
        if input.peek(syn::Lit)
            || input.peek(syn::Ident)
            || input.peek(syn::token::Paren)
            || input.peek(Token![-])
        {
            end = Some(input.parse::<Expr>()?);
        }
        if input.peek(Token![:]) {
            input.parse::<Token![:]>()?;
        }
        if input.peek(syn::Lit)
            || input.peek(syn::Ident)
            || input.peek(syn::token::Paren)
            || input.peek(Token![-])
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

/// implement float out binary trait
#[proc_macro]
pub fn float_out_binary(_: TokenStream) -> TokenStream {
    impl_float_out_binary()
}

#[cfg(feature = "cuda")]
/// implement float out binary trait for cuda
#[proc_macro]
pub fn float_out_binary_cuda(_: TokenStream) -> TokenStream {
    impl_cuda_float_out_binary()
}

/// implement simd float out binary trait
#[proc_macro]
pub fn float_out_binary_simd(_: TokenStream) -> TokenStream {
    impl_simd_binary_out_float()
}

/// implement simd float out binary trait with rhs scalar
#[proc_macro]
pub fn float_out_binary_simd_with_rhs_scalar(_: TokenStream) -> TokenStream {
    impl_simd_binary_out_float_rhs_scalar()
}

/// implement simd float out binary trait with lhs scalar
#[proc_macro]
pub fn float_out_binary_simd_with_lhs_scalar(_: TokenStream) -> TokenStream {
    impl_simd_binary_out_float_lhs_scalar()
}

/// implement float out unary trait
#[proc_macro]
pub fn float_out_unary(_: TokenStream) -> TokenStream {
    impl_float_out_unary()
}

#[cfg(feature = "cuda")]
/// implement float out unary trait for cuda
#[proc_macro]
pub fn float_out_unary_cuda(_: TokenStream) -> TokenStream {
    crate::float_unary::impl_cuda_float_out_unary()
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

#[cfg(feature = "cuda")]
/// generate notmal out trait
#[proc_macro]
pub fn impl_cuda_normal_out_binary(_: TokenStream) -> TokenStream {
    crate::normal_out::__impl_cuda_normal_out_binary()
}

/// gemerate normal out unary trait
#[proc_macro]
pub fn impl_normal_out_unary(_: TokenStream) -> TokenStream {
    normal_out_unary::__impl_normal_out_unary()
}

#[cfg(feature = "cuda")]
/// gemerate normal out unary trait
#[proc_macro]
pub fn impl_normal_out_unary_cuda(_: TokenStream) -> TokenStream {
    normal_out_unary::__impl_normal_out_unary_cuda()
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

/// implement simd normal out trait with rhs scalar
#[proc_macro]
pub fn impl_normal_out_simd_with_rhs_scalar(_: TokenStream) -> TokenStream {
    impl_simd_normal_out_with_rhs_scalar()
}

/// implement simd normal out trait with lhs scalar
#[proc_macro]
pub fn impl_normal_out_simd_with_lhs_scalar(_: TokenStream) -> TokenStream {
    impl_simd_normal_out_with_lhs_scalar()
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

#[cfg(feature = "cuda")]
/// implment into cuda scalar trait
#[proc_macro]
pub fn impl_into_cuda_scalar(_: TokenStream) -> TokenStream {
    into_cuda_scalar::__impl_into_cuda_scalar().into()
}

/// implment into scalar trait
#[proc_macro]
pub fn impl_into_scalar(_: TokenStream) -> TokenStream {
    into_scalar::__impl_into_scalar().into()
}

/// implement bitwise out trait
#[proc_macro]
pub fn impl_bitwise_out(_: TokenStream) -> TokenStream {
    let mut ret = proc_macro2::TokenStream::new();

    let types = [
        "bool", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "isize", "usize",
    ];

    for lhs in types.iter() {
        for rhs in types.iter() {
            let lhs_type = TypeInfo::new(lhs);
            let rhs_type = TypeInfo::new(rhs);
            let lhs_dtype = lhs_type.dtype;
            let rhs_dtype = rhs_type.dtype;
            let res = if lhs_dtype == rhs_dtype {
                quote! {
                    impl BitWiseOut<#rhs_dtype> for #lhs_dtype {
                        type Output = <#lhs_dtype as NormalOutPromote<#rhs_dtype>>::Output;
                        #[inline(always)]
                        fn _bitand(self, rhs: #rhs_dtype) -> Self::Output {
                            self.__bitand(rhs)
                        }
                        #[inline(always)]
                        fn _bitor(self, rhs: #rhs_dtype) -> Self::Output {
                            self.__bitor(rhs)
                        }
                        #[inline(always)]
                        fn _bitxor(self, rhs: #rhs_dtype) -> Self::Output {
                            self.__bitxor(rhs)
                        }
                        #[inline(always)]
                        fn _not(self) -> Self::Output {
                            self.__not()
                        }
                        #[inline(always)]
                        fn _shl(self, rhs: #rhs_dtype) -> Self::Output {
                            self.__shl(rhs)
                        }
                        #[inline(always)]
                        fn _shr(self, rhs: #rhs_dtype) -> Self::Output {
                            self.__shr(rhs)
                        }
                    }
                }
            } else {
                quote! {
                    impl BitWiseOut<#rhs_dtype> for #lhs_dtype {
                        type Output = <#lhs_dtype as NormalOutPromote<#rhs_dtype>>::Output;
                        #[inline(always)]
                        fn _bitand(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs: Self::Output = self.into_scalar();
                            let rhs: Self::Output = rhs.into_scalar();
                            lhs.__bitand(rhs)
                        }
                        #[inline(always)]
                        fn _bitor(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs: Self::Output = self.into_scalar();
                            let rhs: Self::Output = rhs.into_scalar();
                            lhs.__bitor(rhs)
                        }
                        #[inline(always)]
                        fn _bitxor(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs: Self::Output = self.into_scalar();
                            let rhs: Self::Output = rhs.into_scalar();
                            lhs.__bitxor(rhs)
                        }
                        #[inline(always)]
                        fn _not(self) -> Self::Output {
                            let lhs: Self::Output = self.into_scalar();
                            lhs.__not()
                        }
                        #[inline(always)]
                        fn _shl(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs: Self::Output = self.into_scalar();
                            let rhs: Self::Output = rhs.into_scalar();
                            lhs.__shl(rhs)
                        }
                        #[inline(always)]
                        fn _shr(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs: Self::Output = self.into_scalar();
                            let rhs: Self::Output = rhs.into_scalar();
                            lhs.__shr(rhs)
                        }
                    }
                }
            };
            ret.extend(res);
        }
    }

    ret.into()
}

/// implement bitwise out trait
#[proc_macro]
pub fn impl_cuda_bitwise_out(_: TokenStream) -> TokenStream {
    let mut ret = proc_macro2::TokenStream::new();

    let types = [
        "bool", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "isize", "usize",
    ];

    for lhs in types.iter() {
        for rhs in types.iter() {
            let lhs_type = TypeInfo::new(lhs);
            let rhs_type = TypeInfo::new(rhs);
            let lhs_dtype = lhs_type.dtype;
            let rhs_dtype = rhs_type.dtype;
            let res = if lhs_dtype == rhs_dtype {
                quote! {
                    impl BitWiseOut<Scalar<#rhs_dtype>> for Scalar<#lhs_dtype> {
                        type Output = <Scalar<#lhs_dtype> as NormalOutPromote<Scalar<#rhs_dtype>>>::Output;
                        #[inline(always)]
                        fn _bitand(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            self.__bitand(rhs)
                        }
                        #[inline(always)]
                        fn _bitor(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            self.__bitor(rhs)
                        }
                        #[inline(always)]
                        fn _bitxor(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            self.__bitxor(rhs)
                        }
                        #[inline(always)]
                        fn _not(self) -> Self::Output {
                            self.__not()
                        }
                        #[inline(always)]
                        fn _shl(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            self.__shl(rhs)
                        }
                        #[inline(always)]
                        fn _shr(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            self.__shr(rhs)
                        }
                    }
                }
            } else {
                quote! {
                    impl BitWiseOut<Scalar<#rhs_dtype>> for Scalar<#lhs_dtype> {
                        type Output = <Scalar<#lhs_dtype> as NormalOutPromote<Scalar<#rhs_dtype>>>::Output;
                        #[inline(always)]
                        fn _bitand(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            let lhs: Self::Output = self.into_scalar();
                            let rhs: Self::Output = rhs.into_scalar();
                            lhs.__bitand(rhs)
                        }
                        #[inline(always)]
                        fn _bitor(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            let lhs: Self::Output = self.into_scalar();
                            let rhs: Self::Output = rhs.into_scalar();
                            lhs.__bitor(rhs)
                        }
                        #[inline(always)]
                        fn _bitxor(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            let lhs: Self::Output = self.into_scalar();
                            let rhs: Self::Output = rhs.into_scalar();
                            lhs.__bitxor(rhs)
                        }
                        #[inline(always)]
                        fn _not(self) -> Self::Output {
                            let lhs: Self::Output = self.into_scalar();
                            lhs.__not()
                        }
                        #[inline(always)]
                        fn _shl(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            let lhs: Self::Output = self.into_scalar();
                            let rhs: Self::Output = rhs.into_scalar();
                            lhs.__shl(rhs)
                        }
                        #[inline(always)]
                        fn _shr(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            let lhs: Self::Output = self.into_scalar();
                            let rhs: Self::Output = rhs.into_scalar();
                            lhs.__shr(rhs)
                        }
                    }
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
        "bool", "f16", "f32", "f64", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "isize",
        "usize", "bf16",
    ];

    for lhs in types.iter() {
        for rhs in types.iter() {
            let lhs_type = TypeInfo::new(lhs);
            let rhs_type = TypeInfo::new(rhs);
            let lhs_dtype = lhs_type.dtype;
            let rhs_dtype = rhs_type.dtype;
            let res = if lhs_dtype == rhs_dtype {
                quote! {
                    impl Cmp<#rhs_dtype> for #lhs_dtype {
                        type Output = bool;
                        fn _eq(self, rhs: #rhs_dtype) -> Self::Output {
                            self == rhs
                        }
                        fn _ne(self, rhs: #rhs_dtype) -> Self::Output {
                            self != rhs
                        }
                        fn _lt(self, rhs: #rhs_dtype) -> Self::Output {
                            self < rhs
                        }

                        fn _le(self, rhs: #rhs_dtype) -> Self::Output {
                            self <= rhs
                        }
                        fn _gt(self, rhs: #rhs_dtype) -> Self::Output {
                            self > rhs
                        }
                        fn _ge(self, rhs: #rhs_dtype) -> Self::Output {
                            self >= rhs
                        }
                    }
                }
            } else {
                quote! {
                    impl Cmp<#rhs_dtype> for #lhs_dtype {
                        type Output = bool;
                        fn _eq(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs: <#lhs_dtype as NormalOutPromote<#rhs_dtype>>::Output = self.into_scalar();
                            let rhs: <#lhs_dtype as NormalOutPromote<#rhs_dtype>>::Output = rhs.into_scalar();
                            lhs == rhs
                        }
                        fn _ne(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs: <#lhs_dtype as NormalOutPromote<#rhs_dtype>>::Output = self.into_scalar();
                            let rhs: <#lhs_dtype as NormalOutPromote<#rhs_dtype>>::Output = rhs.into_scalar();
                            lhs != rhs
                        }
                        fn _lt(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs: <#lhs_dtype as NormalOutPromote<#rhs_dtype>>::Output = self.into_scalar();
                            let rhs: <#lhs_dtype as NormalOutPromote<#rhs_dtype>>::Output = rhs.into_scalar();
                            lhs < rhs
                        }

                        fn _le(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs: <#lhs_dtype as NormalOutPromote<#rhs_dtype>>::Output = self.into_scalar();
                            let rhs: <#lhs_dtype as NormalOutPromote<#rhs_dtype>>::Output = rhs.into_scalar();
                            lhs <= rhs
                        }
                        fn _gt(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs: <#lhs_dtype as NormalOutPromote<#rhs_dtype>>::Output = self.into_scalar();
                            let rhs: <#lhs_dtype as NormalOutPromote<#rhs_dtype>>::Output = rhs.into_scalar();
                            lhs > rhs
                        }
                        fn _ge(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs: <#lhs_dtype as NormalOutPromote<#rhs_dtype>>::Output = self.into_scalar();
                            let rhs: <#lhs_dtype as NormalOutPromote<#rhs_dtype>>::Output = rhs.into_scalar();
                            lhs >= rhs
                        }
                    }
                }
            };
            ret.extend(res);
        }
    }

    ret.into()
}

/// implement compare trait
#[proc_macro]
pub fn impl_cmp_cuda(_: TokenStream) -> TokenStream {
    let mut ret = proc_macro2::TokenStream::new();

    let types = [
        "bool", "f16", "f32", "f64", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "isize",
        "usize", "bf16",
    ];

    for lhs in types.iter() {
        for rhs in types.iter() {
            let lhs_type = TypeInfo::new(lhs);
            let rhs_type = TypeInfo::new(rhs);
            let lhs_dtype = lhs_type.dtype;
            let rhs_dtype = rhs_type.dtype;
            let res = if lhs_dtype == rhs_dtype {
                quote! {
                    impl Cmp<Scalar<#rhs_dtype>> for Scalar<#lhs_dtype> {
                        type Output = Scalar<bool>;
                        fn _eq(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            self.__eq(rhs)
                        }
                        fn _ne(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            self.__ne(rhs)
                        }
                        fn _lt(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            self.__lt(rhs)
                        }

                        fn _le(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            self.__le(rhs)
                        }
                        fn _gt(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            self.__gt(rhs)
                        }
                        fn _ge(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            self.__ge(rhs)
                        }
                    }
                }
            } else {
                quote! {
                    impl Cmp<Scalar<#rhs_dtype>> for Scalar<#lhs_dtype> {
                        type Output = Scalar<bool>;
                        fn _eq(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            let lhs: <Scalar<#lhs_dtype> as NormalOutPromote<Scalar<#rhs_dtype>>>::Output = self.into_scalar();
                            let rhs: <Scalar<#lhs_dtype> as NormalOutPromote<Scalar<#rhs_dtype>>>::Output = rhs.into_scalar();
                            lhs.__eq(rhs)
                        }
                        fn _ne(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            let lhs: <Scalar<#lhs_dtype> as NormalOutPromote<Scalar<#rhs_dtype>>>::Output = self.into_scalar();
                            let rhs: <Scalar<#lhs_dtype> as NormalOutPromote<Scalar<#rhs_dtype>>>::Output = rhs.into_scalar();
                            lhs.__ne(rhs)
                        }
                        fn _lt(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            let lhs: <Scalar<#lhs_dtype> as NormalOutPromote<Scalar<#rhs_dtype>>>::Output = self.into_scalar();
                            let rhs: <Scalar<#lhs_dtype> as NormalOutPromote<Scalar<#rhs_dtype>>>::Output = rhs.into_scalar();
                            lhs.__lt(rhs)
                        }

                        fn _le(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            let lhs: <Scalar<#lhs_dtype> as NormalOutPromote<Scalar<#rhs_dtype>>>::Output = self.into_scalar();
                            let rhs: <Scalar<#lhs_dtype> as NormalOutPromote<Scalar<#rhs_dtype>>>::Output = rhs.into_scalar();
                            lhs.__le(rhs)
                        }
                        fn _gt(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            let lhs: <Scalar<#lhs_dtype> as NormalOutPromote<Scalar<#rhs_dtype>>>::Output = self.into_scalar();
                            let rhs: <Scalar<#lhs_dtype> as NormalOutPromote<Scalar<#rhs_dtype>>>::Output = rhs.into_scalar();
                            lhs.__gt(rhs)
                        }
                        fn _ge(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            let lhs: <Scalar<#lhs_dtype> as NormalOutPromote<Scalar<#rhs_dtype>>>::Output = self.into_scalar();
                            let rhs: <Scalar<#lhs_dtype> as NormalOutPromote<Scalar<#rhs_dtype>>>::Output = rhs.into_scalar();
                            lhs.__ge(rhs)
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
        "bool", "f16", "f32", "f64", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "isize",
        "usize", "bf16",
    ];

    for lhs in types.iter() {
        let lhs_type = TypeInfo::new(lhs);
        let lhs_dtype = lhs_type.dtype;

        let res = quote! {
            impl Eval for #lhs_dtype {
                type Output = bool;
                #[inline(always)]
                fn _is_nan(&self) -> bool {
                    self.__is_nan()
                }
                #[inline(always)]
                fn _is_true(&self) -> bool {
                    self.__is_true()
                }
                #[inline(always)]
                fn _is_inf(&self) -> bool {
                    self.__is_inf()
                }
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
pub fn conv2d_microkernel_gen_pad_inps(input: TokenStream) -> TokenStream {
    conv2d::conv2d_microkernel_gen_pad_inps(input)
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

/// generate save trait
#[proc_macro_derive(Save, attributes(compress))]
pub fn impl_save(input: TokenStream) -> TokenStream {
    let ast = syn::parse_macro_input!(input as syn::DeriveInput);
    let name = &ast.ident;
    let meta_name = format_ident!("{}Meta", name);

    let visibility = &ast.vis;
    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();
    let fields = match &ast.data {
        syn::Data::Struct(s) => &s.fields,
        _ => panic!("Save can only be derived for structs"),
    };

    let mut compressions = vec![];
    let mut endians = vec![];
    let mut compress_levels = vec![];

    let meta_fields = fields
        .iter()
        .map(|f| {
            let mut compression_algo = None;
            let mut endian = None;
            let mut level = None;

            for attr in &f.attrs {
                if attr.path().is_ident("compress") {
                    attr.parse_nested_meta(|meta| {
                        if meta.path.is_ident("algo") {
                            let value: syn::LitStr = meta.value()?.parse()?;
                            let algo = match value.value().as_str().to_lowercase().as_str() {
                                "gzip" => quote!(Gzip),
                                "deflate" => quote!(Deflate),
                                "zlib" => quote!(Zlib),
                                "none" => quote!(NoCompression),
                                _ => panic!("Unsupported compression algorithm, supported: gzip, deflate, zlib, none"),
                            };
                            compression_algo = Some(quote!(tensor_dyn::CompressionAlgo::#algo));
                        } else if meta.path.is_ident("level") {
                            let value: syn::LitStr = meta.value()?.parse()?;
                            let tmp: u32 = value.value().parse().map_err(|e| {
                                syn::Error::new(value.span(), format!("Invalid level: {}", e))
                            })?;
                            level = Some(quote!(#tmp));
                        } else if meta.path.is_ident("endian") {
                            let value: syn::LitStr = meta.value()?.parse()?;
                            let tmp = match value.value().as_str() {
                                "native" => quote!(Native),
                                "little" => quote!(Little),
                                "big" => quote!(Big),
                                _ => panic!("Unsupported endianness, supported: native, little, big"),
                            };
                            endian = Some(quote!(tensor_dyn::Endian::#tmp));
                        }
                        Ok(())
                    })
                    .unwrap();
                }
            }
            compressions.push(compression_algo);
            endians.push(endian);
            compress_levels.push(level);
            let name = &f.ident;
            let ty = &f.ty;
            quote! {
                pub #name: <#ty as Save>::Meta
            }
        })
        .collect::<Vec<_>>();

    let call_save = fields.iter().enumerate().map(|(idx, f)| {
        let name = &f.ident;
        let ty = &f.ty;
        let ident = format_ident!("field_{}", idx);
        let compression_algo = compressions[idx].clone().unwrap_or(quote!(compression_algo));
        let endian = endians[idx].clone().unwrap_or(quote!(endian));
        let level = compress_levels[idx].clone().unwrap_or(quote!(level));
        if let Some(name) = name {
            quote! {
                let #ident = <#ty as Save>::__save(&data.#name, file, len_so_far, global_cnt, #compression_algo, #endian, #level)?;
            }
        } else {
            quote! {
                let #ident = <#ty as Save>::__save(&data.#idx, file, len_so_far, global_cnt, #compression_algo, #endian, #level)?;
            }
        }
    });

    let construct_fields = fields.iter().enumerate().map(|(idx, f)| {
        let name = &f.ident;
        let ident = format_ident!("field_{}", idx);
        quote! {
            #name: #ident
        }
    });

    let expanded = quote! {
        #[derive(serde::Deserialize, serde::Serialize)]
        #visibility struct #meta_name #ty_generics #where_clause  {
            #(#meta_fields,)*
        }
        impl #impl_generics Save for #name #ty_generics #where_clause {
            type Meta = #meta_name #ty_generics;
            fn __save(
                data: &Self,
                file: &mut std::fs::File,
                len_so_far: &mut usize,
                global_cnt: &mut usize,
                compression_algo: tensor_dyn::CompressionAlgo,
                endian: tensor_dyn::Endian,
                level: u32,
            ) -> std::io::Result<Self::Meta> {
                #(#call_save)*
                Ok(Self::Meta {
                    #(#construct_fields),*
                })
            }
        }
    };

    expanded.into()
}

/// generate load trait
#[proc_macro_derive(Load)]
pub fn impl_load(input: TokenStream) -> TokenStream {
    let ast = syn::parse_macro_input!(input as syn::DeriveInput);
    let name = &ast.ident;
    let meta_name = format_ident!("{}Meta", name);
    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();

    let fields = match &ast.data {
        syn::Data::Struct(s) => &s.fields,
        _ => panic!("Load can only be derived for structs"),
    };

    let call_load = fields.iter().enumerate().map(|(idx, f)| {
        let name = &f.ident;
        let ident = format_ident!("field_{}", idx);
        if let Some(name) = name {
            quote! {
                let #ident = self.#name.load(file)?;
            }
        } else {
            quote! {
                let #ident = self.#idx.load(file)?;
            }
        }
    });

    let construct_fields = fields.iter().enumerate().map(|(idx, f)| {
        let name = &f.ident;
        let ident = format_ident!("field_{}", idx);
        quote! {
            #name: #ident
        }
    });

    let expanded = quote! {
        impl #impl_generics MetaLoad for #meta_name #ty_generics #where_clause {
            type Output = #name #ty_generics;
            fn load(&self, file: &mut std::fs::File) -> std::io::Result<Self::Output> {
                #(#call_load)*
                Ok(#name {
                    #(#construct_fields),*
                })
            }
        }
        impl #impl_generics Load for #name #ty_generics #where_clause {
            fn load(path: &str) -> std::io::Result<Self> {
                let meta = parse_header_compressed::<Self>(path).expect(format!("failed to parse header for {}", stringify!(#name)).as_str());
                let mut file = File::open(path)?;
                meta.load(&mut file)
            }
        }
    };

    expanded.into()
}
