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
use kernel_gen_helper::{
    __gen_fast_layernorm_simd_helper, __gen_fast_reduce_simd_helper,
    __gen_reduce_dim_not_include_simd_helper,
};
use normal_out::__impl_normal_out_binary;
use proc_macro::TokenStream;
use scalar_convert::__impl_scalar_convert;
use simd_bitwise::impl_simd_bitwise_out;
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
    skip: bool,
}

struct Selections {
    selections: Vec<TokenStream>,
}

impl parse::Parse for SelectionParser {
    fn parse(input: parse::ParseStream) -> syn::Result<Self> {
        let mut start: Option<Expr> = None;
        let mut end: Option<Expr> = None;
        let mut step: Option<Expr> = None;
        if input.peek(Token![..]) {
            input.parse::<Token![..]>()?;
            return Ok(Self {
                start,
                end,
                step,
                skip: true,
            });
        }
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
            return Ok(Self {
                start,
                end,
                step,
                skip: false,
            });
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
        Ok(Self {
            start,
            end,
            step,
            skip: false,
        })
    }
}

impl parse::Parse for Selections {
    fn parse(input: parse::ParseStream) -> syn::Result<Self> {
        let mut selections: Vec<TokenStream> = vec![];

        while !input.is_empty() {
            let mut item_tokens = TokenStream2::new();
            while !input.is_empty() && !input.peek(Token![,]) {
                let token = input.parse::<TokenTree>()?;
                item_tokens.extend(quote!(#token));
            }

            selections.push(item_tokens.into());

            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(Self { selections })
    }
}

/// parse the input and generate the corresponding slice
#[proc_macro]
pub fn select(input: TokenStream) -> TokenStream {
    let res: Selections = parse_macro_input!(input as Selections);
    let mut slices: Vec<SelectionParser> = vec![];
    for x in res.selections {
        slices.push(parse_macro_input!(x as SelectionParser));
    }
    let mut ret_stream = TokenStream2::new();
    let len = slices.len();
    let mut skipped = false;
    for (idx, x) in slices.into_iter().enumerate() {
        if x.skip {
            if skipped {
                return syn::Error::new(
                    proc_macro2::Span::call_site(),
                    "unexpected token, slicing only support `..` once",
                )
                .to_compile_error()
                .into();
            }
            ret_stream.extend(quote!((0, 0, 0x7FFFFFFFFFFFFFFF)));
            skipped = true;
            if idx != len - 1 {
                ret_stream.extend(quote!(,));
            }
            continue;
        }
        match (x.start, x.end, x.step) {
            (None, None, None) => {
                ret_stream.extend(quote!(((0, 0x7FFFFFFFFFFFFFFF, 1))));
            }
            (None, None, Some(step)) => {
                ret_stream.extend(quote!((0, 0x7FFFFFFFFFFFFFFF, #step)));
            }
            (None, Some(end), None) => {
                ret_stream.extend(quote!((0, #end, 1)));
            }
            (None, Some(end), Some(step)) => {
                ret_stream.extend(quote!((0, #end, #step)));
            }
            (Some(start), None, None) => {
                ret_stream.extend(quote!((#start, 0x7FFFFFFFFFFFFFFF, 1)));
            }
            (Some(start), None, Some(step)) => {
                ret_stream.extend(quote!((#start, 0x7FFFFFFFFFFFFFFF, #step)));
            }
            (Some(start), Some(end), None) => {
                ret_stream.extend(quote!((#start, #end, 1)));
            }
            (Some(start), Some(end), Some(step)) => {
                ret_stream.extend(quote!((#start, #end, #step)));
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
                            let lhs: Self::Output = self.cast();
                            let rhs: Self::Output = rhs.cast();
                            lhs.__bitand(rhs)
                        }
                        #[inline(always)]
                        fn _bitor(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs: Self::Output = self.cast();
                            let rhs: Self::Output = rhs.cast();
                            lhs.__bitor(rhs)
                        }
                        #[inline(always)]
                        fn _bitxor(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs: Self::Output = self.cast();
                            let rhs: Self::Output = rhs.cast();
                            lhs.__bitxor(rhs)
                        }
                        #[inline(always)]
                        fn _not(self) -> Self::Output {
                            let lhs: Self::Output = self.cast();
                            lhs.__not()
                        }
                        #[inline(always)]
                        fn _shl(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs: Self::Output = self.cast();
                            let rhs: Self::Output = rhs.cast();
                            lhs.__shl(rhs)
                        }
                        #[inline(always)]
                        fn _shr(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs: Self::Output = self.cast();
                            let rhs: Self::Output = rhs.cast();
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
                            let lhs: Self::Output = self.cast();
                            let rhs: Self::Output = rhs.cast();
                            lhs.__bitand(rhs)
                        }
                        #[inline(always)]
                        fn _bitor(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            let lhs: Self::Output = self.cast();
                            let rhs: Self::Output = rhs.cast();
                            lhs.__bitor(rhs)
                        }
                        #[inline(always)]
                        fn _bitxor(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            let lhs: Self::Output = self.cast();
                            let rhs: Self::Output = rhs.cast();
                            lhs.__bitxor(rhs)
                        }
                        #[inline(always)]
                        fn _not(self) -> Self::Output {
                            let lhs: Self::Output = self.cast();
                            lhs.__not()
                        }
                        #[inline(always)]
                        fn _shl(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            let lhs: Self::Output = self.cast();
                            let rhs: Self::Output = rhs.cast();
                            lhs.__shl(rhs)
                        }
                        #[inline(always)]
                        fn _shr(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            let lhs: Self::Output = self.cast();
                            let rhs: Self::Output = rhs.cast();
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
                            let lhs: <#lhs_dtype as NormalOutPromote<#rhs_dtype>>::Output = self.cast();
                            let rhs: <#lhs_dtype as NormalOutPromote<#rhs_dtype>>::Output = rhs.cast();
                            lhs == rhs
                        }
                        fn _ne(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs: <#lhs_dtype as NormalOutPromote<#rhs_dtype>>::Output = self.cast();
                            let rhs: <#lhs_dtype as NormalOutPromote<#rhs_dtype>>::Output = rhs.cast();
                            lhs != rhs
                        }
                        fn _lt(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs: <#lhs_dtype as NormalOutPromote<#rhs_dtype>>::Output = self.cast();
                            let rhs: <#lhs_dtype as NormalOutPromote<#rhs_dtype>>::Output = rhs.cast();
                            lhs < rhs
                        }

                        fn _le(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs: <#lhs_dtype as NormalOutPromote<#rhs_dtype>>::Output = self.cast();
                            let rhs: <#lhs_dtype as NormalOutPromote<#rhs_dtype>>::Output = rhs.cast();
                            lhs <= rhs
                        }
                        fn _gt(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs: <#lhs_dtype as NormalOutPromote<#rhs_dtype>>::Output = self.cast();
                            let rhs: <#lhs_dtype as NormalOutPromote<#rhs_dtype>>::Output = rhs.cast();
                            lhs > rhs
                        }
                        fn _ge(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs: <#lhs_dtype as NormalOutPromote<#rhs_dtype>>::Output = self.cast();
                            let rhs: <#lhs_dtype as NormalOutPromote<#rhs_dtype>>::Output = rhs.cast();
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
                            let lhs: <Scalar<#lhs_dtype> as NormalOutPromote<Scalar<#rhs_dtype>>>::Output = self.cast();
                            let rhs: <Scalar<#lhs_dtype> as NormalOutPromote<Scalar<#rhs_dtype>>>::Output = rhs.cast();
                            lhs.__eq(rhs)
                        }
                        fn _ne(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            let lhs: <Scalar<#lhs_dtype> as NormalOutPromote<Scalar<#rhs_dtype>>>::Output = self.cast();
                            let rhs: <Scalar<#lhs_dtype> as NormalOutPromote<Scalar<#rhs_dtype>>>::Output = rhs.cast();
                            lhs.__ne(rhs)
                        }
                        fn _lt(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            let lhs: <Scalar<#lhs_dtype> as NormalOutPromote<Scalar<#rhs_dtype>>>::Output = self.cast();
                            let rhs: <Scalar<#lhs_dtype> as NormalOutPromote<Scalar<#rhs_dtype>>>::Output = rhs.cast();
                            lhs.__lt(rhs)
                        }

                        fn _le(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            let lhs: <Scalar<#lhs_dtype> as NormalOutPromote<Scalar<#rhs_dtype>>>::Output = self.cast();
                            let rhs: <Scalar<#lhs_dtype> as NormalOutPromote<Scalar<#rhs_dtype>>>::Output = rhs.cast();
                            lhs.__le(rhs)
                        }
                        fn _gt(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            let lhs: <Scalar<#lhs_dtype> as NormalOutPromote<Scalar<#rhs_dtype>>>::Output = self.cast();
                            let rhs: <Scalar<#lhs_dtype> as NormalOutPromote<Scalar<#rhs_dtype>>>::Output = rhs.cast();
                            lhs.__gt(rhs)
                        }
                        fn _ge(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            let lhs: <Scalar<#lhs_dtype> as NormalOutPromote<Scalar<#rhs_dtype>>>::Output = self.cast();
                            let rhs: <Scalar<#lhs_dtype> as NormalOutPromote<Scalar<#rhs_dtype>>>::Output = rhs.cast();
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

/// generate fast layernorm simd helper
#[proc_macro]
pub fn gen_fast_layernorm_simd_helper(input: TokenStream) -> TokenStream {
    __gen_fast_layernorm_simd_helper(input)
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
    let mut compress_levels = vec![];

    let meta_fields = fields
        .iter()
        .map(|f| {
            let mut compression_algo = None;
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
                            compression_algo = Some(quote!(hpt::save_load::CompressionAlgo::#algo));
                        } else if meta.path.is_ident("level") {
                            let value: syn::LitStr = meta.value()?.parse()?;
                            let tmp: u32 = value.value().parse().map_err(|e| {
                                syn::Error::new(value.span(), format!("Invalid level: {}", e))
                            })?;
                            level = Some(quote!(#tmp));
                        }
                        Ok(())
                    })
                    .unwrap();
                }
            }
            compressions.push(compression_algo);
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
        let level = compress_levels[idx].clone().unwrap_or(quote!(level));
        if let Some(name) = name {
            quote! {
                let #ident = <#ty as Save>::__save(&data.#name, file, len_so_far, global_cnt, #compression_algo, #level)?;
            }
        } else {
            quote! {
                let #ident = <#ty as Save>::__save(&data.#idx, file, len_so_far, global_cnt, #compression_algo, #level)?;
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
        #[derive(hpt::re_exports::serde::Deserialize, hpt::re_exports::serde::Serialize)]
        #[serde(crate = "hpt::re_exports::serde")]
        #visibility struct #meta_name #ty_generics #where_clause  {
            #(#meta_fields,)*
        }
        impl #impl_generics hpt::Save for #name #ty_generics #where_clause {
            type Meta = #meta_name #ty_generics;
            fn __save(
                data: &Self,
                file: &mut std::fs::File,
                len_so_far: &mut usize,
                global_cnt: &mut usize,
                compression_algo: hpt::save_load::CompressionAlgo,
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
        impl #impl_generics hpt::save_load::MetaLoad for #meta_name #ty_generics #where_clause {
            type Output = #name #ty_generics;
            fn load(&self, file: &mut std::fs::File) -> std::io::Result<Self::Output> {
                use hpt::save_load::MetaLoad;
                #(#call_load)*
                Ok(#name {
                    #(#construct_fields),*
                })
            }
        }
        impl #impl_generics hpt::Load for #name #ty_generics #where_clause {
            fn load<P: Into<std::path::PathBuf>>(path: P) -> std::io::Result<Self> {
                use hpt::save_load::MetaLoad;
                let path: std::path::PathBuf = path.into();
                let meta = hpt::save_load::parse_header_compressed::<Self, _>(&path).expect(format!("failed to parse header for {}", stringify!(#name)).as_str());
                let mut file = std::fs::File::open(path)?;
                meta.load(&mut file)
            }
        }
    };

    expanded.into()
}

/// generate from safetensors trait
#[proc_macro_derive(FromSafeTensors, attributes(map))]
pub fn impl_from_safetensors(input: TokenStream) -> TokenStream {
    let ast = syn::parse_macro_input!(input as syn::DeriveInput);
    let struct_name = &ast.ident;
    let fields = match &ast.data {
        syn::Data::Struct(s) => &s.fields,
        _ => panic!("FromSafeTensors can only be derived for structs"),
    };
    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();
    let mut construct_fields = vec![];
    for (_, field) in fields.iter().enumerate() {
        let ty = &field.ty;
        let name = &field.ident;
        let mut value_construct = vec![];
        let mut from_construct = vec![];
        let mut params = vec![];
        let mut vec_len = None;
        for attr in &field.attrs {
            if attr.path().is_ident("map") {
                let mut path = None;
                let mut value = None;
                let mut tensor_name = None;
                let mut inner_type = None;
                attr.parse_nested_meta(|meta| {
                    if meta.path.is_ident("path") {
                        let value: syn::LitStr = meta.value()?.parse()?;
                        path = Some(value.value());
                    } else if meta.path.is_ident("value") {
                        let val: syn::Expr = meta.value()?.parse()?;
                        value = Some(val);
                    } else if meta.path.is_ident("tensor_name") {
                        let value: syn::LitStr = meta.value()?.parse()?;
                        tensor_name = Some(value.value());
                    } else if meta.path.is_ident("vec_len") {
                        let value: syn::LitInt = meta.value()?.parse()?;
                        vec_len = Some(value.base10_parse::<usize>().unwrap());
                    } else if meta.path.is_ident("inner_type") {
                        let value: syn::Ident = meta.value()?.parse()?;
                        inner_type = Some(value);
                    }
                    Ok(())
                })
                .unwrap_or_else(|err| println!("Failed to parse attribute: {}", err));
                params.push((path, value, tensor_name, vec_len, inner_type));
            }
        }
        let param_count = params.len();
        for (path, value, tensor_name, vec_len, inner_type) in params {
            if let Some(vec_len) = vec_len {
                let inner_type = inner_type.expect("inner_type is required for vec");
                if let Some(path) = path {
                    from_construct.push(quote! {
                        #path => {
                            let mut vec = vec![];
                            for i in 0..#vec_len {
                                vec.push(<#inner_type as FromSafeTensors>::from_safe_tensors(data, &format!("{}.{}", path, i)));
                            }
                            vec
                        }
                    });
                } else {
                    value_construct.push(quote! {
                        {
                            let mut vec = vec![];
                            for i in 0..#vec_len {
                                vec.push(<#inner_type as FromSafeTensors>::from_safe_tensors(data, &format!("{}.{}", path, i)));
                            }
                            vec
                        }
                    });
                }
            } else {
                match (path, value, tensor_name) {
                    (None, None, Some(tensor_name)) => {
                        value_construct.push(quote! {
                            <#ty as FromSafeTensors>::from_safe_tensors(data, #tensor_name)
                        });
                    }
                    (None, Some(value), None) => {
                        if param_count > 1 {
                            panic!("value without path means generic assignment, there can only be one value without path");
                        }
                        value_construct.push(quote! {
                            #value
                        });
                    }
                    (Some(path), None, Some(tensor_name)) => {
                        from_construct.push(quote! {
                            #path => <#ty as FromSafeTensors>::from_safe_tensors(data, #tensor_name),
                        });
                    }
                    (Some(path), Some(value), None) => {
                        from_construct.push(quote! {
                            #path => #value,
                        });
                    }

                    (None, Some(_), Some(_)) | (Some(_), Some(_), Some(_)) => {
                        panic!("value and tensor_name cannot be used together");
                    }
                    (Some(_), None, None) | (None, None, None) => {
                        panic!("path and value are not present");
                    }
                }
            }
        }
        if !value_construct.is_empty() {
            construct_fields.push(quote! {
                #name: #(#value_construct)*
            });
        } else if !from_construct.is_empty() {
            construct_fields.push(quote! {
                #name: match path {
                    #(#from_construct)*
                    _ => panic!("unknown field for field {} in struct {}: `path: {}`", stringify!(#name), stringify!(#struct_name), path),
                }
            });
        } else {
            construct_fields.push(quote! {
                #name: <#ty as FromSafeTensors>::from_safe_tensors(data, &format!("{}.{}", path, stringify!(#name)))
            });
        }
    }
    let expanded = quote! {
        impl #impl_generics FromSafeTensors for #struct_name #ty_generics #where_clause {
            fn from_safe_tensors(data: &SafeTensors, path: &str) -> Self {
                Self {
                    #(#construct_fields),*
                }
            }
        }
    };
    // let syntax_tree = syn::parse2(expanded.clone()).expect(&format!(
    //     "failed to parse expanded: {}",
    //     expanded.to_string()
    // ));
    // let formatted = prettyplease::unparse(&syntax_tree);
    // println!("{}", formatted);
    expanded.into()
}
