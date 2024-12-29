use crate::type_utils::TypeInfo;
use proc_macro::TokenStream;
use quote::quote;

pub(crate) fn __impl_normal_out_unary() -> TokenStream {
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
        "Complex32",
        "Complex64",
    ];

    for lhs in types.iter() {
        let lhs_type = TypeInfo::new(lhs);
        let lhs_dtype = lhs_type.dtype;

        let res = quote! {
            impl NormalOutUnary for #lhs_dtype {
                type Base = <Self as NormalOutPromote<Self>>::Output;
                #[inline(always)]
                fn _square(self) -> Self {
                    self._mul(self)
                }
                #[inline(always)]
                fn _neg(self) -> Self {
                    self.__neg()
                }
                #[inline(always)]
                fn _abs(self) -> Self {
                    self.__abs()
                }
                #[inline(always)]
                fn _ceil(self) -> Self {
                    self.__ceil()
                }
                #[inline(always)]
                fn _floor(self) -> Self {
                    self.__floor()
                }
                #[inline(always)]
                fn _round(self) -> Self {
                    self.__round()
                }
                #[inline(always)]
                fn _signum(self) -> Self {
                    self.__signum()
                }
                #[inline(always)]
                fn _relu(self) -> Self {
                    self.__relu()
                }
                #[inline(always)]
                fn _relu6(self) -> Self {
                    self.__relu6()
                }
                #[inline(always)]
                fn _leaky_relu(self, alpha: Self::Base) -> Self {
                    self.__leaky_relu(alpha)
                }
            }
        };
        ret.extend(res);
    }

    ret.into()
}

#[cfg(feature = "cuda")]
pub(crate) fn __impl_normal_out_unary_cuda() -> TokenStream {
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
        "Complex32",
        "Complex64",
    ];

    for lhs in types.iter() {
        let lhs_type = TypeInfo::new(lhs);
        let lhs_dtype = lhs_type.dtype;

        let res = quote! {
            impl NormalOutUnary for Scalar<#lhs_dtype> {
                type Base = <Self as NormalOutPromote<Self>>::Output;
                #[inline(always)]
                fn _square(self) -> Self {
                    self._square()
                }
                #[inline(always)]
                fn _neg(self) -> Self {
                    self.__neg()
                }
                #[inline(always)]
                fn _abs(self) -> Self {
                    self.__abs()
                }
                #[inline(always)]
                fn _ceil(self) -> Self {
                    self.__ceil()
                }
                #[inline(always)]
                fn _floor(self) -> Self {
                    self.__floor()
                }
                #[inline(always)]
                fn _round(self) -> Self {
                    self.__round()
                }
                #[inline(always)]
                fn _signum(self) -> Self {
                    self.__signum()
                }
                #[inline(always)]
                fn _relu(self) -> Self {
                    self.__relu()
                }
                #[inline(always)]
                fn _relu6(self) -> Self {
                    self.__relu6()
                }
                #[inline(always)]
                fn _leaky_relu(self, alpha: Self::Base) -> Self {
                    self.clone().__leaky_relu(alpha)
                }
            }
        };
        ret.extend(res);
    }

    ret.into()
}