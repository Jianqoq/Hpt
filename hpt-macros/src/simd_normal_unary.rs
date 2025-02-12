use crate::type_utils::SimdType;
use proc_macro::TokenStream;
use quote::quote;

pub(crate) fn impl_simd_normal_out_unary() -> TokenStream {
    let mut ret = proc_macro2::TokenStream::new();

    let types = [
        "bool",
        "bf16",
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
        "complex32",
        "complex64",
    ];

    for lhs in types.iter() {
        let lhs_simd: SimdType = (*lhs).into();

        let res = quote! {
            impl NormalOutUnary for #lhs_simd {
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
                fn _trunc(self) -> Self {
                    self.__trunc()
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
                fn _leaky_relu(self, alpha: Self) -> Self {
                    self.__leaky_relu(alpha)
                }
                #[inline(always)]
                fn _copysign(self, rhs: Self) -> Self {
                    self.__copysign(rhs)
                }
            }
        };
        ret.extend(res);
    }

    ret.into()
}
