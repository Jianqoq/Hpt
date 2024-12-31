use crate::type_utils::{type_simd_lanes, SimdType, TypeInfo};
use proc_macro::TokenStream;
use quote::quote;

pub fn impl_float_out_unary() -> TokenStream {
    let mut ret = proc_macro2::TokenStream::new();

    let types = [
        ("bool", type_simd_lanes("bool"), "bool"),
        ("bf16", type_simd_lanes("bf16"), "bf16"),
        ("f16", type_simd_lanes("f16"), "f16"),
        ("f32", type_simd_lanes("f32"), "f32"),
        ("f64", type_simd_lanes("f64"), "f64"),
        ("i8", type_simd_lanes("i8"), "i8"),
        ("i16", type_simd_lanes("i16"), "i16"),
        ("i32", type_simd_lanes("i32"), "i32"),
        ("i64", type_simd_lanes("i64"), "i64"),
        ("u8", type_simd_lanes("u8"), "u8"),
        ("u16", type_simd_lanes("u16"), "u16"),
        ("u32", type_simd_lanes("u32"), "u32"),
        ("u64", type_simd_lanes("u64"), "u64"),
        ("isize", type_simd_lanes("isize"), "isize"),
        ("usize", type_simd_lanes("usize"), "usize"),
        ("Complex32", type_simd_lanes("complex32"), "complex32"),
        ("Complex64", type_simd_lanes("complex64"), "complex64"),
    ];

    for (lhs_ty, _, lhs) in types.iter() {
        let lhs_type = TypeInfo::new(&lhs_ty.to_lowercase());
        let lhs_simd: SimdType = (*lhs).into();
        if lhs_type.dtype.is_float() || lhs_type.dtype.is_cplx() {
            let res = quote! {
                impl FloatOutUnary for #lhs_simd {
                    type Output = <#lhs_simd as FloatOutUnaryPromote>::Output;
                    #[inline(always)]
                    fn _sin(self) -> Self::Output {
                        SimdMath::sin(self)
                    }
                    #[inline(always)]
                    fn _cos(self) -> Self::Output {
                        SimdMath::cos(self)
                    }
                    #[inline(always)]
                    fn _tan(self) -> Self::Output {
                        SimdMath::tan(self)
                    }
                    #[inline(always)]
                    fn _asin(self) -> Self::Output {
                        SimdMath::asin(self)
                    }
                    #[inline(always)]
                    fn _acos(self) -> Self::Output {
                        SimdMath::acos(self)
                    }
                    #[inline(always)]
                    fn _atan(self) -> Self::Output {
                        SimdMath::atan(self)
                    }
                    #[inline(always)]
                    fn _sinh(self) -> Self::Output {
                        SimdMath::sinh(self)
                    }
                    #[inline(always)]
                    fn _cosh(self) -> Self::Output {
                        SimdMath::cosh(self)
                    }
                    #[inline(always)]
                    fn _tanh(self) -> Self::Output {
                        SimdMath::tanh(self)
                    }
                    #[inline(always)]
                    fn _asinh(self) -> Self::Output {
                        SimdMath::asinh(self)
                    }
                    #[inline(always)]
                    fn _erf(self) -> Self::Output {
                        SimdMath::erf(self)
                    }
                    #[inline(always)]
                    fn _ln(self) -> Self::Output {
                        SimdMath::ln(self)
                    }
                    #[inline(always)]
                    fn _exp(self) -> Self::Output {
                        SimdMath::exp(self)
                    }
                    #[inline(always)]
                    fn _log2(self) -> Self::Output {
                        SimdMath::log2(self)
                    }
                    #[inline(always)]
                    fn _log10(self) -> Self::Output {
                        SimdMath::log10(self)
                    }
                    #[inline(always)]
                    fn _sqrt(self) -> Self::Output {
                        SimdMath::sqrt(self)
                    }
                    #[inline(always)]
                    fn _cbrt(self) -> Self::Output {
                        SimdMath::cbrt(self)
                    }
                    #[inline(always)]
                    fn _atanh(self) -> Self::Output {
                        SimdMath::atanh(self)
                    }
                    #[inline(always)]
                    fn _acosh(self) -> Self::Output {
                        SimdMath::acosh(self)
                    }
                    #[inline(always)]
                    fn _exp2(self) -> Self::Output {
                        SimdMath::exp2(self)
                    }
                    #[inline(always)]
                    fn _recip(self) -> Self::Output {
                        SimdMath::recip(self)
                    }
                    #[inline(always)]
                    fn _sigmoid(self) -> Self::Output {
                        SimdMath::sigmoid(self)
                    }
                    #[inline(always)]
                    fn _gelu(self) -> Self::Output {
                        SimdMath::gelu(self)
                    }
                    #[inline(always)]
                    fn _softplus(self) -> Self::Output {
                        SimdMath::softplus(self)
                    }
                    #[inline(always)]
                    fn _softsign(self) -> Self::Output {
                        SimdMath::softsign(self)
                    }
                    #[inline(always)]
                    fn _mish(self) -> Self::Output {
                        SimdMath::mish(self)
                    }
                    #[inline(always)]
                    fn _celu(self, alpha: Self::Output) -> Self::Output {
                        SimdMath::celu(self, alpha)
                    }
                    #[inline(always)]
                    fn _selu(self, alpha: Self::Output, scale: Self::Output)-> Self::Output {
                        SimdMath::selu(self, alpha, scale)
                    }
                    #[inline(always)]
                    fn _elu(self, alpha: Self::Output) -> Self::Output {
                        SimdMath::elu(self, alpha)
                    }
                    #[inline(always)]
                    fn _hard_swish(self) -> Self::Output {
                        SimdMath::hard_swish(self)
                    }
                    #[inline(always)]
                    fn _hard_sigmoid(self) -> Self::Output {
                        SimdMath::hard_sigmoid(self)
                    }
                    #[inline(always)]
                    fn _fast_hard_sigmoid(self) -> Self::Output {
                        SimdMath::fast_hard_sigmoid(self)
                    }
                }
            };
            ret.extend(res);
        } else {
            let res = quote! {
                impl FloatOutUnary for #lhs_simd {
                    type Output = <#lhs_simd as FloatOutUnaryPromote>::Output;
                    #[inline(always)]
                    fn _sin(self) -> Self::Output {
                        SimdMath::sin(self.into_vec())
                    }
                    #[inline(always)]
                    fn _cos(self) -> Self::Output {
                        SimdMath::cos(self.into_vec())
                    }
                    #[inline(always)]
                    fn _tan(self) -> Self::Output {
                        SimdMath::tan(self.into_vec())
                    }
                    #[inline(always)]
                    fn _asin(self) -> Self::Output {
                        SimdMath::asin(self.into_vec())
                    }
                    #[inline(always)]
                    fn _acos(self) -> Self::Output {
                        SimdMath::acos(self.into_vec())
                    }
                    #[inline(always)]
                    fn _atan(self) -> Self::Output {
                        SimdMath::atan(self.into_vec())
                    }
                    #[inline(always)]
                    fn _sinh(self) -> Self::Output {
                        SimdMath::sinh(self.into_vec())
                    }
                    #[inline(always)]
                    fn _cosh(self) -> Self::Output {
                        SimdMath::cosh(self.into_vec())
                    }
                    #[inline(always)]
                    fn _tanh(self) -> Self::Output {
                        SimdMath::tanh(self.into_vec())
                    }
                    #[inline(always)]
                    fn _asinh(self) -> Self::Output {
                        SimdMath::asinh(self.into_vec())
                    }
                    #[inline(always)]
                    fn _erf(self) -> Self::Output {
                        SimdMath::erf(self.into_vec())
                    }
                    #[inline(always)]
                    fn _ln(self) -> Self::Output {
                        SimdMath::ln(self.into_vec())
                    }
                    #[inline(always)]
                    fn _exp(self) -> Self::Output {
                        SimdMath::exp(self.into_vec())
                    }
                    #[inline(always)]
                    fn _log2(self) -> Self::Output {
                        SimdMath::log2(self.into_vec())
                    }
                    #[inline(always)]
                    fn _log10(self) -> Self::Output {
                        SimdMath::log10(self.into_vec())
                    }
                    #[inline(always)]
                    fn _sqrt(self) -> Self::Output {
                        SimdMath::sqrt(self.into_vec())
                    }
                    #[inline(always)]
                    fn _cbrt(self) -> Self::Output {
                        SimdMath::cbrt(self.into_vec())
                    }
                    #[inline(always)]
                    fn _atanh(self) -> Self::Output {
                        SimdMath::atanh(self.into_vec())
                    }
                    #[inline(always)]
                    fn _acosh(self) -> Self::Output {
                        SimdMath::acosh(self.into_vec())
                    }
                    #[inline(always)]
                    fn _exp2(self) -> Self::Output {
                        SimdMath::exp2(self.into_vec())
                    }
                    #[inline(always)]
                    fn _recip(self) -> Self::Output {
                        SimdMath::recip(self.into_vec())
                    }
                    #[inline(always)]
                    fn _sigmoid(self) -> Self::Output {
                        SimdMath::sigmoid(self.into_vec())
                    }
                    #[inline(always)]
                    fn _gelu(self) -> Self::Output {
                        SimdMath::gelu(self.into_vec())
                    }
                    #[inline(always)]
                    fn _softplus(self) -> Self::Output {
                        SimdMath::softplus(self.into_vec())
                    }
                    #[inline(always)]
                    fn _softsign(self) -> Self::Output {
                        SimdMath::softsign(self.into_vec())
                    }
                    #[inline(always)]
                    fn _mish(self) -> Self::Output {
                        SimdMath::mish(self.into_vec())
                    }
                    #[inline(always)]
                    fn _celu(self, alpha: Self::Output) -> Self::Output {
                        SimdMath::celu(self.into_vec(), alpha.into_vec())
                    }
                    #[inline(always)]
                    fn _selu(self, alpha: Self::Output, scale: Self::Output) -> Self::Output {
                        SimdMath::selu(self.into_vec(), alpha.into_vec(), scale.into_vec())
                    }
                    #[inline(always)]
                    fn _elu(self, alpha: Self::Output) -> Self::Output {
                        SimdMath::elu(self.into_vec(), alpha.into_vec())
                    }
                    #[inline(always)]
                    fn _hard_swish(self) -> Self::Output {
                        SimdMath::hard_swish(self.into_vec())
                    }
                    #[inline(always)]
                    fn _hard_sigmoid(self) -> Self::Output {
                        SimdMath::hard_sigmoid(self.into_vec())
                    }
                    #[inline(always)]
                    fn _fast_hard_sigmoid(self) -> Self::Output {
                        SimdMath::fast_hard_sigmoid(self.into_vec())
                    }
                }

            };
            ret.extend(res);
        }
    }

    ret.into()
}
