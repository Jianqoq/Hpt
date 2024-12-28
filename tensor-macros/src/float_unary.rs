use crate::type_utils::TypeInfo;
use proc_macro::TokenStream;
use quote::quote;

pub fn impl_float_out_unary() -> TokenStream {
    let mut ret = proc_macro2::TokenStream::new();

    let types = [
        "bool",
        "f16",
        "bf16",
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
        "Complex32",
        "Complex64",
    ];

    for lhs in types.iter() {
        let lhs_type = TypeInfo::new(lhs);
        let lhs_dtype = lhs_type.dtype;

        let res = if lhs_dtype.is_float() || lhs_dtype.is_cplx32() || lhs_dtype.is_cplx64() {
            quote! {
                impl FloatOutUnary for #lhs_dtype {
                    type Output = <#lhs_dtype as FloatOutUnaryPromote>::Output;
                    fn _exp(self) -> Self::Output {
                        self.__exp()
                    }
                    fn _exp2(self) -> Self::Output {
                        self.__exp2()
                    }
                    fn _ln(self) -> Self::Output {
                        self.__ln()
                    }
                    fn _log2(self) -> Self::Output {
                        self.__log2()
                    }
                    fn _log10(self) -> Self::Output {
                        self.__log10()
                    }
                    fn _sqrt(self) -> Self::Output {
                        self.__sqrt()
                    }
                    fn _sin(self) -> Self::Output {
                        self.sin()
                    }
                    fn _cos(self) -> Self::Output {
                        self.cos()
                    }
                    fn _tan(self) -> Self::Output {
                        self.tan()
                    }
                    fn _asin(self) -> Self::Output {
                        self.asin()
                    }
                    fn _acos(self) -> Self::Output {
                        self.acos()
                    }
                    fn _atan(self) -> Self::Output {
                        self.atan()
                    }
                    fn _sinh(self) -> Self::Output {
                        self.sinh()
                    }
                    fn _cosh(self) -> Self::Output {
                        self.cosh()
                    }
                    fn _tanh(self) -> Self::Output {
                        self.tanh()
                    }
                    fn _asinh(self) -> Self::Output {
                        self.asinh()
                    }
                    fn _acosh(self) -> Self::Output {
                        self.acosh()
                    }
                    fn _atanh(self) -> Self::Output {
                        self.atanh()
                    }
                    fn _recip(self) -> Self::Output {
                        self.__recip()
                    }
                    fn _erf(self) -> Self::Output {
                        self.__erf()
                    }
                    fn _celu(self, alpha: Self::Output) -> Self::Output {
                        self.__celu(alpha)
                    }
                    fn _sigmoid(self) -> Self::Output {
                        self.__sigmoid()
                    }
                    fn _elu(self, alpha: Self::Output) -> Self::Output {
                        self.__elu(alpha)
                    }
                    fn _gelu(self) -> Self::Output {
                        self.__gelu()
                    }
                    fn _selu(self, alpha: Self::Output, scale: Self::Output) -> Self::Output {
                        self.__selu(alpha, scale)
                    }
                    fn _hard_sigmoid(self) -> Self::Output {
                        self.__hard_sigmoid()
                    }
                    fn _fast_hard_sigmoid(self) -> Self::Output {
                        self.__fast_hard_sigmoid()
                    }
                    fn _hard_swish(self) -> Self::Output {
                        self.__hard_swish()
                    }
                    fn _softplus(self) -> Self::Output {
                        self.__softplus()
                    }
                    fn _softsign(self) -> Self::Output {
                        self.__softsign()
                    }
                    fn _mish(self) -> Self::Output {
                        self.__mish()
                    }
                    fn _cbrt(self) -> Self::Output {
                        self.__cbrt()
                    }
                }
            }
        } else {
            quote! {
                impl FloatOutUnary for #lhs_dtype {
                    type Output = <#lhs_dtype as FloatOutUnaryPromote>::Output;
                    fn _exp(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__exp()
                    }
                    fn _exp2(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__exp2()
                    }
                    fn _ln(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__ln()
                    }
                    fn _log2(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__log2()
                    }
                    fn _log10(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__log10()
                    }
                    fn _sqrt(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__sqrt()
                    }
                    fn _sin(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__sin()
                    }
                    fn _cos(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__cos()
                    }
                    fn _tan(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__tan()
                    }
                    fn _asin(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__asin()
                    }
                    fn _acos(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__acos()
                    }
                    fn _atan(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__atan()
                    }
                    fn _sinh(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__sinh()
                    }
                    fn _cosh(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__cosh()
                    }
                    fn _tanh(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__tanh()
                    }
                    fn _asinh(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__asinh()
                    }
                    fn _acosh(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__acosh()
                    }
                    fn _atanh(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__atanh()
                    }
                    fn _recip(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__recip()
                    }
                    fn _erf(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__erf()
                    }
                    fn _celu(self, alpha: Self::Output) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        let alpha: Self::Output = alpha.into_scalar();
                        lhs.__celu(alpha)
                    }
                    fn _sigmoid(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__sigmoid()
                    }
                    fn _elu(self, alpha: Self::Output) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        let alpha: Self::Output = alpha.into_scalar();
                        lhs.__elu(alpha)
                    }
                    fn _gelu(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__gelu()
                    }
                    fn _selu(self, alpha: Self::Output, scale: Self::Output) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        let alpha: Self::Output = alpha.into_scalar();
                        let scale: Self::Output = scale.into_scalar();
                        lhs.__selu(alpha, scale)
                    }
                    fn _hard_sigmoid(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__hard_sigmoid()
                    }
                    fn _fast_hard_sigmoid(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__fast_hard_sigmoid()
                    }
                    fn _hard_swish(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__hard_swish()
                    }
                    fn _softplus(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__softplus()
                    }
                    fn _softsign(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__softsign()
                    }
                    fn _mish(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__mish()
                    }
                    fn _cbrt(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__cbrt()
                    }
                }
            }
        };
        ret.extend(res);
    }

    ret.into()
}

pub fn impl_cuda_float_out_unary() -> TokenStream {
    let mut ret = proc_macro2::TokenStream::new();

    let types = [
        "bool",
        "f16",
        "bf16",
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
        "Complex32",
        "Complex64",
    ];

    for lhs in types.iter() {
        let lhs_type = TypeInfo::new(lhs);
        let lhs_dtype = lhs_type.dtype;

        let res = if lhs_dtype.is_float() || lhs_dtype.is_cplx32() || lhs_dtype.is_cplx64() {
            quote! {
                impl FloatOutUnary for Scalar<#lhs_dtype> {
                    type Output = <Scalar<#lhs_dtype> as FloatOutUnaryPromote>::Output;
                    type Base = <Scalar<#lhs_dtype> as FloatOutUnaryPromote>::Output;
                    fn _exp(self) -> Self::Output {
                        self.__exp()
                    }
                    fn _exp2(self) -> Self::Output {
                        self.__exp2()
                    }
                    fn _ln(self) -> Self::Output {
                        self.__ln()
                    }
                    fn _log2(self) -> Self::Output {
                        self.__log2()
                    }
                    fn _log10(self) -> Self::Output {
                        self.__log10()
                    }
                    fn _sqrt(self) -> Self::Output {
                        self.__sqrt()
                    }
                    fn _sin(self) -> Self::Output {
                        self.__sin()
                    }
                    fn _cos(self) -> Self::Output {
                        self.__cos()
                    }
                    fn _tan(self) -> Self::Output {
                        self.__tan()
                    }
                    fn _asin(self) -> Self::Output {
                        self.__asin()
                    }
                    fn _acos(self) -> Self::Output {
                        self.__acos()
                    }
                    fn _atan(self) -> Self::Output {
                        self.__atan()
                    }
                    fn _sinh(self) -> Self::Output {
                        self.__sinh()
                    }
                    fn _cosh(self) -> Self::Output {
                        self.__cosh()
                    }
                    fn _tanh(self) -> Self::Output {
                        self.__tanh()
                    }
                    fn _asinh(self) -> Self::Output {
                        self.__asinh()
                    }
                    fn _acosh(self) -> Self::Output {
                        self.__acosh()
                    }
                    fn _atanh(self) -> Self::Output {
                        self.__atanh()
                    }
                    fn _recip(self) -> Self::Output {
                        self.__recip()
                    }
                    fn _erf(self) -> Self::Output {
                        self.__erf()
                    }
                    fn _celu(self, alpha: Self) -> Self::Output {
                        self.__celu(alpha)
                    }
                    fn _sigmoid(self) -> Self::Output {
                        self.__sigmoid()
                    }
                    fn _elu(self, alpha: Self) -> Self::Output {
                        self.__elu(alpha)
                    }
                    fn _gelu(self) -> Self::Output {
                        self.__gelu()
                    }
                    fn _selu(self, alpha: Self, scale: Self) -> Self::Output {
                        self.__selu(alpha, scale)
                    }
                    fn _hard_sigmoid(self) -> Self::Output {
                        self.__hard_sigmoid()
                    }
                    fn _fast_hard_sigmoid(self) -> Self::Output {
                        self.__fast_hard_sigmoid()
                    }
                    fn _hard_swish(self) -> Self::Output {
                        self.__hard_swish()
                    }
                    fn _softplus(self) -> Self::Output {
                        self.__softplus()
                    }
                    fn _softsign(self) -> Self::Output {
                        self.__softsign()
                    }
                    fn _mish(self) -> Self::Output {
                        self.__mish()
                    }
                    fn _cbrt(self) -> Self::Output {
                        self.__cbrt()
                    }
                }
            }
        } else {
            quote! {
                impl FloatOutUnary for Scalar<#lhs_dtype> {
                    type Output = <Scalar<#lhs_dtype> as FloatOutUnaryPromote>::Output;
                    type Base = <Scalar<#lhs_dtype> as FloatOutUnaryPromote>::Output;
                    fn _exp(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__exp()
                    }
                    fn _exp2(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__exp2()
                    }
                    fn _ln(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__ln()
                    }
                    fn _log2(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__log2()
                    }
                    fn _log10(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__log10()
                    }
                    fn _sqrt(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__sqrt()
                    }
                    fn _sin(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__sin()
                    }
                    fn _cos(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__cos()
                    }
                    fn _tan(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__tan()
                    }
                    fn _asin(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__asin()
                    }
                    fn _acos(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__acos()
                    }
                    fn _atan(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__atan()
                    }
                    fn _sinh(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__sinh()
                    }
                    fn _cosh(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__cosh()
                    }
                    fn _tanh(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__tanh()
                    }
                    fn _asinh(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__asinh()
                    }
                    fn _acosh(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__acosh()
                    }
                    fn _atanh(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__atanh()
                    }
                    fn _recip(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__recip()
                    }
                    fn _erf(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__erf()
                    }
                    fn _celu(self, alpha: Self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__celu(alpha)
                    }
                    fn _sigmoid(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__sigmoid()
                    }
                    fn _elu(self, alpha: Self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__elu(alpha)
                    }
                    fn _gelu(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__gelu()
                    }
                    fn _selu(self, alpha: Self, scale: Self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__selu(alpha, scale)
                    }
                    fn _hard_sigmoid(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__hard_sigmoid()
                    }
                    fn _fast_hard_sigmoid(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__fast_hard_sigmoid()
                    }
                    fn _hard_swish(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__hard_swish()
                    }
                    fn _softplus(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__softplus()
                    }
                    fn _softsign(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__softsign()
                    }
                    fn _mish(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__mish()
                    }
                    fn _cbrt(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__cbrt()
                    }
                }
            }
        };
        ret.extend(res);
    }

    ret.into()
}