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
                    type Base = <#lhs_dtype as FloatOutUnaryPromote>::Output;
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
                    fn _celu(self, alpha: Self::Base) -> Self::Output {
                        self.__celu(alpha)
                    }
                    fn _sigmoid(self) -> Self::Output {
                        self.__sigmoid()
                    }
                    fn _elu(self, alpha: Self::Base) -> Self::Output {
                        self.__elu(alpha)
                    }
                    fn _gelu(self) -> Self::Output {
                        self.__gelu()
                    }
                    fn _selu(self, alpha: Self::Base, scale: Self::Base) -> Self::Output {
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
                    type Base = <#lhs_dtype as FloatOutUnaryPromote>::Output;
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
                    fn _celu(self, alpha: Self::Base) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__celu(alpha)
                    }
                    fn _sigmoid(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__sigmoid()
                    }
                    fn _elu(self, alpha: Self::Base) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__elu(alpha)
                    }
                    fn _gelu(self) -> Self::Output {
                        let lhs: Self::Output = self.into_scalar();
                        lhs.__gelu()
                    }
                    fn _selu(self, alpha: Self::Base, scale: Self::Base) -> Self::Output {
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
        let res_type = lhs_type.infer_float_res_type_uary();
        let selu = cuda_selu(res_type);
        let gelu = cuda_gelu(res_type);
        let erf = cuda_builtin_unary(res_type, "erf");
        let sin = cuda_builtin_unary(res_type, "sin");
        let cos = cuda_builtin_unary(res_type, "cos");
        let tan = cuda_builtin_unary(res_type, "tan");
        let asin = cuda_builtin_unary(res_type, "asin");
        let acos = cuda_builtin_unary(res_type, "acos");
        let atan = cuda_builtin_unary(res_type, "atan");
        let sinh = cuda_builtin_unary(res_type, "sinh");
        let cosh = cuda_builtin_unary(res_type, "cosh");
        let tanh = cuda_builtin_unary(res_type, "tanh");
        let asinh = cuda_builtin_unary(res_type, "asinh");
        let acosh = cuda_builtin_unary(res_type, "acosh");
        let atanh = cuda_builtin_unary(res_type, "atanh");
        let exp = cuda_builtin_unary(res_type, "exp");
        let exp2 = cuda_builtin_unary(res_type, "exp2");
        let ln = cuda_builtin_unary(res_type, "log");
        let log2 = cuda_builtin_unary(res_type, "log2");
        let log10 = cuda_builtin_unary(res_type, "log10");
        let sqrt = cuda_builtin_unary(res_type, "sqrt");
        let recip = cuda_recip(res_type);
        let elu = cuda_elu(res_type);
        let celu = cuda_celu(res_type);
        let _fast_hard_sigmoid = cuda_fast_hard_sigmoid(res_type);
        let hard_sigmoid = cuda_hard_sigmoid(res_type);
        let hard_swish = cuda_hard_swish(res_type);
        let softplus = cuda_softplus(res_type);
        let sigmoid = cuda_sigmoid(res_type);
        let soft_sign = cuda_softsign(res_type);
        let mish = cuda_mish(res_type);
        let cbrt = cuda_cbrt(res_type);
        let res = quote! {
            impl FloatOutUnary for Scalar<#lhs_dtype> {
                type Output = Scalar<#res_type>;
                type Base = Scalar<#res_type>;
                #exp #exp2 #ln #log2 #log10 #sqrt #sin #cos #tan #asin #acos #atan
                #sinh #cosh #tanh #asinh #acosh #atanh #recip #erf #celu #sigmoid #elu
                #gelu #selu #hard_sigmoid #_fast_hard_sigmoid #hard_swish #softplus
                #soft_sign #mish #cbrt
            }
        };
        ret.extend(res);
    }

    ret.into()
}

use crate::type_utils::Type;
fn cuda_selu(res_type: Type) -> proc_macro2::TokenStream {
    let alpha = 1.6732632423543772848170429916717;
    let gamma = 1.0507009873554804934193349852946;
    let tks = match res_type {
        Type::BF16 => quote! {
            Scalar::new(format!(
                "__float2bfloat16_rn({}f * ({} > 0.0f ? {} : {}f * (expf({}) - 1.0f)))",
                #gamma, self.to_f32().val, self.to_f32().val, #alpha, self.to_f32().val
            ))
        },
        Type::F16 => quote! {
            Scalar::new(format!(
                "__float2half_rn({}f * ({} > 0.0f ? {} : {}f * (expf({}) - 1.0f)))",
                #gamma, self.to_f32().val, self.to_f32().val, #alpha, self.to_f32().val
            ))
        },
        Type::F32 => quote! {
            Scalar::new(format!(
                "({}f * ({} > 0.0f ? {} : {}f * (expf({}) - 1.0f)))",
                #gamma, self.to_f32().val, self.to_f32().val, #alpha, self.to_f32().val
            ))
        },
        Type::F64 => quote! {
            Scalar::new(format!(
                "({} * ({} > 0.0 ? {} : {} * (exp({}) - 1.0)))",
                #gamma, self.to_f64().val, self.to_f64().val, #alpha, self.to_f64().val
            ))
        },
        Type::Complex32 => quote! {
            Scalar::new(format!(
                "cuCmulf(make_cuComplex({}f, 0.0f),
                    cuCabsf({}) > 0.0f ? {} : 
                    cuCmulf(make_cuComplex({}f, 0.0f), 
                        cuCsubf(cuCexpf({}), make_cuComplex(1.0f, 0.0f))))",
                #gamma, self.to_complex32().val, self.to_complex32().val, #alpha, self.to_complex32().val
            ))
        },
        Type::Complex64 => quote! {
            Scalar::new(format!(
                "cuCmul(make_cuDoubleComplex({}, 0.0),
                    cuCabs({}) > 0.0 ? {} : 
                    cuCmul(make_cuDoubleComplex({}, 0.0), 
                        cuCsub(cuCexp({}), make_cuDoubleComplex(1.0, 0.0))))",
                #gamma, self.to_complex64().val, self.to_complex64().val, #alpha, self.to_complex64().val
            ))
        },
        _ => unreachable!(),
    };
    quote! {
        fn _selu(self, alpha: Self::Base, scale: Self::Base) -> Self::Output {
            #tks
        }
    }
}

fn cuda_gelu(res_type: Type) -> proc_macro2::TokenStream {
    let tks = match res_type {
        Type::BF16 => quote! {
            Scalar::new(format!(
                "__float2bfloat16_rn(0.5f * {} * (1.0f + erff({} * {}f))",
                self.to_f32().val, self.to_f32().val, std::f32::consts::FRAC_1_SQRT_2
            ))
        },
        Type::F16 => quote! {
            Scalar::new(format!(
                "__float2half_rn(0.5f * {} * (1.0f + erff({} * {}f)))",
                self.to_f32().val, self.to_f32().val, std::f32::consts::FRAC_1_SQRT_2
            ))
        },
        Type::F32 => quote! {
            Scalar::new(format!(
                "(0.5f * {} * (1.0f + erff({} * {}f)))",
                self.to_f32().val, self.to_f32().val, std::f32::consts::FRAC_1_SQRT_2
            ))
        },
        Type::F64 => quote! {
            Scalar::new(format!(
                "(0.5 * {} * (1.0 + erf({} * {}f)))",
                self.to_f64().val, self.to_f64().val, std::f64::consts::FRAC_1_SQRT_2
            ))
        },
        Type::Complex32 | Type::Complex64 => {
            quote! {
                panic!("GELU is not implemented for complex numbers")
            }
        }
        _ => unreachable!(),
    };
    quote! {
        fn _gelu(self) -> Self::Output {
            #tks
        }
    }
}

fn cuda_builtin_unary(res_type: Type, method: &str) -> proc_macro2::TokenStream {
    let method_name = if method == "log" { "ln" } else { method };
    let method_ident =
        syn::Ident::new(&format!("_{}", method_name), proc_macro2::Span::call_site());
    let method_f = format!("{}f", method); // float 版本，如 sinf
    let method_c32 = format!("cuC{}f", method); // complex32 版本，如 cuCsinf
    let method_c64 = format!("cuC{}", method); // complex64 版本，如 cuCsin
    let tks = match res_type {
        Type::BF16 => quote! {
            Scalar::new(format!(
                "__float2bfloat16_rn({}({}))",
                #method_f, self.to_f32().val
            ))
        },
        Type::F16 => quote! {
            Scalar::new(format!(
                "__float2half_rn({}({}))",
                #method_f, self.to_f32().val
            ))
        },
        Type::F32 => quote! {
            Scalar::new(format!(
                "{}({})",
                #method_f, self.to_f32().val
            ))
        },
        Type::F64 => quote! {
            Scalar::new(format!(
                "{}({})",
                #method, self.to_f64().val
            ))
        },
        Type::Complex32 => quote! {
            Scalar::new(format!(
                "{}({})",
                #method_c32, self.to_complex32().val
            ))
        },
        Type::Complex64 => quote! {
            Scalar::new(format!(
                "{}({})",
                #method_c64, self.to_complex64().val
            ))
        },
        _ => unreachable!(),
    };
    quote! {
        #[inline(always)]
        fn #method_ident(self) -> Self::Output {
            #tks
        }
    }
}

fn cuda_recip(res_type: Type) -> proc_macro2::TokenStream {
    let tks = match res_type {
        Type::BF16 => quote! {
            Scalar::new(format!(
                "__float2bfloat16_rn(1.0f / {})",
                self.to_f32().val
            ))
        },
        Type::F16 => quote! {
            Scalar::new(format!(
                "__float2half_rn(1.0f / {})",
                self.to_f32().val
            ))
        },
        Type::F32 => quote! {
            Scalar::new(format!(
                "(1.0f / {})",
                self.to_f32().val
            ))
        },
        Type::F64 => quote! {
            Scalar::new(format!(
                "(1.0 / {})",
                self.to_f64().val
            ))
        },
        Type::Complex32 => quote! {
            Scalar::new(format!(
                "cuCdivf(make_cuComplex(1.0f, 0.0f), {})",
                self.to_complex32().val
            ))
        },
        Type::Complex64 => quote! {
            Scalar::new(format!(
                "cuCdiv(make_cuDoubleComplex(1.0, 0.0), {})",
                self.to_complex64().val
            ))
        },
        _ => unreachable!(),
    };
    quote! {
        #[inline(always)]
        fn _recip(self) -> Self::Output {
            #tks
        }
    }
}

fn cuda_elu(res_type: Type) -> proc_macro2::TokenStream {
    let tks = match res_type {
        Type::BF16 => quote! {
            Scalar::new(format!(
                "__float2bfloat16_rn({} > 0.0f ? {} : (expf({}) - 1.0f))",
                self.to_f32().val, self.to_f32().val, self.to_f32().val
            ))
        },
        Type::F16 => quote! {
            Scalar::new(format!(
                "__float2half_rn({} > 0.0f ? {} : (expf({}) - 1.0f))",
                self.to_f32().val, self.to_f32().val, self.to_f32().val
            ))
        },
        Type::F32 => quote! {
            Scalar::new(format!(
                "({} > 0.0f ? {} : (expf({}) - 1.0f))",
                self.to_f32().val, self.to_f32().val, self.to_f32().val
            ))
        },
        Type::F64 => quote! {
            Scalar::new(format!(
                "({} > 0.0 ? {} : (exp({}) - 1.0))",
                self.to_f64().val, self.to_f64().val, self.to_f64().val
            ))
        },
        Type::Complex32 => quote! {
            Scalar::new(format!(
                "cuCabsf({}) > 0.0f ? {} : cuCsubf(cuCexpf({}), make_cuComplex(1.0f, 0.0f))",
                self.to_complex32().val, self.to_complex32().val, self.to_complex32().val
            ))
        },
        Type::Complex64 => quote! {
            Scalar::new(format!(
                "cuCabs({}) > 0.0 ? {} : cuCsub(cuCexp({}), make_cuDoubleComplex(1.0, 0.0))",
                self.to_complex64().val, self.to_complex64().val, self.to_complex64().val
            ))
        },
        _ => unreachable!(),
    };
    quote! {
        #[inline(always)]
        fn _elu(self, alpha: Self::Base) -> Self::Output {
            #tks
        }
    }
}

fn cuda_celu(res_type: Type) -> proc_macro2::TokenStream {
    let tks = match res_type {
        Type::BF16 => quote! {
            Scalar::new(format!(
                "__float2bfloat16_rn({} > 0.0f ? {} : {} * (expf({} / {}) - 1.0f))",
                self.to_f32().val, self.to_f32().val, alpha.to_f32().val, self.to_f32().val, alpha.to_f32().val
            ))
        },
        Type::F16 => quote! {
            Scalar::new(format!(
                "__float2half_rn({} > 0.0f ? {} : {} * (expf({} / {}) - 1.0f))",
                self.to_f32().val, self.to_f32().val, alpha.to_f32().val, self.to_f32().val, alpha.to_f32().val
            ))
        },
        Type::F32 => quote! {
            Scalar::new(format!(
                "({} > 0.0f ? {} : {} * (expf({} / {}) - 1.0f))",
                self.to_f32().val, self.to_f32().val, alpha.to_f32().val, self.to_f32().val, alpha.to_f32().val
            ))
        },
        Type::F64 => quote! {
            Scalar::new(format!(
                "({} > 0.0 ? {} : {} * (exp({} / {}) - 1.0))",
                self.to_f64().val, self.to_f64().val, alpha.to_f64().val, self.to_f64().val, alpha.to_f64().val
            ))
        },
        Type::Complex32 => quote! {
            Scalar::new(format!(
                "cuCabsf({}) > 0.0f ? {} : cuCmulf(make_cuComplex({}, 0.0f), cuCsubf(cuCexpf(cuCdivf({}, make_cuComplex({}, 0.0f))), make_cuComplex(1.0f, 0.0f)))",
                self.to_complex32().val, self.to_complex32().val, alpha.to_f32().val, self.to_complex32().val, alpha.to_f32().val
            ))
        },
        Type::Complex64 => quote! {
            Scalar::new(format!(
                "cuCabs({}) > 0.0 ? {} : cuCmul(make_cuDoubleComplex({}, 0.0), cuCsub(cuCexp(cuCdiv({}, make_cuDoubleComplex({}, 0.0))), make_cuDoubleComplex(1.0, 0.0)))",
                self.to_complex64().val, self.to_complex64().val, alpha.to_f64().val, self.to_complex64().val, alpha.to_f64().val
            ))
        },
        _ => unreachable!(),
    };
    quote! {
        #[inline(always)]
        fn _celu(self, alpha: Self::Base) -> Self::Output {
            #tks
        }
    }
}

fn cuda_fast_hard_sigmoid(res_type: Type) -> proc_macro2::TokenStream {
    let tks = match res_type {
        Type::BF16 => quote! {
            Scalar::new(format!(
                "__float2bfloat16_rn(fmaxf(0.0f, fminf(1.0f, 0.2f * {} + 0.5f)))",
                self.to_f32().val
            ))
        },
        Type::F16 => quote! {
            Scalar::new(format!(
                "__float2half_rn(fmaxf(0.0f, fminf(1.0f, 0.2f * {} + 0.5f)))",
                self.to_f32().val
            ))
        },
        Type::F32 => quote! {
            Scalar::new(format!(
                "fmaxf(0.0f, fminf(1.0f, 0.2f * {} + 0.5f))",
                self.to_f32().val
            ))
        },
        Type::F64 => quote! {
            Scalar::new(format!(
                "fmax(0.0, fmin(1.0, 0.2 * {} + 0.5))",
                self.to_f64().val
            ))
        },
        Type::Complex32 => quote! {
            Scalar::new(format!(
                "make_cuComplex(fmaxf(0.0f, fminf(1.0f, 0.2f * cuCrealf({}) + 0.5f)), 0.0f)",
                self.to_f32().val
            ))
        },
        Type::Complex64 => quote! {
            Scalar::new(format!(
                "make_cuDoubleComplex(fmax(0.0, fmin(1.0, 0.2 * cuCreal({}) + 0.5)), 0.0)",
                self.to_f64().val
            ))
        },
        _ => unreachable!(),
    };
    quote! {
        #[inline(always)]
        fn _fast_hard_sigmoid(self) -> Self::Output {
            #tks
        }
    }
}

fn cuda_hard_sigmoid(res_type: Type) -> proc_macro2::TokenStream {
    let tks = match res_type {
        Type::BF16 => quote! {
            Scalar::new(format!(
                "__float2bfloat16_rn({} > 2.5f ? 1.0f : ({}) < -2.5f ? 0.0f : 0.2f * ({}) + 0.5f)",
                self.to_f32().val, self.to_f32().val, self.to_f32().val
            ))
        },
        Type::F16 => quote! {
            Scalar::new(format!(
                "__float2half_rn({} > 2.5f ? 1.0f : ({}) < -2.5f ? 0.0f : 0.2f * ({}) + 0.5f)",
                self.to_f32().val, self.to_f32().val, self.to_f32().val
            ))
        },
        Type::F32 => quote! {
            Scalar::new(format!(
                "({} > 2.5f ? 1.0f : ({} < -2.5f ? 0.0f : 0.2f * {} + 0.5f))",
                self.to_f32().val, self.to_f32().val, self.to_f32().val
            ))
        },
        Type::F64 => quote! {
            Scalar::new(format!(
                "({} > 2.5 ? 1.0 : ({} < -2.5 ? 0.0 : 0.2 * {} + 0.5))",
                self.to_f64().val, self.to_f64().val, self.to_f64().val
            ))
        },
        Type::Complex32 => quote! {
            Scalar::new(format!(
                "cuCrealf({}) > 2.5f ? make_cuComplex(1.0f, 0.0f) : (cuCrealf({}) < -2.5f ? make_cuComplex(0.0f, 0.0f) : make_cuComplex(0.2f * cuCrealf({}) + 0.5f, 0.0f))",
                self.to_complex32().val, self.to_complex32().val, self.to_complex32().val
            ))
        },
        Type::Complex64 => quote! {
            Scalar::new(format!(
                "cuCreal({}) > 2.5 ? make_cuDoubleComplex(1.0, 0.0) : (cuCreal({}) < -2.5 ? make_cuDoubleComplex(0.0, 0.0) : make_cuDoubleComplex(0.2 * cuCreal({}) + 0.5, 0.0))",
                self.to_complex64().val, self.to_complex64().val, self.to_complex64().val
            ))
        },
        _ => unreachable!(),
    };
    quote! {
        #[inline(always)]
        fn _hard_sigmoid(self) -> Self::Output {
            #tks
        }
    }
}

fn cuda_hard_swish(res_type: Type) -> proc_macro2::TokenStream {
    let tks = match res_type {
        Type::BF16 => quote! {
            Scalar::new(format!(
                "__float2bfloat16_rn({} * ({} > 3.0f ? 1.0f : ({} < -3.0f ? 0.0f : ({} + 3.0f) / 6.0f)))",
                self.to_f32().val, self.to_f32().val, self.to_f32().val, self.to_f32().val
            ))
        },
        Type::F16 => quote! {
            Scalar::new(format!(
                "__float2half_rn({} * ({} > 3.0f ? 1.0f : ({} < -3.0f ? 0.0f : ({} + 3.0f) / 6.0f)))",
                self.to_f32().val, self.to_f32().val, self.to_f32().val, self.to_f32().val
            ))
        },
        Type::F32 => quote! {
            Scalar::new(format!(
                "({} * ({} > 3.0f ? 1.0f : ({} < -3.0f ? 0.0f : ({} + 3.0f) / 6.0f)))",
                self.to_f32().val, self.to_f32().val, self.to_f32().val, self.to_f32().val
            ))
        },
        Type::F64 => quote! {
            Scalar::new(format!(
                "({} * ({} > 3.0 ? 1.0 : ({} < -3.0 ? 0.0 : ({} + 3.0) / 6.0)))",
                self.to_f64().val, self.to_f64().val, self.to_f64().val, self.to_f64().val
            ))
        },
        Type::Complex32 => quote! {
            Scalar::new(format!(
                "cuCmulf({}, make_cuComplex(cuCrealf({}) > 3.0f ? 1.0f : (cuCrealf({}) < -3.0f ? 0.0f : (cuCrealf({}) + 3.0f) / 6.0f), 0.0f))",
                self.to_complex32().val, self.to_complex32().val, self.to_complex32().val, self.to_complex32().val
            ))
        },
        Type::Complex64 => quote! {
            Scalar::new(format!(
                "cuCmul({}, make_cuDoubleComplex(cuCreal({}) > 3.0 ? 1.0 : (cuCreal({}) < -3.0 ? 0.0 : (cuCreal({}) + 3.0) / 6.0), 0.0))",
                self.to_complex64().val, self.to_complex64().val, self.to_complex64().val, self.to_complex64().val
            ))
        },
        _ => unreachable!(),
    };
    quote! {
        #[inline(always)]
        fn _hard_swish(self) -> Self::Output {
            #tks
        }
    }
}

fn cuda_softplus(res_type: Type) -> proc_macro2::TokenStream {
    let tks = match res_type {
        Type::BF16 => quote! {
            Scalar::new(format!(
                "__float2bfloat16_rn(logf(1.0f + expf({})))",
                self.to_f32().val
            ))
        },
        Type::F16 => quote! {
            Scalar::new(format!(
                "__float2half_rn(logf(1.0f + expf({})))",
                self.to_f32().val
            ))
        },
        Type::F32 => quote! {
            Scalar::new(format!(
                "logf(1.0f + expf({}))",
                self.to_f32().val
            ))
        },
        Type::F64 => quote! {
            Scalar::new(format!(
                "log(1.0 + exp({}))",
                self.to_f64().val
            ))
        },
        Type::Complex32 => quote! {
            Scalar::new(format!(
                "cuClogf(cuCaddf(make_cuComplex(1.0f, 0.0f), cuCexpf({})))",
                self.to_complex32().val
            ))
        },
        Type::Complex64 => quote! {
            Scalar::new(format!(
                "cuClog(cuCadd(make_cuDoubleComplex(1.0, 0.0), cuCexp({})))",
                self.to_complex64().val
            ))
        },
        _ => unreachable!(),
    };
    quote! {
        #[inline(always)]
        fn _softplus(self) -> Self::Output {
            #tks
        }
    }
}

fn cuda_sigmoid(res_type: Type) -> proc_macro2::TokenStream {
    let tks = match res_type {
        Type::BF16 => quote! {
            Scalar::new(format!(
                "__float2bfloat16_rn(1.0f / (1.0f + expf(-{})))",
                self.to_f32().val
            ))
        },
        Type::F16 => quote! {
            Scalar::new(format!(
                "__float2half_rn(1.0f / (1.0f + expf(-{})))",
                self.to_f32().val
            ))
        },
        Type::F32 => quote! {
            Scalar::new(format!(
                "(1.0f / (1.0f + expf(-{})))",
                self.to_f32().val
            ))
        },
        Type::F64 => quote! {
            Scalar::new(format!(
                "(1.0 / (1.0 + exp(-{})))",
                self.to_f64().val
            ))
        },
        Type::Complex32 => quote! {
            Scalar::new(format!(
                "cuCdivf(make_cuComplex(1.0f, 0.0f), cuCaddf(make_cuComplex(1.0f, 0.0f), cuCexpf(cuCmulf(make_cuComplex(-1.0f, 0.0f), {}))))",
                self.to_complex32().val
            ))
        },
        Type::Complex64 => quote! {
            Scalar::new(format!(
                "cuCdiv(make_cuDoubleComplex(1.0, 0.0), cuCadd(make_cuDoubleComplex(1.0, 0.0), cuCexp(cuCmul(make_cuDoubleComplex(-1.0, 0.0), {}))))",
                self.to_complex64().val
            ))
        },
        _ => unreachable!(),
    };
    quote! {
        #[inline(always)]
        fn _sigmoid(self) -> Self::Output {
            #tks
        }
    }
}

fn cuda_softsign(res_type: Type) -> proc_macro2::TokenStream {
    let tks = match res_type {
        Type::BF16 => quote! {
            Scalar::new(format!(
                "__float2bfloat16_rn({} / (1.0f + fabsf({})))",
                self.to_f32().val, self.to_f32().val
            ))
        },
        Type::F16 => quote! {
            Scalar::new(format!(
                "__float2half_rn({} / (1.0f + fabsf({})))",
                self.to_f32().val, self.to_f32().val
            ))
        },
        Type::F32 => quote! {
            Scalar::new(format!(
                "({} / (1.0f + fabsf({})))",
                self.to_f32().val, self.to_f32().val
            ))
        },
        Type::F64 => quote! {
            Scalar::new(format!(
                "({} / (1.0 + fabs({})))",
                self.to_f64().val, self.to_f64().val
            ))
        },
        Type::Complex32 => quote! {
            Scalar::new(format!(
                "cuCdivf({}, cuCaddf(make_cuComplex(1.0f, 0.0f), make_cuComplex(cuCabsf({}), 0.0f)))",
                self.to_complex32().val, self.to_complex32().val
            ))
        },
        Type::Complex64 => quote! {
            Scalar::new(format!(
                "cuCdiv({}, cuCadd(make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(cuCabs({}), 0.0)))",
                self.to_complex64().val, self.to_complex64().val
            ))
        },
        _ => unreachable!(),
    };
    quote! {
        #[inline(always)]
        fn _softsign(self) -> Self::Output {
            #tks
        }
    }
}

fn cuda_mish(res_type: Type) -> proc_macro2::TokenStream {
    let tks = match res_type {
        Type::BF16 => quote! {
            Scalar::new(format!(
                "__float2bfloat16_rn({} * tanhf(logf(1.0f + expf({}))))",
                self.to_f32().val, self.to_f32().val
            ))
        },
        Type::F16 => quote! {
            Scalar::new(format!(
                "__float2half_rn({} * tanhf(logf(1.0f + expf({}))))",
                self.to_f32().val, self.to_f32().val
            ))
        },
        Type::F32 => quote! {
            Scalar::new(format!(
                "({} * tanhf(logf(1.0f + expf({}))))",
                self.to_f32().val, self.to_f32().val
            ))
        },
        Type::F64 => quote! {
            Scalar::new(format!(
                "({} * tanh(log(1.0 + exp({}))))",
                self.to_f64().val, self.to_f64().val
            ))
        },
        Type::Complex32 => quote! {
            Scalar::new(format!(
                "cuCmulf({}, cuCtanhf(cuClogf(cuCaddf(make_cuComplex(1.0f, 0.0f), cuCexpf({})))))",
                self.to_complex32().val, self.to_complex32().val
            ))
        },
        Type::Complex64 => quote! {
            Scalar::new(format!(
                "cuCmul({}, cuCtanh(cuClog(cuCadd(make_cuDoubleComplex(1.0, 0.0), cuCexp({})))))",
                self.to_complex64().val, self.to_complex64().val
            ))
        },
        _ => unreachable!(),
    };
    quote! {
        #[inline(always)]
        fn _mish(self) -> Self::Output {
            #tks
        }
    }
}

fn cuda_cbrt(res_type: Type) -> proc_macro2::TokenStream {
    let tks = match res_type {
        Type::BF16 => quote! {
            Scalar::new(format!(
                "__float2bfloat16_rn(cbrtf({})))",
                self.to_f32().val
            ))
        },
        Type::F16 => quote! {
            Scalar::new(format!(
                "__float2half_rn(cbrtf({})))",
                self.to_f32().val
            ))
        },
        Type::F32 => quote! {
            Scalar::new(format!(
                "cbrtf({})",
                self.to_f32().val
            ))
        },
        Type::F64 => quote! {
            Scalar::new(format!(
                "cbrt({})",
                self.to_f64().val
            ))
        },
        Type::Complex32 => quote! {
            Scalar::new(format!(
                "cuCpowf({}, make_cuComplex(0.333333333333333f, 0.0f))",
                self.to_complex32().val
            ))
        },
        Type::Complex64 => quote! {
            Scalar::new(format!(
                "cuCpow({}, make_cuDoubleComplex(0.333333333333333, 0.0))",
                self.to_complex64().val
            ))
        },
        _ => unreachable!(),
    };
    quote! {
        #[inline(always)]
        fn _cbrt(self) -> Self::Output {
            #tks
        }
    }
}
