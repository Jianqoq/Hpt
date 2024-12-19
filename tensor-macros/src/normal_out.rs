use crate::type_utils::{Type, TypeInfo};
use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;

pub(crate) fn __impl_normal_out_binary() -> TokenStream {
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
        for rhs in types.iter() {
            let lhs_type = TypeInfo::new(lhs);
            let rhs_type = TypeInfo::new(rhs);
            let lhs_dtype = lhs_type.dtype;
            let rhs_dtype = rhs_type.dtype;
            let res_type = lhs_type.infer_normal_res_type(&rhs_type);
            let res_type_ident = Ident::new(
                &res_type.to_string().to_lowercase(),
                proc_macro2::Span::call_site(),
            );
            let to_res_type = Ident::new(
                &format!("to_{}", res_type.to_string().to_lowercase()),
                proc_macro2::Span::call_site(),
            );
            let mul_add_method = mul_add(rhs_dtype, res_type, to_res_type.clone());
            let pow_method = pow(rhs_dtype, res_type, to_res_type.clone());
            let cmp_method = cmp(rhs_dtype, res_type, to_res_type.clone());
            let std_ops = std_ops(rhs_dtype, res_type, to_res_type.clone());

            let clamp = if res_type.is_cplx() {
                quote! {
                    #[inline(always)]
                    fn _clip(self, min: #rhs_dtype, max: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            let c = self.[<to_ #res_type_ident>]();
                            let min = min.[<to_ #res_type_ident>]();
                            let max = max.[<to_ #res_type_ident>]();
                            let clamped_re = c.re.clamp(min.re, max.re);
                            let clamped_im = c.im.clamp(min.im, max.im);
                            #res_type::new(clamped_re, clamped_im)
                        }
                    }
                }
            } else {
                quote! {
                    #[inline(always)]
                    fn _clip(self, min: #rhs_dtype, max: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            let a = self.[<to_ #res_type_ident>]();
                            let min = min.[<to_ #res_type_ident>]();
                            let max = max.[<to_ #res_type_ident>]();
                            if a < min { min } else if a > max { max } else { a }
                        }
                    }
                }
            };

            let res = quote! {
                impl NormalOut<#rhs_dtype> for #lhs_dtype {
                    type Output = #res_type;
                    #pow_method
                    #clamp
                    #mul_add_method
                    #std_ops
                    #cmp_method
                }
            };
            ret.extend(res);
        }
    }

    ret.into()
}

fn mul_add(rhs_type: Type, res_type: Type, to_res_type: Ident) -> proc_macro2::TokenStream {
    let mul_add_body = if res_type.is_float() {
        quote! {
            self.#to_res_type() * a.#to_res_type() + b.#to_res_type()
        }
    } else if res_type.is_bool() {
        quote! {
            self || a && b
        }
    } else if res_type.is_cplx() {
        quote! {
            self.#to_res_type() * a.#to_res_type() + b.#to_res_type()
        }
    } else {
        quote! {
                self.#to_res_type().wrapping_mul(a.#to_res_type()) + b.#to_res_type()
        }
    };
    quote! {
        #[inline(always)]
        fn _mul_add(self, a: #rhs_type, b: #rhs_type) -> Self::Output {
            #mul_add_body
        }
    }
}

fn pow(rhs_dtype: Type, res_type: Type, to_res_type: Ident) -> proc_macro2::TokenStream {
    let pow_body = if res_type.is_float() {
        quote! {
            self.#to_res_type().powf(rhs.#to_res_type())
        }
    } else if res_type.is_cplx() {
        quote! {
            self.#to_res_type().powc(rhs.#to_res_type())
        }
    } else {
        if res_type.is_bool() {
            quote! {
                self || rhs
            }
        } else {
            quote! {
                self.#to_res_type().pow(rhs.to_u32())
            }
        }
    };
    quote! {
        #[inline(always)]
        fn _pow(self, rhs: #rhs_dtype) -> Self::Output {
            #pow_body
        }
    }
}

fn cmp(rhs_dtype: Type, res_type: Type, to_res_type: Ident) -> proc_macro2::TokenStream {
    let (max_body, min_body) = if res_type.is_bool() {
        let max_body = quote! {
            self | rhs
        };
        let min_body = quote! {
            self & rhs
        };
        (max_body, min_body)
    } else if res_type.is_cplx() {
        let max_body = quote! {
            panic!("max method is not supported for complex number")
        };
        let min_body = quote! {
            panic!("min method is not supported for complex number")
        };
        (max_body, min_body)
    } else {
        let max_body = quote! {
            self.#to_res_type().max(rhs.#to_res_type())
        };
        let min_body = quote! {
            self.#to_res_type().min(rhs.#to_res_type())
        };
        (max_body, min_body)
    };
    quote! {
        #[inline(always)]
        fn _max(self, rhs: #rhs_dtype) -> Self::Output {
            #max_body
        }
        #[inline(always)]
        fn _min(self, rhs: #rhs_dtype) -> Self::Output {
            #min_body
        }
    }
}

fn std_ops(rhs_dtype: Type, res_type: Type, to_res_type: Ident) -> proc_macro2::TokenStream {
    let (add, sub, mul, rem) = if res_type.is_bool() {
        let add = quote! {
            let res = self.to_i8() + rhs.to_i8();
            res != 0
        };
        let sub = quote! {
            let res = self.to_i8() - rhs.to_i8();
            res != 0
        };
        let mul = quote! {
            let res = self.to_i8() * rhs.to_i8();
            res != 0
        };
        let rem = quote! {
            let res = self.to_i8() % rhs.to_i8();
            res != 0
        };
        (add, sub, mul, rem)
    } else if res_type.is_cplx() || res_type.is_float() {
        let add = quote! {
            self.#to_res_type() + rhs.#to_res_type()
        };
        let sub = quote! {
            self.#to_res_type() - rhs.#to_res_type()
        };
        let mul = quote! {
            self.#to_res_type() * rhs.#to_res_type()
        };
        let rem = quote! {
            self.#to_res_type() % rhs.#to_res_type()
        };
        (add, sub, mul, rem)
    } else {
        let add = quote! {
            self.#to_res_type().wrapping_add(rhs.#to_res_type())
        };
        let sub = quote! {
            self.#to_res_type().wrapping_sub(rhs.#to_res_type())
        };
        let mul = quote! {
            self.#to_res_type().wrapping_mul(rhs.#to_res_type())
        };
        let rem = quote! {
            self.#to_res_type().wrapping_rem(rhs.#to_res_type())
        };
        (add, sub, mul, rem)
    };
    quote! {
        #[inline(always)]
        fn _add(self, rhs: #rhs_dtype) -> Self::Output {
            #add
        }
        #[inline(always)]
        fn _sub(self, rhs: #rhs_dtype) -> Self::Output {
            #sub
        }
        #[inline(always)]
        fn _mul(self, rhs: #rhs_dtype) -> Self::Output {
            #mul
        }
        #[inline(always)]
        fn _rem(self, rhs: #rhs_dtype) -> Self::Output {
            #rem
        }
    }
}

pub(crate) fn __impl_cuda_normal_out_binary() -> TokenStream {
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
        for rhs in types.iter() {
            let lhs_type = TypeInfo::new(lhs);
            let rhs_type = TypeInfo::new(rhs);
            let lhs_dtype = lhs_type.dtype;
            let rhs_dtype = rhs_type.dtype;
            let res_type = lhs_type.infer_normal_res_type(&rhs_type);
            let to_res_type = Ident::new(
                &format!("to_{}", res_type.to_string().to_lowercase()),
                proc_macro2::Span::call_site(),
            );
            let mul_add_method = cuda_mul_add(rhs_dtype, res_type, to_res_type.clone());
            let clamp_method = cuda_clamp(rhs_dtype, res_type, to_res_type.clone());
            let pow_method = cuda_pow(rhs_dtype, res_type, to_res_type.clone());
            let cmp_method = cuda_cmp(rhs_dtype, res_type, to_res_type.clone());
            let std_ops = cuda_std_ops(rhs_dtype, res_type, to_res_type.clone());

            let res = quote! {
                impl NormalOut<Scalar<#rhs_dtype>> for Scalar<#lhs_dtype> {
                    type Output = Scalar<#res_type>;
                    #pow_method
                    #clamp_method
                    #mul_add_method
                    #std_ops
                    #cmp_method
                }
            };
            ret.extend(res);
        }
    }

    ret.into()
}

fn cuda_mul_add(rhs_type: Type, res_type: Type, to_res_type: Ident) -> proc_macro2::TokenStream {
    let tks = if res_type.is_bf16() {
        quote! {
            Scalar::new(format!(
                "__float2bfloat16_rn(fmaf({}, {}, {}))",
                self.to_f32().val, a.to_f32().val, b.to_f32().val
            ))
        }
    } else if res_type.is_f16() {
        quote! {
            Scalar::new(format!(
                "__float2half_rn(fmaf({}, {}, {}))",
                self.to_f32().val, a.to_f32().val, b.to_f32().val
            ))
        }
    } else if res_type.is_f32() {
        quote! {
            Scalar::new(format!(
                "fmaf({}, {}, {})",
                self.to_f32().val, a.to_f32().val, b.to_f32().val
            ))
        }
    } else if res_type.is_f64() {
        quote! {
            Scalar::new(format!(
                "fma({}, {}, {})",
                self.to_f64().val, a.to_f64().val, b.to_f64().val
            ))
        }
    } else if res_type.is_cplx32() {
        quote! {
            Scalar::new(format!(
                "cuCaddf(cuCmulf(make_cuComplex((float){}, 0.0f), {}), {})",
                self.to_complex32().val, a.to_complex32().val, b.to_complex32().val
            ))
        }
    } else if res_type.is_cplx64() {
        quote! {
            Scalar::new(format!(
                "cuCadd(cuCmul(make_cuDoubleComplex((double){}, 0.0), {}), {})",
                self.to_complex64().val, a.to_complex64().val, b.to_complex64().val
            ))
        }
    } else {
        quote! {
            Scalar::new(format!("({} * {} + {})",
            self.#to_res_type().val, a.#to_res_type().val, b.#to_res_type().val))
        }
    };
    quote! {
        #[inline(always)]
        fn _mul_add(self, a: Scalar<#rhs_type>, b: Scalar<#rhs_type>) -> Self::Output {
            #tks
        }
    }
}

fn cuda_clamp(rhs_type: Type, res_type: Type, to_res_type: Ident) -> proc_macro2::TokenStream {
    let tks = if res_type.is_bf16() {
        quote! {
            Scalar::new(format!(
                "__float2bfloat16_rn(fminf(fmaxf({}, {}), {}))",
                self.to_f32().val, min.to_f32().val, max.to_f32().val
            ))
        }
    } else if res_type.is_f16() {
        quote! {
            Scalar::new(format!(
                "__float2half_rn(fminf(fmaxf({}, {}), {}))",
                self.to_f32().val, min.to_f32().val, max.to_f32().val
            ))
        }
    } else if res_type.is_f32() {
        quote! {
            Scalar::new(format!(
                "fminf(fmaxf({}, {}), {})",
                self.to_f32().val, min.to_f32().val, max.to_f32().val
            ))
        }
    } else if res_type.is_f64() {
        quote! {
            Scalar::new(format!(
                "fmin(fmax({}, {}), {})",
                self.to_f64().val, min.to_f64().val, max.to_f64().val
            ))
        }
    } else if res_type.is_cplx32() || res_type.is_cplx64() {
        quote! {
            unimplemented!("clamp method is not supported for complex number")
        }
    } else {
        quote! {
            Scalar::new(format!(
                "min(max(({}, {}), {})",
                self.#to_res_type().val, min.#to_res_type().val, max.#to_res_type().val
            ))
        }
    };
    quote! {
        #[inline(always)]
        fn _clip(self, min: Scalar<#rhs_type>, max: Scalar<#rhs_type>) -> Self::Output {
            #tks
        }
    }
}

fn cuda_pow(rhs_dtype: Type, res_type: Type, to_res_type: Ident) -> proc_macro2::TokenStream {
    let tks = if res_type.is_bf16() {
        quote! {
            Scalar::new(format!(
                "__float2bfloat16_rn(powf({}, {}))",
                self.to_f32().val, rhs.to_f32().val
            ))
        }
    } else if res_type.is_f16() {
        quote! {
            Scalar::new(format!(
                "__float2half_rn(powf({}, {}))",
                self.to_f32().val, rhs.to_f32().val
            ))
        }
    } else if res_type.is_f32() {
        quote! {
            Scalar::new(format!(
                "powf({}, {})",
                self.to_f32().val, rhs.to_f32().val
            ))
        }
    } else if res_type.is_f64() {
        quote! {
            Scalar::new(format!(
                "pow({}, {})",
                self.to_f64().val, rhs.to_f64().val
            ))
        }
    } else if res_type.is_cplx32() {
        quote! {
            Scalar::new(format!(
                "cuCpowf({}, {})",
                self.to_complex32().val, rhs.to_complex32().val
            ))
        }
    } else if res_type.is_cplx64() {
        quote! {
            Scalar::new(format!(
                "cuCpow({}, {})",
                self.to_complex64().val, rhs.to_complex64().val
            ))
        }
    } else {
        let res_cuda_type = res_type.to_cuda_type();
        quote! {
            Scalar::new(format!("(({})pow((double){}, (double){}))",
                #res_cuda_type, self.#to_res_type().val, rhs.#to_res_type().val))
        }
    };
    quote! {
        #[inline(always)]
        fn _pow(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
            #tks
        }
    }
}

fn cuda_cmp(rhs_dtype: Type, res_type: Type, to_res_type: Ident) -> proc_macro2::TokenStream {
    let min_tks = if res_type.is_bf16() {
        quote! {
            Scalar::new(format!(
                "__float2bfloat16_rn(fminf({}, {}))",
                self.to_f32().val, rhs.to_f32().val
            ))
        }
    } else if res_type.is_f16() {
        quote! {
            Scalar::new(format!(
                "__float2half_rn(fminf({}, {}))",
                self.to_f32().val, rhs.to_f32().val
            ))
        }
    } else if res_type.is_f32() {
        quote! {
            Scalar::new(format!(
                "fminf({}, {})",
                self.to_f32().val, rhs.to_f32().val
            ))
        }
    } else if res_type.is_f64() {
        quote! {
            Scalar::new(format!(
                "fmin({}, {})",
                self.to_f64().val, rhs.to_f64().val
            ))
        }
    } else if res_type.is_cplx32() || res_type.is_cplx64() {
        quote! {
            unimplemented!("min method is not supported for complex number")
        }
    } else {
        quote! {
            Scalar::new(format!("min({}, {})", self.#to_res_type().val, rhs.#to_res_type().val))
        }
    };
    let max_tks = if res_type.is_bf16() {
        quote! {
            Scalar::new(format!(
                "__float2bfloat16_rn(fmaxf({}, {}))",
                self.to_f32().val, rhs.to_f32().val
            ))
        }
    } else if res_type.is_f16() {
        quote! {
            Scalar::new(format!(
                "__float2half_rn(fmaxf({}, {}))",
                self.to_f32().val, rhs.to_f32().val
            ))
        }
    } else if res_type.is_f32() {
        quote! {
            Scalar::new(format!(
                "fmaxf({}, {})",
                self.to_f32().val, rhs.to_f32().val
            ))
        }
    } else if res_type.is_f64() {
        quote! {
            Scalar::new(format!(
                "fmax({}, {})",
                self.to_f64().val, rhs.to_f64().val
            ))
        }
    } else if res_type.is_cplx32() || res_type.is_cplx64() {
        quote! {
            unimplemented!("max method is not supported for complex number")
        }
    } else {
        quote! {
            Scalar::new(format!("max({}, {})", self.#to_res_type().val, rhs.#to_res_type().val))
        }
    };
    quote! {
        #[inline(always)]
        fn _min(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
            #min_tks
        }
         #[inline(always)]
        fn _max(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
            #max_tks
        }
    }
}

fn cuda_std_ops(rhs_dtype: Type, res_type: Type, to_res_type: Ident) -> proc_macro2::TokenStream {
    let (add, sub, mul, rem) = if res_type.is_bf16() {
        (
            quote! {
                Scalar::new(format!(
                    "__float2bfloat16_rn({} +{})",
                    self.to_f32().val, rhs.to_f32().val
                ))
            },
            quote! {
                Scalar::new(format!(
                    "__float2bfloat16_rn({} - {})",
                    self.to_f32().val, rhs.to_f32().val
                ))
            },
            quote! {
                Scalar::new(format!(
                    "__float2bfloat16_rn({} * {})",
                    self.to_f32().val, rhs.to_f32().val
                ))
            },
            quote! {
                Scalar::new(format!(
                    "__float2bfloat16_rn(fmodf({}, {}))",
                    self.to_f32().val, rhs.to_f32().val
                ))
            },
        )
    } else if res_type.is_f16() {
        (
            quote! {
                Scalar::new(format!(
                    "__float2half_rn({} +{})",
                    self.to_f32().val, rhs.to_f32().val
                ))
            },
            quote! {
                Scalar::new(format!(
                    "__float2half_rn({} - {})",
                    self.to_f32().val, rhs.to_f32().val
                ))
            },
            quote! {
                Scalar::new(format!(
                    "__float2half_rn({} * {})",
                    self.to_f32().val, rhs.to_f32().val
                ))
            },
            quote! {
                Scalar::new(format!(
                    "__float2half_rn(fmodf({}, {}))",
                    self.to_f32().val, rhs.to_f32().val
                ))
            },
        )
    } else if res_type.is_cplx32() {
        (
            quote! {
                Scalar::new(format!(
                    "cuCaddf({}, {})",
                    self.to_complex32().val, rhs.to_complex32().val
                ))
            },
            quote! {
                Scalar::new(format!(
                    "cuCsubf({}, {})",
                    self.to_complex32().val, rhs.to_complex32().val
                ))
            },
            quote! {
                Scalar::new(format!(
                    "cuCmulf({}, {})",
                    self.to_complex32().val, rhs.to_complex32().val
                ))
            },
            quote! {
                // 复数取余通常不定义，返回0
                Scalar::new("make_cuComplex(0.0f, 0.0f)".to_string())
            },
        )
    } else if res_type.is_cplx64() {
        (
            quote! {
                Scalar::new(format!(
                    "cuCadd({}, {})",
                    self.to_complex64().val, rhs.to_complex64().val
                ))
            },
            quote! {
                Scalar::new(format!(
                    "cuCsub({}, {})",
                    self.to_complex64().val, rhs.to_complex64().val
                ))
            },
            quote! {
                Scalar::new(format!(
                    "cuCmul({}, {})",
                    self.to_complex64().val, rhs.to_complex64().val
                ))
            },
            quote! {
                Scalar::new("make_cuDoubleComplex(0.0, 0.0)".to_string())
            },
        )
    } else {
        (
            quote! {
                Scalar::new(format!("(({}) + ({}))", self.#to_res_type().val, rhs.#to_res_type().val))
            },
            quote! {
                Scalar::new(format!("(({}) - ({}))", self.#to_res_type().val, rhs.#to_res_type().val))
            },
            quote! {
                Scalar::new(format!("(({}) * ({}))", self.#to_res_type().val, rhs.#to_res_type().val))
            },
            quote! {
                Scalar::new(format!("(({} != 0) ? (({}) % ({})) : 0)", rhs.#to_res_type().val, self.#to_res_type().val, rhs.#to_res_type().val))
            },
        )
    };
    quote! {
        #[inline(always)]
        fn _add(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
            #add
        }
        #[inline(always)]
        fn _sub(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
            #sub
        }
        #[inline(always)]
        fn _mul(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
            #mul
        }
        #[inline(always)]
        fn _rem(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
            #rem
        }
    }
}
