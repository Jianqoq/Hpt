use crate::type_utils::{Type, TypeInfo};
use proc_macro::TokenStream;
use proc_macro2::Ident;
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
        let neg_method = neg(lhs_dtype);
        let abs_method = abs(lhs_dtype);
        let ceil_method = ceil_floor_round(lhs_dtype, 1);
        let floor_method = ceil_floor_round(lhs_dtype, 2);
        let sign_method = sign(lhs_dtype);
        let round_method = ceil_floor_round(lhs_dtype, 0);
        let relu_method = relu();
        let relu6_method = relu6();
        let leaky_relu_method = leaky_relu();
        let res = quote! {
            impl NormalOutUnary for #lhs_dtype {
                type Base = Self;
                #[inline(always)]
                fn _square(self) -> Self {
                    self._mul(self)
                }
                #neg_method
                #abs_method
                #ceil_method
                #floor_method
                #sign_method
                #round_method
                #relu_method
                #relu6_method
                #leaky_relu_method
            }
        };
        ret.extend(res);
    }

    ret.into()
}

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
        let neg_method = cuda_neg(lhs_dtype);
        let abs_method = cuda_abs(lhs_dtype);
        let ceil_method = cuda_ceil_floor_round(lhs_dtype, 1);
        let floor_method = cuda_ceil_floor_round(lhs_dtype, 2);
        let sign_method = cuda_sign(lhs_dtype);
        let round_method = cuda_ceil_floor_round(lhs_dtype, 0);
        let relu_method = cuda_relu(lhs_dtype);
        let relu6_method = cuda_relu6(lhs_dtype);
        let leaky_relu_method = cuda_leaky_relu(lhs_dtype);
        let res = quote! {
            impl NormalOutUnary for Scalar<#lhs_dtype> {
                type Base = Scalar<#lhs_dtype>;
                #[inline(always)]
                fn _square(self) -> Self {
                    self.clone()._mul(self)
                }
                #neg_method
                #abs_method
                #ceil_method
                #floor_method
                #sign_method
                #round_method
                #relu_method
                #relu6_method
                #leaky_relu_method
            }
        };
        ret.extend(res);
    }

    ret.into()
}

fn neg(lhs_dtype: Type) -> proc_macro2::TokenStream {
    let neg_body = if lhs_dtype.is_float() {
        quote! {
            -self
        }
    } else if lhs_dtype.is_bool() {
        quote! {
            !self
        }
    } else if lhs_dtype.is_unsigned() {
        quote! {
            !self + 1
        }
    } else {
        quote! {
            -self
        }
    };
    quote! {
        #[inline(always)]
        fn _neg(self) -> Self {
            #neg_body
        }
    }
}

fn abs(lhs_dtype: Type) -> proc_macro2::TokenStream {
    let abs_body = if lhs_dtype.is_unsigned() {
        quote! {
            self
        }
    } else if lhs_dtype.is_cplx() {
        quote! {
                panic!("abs method is not supported for complex number")
        }
    } else {
        quote! {
            self.abs()
        }
    };
    quote! {
        #[inline(always)]
        fn _abs(self) -> Self {
            #abs_body
        }
    }
}

fn ceil_floor_round(lhs_dtype: Type, mode: u8) -> proc_macro2::TokenStream {
    let ceil_body = if lhs_dtype.is_float() {
        match mode {
            0 => quote! {
                self.round()
            },
            1 => quote! {
                self.ceil()
            },
            2 => quote! {
                self.floor()
            },
            _ => unreachable!(),
        }
    } else {
        quote! {
            self
        }
    };
    let method = match mode {
        0 => "_round",
        1 => "_ceil",
        2 => "_floor",
        _ => unreachable!(),
    };
    let method_ident = Ident::new(method, proc_macro2::Span::call_site());
    quote! {
        #[inline(always)]
        fn #method_ident(self) -> Self {
            #ceil_body
        }
    }
}

fn sign(res_type: Type) -> proc_macro2::TokenStream {
    let sign_body = if res_type.is_float() {
        quote! {
            self.signum()
        }
    } else if res_type.is_cplx() {
        quote! {
            panic!("sign method is not supported for complex number")
        }
    } else if res_type.is_unsigned() {
        quote! {
            #res_type::ZERO
        }
    } else {
        quote! {
            self.signum()
        }
    };
    quote! {
        #[inline(always)]
        fn _sign(self) -> Self {
            #sign_body
        }
    }
}

fn relu() -> proc_macro2::TokenStream {
    quote! {
        #[inline(always)]
        fn _relu(self) -> Self {
            self._max(Self::ZERO)
        }
    }
}

fn relu6() -> proc_macro2::TokenStream {
    quote! {
        #[inline(always)]
        fn _relu6(self) -> Self {
            self._max(Self::ZERO)._min(Self::SIX)
        }
    }
}

fn leaky_relu() -> proc_macro2::TokenStream {
    quote! {
        #[inline(always)]
        fn _leaky_relu(self, alpha: Self::Base) -> Self {
            self._max(Self::ZERO)._add(alpha._mul(self._min(Self::ZERO)))
        }
    }
}

fn cuda_neg(lhs_dtype: Type) -> proc_macro2::TokenStream {
    let neg_body = if lhs_dtype.is_bf16() {
        quote! {
            Scalar::new(format!(
                "__float2bfloat16_rn(-{})",
                self.to_f32().val
            ))
        }
    } else if lhs_dtype.is_f16() {
        quote! {
            Scalar::new(format!(
                "__hneg({})",
                self.val
            ))
        }
    } else if lhs_dtype.is_float() {
        quote! {
            Scalar::new(format!("(-({}))", self.val))
        }
    } else if lhs_dtype.is_bool() {
        quote! {
            Scalar::new(format!("(!({}))", self.val))
        }
    } else if lhs_dtype.is_unsigned() {
        quote! {
            Scalar::new("0".to_string())
        }
    } else if lhs_dtype.is_cplx32() {
        quote! {
            Scalar::new(format!(
                "make_cuComplex(-cuCrealf({}), -cuCimagf({}))",
                self.val, self.val
            ))
        }
    } else if lhs_dtype.is_cplx64() {
        quote! {
            Scalar::new(format!(
                "make_cuDoubleComplex(-cuCreal({}), -cuCimag({}))",
                self.val, self.val
            ))
        }
    } else {
        quote! {
            Scalar::new(format!("(-({}))", self.val))
        }
    };
    quote! {
        #[inline(always)]
        fn _neg(self) -> Self {
            #neg_body
        }
    }
}

fn cuda_abs(lhs_dtype: Type) -> proc_macro2::TokenStream {
    let abs_body = if lhs_dtype.is_bf16() {
        quote! {
            Scalar::new(format!(
                "__float2bfloat16_rn(fabsf({}))",
                self.to_f32().val
            ))
        }
    } else if lhs_dtype.is_f16() {
        quote! {
            Scalar::new(format!(
                "__habs({})",
                self.val
            ))
        }
    } else if lhs_dtype.is_f32() {
        quote! {
            Scalar::new(format!("fabsf({})", self.val))
        }
    } else if lhs_dtype.is_f64() {
        quote! {
            Scalar::new(format!("fabs({})", self.val))
        }
    } else if lhs_dtype.is_bool() {
        quote! {
            self
        }
    } else if lhs_dtype.is_unsigned() {
        quote! {
            self.clone()
        }
    } else if lhs_dtype.is_cplx32() {
        quote! {
            Scalar::new(format!("cuCabsf({})", self.val))
        }
    } else if lhs_dtype.is_cplx64() {
        quote! {
            Scalar::new(format!("cuCabs({})", self.val))
        }
    } else {
        quote! {
            Scalar::new(format!("abs({})", self.val))
        }
    };
    quote! {
        #[inline(always)]
        fn _abs(self) -> Self {
            #abs_body
        }
    }
}

fn cuda_ceil_floor_round(lhs_dtype: Type, mode: u8) -> proc_macro2::TokenStream {
    let ceil_body = if lhs_dtype.is_bf16() {
        match mode {
            0 => quote! {
                Scalar::new(format!(
                    "__float2bfloat16_rn(roundf({}))",
                    self.to_f32().val
                ))
            },
            1 => quote! {
                Scalar::new(format!(
                    "__float2bfloat16_rn(ceilf({}))",
                    self.to_f32().val
                ))
            },
            2 => quote! {
                Scalar::new(format!(
                    "__float2bfloat16_rn(floorf({}))",
                    self.to_f32().val
                ))
            },
            _ => unreachable!(),
        }
    } else if lhs_dtype.is_f16() {
        match mode {
            0 => quote! {
                Scalar::new(format!(
                    "__float2half_rn(rintf({}))",
                    self.to_f32().val
                ))
            },
            1 => quote! {
                Scalar::new(format!(
                    "__float2half_rn(ceilf({}))",
                    self.to_f32().val
                ))
            },
            2 => quote! {
                Scalar::new(format!(
                    "__float2half_rn(floorf({}))",
                    self.to_f32().val
                ))
            },
            _ => unreachable!(),
        }
    } else if lhs_dtype.is_f32() {
        match mode {
            0 => quote! {
                Scalar::new(format!("roundf({})", self.val))
            },
            1 => quote! {
                Scalar::new(format!("ceilf({})", self.val))
            },
            2 => quote! {
                Scalar::new(format!("floorf({})", self.val))
            },
            _ => unreachable!(),
        }
    } else if lhs_dtype.is_f64() {
        match mode {
            0 => quote! {
                Scalar::new(format!("round({})", self.val))
            },
            1 => quote! {
                Scalar::new(format!("ceil({})", self.val))
            },
            2 => quote! {
                Scalar::new(format!("floor({})", self.val))
            },
            _ => unreachable!(),
        }
    } else {
        quote! {
            self
        }
    };
    let method = match mode {
        0 => "_round",
        1 => "_ceil",
        2 => "_floor",
        _ => unreachable!(),
    };
    let method_ident = Ident::new(method, proc_macro2::Span::call_site());
    quote! {
        #[inline(always)]
        fn #method_ident(self) -> Self {
            #ceil_body
        }
    }
}

fn cuda_sign(lhs_dtype: Type) -> proc_macro2::TokenStream {
    let sign_body = if lhs_dtype.is_bf16() {
        quote! {
            Scalar::new(format!(
                "__float2bfloat16_rn(copysignf(1.0f, {}))",
                self.to_f32().val
            ))
        }
    } else if lhs_dtype.is_f16() {
        quote! {
            Scalar::new(format!(
                "__float2half_rn(copysignf(1.0f, {}))",
                self.to_f32().val
            ))
        }
    } else if lhs_dtype.is_f32() {
        quote! {
            Scalar::new(format!(
                "copysignf(1.0f, {})",
                self.to_f32().val
            ))
        }
    } else if lhs_dtype.is_f64() {
        quote! {
            Scalar::new(format!(
                "copysign(1.0, {})",
                self.to_f64().val
            ))
        }
    } else if lhs_dtype.is_cplx32() || lhs_dtype.is_cplx64() {
        quote! {
            panic!("sign method is not supported for complex number")
        }
    } else if lhs_dtype.is_unsigned() {
        quote! {
            Scalar::new(format!("({} > 0) ? 1 : 0", self.val))
        }
    } else {
        quote! {
            Scalar::new(format!("({0} > 0) ? 1 : ({0} < 0) ? -1 : 0", self.val))
        }
    };
     quote! {
        #[inline(always)]
        fn _sign(self) -> Self {
            #sign_body
        }
    }
}

fn cuda_relu(lhs_dtype: Type) -> proc_macro2::TokenStream {
    let relu_body = if lhs_dtype.is_bf16() {
        quote! {
            Scalar::new(format!(
                "__float2bfloat16_rn(fmaxf({}), 0.0f))",
                self.to_f32().val
            ))
        }
    } else if lhs_dtype.is_f16() {
        quote! {
            Scalar::new(format!(
                "__float2half_rn(fmaxf({}), 0.0f))",
                self.to_f32().val
            ))
        }
    } else if lhs_dtype.is_f32() {
        quote! {
            Scalar::new(format!("fmaxf({}, 0.0f)", self.to_f32().val))
        }
    } else if lhs_dtype.is_f64() {
        quote! {
            Scalar::new(format!("fmax({}, 0.0)", self.to_f64().val))
        }
    } else if lhs_dtype.is_cplx32() {
        quote! {
            Scalar::new(format!(
                "cuCabsf({}) > 0.0f ? {} : make_cuComplex(0.0f, 0.0f)",
                self.to_complex32().val, self.to_complex32().val
            ))
        }
    } else if lhs_dtype.is_cplx64() {
        quote! {
            Scalar::new(format!(
                "cuCabs({}) > 0.0 ? {} : make_cuDoubleComplex(0.0, 0.0)",
                self.to_complex64().val, self.to_complex64().val
            ))
        }
    } else if lhs_dtype.is_unsigned() {
        quote! {
            self
        }
    } else {
        quote! {
            Scalar::new(format!("max({}, 0)", self.to_f32().val))
        }
    };
     quote! {
        #[inline(always)]
        fn _relu(self) -> Self {
            #relu_body
        }
    }
}

fn cuda_relu6(lhs_dtype: Type) -> proc_macro2::TokenStream {
    let relu6_body = if lhs_dtype.is_bf16() {
        quote! {
            Scalar::new(format!(
                "__float2bfloat16_rn(fminf(fmaxf({}), 0.0f), 6.0f))",
                self.to_f32().val
            ))
        }
    } else if lhs_dtype.is_f16() {
        quote! {
            Scalar::new(format!(
                "__float2half_rn(fminf(fmaxf({}), 0.0f), 6.0f))",
                self.to_f32().val
            ))
        }
    } else if lhs_dtype.is_f32() {
        quote! {
            Scalar::new(format!("fminf(fmaxf({}, 0.0f), 6.0f)", self.to_f32().val))
        }
    } else if lhs_dtype.is_f64() {
        quote! {
            Scalar::new(format!("fmin(fmax({}, 0.0), 6.0)", self.to_f64().val))
        }
    } else if lhs_dtype.is_cplx32() {
        quote! {
            Scalar::new(format!(
                "cuCabsf({}) > 0.0f ? (cuCabsf({}) < 6.0f ? {} : make_cuComplex(6.0f, 0.0f)) : make_cuComplex(0.0f, 0.0f)",
                self.to_complex32().val, self.to_complex32().val, self.to_complex32().val
            ))
        }
    } else if lhs_dtype.is_cplx64() {
        quote! {
            Scalar::new(format!(
                "cuCabs({}) > 0.0 ? (cuCabs({}) < 6.0 ? {} : make_cuDoubleComplex(6.0, 0.0)) : make_cuDoubleComplex(0.0, 0.0)",
                self.to_complex64().val, self.to_complex64().val, self.to_complex64().val
            ))
        }
    } else if lhs_dtype.is_unsigned() {
        quote! {
            Scalar::new(format!("min({}, 6)", self.val))
        }
    } else {
        quote! {
            Scalar::new(format!("min(max({}, 0), 6)", self.val))
        }
    };
     quote! {
        #[inline(always)]
        fn _relu6(self) -> Self {
            #relu6_body
        }
    }
}

fn cuda_leaky_relu(lhs_dtype: Type) -> proc_macro2::TokenStream {
    let leaky_body = if lhs_dtype.is_bf16() {
        quote! {
            Scalar::new(format!(
                "__float2bfloat16_rn(({} > 0.0f) ? {} : ({} * {}))",
                self.to_f32().val, self.to_f32().val, alpha.to_f32().val, self.to_f32().val
            ))
        }
    } else if lhs_dtype.is_f16() {
        quote! {
            Scalar::new(format!(
                "__float2half_rn(({} > 0.0f) ? {} : ({} * {}))",
                self.to_f32().val, self.to_f32().val, alpha.to_f32().val, self.to_f32().val
            ))
        }
    } else if lhs_dtype.is_f32() {
        quote! {
            Scalar::new(format!(
                "({} > 0.0f) ? {} : ({} * {})",
                self.to_f32().val, self.to_f32().val, alpha.to_f32().val, self.to_f32().val
            ))
        }
    } else if lhs_dtype.is_f64() {
        quote! {
            Scalar::new(format!(
                "({} > 0.0) ? {} : ({} * {})",
                self.to_f64().val, self.to_f64().val, alpha.to_f64().val, self.to_f64().val
            ))
        }
    } else if lhs_dtype.is_cplx32() {
        quote! {
            Scalar::new(format!(
                "cuCabsf({}) > 0.0f ? {} : cuCmulf(make_cuComplex({}, 0.0f), {})",
                self.to_complex32().val, self.to_complex32().val, alpha.to_complex32().val, self.to_complex32().val
            ))
        }
    } else if lhs_dtype.is_cplx64() {
        quote! {
            Scalar::new(format!(
                "cuCabs({}) > 0.0 ? {} : cuCmul(make_cuDoubleComplex({}, 0.0), {})",
                self.to_complex64().val, self.to_complex64().val, alpha.to_complex64().val, self.to_complex64().val
            ))
        }
    } else if lhs_dtype.is_unsigned() {
        // 无符号类型不需要 leaky，直接返回原值
        quote! {
            self
        }
    } else {
        quote! {
            Scalar::new(format!(
                "({} > 0) ? {} : ({} * {})",
                self.val, self.val, alpha.val, self.val
            ))
        }
    };

    quote! {
        #[inline(always)]
        fn _leaky_relu(self, alpha: Self::Base) -> Self {
            #leaky_body
        }
    }
}
