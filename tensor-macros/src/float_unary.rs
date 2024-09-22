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
        let res_type = lhs_type.infer_float_res_type_uary();
        let to_res_type =
            proc_macro2::Ident::new(&format!("to_{}", res_type), proc_macro2::Span::call_site());
        let selu = if res_type.is_cplx() {
            quote! {
                panic!("selu is not supported for complex numbers")
            }
        } else if res_type.is_f16() && !lhs_dtype.is_f16() {
            quote! {
                let x = self.#to_res_type();
                let alpha = alpha.#to_res_type();
                let scale = scale.#to_res_type();
                let positive_part = x.max(#res_type::ZERO);
                let negative_part = alpha * (x.exp() - #res_type::ONE) * (x <= #res_type::ZERO).#to_res_type();
                scale * positive_part + negative_part
            }
        } else if res_type.is_f16() {
            quote! {
                let scale = scale.to_f32();
                if self.to_bits() > 0 {
                    (self.to_f32() * scale).to_f16()
                } else {
                    let alpha = alpha.to_f32();
                    let x = self.to_f32();
                    (alpha * (x.exp() - f32::ONE) * scale).to_f16()
                }
            }
        } else if res_type.is_bf16() {
            quote! {
                let scale = scale.to_f32();
                if self.to_bits() > 0 {
                    (self.to_f32() * scale).to_bf16()
                } else {
                    let alpha = alpha.to_f32();
                    let x = self.to_f32();
                    (alpha * (x.exp() - f32::ONE) * scale).to_bf16()
                }
            }
        } else if res_type.is_f32() || res_type.is_f64() {
            quote! {
                let x = self.#to_res_type();
                let alpha = alpha.#to_res_type();
                let scale = scale.#to_res_type();
                scale * x._elu(alpha)
            }
        } else {
            quote! {
                let x = self.#to_res_type();
                let alpha = alpha.#to_res_type();
                let scale = scale.#to_res_type();
                fn select(mask: #res_type, a: #res_type, b: #res_type) -> #res_type {
                    (mask & a) | (!mask & b)
                }
                select((x > #res_type::ZERO).#to_res_type(), scale * x, alpha * (x.exp() - #res_type::ONE) * scale)
            }
        };
        let gelu = if res_type.is_cplx() {
            quote! {
                fn _gelu(self) -> Self::Output {
                    panic!("gelu is not supported for complex numbers")
                }
            }
        } else if res_type.is_bf16() {
            quote! {
                fn _gelu(self) -> Self::Output {
                    let x = self.to_f32();
                    x._gelu().to_bf16()
                }
            }
        } else if res_type.is_f16() {
            quote! {
                fn _gelu(self) -> Self::Output {
                    let x = self.to_f32();
                    x._gelu().to_f16()
                }
            }
        } else if res_type.is_f32() {
            quote! {
                fn _gelu(self) -> Self::Output {
                    let x = self.to_f32();
                    0.5 * x * (1.0 + Sleef::erf(f32::FRAC_1_SQRT_2 * x))
                }
            }
        } else {
            quote! {
                fn _gelu(self) -> Self::Output {
                    let x = self.to_f64();
                    0.5 * x * (1.0 + Sleef::erf(f64::FRAC_1_SQRT_2 * x))
                }
            }
        };
        let erf = if res_type.is_bf16() || res_type.is_f16() {
            quote! {
                fn _erf(self) -> Self::Output {
                    self.to_f32().erf().#to_res_type()
                }
            }
        } else if res_type.is_cplx() {
            quote! {
                fn _erf(self) -> Self::Output {
                    panic!("erf is not supported for complex numbers")
                }
            }
        } else {
            quote! {
                fn _erf(self) -> Self::Output {
                    self.#to_res_type().erf()
                }
            }
        };

        let recip = if res_type.is_cplx() {
            quote! {
                fn _recip(self) -> Self::Output {
                    panic!("recip is not supported for complex numbers")
                }
            }
        } else {
            quote! {
                fn _recip(self) -> Self::Output {
                    self.#to_res_type().recip()
                }
            }
        };
        let celu = if res_type.is_cplx() {
            quote! {
                fn _celu(self, alpha: Self::Base) -> Self::Output {
                    panic!("celu is not supported for complex numbers")
                }
            }
        } else {
            quote! {
                fn _celu(self, alpha: Self::Base) -> Self::Output {
                    let x = self.#to_res_type();
                    let alpha = alpha.#to_res_type();
                    x.max(#res_type::ZERO) + (alpha * (x / alpha).exp() - #res_type::ONE).min(#res_type::ZERO)
                }
            }
        };
        let elu = if res_type.is_cplx() {
            quote! {
                fn _elu(self, alpha: Self::Base) -> Self::Output {
                    panic!("elu is not supported for complex numbers")
                }
            }
        } else {
            quote! {
                fn _elu(self, alpha: Self::Base) -> Self::Output {
                    let x = self.#to_res_type();
                    let alpha = alpha.#to_res_type();
                    if x > #res_type::ZERO {
                        x
                    } else {
                        alpha * (x.exp_m1())
                    }
                }
            }
        };
        let leaky_relu = if res_type.is_cplx() {
            quote! {
                fn _leaky_relu(self, alpha: Self::Base) -> Self::Output {
                    panic!("leaky_relu is not supported for complex numbers")
                }
            }
        } else {
            quote! {
                fn _leaky_relu(self, alpha: Self::Base) -> Self::Output {
                    let x = self.#to_res_type();
                    let alpha = alpha.#to_res_type();
                    if x >= #res_type::ZERO {
                        x
                    } else {
                        alpha * x
                    }
                }
            }
        };
        let _fast_hard_sigmoid = if res_type.is_cplx() {
            quote! {
                fn _fast_hard_sigmoid(self) -> Self::Output {
                    panic!("fast_hard_sigmoid is not supported for complex numbers")
                }
            }
        } else {
            quote! {
                fn _fast_hard_sigmoid(self) -> Self::Output {
                    let x = self.#to_res_type();
                    if x <= -#res_type::THREE {
                        #res_type::ZERO
                    } else if x >= #res_type::THREE {
                        #res_type::ONE
                    } else {
                        (x / #res_type::SIX) + #res_type::HALF
                    }
                }
            }
        };
        let hard_sigmoid = if res_type.is_cplx() {
            quote! {
                fn _hard_sigmoid(self) -> Self::Output {
                    panic!("hard_sigmoid is not supported for complex numbers")
                }
            }
        } else {
            quote! {
                fn _hard_sigmoid(self) -> Self::Output {
                    let x = self.#to_res_type();
                    #res_type::ZERO.max(#res_type::ONE.min(#res_type::POINT_TWO * x + #res_type::HALF))
                }
            }
        };
        let relu6 = if res_type.is_cplx() {
            quote! {
                fn _relu6(self) -> Self::Output {
                    panic!("relu6 is not supported for complex numbers")
                }
            }
        } else {
            quote! {
                fn _relu6(self) -> Self::Output {
                    self.#to_res_type().max(#res_type::ZERO).min(#res_type::SIX)
                }
            }
        };
        let relu = if res_type.is_cplx() {
            quote! {
                fn _relu(self) -> Self::Output {
                    panic!("relu is not supported for complex numbers")
                }
            }
        } else {
            quote! {
                fn _relu(self) -> Self::Output {
                    self.#to_res_type().max(#res_type::ZERO)
                }
            }
        };
        let hard_swish = if res_type.is_cplx() {
            quote! {
                fn _hard_swish(self) -> Self::Output {
                    panic!("hard_swish is not supported for complex numbers")
                }
            }
        } else {
            quote! {
                fn _hard_swish(self) -> Self::Output {
                    let x = self.#to_res_type();
                    x * ((x + #res_type::THREE).max(#res_type::ZERO).min(#res_type::SIX)) / #res_type::SIX
                }
            }
        };
        let soft_sign = if res_type.is_cplx() {
            quote! {
                fn _softsign(self) -> Self::Output {
                    panic!("softsign is not supported for complex numbers")
                }
            }
        } else {
            quote! {
                fn _softsign(self) -> Self::Output {
                    let x = self.#to_res_type();
                    x / (#res_type::ONE + x.abs())
                }
            }
        };
        let res = quote! {
            impl FloatOutUnary for #lhs_dtype {
                type Output = #res_type;
                type Base = #res_type;
                fn _exp(self) -> Self::Output {
                    self.#to_res_type().exp()
                }
                fn _exp2(self) -> Self::Output {
                    self.#to_res_type().exp2()
                }
                fn _ln(self) -> Self::Output {
                    self.#to_res_type().ln()
                }
                fn _log2(self) -> Self::Output {
                    self.#to_res_type().log2()
                }
                fn _log10(self) -> Self::Output {
                    self.#to_res_type().log10()
                }
                fn _sqrt(self) -> Self::Output {
                    self.#to_res_type().sqrt()
                }
                fn _sin(self) -> Self::Output {
                    self.#to_res_type().sin()
                }
                fn _cos(self) -> Self::Output {
                    self.#to_res_type().cos()
                }
                fn _tan(self) -> Self::Output {
                    self.#to_res_type().tan()
                }
                fn _asin(self) -> Self::Output {
                    self.#to_res_type().asin()
                }
                fn _acos(self) -> Self::Output {
                    self.#to_res_type().acos()
                }
                fn _atan(self) -> Self::Output {
                    self.#to_res_type().atan()
                }
                fn _sinh(self) -> Self::Output {
                    self.#to_res_type().sinh()
                }
                fn _cosh(self) -> Self::Output {
                    self.#to_res_type().cosh()
                }
                fn _tanh(self) -> Self::Output {
                    self.#to_res_type().tanh()
                }
                fn _asinh(self) -> Self::Output {
                    self.#to_res_type().asinh()
                }
                fn _acosh(self) -> Self::Output {
                    self.#to_res_type().acosh()
                }
                fn _atanh(self) -> Self::Output {
                    self.#to_res_type().atanh()
                }
                #recip
                #erf
                #celu
                fn _sigmoid(self) -> Self::Output {
                    #res_type::ONE / (#res_type::ONE + (-self.#to_res_type()).exp())
                }
                #elu
                #leaky_relu
                #relu
                #gelu
                fn _selu(self, alpha: Self::Base, scale: Self::Base) -> Self::Output {
                    #selu
                }
                #hard_sigmoid
                #_fast_hard_sigmoid
                #relu6
                #hard_swish
                fn _softplus(self) -> Self::Output {
                    let x = self.#to_res_type();
                    (#res_type::ONE + x.exp()).ln()
                }
                #soft_sign
                fn _mish(self) -> Self::Output {
                    let x = self.#to_res_type();
                    x * (#res_type::ONE + x.exp()).ln().tanh()
                }
                fn _cbrt(self) -> Self::Output {
                    self.#to_res_type().cbrt()
                }
            }
        };
        ret.extend(res);
    }

    ret.into()
}
