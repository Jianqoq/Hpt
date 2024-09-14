use proc_macro::TokenStream;
use quote::quote;
use crate::type_utils::TypeInfo;

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
    ];

    for lhs in types.iter() {
        let lhs_type = TypeInfo::new(lhs);
        let lhs_dtype = lhs_type.dtype;
        let res_type = lhs_type.infer_float_res_type_uary();
        let selu = if res_type.is_f16() && !lhs_dtype.is_f16() {
            quote! {
                    paste::paste! {
                    let x = self.[<to_ #res_type>]();
                    let alpha = alpha.[<to_ #res_type>]();
                    let scale = scale.[<to_ #res_type>]();
                    let positive_part = x.max(#res_type::ZERO);
                    let negative_part = alpha * (x.exp() - #res_type::ONE) * (x <= #res_type::ZERO).[<to_ #res_type>]();
                    scale * positive_part + negative_part
                }
            }
        } else if res_type.is_f16() {
            quote! {
                        paste::paste! {
                        let scale = scale.to_f32();
                        if self.to_bits() > 0 {
                            (self.to_f32() * scale).to_f16()
                        } else {
                            let alpha = alpha.to_f32();
                            let x = self.to_f32();
                            (alpha * (x.exp() - f32::ONE) * scale).to_f16()
                        }
                    }
                }
        } else if res_type.is_bf16() {
            quote! {
                        paste::paste! {
                        let scale = scale.to_f32();
                        if self.to_bits() > 0 {
                            (self.to_f32() * scale).to_bf16()
                        } else {
                            let alpha = alpha.to_f32();
                            let x = self.to_f32();
                            (alpha * (x.exp() - f32::ONE) * scale).to_bf16()
                        }
                    }
                }
        } else if res_type.is_f32() || res_type.is_f64() {
            quote! {
                        paste::paste! {
                        let x = self.[<to_ #res_type>]();
                        let alpha = alpha.[<to_ #res_type>]();
                        let scale = scale.[<to_ #res_type>]();
                        scale * x._elu(alpha)
                    }
                }
        } else {
            quote! {
                    paste::paste! {
                    let x = self.[<to_ #res_type>]();
                    let alpha = alpha.[<to_ #res_type>]();
                    let scale = scale.[<to_ #res_type>]();
                    fn select(mask: #res_type, a: #res_type, b: #res_type) -> #res_type {
                        (mask & a) | (!mask & b)
                    }
                    select((x > #res_type::ZERO).[<to_ #res_type>](), scale * x, alpha * (x.exp() - #res_type::ONE) * scale)
                }
            }
        };
        let gelu = if res_type.is_bf16() {
            quote! {
                fn _gelu(self) -> Self::Output {
                    let x = self.to_f32();
                    let sqrt2_over_2 = std::f32::consts::FRAC_1_SQRT_2;
                    (0.5 * x * (f32::ONE + (x * sqrt2_over_2).erf())).to_bf16()
                }
            }
        } else if res_type.is_f16() {
            quote! {
                fn _gelu(self) -> Self::Output {
                    let x = self.to_f32();
                    let sqrt2_over_2 = std::f32::consts::FRAC_1_SQRT_2;
                    (0.5 * x * (f32::ONE + (x * sqrt2_over_2).erf())).to_f16()
                }
            }
        } else if res_type.is_f32() {
            quote! {
                fn _gelu(self) -> Self::Output {
                    let x = self.to_f32();
                    let sqrt2_over_2 = std::f32::consts::FRAC_1_SQRT_2;
                    0.5 * x * (f32::ONE + (x * sqrt2_over_2).erf())
                }
            }
        } else {
            quote! {
                fn _gelu(self) -> Self::Output {
                    let x = self.to_f64();
                    let sqrt2_over_2 = std::f64::consts::FRAC_1_SQRT_2;
                    f64::HALF * x * (f64::ONE + (x * sqrt2_over_2).erf())
                }
            }
        };
        let erf = if res_type.is_bf16() || res_type.is_f16() {
            quote! {
                fn _erf(self) -> Self::Output {
                    paste::paste! {
                        self.to_f32().erf().[<to_ #res_type>]()
                    }
                }
            }
        } else {
            quote! {
                fn _erf(self) -> Self::Output {
                    paste::paste! {
                        self.[<to_ #res_type>]().erf()
                    }
                }
            }
        };
        let res =
            quote! {
                impl FloatOutUnary for #lhs_dtype {
                    type Output = #res_type;
                    type Base = #res_type;
                    fn _exp(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().exp()
                        }
                    }
                    fn _exp2(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().exp2()
                        }
                    }
                    fn _ln(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().ln()
                        }
                    }
                    fn _log2(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().log2()
                        }
                    }
                    fn _log10(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().log10()
                        }
                    }
                    fn _sqrt(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().sqrt()
                        }
                    }
                    fn _sin(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().sin()
                        }
                    }
                    fn _cos(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().cos()
                        }
                    }
                    fn _tan(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().tan()
                        }
                    }
                    fn _asin(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().asin()
                        }
                    }
                    fn _acos(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().acos()
                        }
                    }
                    fn _atan(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().atan()
                        }
                    }
                    fn _sinh(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().sinh()
                        }
                    }
                    fn _cosh(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().cosh()
                        }
                    }
                    fn _tanh(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().tanh()
                        }
                    }
                    fn _asinh(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().asinh()
                        }
                    }
                    fn _acosh(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().acosh()
                        }
                    }
                    fn _atanh(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().atanh()
                        }
                    }
                    fn _recip(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().recip()
                        }
                    }
                    #erf
                    fn _celu(self, alpha: Self::Base) -> Self::Output {
                        paste::paste! {
                            let x = self.[<to_ #res_type>]();
                            let alpha = alpha.[<to_ #res_type>]();
                            x.max(#res_type::ZERO) + (alpha * (x / alpha).exp() - #res_type::ONE).min(#res_type::ZERO)
                        }
                    }
                    fn _sigmoid(self) -> Self::Output {
                        paste::paste! {
                            #res_type::ONE / (#res_type::ONE + (-self.[<to_ #res_type>]()).exp())
                        }
                    }
                    fn _elu(self, alpha: Self::Base) -> Self::Output {
                        paste::paste! {
                            let x = self.[<to_ #res_type>]();
                            let alpha = alpha.[<to_ #res_type>]();
                            if x > #res_type::ZERO {
                                x
                            } else {
                                alpha * (x.exp_m1())
                            }
                        }
                    }
                    fn _leaky_relu(self, alpha: Self::Base) -> Self::Output {
                        paste::paste! {
                            let x = self.[<to_ #res_type>]();
                            let alpha = alpha.[<to_ #res_type>]();
                            if x >= #res_type::ZERO {
                                x
                            } else {
                                alpha * x
                            }
                        }
                    }
                    fn _relu(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().max(#res_type::ZERO)
                        }
                    }
                    #gelu
                    fn _selu(self, alpha: Self::Base, scale: Self::Base) -> Self::Output {
                        #selu
                    }
                    fn _hard_sigmoid(self) -> Self::Output {
                        paste::paste! {
                            let x = self.[<to_ #res_type>]();
                            #res_type::ONE.min(#res_type::ZERO.max(#res_type::POINT_TWO * x + #res_type::HALF))
                        }
                    }
                    fn _relu6(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().max(#res_type::ZERO).min(#res_type::SIX)
                        }
                    }
                    fn _hard_swish(self) -> Self::Output {
                        paste::paste! {
                            let x = self.[<to_ #res_type>]();
                            x * ((x + #res_type::THREE).max(#res_type::ZERO).min(#res_type::SIX)) / #res_type::SIX
                        }
                    }
                    fn _softplus(self) -> Self::Output {
                        paste::paste! {
                            let x = self.[<to_ #res_type>]();
                            (#res_type::ONE + x.exp()).ln()
                        }
                    }
                    fn _softsign(self) -> Self::Output {
                        paste::paste! {
                            let x = self.[<to_ #res_type>]();
                            x / (#res_type::ONE + x.abs())
                        }
                    }
                    fn _mish(self) -> Self::Output {
                        paste::paste! {
                            let x = self.[<to_ #res_type>]();
                            x * (#res_type::ONE + x.exp()).ln().tanh()
                        }
                    }
                    fn _cbrt(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().cbrt()
                        }
                    }
                }
            };
        ret.extend(res);
    }

    ret.into()
}
