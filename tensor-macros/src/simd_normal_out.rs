use proc_macro::TokenStream;
use crate::type_utils::{ type_simd_lanes, SimdType, TypeInfo };
use quote::quote;
use crate::TokenStream2;

pub fn impl_simd_normal_out() -> TokenStream {
    let mut ret = proc_macro2::TokenStream::new();

    let types = [
        (format!("boolx{}", type_simd_lanes("bool")), "bool"),
        (format!("bf16x{}", type_simd_lanes("bf16")), "bf16"),
        (format!("f16x{}", type_simd_lanes("f16")), "f16"),
        (format!("f32x{}", type_simd_lanes("f32")), "f32"),
        (format!("f64x{}", type_simd_lanes("f64")), "f64"),
        (format!("i8x{}", type_simd_lanes("i8")), "i8"),
        (format!("i16x{}", type_simd_lanes("i16")), "i16"),
        (format!("i32x{}", type_simd_lanes("i32")), "i32"),
        (format!("i64x{}", type_simd_lanes("i64")), "i64"),
        (format!("u8x{}", type_simd_lanes("u8")), "u8"),
        (format!("u16x{}", type_simd_lanes("u16")), "u16"),
        (format!("u32x{}", type_simd_lanes("u32")), "u32"),
        (format!("u64x{}", type_simd_lanes("u64")), "u64"),
        (format!("isizex{}", type_simd_lanes("isize")), "isize"),
        (format!("usizex{}", type_simd_lanes("usize")), "usize"),
    ];

    for (lhs_simd_ty, lhs) in types.iter() {
        for (rhs_simd_ty, rhs) in types.iter() {
            let lhs_lanes = type_simd_lanes(lhs);
            let rhs_lanes = type_simd_lanes(rhs);
            let lhs_type = TypeInfo::new(lhs);
            let rhs_type = TypeInfo::new(rhs);
            let lhs_dtype = lhs_type.dtype;
            let rhs_dtype = rhs_type.dtype;
            let res_type = lhs_type.infer_normal_res_type(&rhs_type);
            if lhs_lanes != rhs_lanes {
                let res_simd_ty = types
                    .iter()
                    .find(|(_, ty)| *ty == res_type.to_string().as_str())
                    .unwrap()
                    .0.clone();
                ret.extend(
                    impl_unreachable((*lhs).into(), (*rhs).into(), (res_type.to_string().as_str()).into())
                );
                continue;
            }
            let pow_method = if res_type.is_float() {
                quote! {
                    fn _pow(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().powf(rhs.[<to_ #res_type>]())
                        }
                    }
                }
            } else {
                quote! {
                    fn _pow(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().pow(rhs.to_u32())
                        }
                    }
                }
            };

            let abs_method = if lhs_dtype.is_unsigned() {
                quote! {
                    fn _abs(self) -> Self {
                        paste::paste! {
                            self
                        }
                    }
                }
            } else {
                quote! {
                    fn _abs(self) -> Self {
                        paste::paste! {
                            self.abs()
                        }
                    }
                }
            };

            let ceil_method = if res_type.is_float() {
                quote! {
                    fn _ceil(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().ceil()
                        }
                    }
                }
            } else {
                quote! {
                    fn _ceil(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]()
                        }
                    }
                }
            };

            let floor_method = if res_type.is_float() {
                quote! {
                    fn _floor(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().floor()
                        }
                    }
                }
            } else {
                quote! {
                    fn _floor(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]()
                        }
                    }
                }
            };

            let sign_method = if res_type.is_float() {
                quote! {
                    fn _sign(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().signum()
                        }
                    }
                }
            } else if res_type.is_unsigned() {
                quote! {
                    fn _sign(self) -> Self::Output {
                        paste::paste! {
                            #res_type::ZERO
                        }
                    }
                }
            } else {
                quote! {
                    fn _sign(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().signum()
                        }
                    }
                }
            };

            let cmp_method =
                quote! {
                    fn _max(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().max(rhs.[<to_ #res_type>]())
                        }
                    }
                    fn _min(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().min(rhs.[<to_ #res_type>]())
                        }
                    }
                };

            let round_method = if res_type.is_float() {
                quote! {
                    fn _round(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]().round()
                        }
                    }
                }
            } else {
                quote! {
                    fn _round(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]()
                        }
                    }
                }
            };

            let res =
                quote! {
                impl NormalOut<#rhs_dtype> for #lhs_dtype {
                    type Output = #res_type;
                    fn _add(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]() + rhs.[<to_ #res_type>]()
                        }
                    }
                    fn _sub(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]() - rhs.[<to_ #res_type>]()
                        }
                    }
                    fn _mul(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]() * rhs.[<to_ #res_type>]()
                        }
                    }
                    #pow_method

                    fn _rem(self, rhs: #rhs_dtype) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]() % rhs.[<to_ #res_type>]()
                        }
                    }
                    fn _square(self) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]() * self.[<to_ #res_type>]()
                        }
                    }
                    fn _clip(self, min: Self::Output, max: Self::Output) -> Self::Output {
                        paste::paste! {
                            let a = self.[<to_ #res_type>]();
                            let min = min.[<to_ #res_type>]();
                            let max = max.[<to_ #res_type>]();
                            if a < min { min } else if a > max { max } else { a }
                        }
                    }
                    #abs_method
                    #ceil_method
                    #floor_method
                    #sign_method
                    #cmp_method
                    #round_method
                }
            };
            // ret.extend(res);
        }
    }

    ret.into()
}

fn impl_unreachable(lhs_dtype: SimdType, rhs_dtype: SimdType, res_type: SimdType) -> TokenStream2 {
    quote! {
        impl NormalOut<#rhs_dtype> for #lhs_dtype {
            type Output = #res_type;
            fn _add(self, rhs: #rhs_dtype) -> Self::Output {
                unreachable!()
            }
            fn _sub(self, rhs: #rhs_dtype) -> Self::Output {
                unreachable!()
            }
            fn _mul(self, rhs: #rhs_dtype) -> Self::Output {
                unreachable!()
            }
            fn _pow(self, rhs: #rhs_dtype) -> Self::Output {
                unreachable!()
            }
            fn _rem(self, rhs: #rhs_dtype) -> Self::Output {
                unreachable!()
            }
            fn _square(self) -> Self::Output {
                unreachable!()
            }
            fn _clip(self, min: Self::Output, max: Self::Output) -> Self::Output {
                unreachable!()
            }
            fn _abs(self) -> Self {
                unreachable!()
            }
            fn _ceil(self) -> Self::Output {
                unreachable!()
            }
            fn _floor(self) -> Self::Output {
                unreachable!()
            }
            fn _sign(self) -> Self::Output {
                unreachable!()
            }
            fn _max(self, rhs: #rhs_dtype) -> Self::Output {
                unreachable!()
            }
            fn _min(self, rhs: #rhs_dtype) -> Self::Output {
                unreachable!()
            }
            fn _round(self) -> Self::Output {
                unreachable!()
            }
        }
    }
}
