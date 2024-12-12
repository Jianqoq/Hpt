use crate::type_utils::{type_simd_is_arr, type_simd_lanes, SimdType, Type, TypeInfo};
use crate::TokenStream2;
use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;

pub(crate) fn impl_simd_normal_out_unary() -> TokenStream {
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

    for (lhs_ty, lhs_lanes, lhs) in types.iter() {
        let lhs_lanes = *lhs_lanes;
        let lhs_type = TypeInfo::new(&lhs_ty.to_lowercase());
        let lhs_dtype = lhs_type.dtype;
        let res_type = lhs_type.clone().dtype;
        let lhs_simd: SimdType = (*lhs).into();
        let res_simd_ty = Ident::new(
            &format!(
                "{}x{}",
                if res_type.is_cplx() {
                    match res_type.to_string().as_str() {
                        "complex32" => "cplx32".to_string(),
                        "complex64" => "cplx64".to_string(),
                        _ => unreachable!(),
                    }
                } else {
                    res_type.to_string()
                },
                type_simd_lanes(&res_type.to_string())
            ),
            proc_macro2::Span::call_site(),
        );

        let neg_method = if lhs_dtype.is_float() {
            if lhs_dtype.is_f32() || lhs_dtype.is_f64() {
                quote! {
                    fn _neg(self) -> Self {
                        SimdMath::neg(self)
                    }
                }
            } else {
                array_cal_single(
                    lhs,
                    lhs_dtype,
                    lhs_lanes,
                    lhs_simd,
                    Ident::new("_neg", proc_macro2::Span::call_site()),
                )
            }
        } else {
            if !type_simd_is_arr(lhs) && lhs_type.is_signed {
                quote! {
                    fn _neg(self) -> Self {
                        SimdMath::neg(self)
                    }
                }
            } else if !type_simd_is_arr(lhs) && !lhs_type.is_signed {
                quote! {
                    fn _neg(self) -> Self {
                        Self(self.wrapping_neg())
                    }
                }
            } else {
                if lhs_type.is_signed {
                    quote! {
                        fn _neg(self) -> Self {
                            let mut arr = [#lhs_dtype::ZERO; #lhs_lanes as usize];
                            for i in 0..#lhs_lanes as usize {
                                arr[i] = self.0[i].neg();
                            }
                            #res_simd_ty::#res_simd_ty(arr.into())
                        }
                    }
                } else {
                    quote! {
                        fn _neg(self) -> Self {
                            self
                        }
                    }
                }
            }
        };

        let abs_method = if lhs_dtype.is_cplx() {
            quote! {
                fn _abs(self) -> Self {
                    panic!("abs not supported for complex numbers")
                }
            }
        } else if lhs_dtype.is_float() {
            if lhs_dtype.is_f32() || lhs_dtype.is_f64() {
                quote! {
                    fn _abs(self) -> Self {
                        SimdMath::abs(self)
                    }
                }
            } else {
                array_cal_single(
                    lhs,
                    lhs_dtype,
                    lhs_lanes,
                    lhs_simd.clone(),
                    Ident::new("_abs", proc_macro2::Span::call_site()),
                )
            }
        } else {
            if !type_simd_is_arr(lhs) && lhs_type.is_signed {
                quote! {
                    fn _abs(self) -> Self {
                        SimdMath::abs(self)
                    }
                }
            } else if !type_simd_is_arr(lhs) && !lhs_type.is_signed {
                quote! {
                    fn _abs(self) -> Self {
                        self
                    }
                }
            } else {
                if lhs_type.is_signed {
                    quote! {
                        fn _abs(self) -> Self {
                            let mut arr = [#lhs_dtype::ZERO; #lhs_lanes as usize];
                            for i in 0..#lhs_lanes as usize {
                                arr[i] = self.0[i].abs();
                            }
                            #lhs_simd(arr.into())
                        }
                    }
                } else {
                    quote! {
                        fn _abs(self) -> Self {
                            self
                        }
                    }
                }
            }
        };

        let unary_no_change_ty_method = if lhs_dtype.is_float() {
            if lhs_dtype.is_f32() || lhs_dtype.is_f64() {
                quote! {
                    fn _floor(self) -> Self {
                        SimdMath::floor(self)
                    }
                    fn _round(self) -> Self {
                        SimdMath::round(self)
                    }
                    fn _ceil(self) -> Self {
                        SimdMath::ceil(self)
                    }
                    fn _square(self) -> Self {
                        SimdMath::square(self)
                    }
                }
            } else {
                let floor = array_cal_single(
                    lhs,
                    lhs_dtype,
                    lhs_lanes,
                    lhs_simd.clone(),
                    Ident::new("_floor", proc_macro2::Span::call_site()),
                );
                let round = array_cal_single(
                    lhs,
                    lhs_dtype,
                    lhs_lanes,
                    lhs_simd.clone(),
                    Ident::new("_round", proc_macro2::Span::call_site()),
                );
                let ceil = array_cal_single(
                    lhs,
                    lhs_dtype,
                    lhs_lanes,
                    lhs_simd.clone(),
                    Ident::new("_ceil", proc_macro2::Span::call_site()),
                );
                let square = array_cal_single(
                    lhs,
                    lhs_dtype,
                    lhs_lanes,
                    lhs_simd.clone(),
                    Ident::new("_square", proc_macro2::Span::call_site()),
                );
                quote! {
                    #floor
                    #round
                    #ceil
                    #square
                }
            }
        } else {
            quote! {
                fn _floor(self) -> Self {
                    self
                }
                fn _round(self) -> Self {
                    self
                }
                fn _ceil(self) -> Self {
                    self
                }
                fn _square(self) -> Self {
                    self * self
                }
            }
        };

        let sign_method = if res_type.is_float() {
            quote! {
                fn _sign(self) -> Self {
                    todo!()
                }
            }
        } else if res_type.is_unsigned() {
            quote! {
                fn _sign(self) -> Self {
                    todo!()
                }
            }
        } else {
            quote! {
                fn _sign(self) -> Self {
                    todo!()
                }
            }
        };

        let relu = quote! {
            fn _relu(self) -> Self {
                self._max(Self::splat(Self::Base::ZERO))
            }
        };

        let relu6 = quote! {
            fn _relu6(self) -> Self {
                self._max(Self::splat(Self::Base::ZERO))._min(Self::splat(Self::Base::SIX))
            }
        };

        let leaky_relu = quote! {
            fn _leaky_relu(self, alpha: Self::Base) -> Self {
                let zero = Self::splat(Self::Base::ZERO);
                self._max(zero)._add(Self::splat(alpha)._mul(self._min(zero)))
            }
        };

        let res = quote! {
            impl NormalOutUnary for #lhs_simd {
                type Base = #lhs_dtype;
                #neg_method
                #abs_method
                #unary_no_change_ty_method
                #sign_method
                #relu
                #relu6
                #leaky_relu
            }
        };
        ret.extend(res);
    }

    ret.into()
}

fn array_cal_single(
    lhs: &str,
    res_type: Type,
    lhs_lanes: u8,
    res_simd_ty: SimdType,
    method: Ident,
) -> TokenStream2 {
    match type_simd_is_arr(lhs) {
        true => {
            quote! {
                fn #method(self) -> Self {
                    let lhs_arr = self.0;
                    let mut arr = [#res_type::ZERO; #lhs_lanes as usize];
                    for i in 0..#lhs_lanes as usize {
                        arr[i] = lhs_arr[i].#method();
                    }
                    #res_simd_ty(arr.into())
                }
            }
        }
        false => {
            quote! {
                fn #method(self) -> Self {
                    let lhs_arr = self.0.as_array();
                    let mut arr = [#res_type::ZERO; #lhs_lanes as usize];
                    for i in 0..#lhs_lanes as usize {
                        arr[i] = lhs_arr[i].#method();
                    }
                    #res_simd_ty(arr.into())
                }
            }
        }
    }
}
