use crate::type_utils::{ type_simd_lanes, SimdType, Type, TypeInfo };
use crate::TokenStream2;
use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;

pub fn impl_simd_binary_out_float() -> TokenStream {
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
        for (rhs_ty, rhs_lanes, rhs) in types.iter() {
            let lhs_lanes = *lhs_lanes;
            let rhs_lanes = *rhs_lanes;
            let lhs_type = TypeInfo::new(&lhs_ty.to_lowercase());
            let rhs_type = TypeInfo::new(&rhs_ty.to_lowercase());
            let res_type = lhs_type.infer_normal_res_type(&rhs_type);
            let res_lanes = type_simd_lanes(&res_type.to_string());
            if lhs_lanes != rhs_lanes || lhs_lanes != res_lanes || rhs_lanes != res_lanes {
                ret.extend(
                    impl_unreachable(
                        (*lhs).into(),
                        (*rhs).into(),
                        res_type.to_string().as_str().into()
                    )
                );
                continue;
            }
            let lhs_simd: SimdType = (*lhs).into();
            let rhs_simd: SimdType = (*rhs).into();
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
                proc_macro2::Span::call_site()
            );
            let to_res_type = Ident::new(
                &format!("to_{}", res_type.to_string().to_lowercase()),
                proc_macro2::Span::call_site()
            );

            let res =
                quote! {
                impl FloatOutBinary<#rhs_simd> for #lhs_simd {
                    type Output = #res_simd_ty::#res_simd_ty;
                    fn _div(self, rhs: #rhs_simd) -> Self::Output {
                        self.#to_res_type() / rhs.#to_res_type()
                    }
                    fn _log(self, base: #rhs_simd) -> Self::Output {
                        todo!()
                    }
                }
            };
            ret.extend(res);
        }
    }

    ret.into()
}

fn impl_unreachable(lhs_dtype: SimdType, rhs_simd: SimdType, res_type: SimdType) -> TokenStream2 {
    quote! {
        impl FloatOutBinary<#rhs_simd> for #lhs_dtype {
            type Output = #res_type;
            fn _div(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _log(self, base: #rhs_simd) -> Self::Output {
                unreachable!()
            }
        }
    }
}

pub fn impl_simd_binary_out_float_lhs_scalar() -> TokenStream {
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

    for (lhs_ty, lhs_lanes, _) in types.iter() {
        for (rhs_ty, rhs_lanes, rhs) in types.iter() {
            let lhs_lanes = *lhs_lanes;
            let rhs_lanes = *rhs_lanes;
            let lhs_type = TypeInfo::new(&lhs_ty.to_lowercase());
            let rhs_type = TypeInfo::new(&rhs_ty.to_lowercase());
            let lhs_dtype = lhs_type.dtype;
            let res_type = lhs_type.infer_normal_res_type(&rhs_type);
            let res_lanes = type_simd_lanes(&res_type.to_string());
            if lhs_lanes != rhs_lanes || lhs_lanes != res_lanes || rhs_lanes != res_lanes {
                ret.extend(
                    impl_unreachable_lhs_scalar(
                        lhs_dtype,
                        (*rhs).into(),
                        res_type.to_string().as_str().into()
                    )
                );
                continue;
            }
            let rhs_simd: SimdType = (*rhs).into();
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
                proc_macro2::Span::call_site()
            );
            let to_res_type = Ident::new(
                &format!("to_{}", res_type.to_string().to_lowercase()),
                proc_macro2::Span::call_site()
            );

            let res =
                quote! {
                impl FloatOutBinary<#rhs_simd> for #lhs_dtype {
                    type Output = #res_simd_ty::#res_simd_ty;
                    fn _div(self, rhs: #rhs_simd) -> Self::Output {
                        #res_simd_ty::#res_simd_ty::splat(self.#to_res_type()) / rhs.#to_res_type()
                    }
                    fn _log(self, base: #rhs_simd) -> Self::Output {
                        todo!()
                    }
                }
            };
            ret.extend(res);
        }
    }

    ret.into()
}

fn impl_unreachable_lhs_scalar(
    lhs_dtype: Type,
    rhs_simd: SimdType,
    res_type: SimdType
) -> TokenStream2 {
    quote! {
        impl FloatOutBinary<#rhs_simd> for #lhs_dtype {
            type Output = #res_type;
            fn _div(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _log(self, base: #rhs_simd) -> Self::Output {
                unreachable!()
            }
        }
    }
}

pub fn impl_simd_binary_out_float_rhs_scalar() -> TokenStream {
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
        for (rhs_ty, rhs_lanes, _) in types.iter() {
            let lhs_lanes = *lhs_lanes;
            let rhs_lanes = *rhs_lanes;
            let lhs_type = TypeInfo::new(&lhs_ty.to_lowercase());
            let rhs_type = TypeInfo::new(&rhs_ty.to_lowercase());
            let rhs_dtype = rhs_type.dtype;
            let res_type = lhs_type.infer_normal_res_type(&rhs_type);
            let res_lanes = type_simd_lanes(&res_type.to_string());
            if lhs_lanes != rhs_lanes || lhs_lanes != res_lanes || rhs_lanes != res_lanes {
                ret.extend(
                    impl_unreachable_rhs_scalar(
                        (*lhs).into(),
                        rhs_dtype,
                        res_type.to_string().as_str().into()
                    )
                );
                continue;
            }
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
                proc_macro2::Span::call_site()
            );
            let to_res_type = Ident::new(
                &format!("to_{}", res_type.to_string().to_lowercase()),
                proc_macro2::Span::call_site()
            );

            let res =
                quote! {
                impl FloatOutBinary<#rhs_dtype> for #lhs_simd {
                    type Output = #res_simd_ty::#res_simd_ty;
                    fn _div(self, rhs: #rhs_dtype) -> Self::Output {
                        self.#to_res_type() / #res_simd_ty::#res_simd_ty::splat(rhs.#to_res_type())
                    }
                    fn _log(self, base: #rhs_dtype) -> Self::Output {
                        todo!()
                    }
                }
            };
            ret.extend(res);
        }
    }

    ret.into()
}

fn impl_unreachable_rhs_scalar(
    lhs_simd: SimdType,
    rhs_dtype: Type,
    res_type: SimdType
) -> TokenStream2 {
    quote! {
        impl FloatOutBinary<#rhs_dtype> for #lhs_simd {
            type Output = #res_type;
            fn _div(self, rhs: #rhs_dtype) -> Self::Output {
                unreachable!()
            }
            fn _log(self, base: #rhs_dtype) -> Self::Output {
                unreachable!()
            }
        }
    }
}
