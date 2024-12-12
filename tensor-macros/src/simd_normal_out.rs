use crate::type_utils::{ type_simd_is_arr, type_simd_lanes, SimdType, Type, TypeInfo };
use crate::TokenStream2;
use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;

pub fn impl_simd_normal_out() -> TokenStream {
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

            let mul_add_method = if res_type.is_float() {
                quote! {
                    fn _mul_add(self, a: #rhs_simd, b: #rhs_simd) -> Self::Output {
                        self.#to_res_type()._mul_add(a.#to_res_type(), b.#to_res_type())
                    }
                }
            } else if res_type.is_bool() {
                quote! {
                    fn _mul_add(self, a: #rhs_simd, b: #rhs_simd) -> Self::Output {
                        self.#to_res_type() & (a.#to_res_type() | b.#to_res_type())
                    }
                }
            } else if !type_simd_is_arr(lhs) && !type_simd_is_arr(rhs) {
                quote! {
                    fn _mul_add(self, a: #rhs_simd, b: #rhs_simd) -> Self::Output {
                            self.#to_res_type() * a.#to_res_type() + b.#to_res_type()
                        }
                }
            } else {
                quote! {
                    fn _mul_add(self, a: #rhs_simd, b: #rhs_simd) -> Self::Output {
                        let lhs_arr = self.0;
                        let a_arr = a.0;
                        let b_arr = b.0;
                        let mut arr = [#res_type::ZERO; #lhs_lanes as usize];
                        for i in 0..#lhs_lanes as usize {
                            arr[i] = lhs_arr[i].#to_res_type() * a_arr[i].#to_res_type() + b_arr[i].#to_res_type();
                        }
                        #res_simd_ty::#res_simd_ty(arr.into())
                    }
                }
            };
            let pow_method = if res_type.is_float() {
                if res_type.is_f32() {
                    quote! {
                        fn _pow(self, rhs: #rhs_simd) -> Self::Output {
                            SimdMath::pow(self.to_f32(), rhs.to_f32())
                        }
                        fn _max(self, rhs: #rhs_simd) -> Self::Output {
                            SimdMath::max(self.to_f32(), rhs.to_f32())
                        }
                        fn _min(self, rhs: #rhs_simd) -> Self::Output {
                            SimdMath::min(self.to_f32(), rhs.to_f32())
                        }
                    }
                } else if res_type.is_f64() {
                    quote! {
                        fn _pow(self, rhs: #rhs_simd) -> Self::Output {
                            SimdMath::pow(self.to_f64(), rhs.to_f64())
                        }
                        fn _max(self, rhs: #rhs_simd) -> Self::Output {
                            SimdMath::max(self.to_f64(), rhs.to_f64())
                        }
                        fn _min(self, rhs: #rhs_simd) -> Self::Output {
                            SimdMath::min(self.to_f64(), rhs.to_f64())
                        }
                    }
                } else {
                    let pow = array_cal(
                        lhs,
                        rhs,
                        res_type,
                        lhs_lanes,
                        rhs_simd,
                        res_simd_ty.clone(),
                        Ident::new("_pow", proc_macro2::Span::call_site())
                    );
                    let max = array_cal(
                        lhs,
                        rhs,
                        res_type,
                        lhs_lanes,
                        rhs_simd,
                        res_simd_ty.clone(),
                        Ident::new("_max", proc_macro2::Span::call_site())
                    );
                    let min = array_cal(
                        lhs,
                        rhs,
                        res_type,
                        lhs_lanes,
                        rhs_simd,
                        res_simd_ty.clone(),
                        Ident::new("_min", proc_macro2::Span::call_site())
                    );
                    quote! {
                        #pow
                        #max
                        #min
                    }
                }
            } else {
                let pow = array_cal(
                    lhs,
                    rhs,
                    res_type,
                    lhs_lanes,
                    rhs_simd,
                    res_simd_ty.clone(),
                    Ident::new("_pow", proc_macro2::Span::call_site())
                );
                let b2 = if !type_simd_is_arr(rhs) && !type_simd_is_arr(lhs) {
                    quote! {
                        fn _max(self, rhs: #rhs_simd) -> Self::Output {
                            SimdMath::max(self.#to_res_type(), rhs.#to_res_type())
                        }
                        fn _min(self, rhs: #rhs_simd) -> Self::Output {
                            SimdMath::min(self.#to_res_type(), rhs.#to_res_type())
                        }
                    }
                } else {
                    let max = array_cal(
                        lhs,
                        rhs,
                        res_type,
                        lhs_lanes,
                        rhs_simd,
                        res_simd_ty.clone(),
                        Ident::new("_max", proc_macro2::Span::call_site())
                    );
                    let min = array_cal(
                        lhs,
                        rhs,
                        res_type,
                        lhs_lanes,
                        rhs_simd,
                        res_simd_ty.clone(),
                        Ident::new("_min", proc_macro2::Span::call_site())
                    );
                    quote! {
                        #max
                        #min
                    }
                };
                quote! {
                    #pow
                    #b2
                }
            };

            let res =
                quote! {
                impl NormalOut<#rhs_simd> for #lhs_simd {
                    type Output = #res_simd_ty::#res_simd_ty;
                    fn _add(self, rhs: #rhs_simd) -> Self::Output {
                        self.#to_res_type() + rhs.#to_res_type()
                    }
                    fn _sub(self, rhs: #rhs_simd) -> Self::Output {
                        self.#to_res_type() - rhs.#to_res_type()
                    }
                    fn _mul(self, rhs: #rhs_simd) -> Self::Output {
                        self.#to_res_type() * rhs.#to_res_type()
                    }
                    #pow_method
                    fn _rem(self, rhs: #rhs_simd) -> Self::Output {
                        self.#to_res_type() % rhs.#to_res_type()
                    }
                    fn _clip(self, min: #rhs_simd, max: #rhs_simd) -> Self::Output {
                        self._max(min.#to_res_type())._min(max.#to_res_type())
                    }
                    #mul_add_method
                }
            };
            ret.extend(res);
        }
    }

    ret.into()
}
fn impl_unreachable(lhs_dtype: SimdType, rhs_simd: SimdType, res_type: SimdType) -> TokenStream2 {
    quote! {
        impl NormalOut<#rhs_simd> for #lhs_dtype {
            type Output = #res_type;
            fn _mul_add(self, a: #rhs_simd, b: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _add(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _sub(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _mul(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _pow(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _rem(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _clip(self, min: #rhs_simd, max: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _max(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _min(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
        }
    }
}
fn array_cal(
    lhs: &str,
    rhs: &str,
    res_type: Type,
    lhs_lanes: u8,
    rhs_simd: SimdType,
    res_simd_ty: Ident,
    method: Ident
) -> TokenStream2 {
    match (type_simd_is_arr(lhs), type_simd_is_arr(rhs)) {
        (true, true) => {
            quote! {
                fn #method(self, rhs: #rhs_simd) -> Self::Output {
                    let lhs_arr = self.0;
                    let rhs_arr = rhs.0;
                    let mut arr = [#res_type::ZERO; #lhs_lanes as usize];
                    for i in 0..#lhs_lanes as usize {
                        arr[i] = lhs_arr[i].#method(rhs_arr[i]);
                    }
                    #res_simd_ty::#res_simd_ty(arr.into())
                }
            }
        }
        (true, false) => {
            quote! {
                fn #method(self, rhs: #rhs_simd) -> Self::Output {
                    let lhs_arr = self.0;
                    let rhs_arr = rhs.0.as_array();
                    let mut arr = [#res_type::ZERO; #lhs_lanes as usize];
                    for i in 0..#lhs_lanes as usize {
                        arr[i] = lhs_arr[i].#method(rhs_arr[i]);
                    }
                    #res_simd_ty::#res_simd_ty(arr.into())
                }
            }
        }
        (false, true) => {
            quote! {
                fn #method(self, rhs: #rhs_simd) -> Self::Output {
                    let lhs_arr = self.0.as_array();
                    let rhs_arr = rhs.0;
                    let mut arr = [#res_type::ZERO; #lhs_lanes as usize];
                    for i in 0..#lhs_lanes as usize {
                        arr[i] = lhs_arr[i].#method(rhs_arr[i]);
                    }
                    #res_simd_ty::#res_simd_ty(arr.into())
                }
            }
        }
        (false, false) => {
            quote! {
                fn #method(self, rhs: #rhs_simd) -> Self::Output {
                    let lhs_arr = self.0.as_array();
                    let rhs_arr = rhs.0.as_array();
                    let mut arr = [#res_type::ZERO; #lhs_lanes as usize];
                    for i in 0..#lhs_lanes as usize {
                        arr[i] = lhs_arr[i].#method(rhs_arr[i]);
                    }
                    #res_simd_ty::#res_simd_ty(arr.into())
                }
            }
        }
    }
}

fn impl_unreachable_with_rhs_scalar(
    lhs_dtype: SimdType,
    rhs_scalar_ty: Type,
    res_type: SimdType
) -> TokenStream2 {
    quote! {
        impl NormalOut<#rhs_scalar_ty> for #lhs_dtype {
            type Output = #res_type;
            fn _mul_add(self, a: #rhs_scalar_ty, b: #rhs_scalar_ty) -> Self::Output {
                unreachable!()
            }
            fn _add(self, rhs: #rhs_scalar_ty) -> Self::Output {
                unreachable!()
            }
            fn _sub(self, rhs: #rhs_scalar_ty) -> Self::Output {
                unreachable!()
            }
            fn _mul(self, rhs: #rhs_scalar_ty) -> Self::Output {
                unreachable!()
            }
            fn _pow(self, rhs: #rhs_scalar_ty) -> Self::Output {
                unreachable!()
            }
            fn _rem(self, rhs: #rhs_scalar_ty) -> Self::Output {
                unreachable!()
            }
            fn _clip(self, min: #rhs_scalar_ty, max: #rhs_scalar_ty) -> Self::Output {
                unreachable!()
            }
            fn _max(self, rhs: #rhs_scalar_ty) -> Self::Output {
                unreachable!()
            }
            fn _min(self, rhs: #rhs_scalar_ty) -> Self::Output {
                unreachable!()
            }
        }
    }
}

pub fn impl_simd_normal_out_with_rhs_scalar() -> TokenStream {
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
            let rhs_dtype = rhs_type.dtype;
            let res_type = lhs_type.infer_normal_res_type(&rhs_type);
            let res_lanes = type_simd_lanes(&res_type.to_string());
            if lhs_lanes != rhs_lanes || lhs_lanes != res_lanes || rhs_lanes != res_lanes {
                ret.extend(
                    impl_unreachable_with_rhs_scalar(
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

            let mul_add_method = if res_type.is_float() {
                quote! {
                    fn _mul_add(self, a: #rhs_dtype, b: #rhs_dtype) -> Self::Output {
                        self.#to_res_type()._mul_add(#res_simd_ty::#res_simd_ty::splat(a.#to_res_type()), #res_simd_ty::#res_simd_ty::splat(b.#to_res_type()))
                    }
                }
            } else if res_type.is_bool() {
                quote! {
                    fn _mul_add(self, a: #rhs_dtype, b: #rhs_dtype) -> Self::Output {
                        self.#to_res_type() | (#res_simd_ty::#res_simd_ty::splat(a.#to_res_type()) & #res_simd_ty::#res_simd_ty::splat(b.#to_res_type()))
                    }
                }
            } else if !type_simd_is_arr(lhs) && !type_simd_is_arr(rhs) {
                quote! {
                    fn _mul_add(self, a: #rhs_dtype, b: #rhs_dtype) -> Self::Output {
                        self.#to_res_type() + #res_simd_ty::#res_simd_ty::splat(a.#to_res_type() * b.#to_res_type())
                    }
                }
            } else {
                quote! {
                    fn _mul_add(self, a: #rhs_dtype, b: #rhs_dtype) -> Self::Output {
                        let lhs_arr = self.0;
                        let mut arr = [#res_type::ZERO; #lhs_lanes as usize];
                        for i in 0..#lhs_lanes as usize {
                            arr[i] = lhs_arr[i].#to_res_type() * a.#to_res_type() + b.#to_res_type();
                        }
                        #res_simd_ty::#res_simd_ty(arr.into())
                    }
                }
            };
            let pow_method = if res_type.is_float() {
                if res_type.is_f32() {
                    quote! {
                        fn _pow(self, rhs: #rhs_dtype) -> Self::Output {
                            SimdMath::pow(self.to_f32(), #res_simd_ty::#res_simd_ty::splat(rhs.to_f32()))
                        }
                        fn _max(self, rhs: #rhs_dtype) -> Self::Output {
                            SimdMath::max(self.to_f32(), #res_simd_ty::#res_simd_ty::splat(rhs.to_f32()))
                        }
                        fn _min(self, rhs: #rhs_dtype) -> Self::Output {
                            SimdMath::min(self.to_f32(), #res_simd_ty::#res_simd_ty::splat(rhs.to_f32()))
                        }
                    }
                } else if res_type.is_f64() {
                    quote! {
                        fn _pow(self, rhs: #rhs_dtype) -> Self::Output {
                            SimdMath::pow(self.to_f64(), #res_simd_ty::#res_simd_ty::splat(rhs.to_f64()))
                        }
                        fn _max(self, rhs: #rhs_dtype) -> Self::Output {
                            SimdMath::max(self.to_f64(), #res_simd_ty::#res_simd_ty::splat(rhs.to_f64()))
                        }
                        fn _min(self, rhs: #rhs_dtype) -> Self::Output {
                            SimdMath::min(self.to_f64(), #res_simd_ty::#res_simd_ty::splat(rhs.to_f64()))
                        }
                    }
                } else {
                    let pow = array_cal_rhs_scalar(
                        lhs,
                        rhs,
                        res_type,
                        lhs_lanes,
                        rhs_dtype,
                        res_simd_ty.clone(),
                        Ident::new("_pow", proc_macro2::Span::call_site())
                    );
                    let max = array_cal_rhs_scalar(
                        lhs,
                        rhs,
                        res_type,
                        lhs_lanes,
                        rhs_dtype,
                        res_simd_ty.clone(),
                        Ident::new("_max", proc_macro2::Span::call_site())
                    );
                    let min = array_cal_rhs_scalar(
                        lhs,
                        rhs,
                        res_type,
                        lhs_lanes,
                        rhs_dtype,
                        res_simd_ty.clone(),
                        Ident::new("_min", proc_macro2::Span::call_site())
                    );
                    quote! {
                        #pow
                        #max
                        #min
                    }
                }
            } else {
                let pow = array_cal_rhs_scalar(
                    lhs,
                    rhs,
                    res_type,
                    lhs_lanes,
                    rhs_dtype,
                    res_simd_ty.clone(),
                    Ident::new("_pow", proc_macro2::Span::call_site())
                );
                let b2 = if !type_simd_is_arr(rhs) && !type_simd_is_arr(lhs) {
                    quote! {
                        fn _max(self, rhs: #rhs_dtype) -> Self::Output {
                            SimdMath::max(self.#to_res_type(), #res_simd_ty::#res_simd_ty::splat(rhs.#to_res_type()))
                        }
                        fn _min(self, rhs: #rhs_dtype) -> Self::Output {
                            SimdMath::min(self.#to_res_type(), #res_simd_ty::#res_simd_ty::splat(rhs.#to_res_type()))
                        }
                    }
                } else {
                    let max = array_cal_rhs_scalar(
                        lhs,
                        rhs,
                        res_type,
                        lhs_lanes,
                        rhs_dtype,
                        res_simd_ty.clone(),
                        Ident::new("_max", proc_macro2::Span::call_site())
                    );
                    let min = array_cal_rhs_scalar(
                        lhs,
                        rhs,
                        res_type,
                        lhs_lanes,
                        rhs_dtype,
                        res_simd_ty.clone(),
                        Ident::new("_min", proc_macro2::Span::call_site())
                    );
                    quote! {
                        #max
                        #min
                    }
                };
                quote! {
                    #pow
                    #b2
                }
            };

            let res =
                quote! {
                impl NormalOut<#rhs_dtype> for #lhs_simd {
                    type Output = #res_simd_ty::#res_simd_ty;
                    fn _add(self, rhs: #rhs_dtype) -> Self::Output {
                        self.#to_res_type() + #res_simd_ty::#res_simd_ty::splat(rhs.#to_res_type())
                    }
                    fn _sub(self, rhs: #rhs_dtype) -> Self::Output {
                        self.#to_res_type() - #res_simd_ty::#res_simd_ty::splat(rhs.#to_res_type())
                    }
                    fn _mul(self, rhs: #rhs_dtype) -> Self::Output {
                        self.#to_res_type() * #res_simd_ty::#res_simd_ty::splat(rhs.#to_res_type())
                    }
                    #pow_method
                    fn _rem(self, rhs: #rhs_dtype) -> Self::Output {
                        self.#to_res_type() % #res_simd_ty::#res_simd_ty::splat(rhs.#to_res_type())
                    }
                    fn _clip(self, min: #rhs_dtype, max: #rhs_dtype) -> Self::Output {
                        self._max(#res_simd_ty::#res_simd_ty::splat(min.#to_res_type()))._min(#res_simd_ty::#res_simd_ty::splat(max.#to_res_type()))
                    }
                    #mul_add_method
                }
            };
            ret.extend(res);
        }
    }

    ret.into()
}

fn array_cal_rhs_scalar(
    lhs: &str,
    rhs: &str,
    res_type: Type,
    lhs_lanes: u8,
    rhs_scalar_ty: Type,
    res_simd_ty: Ident,
    method: Ident
) -> TokenStream2 {
    match (type_simd_is_arr(lhs), type_simd_is_arr(rhs)) {
        (true, true) => {
            quote! {
                fn #method(self, rhs: #rhs_scalar_ty) -> Self::Output {
                    let lhs_arr = self.0;
                    let mut arr = [#res_type::ZERO; #lhs_lanes as usize];
                    for i in 0..#lhs_lanes as usize {
                        arr[i] = lhs_arr[i].#method(rhs);
                    }
                    #res_simd_ty::#res_simd_ty(arr.into())
                }
            }
        }
        (true, false) => {
            quote! {
                fn #method(self, rhs: #rhs_scalar_ty) -> Self::Output {
                    let lhs_arr = self.0;
                    let mut arr = [#res_type::ZERO; #lhs_lanes as usize];
                    for i in 0..#lhs_lanes as usize {
                        arr[i] = lhs_arr[i].#method(rhs);
                    }
                    #res_simd_ty::#res_simd_ty(arr.into())
                }
            }
        }
        (false, true) => {
            quote! {
                fn #method(self, rhs: #rhs_scalar_ty) -> Self::Output {
                    let lhs_arr = self.0.as_array();
                    let mut arr = [#res_type::ZERO; #lhs_lanes as usize];
                    for i in 0..#lhs_lanes as usize {
                        arr[i] = lhs_arr[i].#method(rhs);
                    }
                    #res_simd_ty::#res_simd_ty(arr.into())
                }
            }
        }
        (false, false) => {
            quote! {
                fn #method(self, rhs: #rhs_scalar_ty) -> Self::Output {
                    let lhs_arr = self.0.as_array();
                    let mut arr = [#res_type::ZERO; #lhs_lanes as usize];
                    for i in 0..#lhs_lanes as usize {
                        arr[i] = lhs_arr[i].#method(rhs);
                    }
                    #res_simd_ty::#res_simd_ty(arr.into())
                }
            }
        }
    }
}


pub fn impl_simd_normal_out_with_lhs_scalar() -> TokenStream {
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

            let mul_add_method = if res_type.is_float() {
                quote! {
                    fn _mul_add(self, a: #rhs_simd, b: #rhs_simd) -> Self::Output {
                        #res_simd_ty::#res_simd_ty::splat(self.#to_res_type())._mul_add(a.#to_res_type(), b.#to_res_type())
                    }
                }
            } else if res_type.is_bool() {
                quote! {
                    fn _mul_add(self, a: #rhs_simd, b: #rhs_simd) -> Self::Output {
                        #res_simd_ty::#res_simd_ty::splat(self.#to_res_type()) | (a.#to_res_type() & b.#to_res_type())
                    }
                }
            } else if !type_simd_is_arr(lhs) && !type_simd_is_arr(rhs) {
                quote! {
                    fn _mul_add(self, a: #rhs_simd, b: #rhs_simd) -> Self::Output {
                        #res_simd_ty::#res_simd_ty::splat(self.#to_res_type()) + a.#to_res_type() * b.#to_res_type()
                        }

                }
            } else {
                quote! {
                    fn _mul_add(self, a: #rhs_simd, b: #rhs_simd) -> Self::Output {
                        let a_arr = a.0;
                        let b_arr = b.0;
                        let mut arr = [#res_type::ZERO; #lhs_lanes as usize];
                        for i in 0..#lhs_lanes as usize {
                            arr[i] = self.#to_res_type() * a_arr[i].#to_res_type() + b_arr[i].#to_res_type();
                        }
                        #res_simd_ty::#res_simd_ty(arr.into())
                    }
                }
            };
            let pow_method = if res_type.is_float() {
                if res_type.is_f32() {
                    quote! {
                        fn _pow(self, rhs: #rhs_simd) -> Self::Output {
                            SimdMath::pow(#res_simd_ty::#res_simd_ty::splat(self.to_f32()), rhs.to_f32())
                        }
                        fn _max(self, rhs: #rhs_simd) -> Self::Output {
                            SimdMath::max(#res_simd_ty::#res_simd_ty::splat(self.to_f32()), rhs.to_f32())
                        }
                        fn _min(self, rhs: #rhs_simd) -> Self::Output {
                            SimdMath::min(#res_simd_ty::#res_simd_ty::splat(self.to_f32()), rhs.to_f32())
                        }
                    }
                } else if res_type.is_f64() {
                    quote! {
                        fn _pow(self, rhs: #rhs_simd) -> Self::Output {
                            SimdMath::pow(#res_simd_ty::#res_simd_ty::splat(self.to_f64()), rhs.to_f64())
                        }
                        fn _max(self, rhs: #rhs_simd) -> Self::Output {
                            SimdMath::max(#res_simd_ty::#res_simd_ty::splat(self.to_f64()), rhs.to_f64())
                        }
                        fn _min(self, rhs: #rhs_simd) -> Self::Output {
                            SimdMath::min(#res_simd_ty::#res_simd_ty::splat(self.to_f64()), rhs.to_f64())
                        }
                    }
                } else {
                    let pow = array_cal_lhs_scalar(
                        lhs,
                        rhs,
                        res_type,
                        lhs_lanes,
                        rhs_simd,
                        res_simd_ty.clone(),
                        Ident::new("_pow", proc_macro2::Span::call_site())
                    );
                    let max = array_cal_lhs_scalar(
                        lhs,
                        rhs,
                        res_type,
                        lhs_lanes,
                        rhs_simd,
                        res_simd_ty.clone(),
                        Ident::new("_max", proc_macro2::Span::call_site())
                    );
                    let min = array_cal_lhs_scalar(
                        lhs,
                        rhs,
                        res_type,
                        lhs_lanes,
                        rhs_simd,
                        res_simd_ty.clone(),
                        Ident::new("_min", proc_macro2::Span::call_site())
                    );
                    quote! {
                        #pow
                        #max
                        #min
                    }
                }
            } else {
                let pow = array_cal_lhs_scalar(
                    lhs,
                    rhs,
                    res_type,
                    lhs_lanes,
                    rhs_simd,
                    res_simd_ty.clone(),
                    Ident::new("_pow", proc_macro2::Span::call_site())
                );
                let b2 = if !type_simd_is_arr(rhs) && !type_simd_is_arr(lhs) {
                    quote! {
                        fn _max(self, rhs: #rhs_simd) -> Self::Output {
                            SimdMath::max(#res_simd_ty::#res_simd_ty::splat(self.#to_res_type()), rhs.#to_res_type())
                        }
                        fn _min(self, rhs: #rhs_simd) -> Self::Output {
                            SimdMath::min(#res_simd_ty::#res_simd_ty::splat(self.#to_res_type()), rhs.#to_res_type())
                        }
                    }
                } else {
                    let max = array_cal_lhs_scalar(
                        lhs,
                        rhs,
                        res_type,
                        lhs_lanes,
                        rhs_simd,
                        res_simd_ty.clone(),
                        Ident::new("_max", proc_macro2::Span::call_site())
                    );
                    let min = array_cal_lhs_scalar(
                        lhs,
                        rhs,
                        res_type,
                        lhs_lanes,
                        rhs_simd,
                        res_simd_ty.clone(),
                        Ident::new("_min", proc_macro2::Span::call_site())
                    );
                    quote! {
                        #max
                        #min
                    }
                };
                quote! {
                    #pow
                    #b2
                }
            };

            let res =
                quote! {
                impl NormalOut<#rhs_simd> for #lhs_dtype {
                    type Output = #res_simd_ty::#res_simd_ty;
                    fn _add(self, rhs: #rhs_simd) -> Self::Output {
                        #res_simd_ty::#res_simd_ty::splat(self.#to_res_type()) + rhs.#to_res_type()
                    }
                    fn _sub(self, rhs: #rhs_simd) -> Self::Output {
                        #res_simd_ty::#res_simd_ty::splat(self.#to_res_type()) - rhs.#to_res_type()
                    }
                    fn _mul(self, rhs: #rhs_simd) -> Self::Output {
                        #res_simd_ty::#res_simd_ty::splat(self.#to_res_type()) * rhs.#to_res_type()
                    }
                    #pow_method
                    fn _rem(self, rhs: #rhs_simd) -> Self::Output {
                        #res_simd_ty::#res_simd_ty::splat(self.#to_res_type()) % rhs.#to_res_type()
                    }
                    fn _clip(self, min: #rhs_simd, max: #rhs_simd) -> Self::Output {
                        #res_simd_ty::#res_simd_ty::splat(self.#to_res_type())._max(min.#to_res_type())._min(max.#to_res_type())
                    }
                    #mul_add_method
                }
            };
            ret.extend(res);
        }
    }

    ret.into()
}

fn impl_unreachable_lhs_scalar(lhs_dtype: Type, rhs_simd: SimdType, res_type: SimdType) -> TokenStream2 {
    quote! {
        impl NormalOut<#rhs_simd> for #lhs_dtype {
            type Output = #res_type;
            fn _mul_add(self, a: #rhs_simd, b: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _add(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _sub(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _mul(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _pow(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _rem(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _clip(self, min: #rhs_simd, max: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _max(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _min(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
        }
    }
}

fn array_cal_lhs_scalar(
    lhs: &str,
    rhs: &str,
    res_type: Type,
    lhs_lanes: u8,
    rhs_simd: SimdType,
    res_simd_ty: Ident,
    method: Ident
) -> TokenStream2 {
    match (type_simd_is_arr(lhs), type_simd_is_arr(rhs)) {
        (true, true) => {
            quote! {
                fn #method(self, rhs: #rhs_simd) -> Self::Output {
                    let rhs_arr = rhs.0;
                    let mut arr = [#res_type::ZERO; #lhs_lanes as usize];
                    for i in 0..#lhs_lanes as usize {
                        arr[i] = self.#method(rhs_arr[i]);
                    }
                    #res_simd_ty::#res_simd_ty(arr.into())
                }
            }
        }
        (true, false) => {
            quote! {
                fn #method(self, rhs: #rhs_simd) -> Self::Output {
                    let rhs_arr = rhs.0.as_array();
                    let mut arr = [#res_type::ZERO; #lhs_lanes as usize];
                    for i in 0..#lhs_lanes as usize {
                        arr[i] = self.#method(rhs_arr[i]);
                    }
                    #res_simd_ty::#res_simd_ty(arr.into())
                }
            }
        }
        (false, true) => {
            quote! {
                fn #method(self, rhs: #rhs_simd) -> Self::Output {
                    let rhs_arr = rhs.0;
                    let mut arr = [#res_type::ZERO; #lhs_lanes as usize];
                    for i in 0..#lhs_lanes as usize {
                        arr[i] = self.#method(rhs_arr[i]);
                    }
                    #res_simd_ty::#res_simd_ty(arr.into())
                }
            }
        }
        (false, false) => {
            quote! {
                fn #method(self, rhs: #rhs_simd) -> Self::Output {
                    let rhs_arr = rhs.0.as_array();
                    let mut arr = [#res_type::ZERO; #lhs_lanes as usize];
                    for i in 0..#lhs_lanes as usize {
                        arr[i] = self.#method(rhs_arr[i]);
                    }
                    #res_simd_ty::#res_simd_ty(arr.into())
                }
            }
        }
    }
}