use proc_macro::TokenStream;
use crate::type_utils::{ type_simd_is_arr, type_simd_lanes, SimdType, Type, TypeInfo };
use quote::quote;
use crate::TokenStream2;
use proc_macro2::Ident;

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
                &format!("{}x{}", res_type.to_string(), type_simd_lanes(&res_type.to_string())),
                proc_macro2::Span::call_site()
            );
            let pow_method = if res_type.is_float() {
                if res_type.is_f32() {
                    quote! {
                        fn _pow(self, rhs: #rhs_simd) -> Self::Output {
                            #res_simd_ty::#res_simd_ty(sleef::f32x::pow_u10(self.to_f32().0, rhs.to_f32().0))
                        }
                        fn _max(self, rhs: #rhs_simd) -> Self::Output {
                            #res_simd_ty::#res_simd_ty(self.to_f32().0.simd_max(rhs.to_f32().0))
                        }
                        fn _min(self, rhs: #rhs_simd) -> Self::Output {
                            #res_simd_ty::#res_simd_ty(self.to_f32().0.simd_min(rhs.to_f32().0))
                        }
                    }
                } else if res_type.is_f64() {
                    quote! {
                        fn _pow(self, rhs: #rhs_simd) -> Self::Output {
                            #res_simd_ty::#res_simd_ty(sleef::f64x::pow_u10(self.to_f64().0, rhs.to_f64().0))
                        }
                        fn _max(self, rhs: #rhs_simd) -> Self::Output {
                            #res_simd_ty::#res_simd_ty(self.to_f64().0.simd_max(rhs.to_f64().0))
                        }
                        fn _min(self, rhs: #rhs_simd) -> Self::Output {
                            #res_simd_ty::#res_simd_ty(self.to_f64().0.simd_min(rhs.to_f64().0))
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
                            paste::paste! {
                                #res_simd_ty::#res_simd_ty(self.[<to_ #res_type>]().0.simd_max(rhs.[<to_ #res_type>]().0))
                            }
                        }
                        fn _min(self, rhs: #rhs_simd) -> Self::Output {
                            paste::paste! {
                                #res_simd_ty::#res_simd_ty(self.[<to_ #res_type>]().0.simd_min(rhs.[<to_ #res_type>]().0))
                            }
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

            let abs_method = if lhs_dtype.is_float() {
                if lhs_dtype.is_f32() {
                    quote! {
                        fn _abs(self) -> Self {
                            #lhs_simd(self.to_f32().0.abs())
                        }
                    }
                } else if lhs_dtype.is_f64() {
                    quote! {
                        fn _abs(self) -> Self {
                            #lhs_simd(self.to_f64().0.abs())
                        }
                    }
                } else {
                    array_cal_single(
                        lhs,
                        lhs_dtype,
                        lhs_lanes,
                        lhs_simd.clone(),
                        Ident::new("_abs", proc_macro2::Span::call_site())
                    )
                }
            } else {
                if !type_simd_is_arr(lhs) && lhs_type.is_signed {
                    quote! {
                        fn _abs(self) -> Self {
                            #lhs_simd(self.0.abs())
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
                if lhs_dtype.is_f32() {
                    quote! {
                        fn _floor(self) -> Self {
                            #lhs_simd(self.to_f32().0.floor())
                        }
                        fn _round(self) -> Self {
                            #lhs_simd(self.to_f32().0.round())
                        }
                        fn _ceil(self) -> Self {
                            #lhs_simd(self.to_f32().0.ceil())
                        }
                        fn _square(self) -> Self {
                            #lhs_simd(self.to_f32().0 * self.to_f32().0)
                        }
                    }
                } else if lhs_dtype.is_f64() {
                    quote! {
                        fn _floor(self) -> Self {
                            #lhs_simd(self.to_f64().0.floor())
                        }
                        fn _round(self) -> Self {
                            #lhs_simd(self.to_f64().0.round())
                        }
                        fn _ceil(self) -> Self {
                            #lhs_simd(self.to_f64().0.ceil())
                        }
                        fn _square(self) -> Self {
                            #lhs_simd(self.to_f64().0 * self.to_f64().0)
                        }
                    }
                } else {
                    let floor = array_cal_single(
                        lhs,
                        lhs_dtype,
                        lhs_lanes,
                        lhs_simd.clone(),
                        Ident::new("_floor", proc_macro2::Span::call_site())
                    );
                    let round = array_cal_single(
                        lhs,
                        lhs_dtype,
                        lhs_lanes,
                        lhs_simd.clone(),
                        Ident::new("_round", proc_macro2::Span::call_site())
                    );
                    let ceil = array_cal_single(
                        lhs,
                        lhs_dtype,
                        lhs_lanes,
                        lhs_simd.clone(),
                        Ident::new("_ceil", proc_macro2::Span::call_site())
                    );
                    let square = array_cal_single(
                        lhs,
                        lhs_dtype,
                        lhs_lanes,
                        lhs_simd.clone(),
                        Ident::new("_square", proc_macro2::Span::call_site())
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
                        self
                    }
                }
            };

            let sign_method = if res_type.is_float() {
                quote! {
                    fn _sign(self) -> Self::Output {
                        todo!()
                    }
                }
            } else if res_type.is_unsigned() {
                quote! {
                    fn _sign(self) -> Self::Output {
                        todo!()
                    }
                }
            } else {
                quote! {
                    fn _sign(self) -> Self::Output {
                        todo!()
                    }
                }
            };

            let res =
                quote! {
                impl NormalOut<#rhs_simd> for #lhs_simd {
                    type Output = #res_simd_ty::#res_simd_ty;
                    fn _add(self, rhs: #rhs_simd) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]() + rhs.[<to_ #res_type>]()
                        }
                    }
                    fn _sub(self, rhs: #rhs_simd) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]() - rhs.[<to_ #res_type>]()
                        }
                    }
                    fn _mul(self, rhs: #rhs_simd) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]() * rhs.[<to_ #res_type>]()
                        }
                    }
                    #pow_method

                    fn _rem(self, rhs: #rhs_simd) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]() % rhs.[<to_ #res_type>]()
                        }
                    }
                    fn _clip(self, min: Self::Output, max: Self::Output) -> Self::Output {
                        todo!()
                    }
                    #abs_method
                    #unary_no_change_ty_method
                    #sign_method
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
            fn _square(self) -> Self {
                unreachable!()
            }
            fn _clip(self, min: Self::Output, max: Self::Output) -> Self::Output {
                unreachable!()
            }
            fn _abs(self) -> Self {
                unreachable!()
            }
            fn _ceil(self) -> Self {
                unreachable!()
            }
            fn _floor(self) -> Self {
                unreachable!()
            }
            fn _sign(self) -> Self::Output {
                unreachable!()
            }
            fn _max(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _min(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _round(self) -> Self {
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

fn array_cal_single(
    lhs: &str,
    res_type: Type,
    lhs_lanes: u8,
    res_simd_ty: SimdType,
    method: Ident
) -> TokenStream2 {
    match type_simd_is_arr(lhs) {
        true => {
            quote! {
                fn #method(self) -> Self {
                    let lhs_arr = self.0;
                    let mut arr = [#res_type::ZERO; #lhs_lanes as usize];
                    for i in 0..#lhs_lanes as usize {
                        arr[i] = <#res_type as NormalOut<#res_type>>::#method(lhs_arr[i]);
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
                        arr[i] = <#res_type as NormalOut<#res_type>>::#method(lhs_arr[i]);
                    }
                    #res_simd_ty(arr.into())
                }
            }
        }
    }
}
