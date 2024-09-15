use proc_macro::TokenStream;
use quote::quote;
use crate::type_utils::type_simd_lanes;

pub fn into_vec() -> TokenStream {
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
        (format!("cplx32x{}", type_simd_lanes("complex32")), "complex32"),
        (format!("cplx64x{}", type_simd_lanes("complex64")), "complex64"),
    ];

    for (lhs_simd_ty, lhs) in types.iter() {
        for (rhs_simd_ty, rhs) in types.iter() {
            let lhs_simd_ty = syn::Ident::new(&lhs_simd_ty, proc_macro2::Span::call_site());
            let rhs_simd_ty = syn::Ident::new(&rhs_simd_ty, proc_macro2::Span::call_site());
            let lhs_lanes = type_simd_lanes(lhs);
            let rhs_lanes = type_simd_lanes(rhs);
            let res = if lhs_lanes != rhs_lanes {
                quote! {
                    impl IntoVec<#rhs_simd_ty::#rhs_simd_ty> for #lhs_simd_ty::#lhs_simd_ty {
                        fn into_vec(self) -> #rhs_simd_ty::#rhs_simd_ty {
                            unreachable!()
                        }
                    }
                }
            } else {
                let into_method = syn::Ident::new(&format!("to_{}", rhs), proc_macro2::Span::call_site());
                quote! {
                    impl IntoVec<#rhs_simd_ty::#rhs_simd_ty> for #lhs_simd_ty::#lhs_simd_ty {
                        fn into_vec(self) -> #rhs_simd_ty::#rhs_simd_ty {
                            self.#into_method()
                        }
                    }
                }
            };
            ret.extend(res);
        }
    }

    ret.into()
}