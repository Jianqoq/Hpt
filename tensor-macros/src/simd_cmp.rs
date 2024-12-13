use crate::type_utils::type_simd_lanes;
use crate::TokenStream2;
use crate::TypeInfo;
use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;

pub fn impl_simd_cmp() -> TokenStream {
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
            let lhs_simd_ty = Ident::new(&lhs_simd_ty, proc_macro2::Span::call_site());
            let rhs_simd_ty = Ident::new(&rhs_simd_ty, proc_macro2::Span::call_site());
            let lhs_type = TypeInfo::new(lhs);
            let rhs_type = TypeInfo::new(rhs);
            let res_type = lhs_type.infer_normal_res_type(&rhs_type);
            let lhs_lanes = type_simd_lanes(lhs);
            let rhs_lanes = type_simd_lanes(rhs);
            let res_lanes = type_simd_lanes(&res_type.to_string());
            let res_simd_ty = Ident::new(
                &format!("{}x{}", res_type.to_string(), res_lanes),
                proc_macro2::Span::call_site(),
            );
            if lhs_lanes != rhs_lanes || lhs_lanes != res_lanes || rhs_lanes != res_lanes {
                ret.extend(impl_unreachable(lhs_simd_ty, rhs_simd_ty, res_simd_ty));
                continue;
            }
            let (res_simd_ty, _) = map_mask(&res_type.to_string());
            let to_res_type = Ident::new(
                &format!("to_{}", res_type.to_string()),
                proc_macro2::Span::call_site(),
            );
            let res = quote! {
                impl SimdCmp<#rhs_simd_ty::#rhs_simd_ty> for #lhs_simd_ty::#lhs_simd_ty {
                    type Output = #res_simd_ty::#res_simd_ty;
                    fn _eq(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                        self.#to_res_type().simd_eq(rhs.#to_res_type()) 
                    }
                    fn _ne(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                        self.#to_res_type().simd_ne(rhs.#to_res_type())
                    }
                    fn _lt(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                        self.#to_res_type().simd_lt(rhs.#to_res_type()) 
                    }
                    fn _le(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                        self.#to_res_type().simd_le(rhs.#to_res_type())
                    }
                    fn _gt(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                        self.#to_res_type().simd_gt(rhs.#to_res_type())
                    }
                    fn _ge(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                        self.#to_res_type().simd_ge(rhs.#to_res_type())
                    }
                }
            };
            ret.extend(res);
        }
    }

    ret.into()
}

fn impl_unreachable(lhs_simd: Ident, rhs_simd: Ident, res_type: Ident) -> TokenStream2 {
    quote! {
        impl SimdCmp<#rhs_simd::#rhs_simd> for #lhs_simd::#lhs_simd {
            type Output = #res_type::#res_type;
            fn _eq(self, rhs: #rhs_simd::#rhs_simd) -> #res_type::#res_type {
                unreachable!()
            }
            fn _ne(self, rhs: #rhs_simd::#rhs_simd) -> #res_type::#res_type {
                unreachable!()
            }
            fn _lt(self, rhs: #rhs_simd::#rhs_simd) -> #res_type::#res_type {
                unreachable!()
            }

            fn _le(self, rhs: #rhs_simd::#rhs_simd) -> #res_type::#res_type {
                unreachable!()
            }
            fn _gt(self, rhs: #rhs_simd::#rhs_simd) -> #res_type::#res_type {
                unreachable!()
            }
            fn _ge(self, rhs: #rhs_simd::#rhs_simd) -> #res_type::#res_type {
                unreachable!()
            }
        }
    }
}

fn map_mask(ty: &str) -> (Ident, Ident) {
    match ty {
        "bool" => (
            Ident::new(
                &format!("i8x{}", type_simd_lanes("bool")),
                proc_macro2::Span::call_site(),
            ),
            Ident::new("i8", proc_macro2::Span::call_site()),
        ),
        "bf16" => (
            Ident::new(
                &format!("i16x{}", type_simd_lanes("bf16")),
                proc_macro2::Span::call_site(),
            ),
            Ident::new("i16", proc_macro2::Span::call_site()),
        ),
        "f16" => (
            Ident::new(
                &format!("i16x{}", type_simd_lanes("f16")),
                proc_macro2::Span::call_site(),
            ),
            Ident::new("i16", proc_macro2::Span::call_site()),
        ),
        "f32" => (
            Ident::new(
                &format!("i32x{}", type_simd_lanes("f32")),
                proc_macro2::Span::call_site(),
            ),
            Ident::new("i32", proc_macro2::Span::call_site()),
        ),
        "f64" => (
            Ident::new(
                &format!("i64x{}", type_simd_lanes("f64")),
                proc_macro2::Span::call_site(),
            ),
            Ident::new("i64", proc_macro2::Span::call_site()),
        ),
        "i8" => (
            Ident::new(
                &format!("i8x{}", type_simd_lanes("i8")),
                proc_macro2::Span::call_site(),
            ),
            Ident::new("i8", proc_macro2::Span::call_site()),
        ),
        "i16" => (
            Ident::new(
                &format!("i16x{}", type_simd_lanes("i16")),
                proc_macro2::Span::call_site(),
            ),
            Ident::new("i16", proc_macro2::Span::call_site()),
        ),
        "i32" => (
            Ident::new(
                &format!("i32x{}", type_simd_lanes("i32")),
                proc_macro2::Span::call_site(),
            ),
            Ident::new("i32", proc_macro2::Span::call_site()),
        ),
        "i64" => (
            Ident::new(
                &format!("i64x{}", type_simd_lanes("i64")),
                proc_macro2::Span::call_site(),
            ),
            Ident::new("i64", proc_macro2::Span::call_site()),
        ),
        "u8" => (
            Ident::new(
                &format!("i8x{}", type_simd_lanes("u8")),
                proc_macro2::Span::call_site(),
            ),
            Ident::new("i8", proc_macro2::Span::call_site()),
        ),
        "u16" => (
            Ident::new(
                &format!("i16x{}", type_simd_lanes("u16")),
                proc_macro2::Span::call_site(),
            ),
            Ident::new("i16", proc_macro2::Span::call_site()),
        ),
        "u32" => (
            Ident::new(
                &format!("i32x{}", type_simd_lanes("u32")),
                proc_macro2::Span::call_site(),
            ),
            Ident::new("i32", proc_macro2::Span::call_site()),
        ),
        "u64" => (
            Ident::new(
                &format!("i64x{}", type_simd_lanes("u64")),
                proc_macro2::Span::call_site(),
            ),
            Ident::new("i64", proc_macro2::Span::call_site()),
        ),
        "isize" => (
            Ident::new(
                &format!("isizex{}", type_simd_lanes("isize")),
                proc_macro2::Span::call_site(),
            ),
            Ident::new("isize", proc_macro2::Span::call_site()),
        ),
        "usize" => (
            Ident::new(
                &format!("isizex{}", type_simd_lanes("usize")),
                proc_macro2::Span::call_site(),
            ),
            Ident::new("isize", proc_macro2::Span::call_site()),
        ),
        _ => panic!("Invalid type"),
    }
}
