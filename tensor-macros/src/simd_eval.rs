use crate::type_utils::type_simd_lanes;
use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;

pub fn impl_simd_eval() -> TokenStream {
    let mut ret = proc_macro2::TokenStream::new();

    let types = [
        (format!("boolx{}", type_simd_lanes("bool"))),
        (format!("bf16x{}", type_simd_lanes("bf16"))),
        (format!("f16x{}", type_simd_lanes("f16"))),
        (format!("f32x{}", type_simd_lanes("f32"))),
        (format!("f64x{}", type_simd_lanes("f64"))),
        (format!("i8x{}", type_simd_lanes("i8"))),
        (format!("i16x{}", type_simd_lanes("i16"))),
        (format!("i32x{}", type_simd_lanes("i32"))),
        (format!("i64x{}", type_simd_lanes("i64"))),
        (format!("u8x{}", type_simd_lanes("u8"))),
        (format!("u16x{}", type_simd_lanes("u16"))),
        (format!("u32x{}", type_simd_lanes("u32"))),
        (format!("u64x{}", type_simd_lanes("u64"))),
        (format!("isizex{}", type_simd_lanes("isize"))),
        (format!("usizex{}", type_simd_lanes("usize"))),
    ];

    for simd_ty in types.iter() {
        let simd_ty = Ident::new(&simd_ty, proc_macro2::Span::call_site());

        let res = quote! {
            impl Eval for #simd_ty::#simd_ty {
                type Output = <#simd_ty::#simd_ty as SimdCmpPromote<#simd_ty::#simd_ty>>::Output;
                fn _is_nan(&self) -> Self::Output {
                    self.__is_nan()
                }
                fn _is_true(&self) -> Self::Output {
                    self.__is_true()
                }
                fn _is_inf(&self) -> Self::Output {
                    self.__is_inf()
                }
            }
        };
        ret.extend(res);
    }

    ret.into()
}