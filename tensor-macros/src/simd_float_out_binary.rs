use proc_macro::TokenStream;
use crate::type_utils::{ type_simd_lanes, SimdType, TypeInfo };
use quote::quote;
use crate::TokenStream2;
use proc_macro2::Ident;

pub fn impl_simd_binary_out_float() -> TokenStream {
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

    for (_, lhs) in types.iter() {
        for (_, rhs) in types.iter() {
            let lhs_lanes = type_simd_lanes(lhs);
            let rhs_lanes = type_simd_lanes(rhs);
            let lhs_type = TypeInfo::new(lhs);
            let rhs_type = TypeInfo::new(rhs);
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

            let res =
                quote! {
                impl FloatOutBinary<#rhs_simd> for #lhs_simd {
                    type Output = #res_simd_ty::#res_simd_ty;
                    fn _div(self, rhs: #rhs_simd) -> Self::Output {
                        paste::paste! {
                            self.[<to_ #res_type>]() / rhs.[<to_ #res_type>]()
                        }
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