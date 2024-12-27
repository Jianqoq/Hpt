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
            let lhs_lanes = type_simd_lanes(lhs);
            let rhs_lanes = type_simd_lanes(rhs);
            if lhs_lanes != rhs_lanes {
                ret.extend(impl_unreachable(lhs_simd_ty, rhs_simd_ty));
                continue;
            }
            let res = if lhs_type.dtype == rhs_type.dtype {
                quote! {
                    impl SimdCmp<#rhs_simd_ty::#rhs_simd_ty> for #lhs_simd_ty::#lhs_simd_ty {
                        type Output = <#lhs_simd_ty::#lhs_simd_ty as SimdCmpPromote<#rhs_simd_ty::#rhs_simd_ty>>::Output;
                        fn _eq(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                            self.simd_eq(rhs)
                        }
                        fn _ne(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                            self.simd_ne(rhs)
                        }
                        fn _lt(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                            self.simd_lt(rhs)
                        }
                        fn _le(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                            self.simd_le(rhs)
                        }
                        fn _gt(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                            self.simd_gt(rhs)
                        }
                        fn _ge(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                            self.simd_ge(rhs)
                        }
                    }
                }
            } else {
                quote! {
                    impl SimdCmp<#rhs_simd_ty::#rhs_simd_ty> for #lhs_simd_ty::#lhs_simd_ty {
                        type Output = <#lhs_simd_ty::#lhs_simd_ty as SimdCmpPromote<#rhs_simd_ty::#rhs_simd_ty>>::Output;
                        fn _eq(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                            let lhs: Self::Output = self.into_vec();
                            let rhs: Self::Output = rhs.into_vec();
                            lhs.simd_eq(rhs)
                        }
                        fn _ne(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                            let lhs: Self::Output = self.into_vec();
                            let rhs: Self::Output = rhs.into_vec();
                            lhs.simd_ne(rhs)
                        }
                        fn _lt(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                            let lhs: Self::Output = self.into_vec();
                            let rhs: Self::Output = rhs.into_vec();
                            lhs.simd_lt(rhs)
                        }
                        fn _le(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                            let lhs: Self::Output = self.into_vec();
                            let rhs: Self::Output = rhs.into_vec();
                            lhs.simd_le(rhs)
                        }
                        fn _gt(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                            let lhs: Self::Output = self.into_vec();
                            let rhs: Self::Output = rhs.into_vec();
                            lhs.simd_gt(rhs)
                        }
                        fn _ge(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                            let lhs: Self::Output = self.into_vec();
                            let rhs: Self::Output = rhs.into_vec();
                            lhs.simd_ge(rhs)
                        }
                    }
                }
            };
            ret.extend(res);
        }
    }

    ret.into()
}

fn impl_unreachable(lhs_simd: Ident, rhs_simd: Ident) -> TokenStream2 {
    quote! {
        impl SimdCmp<#rhs_simd::#rhs_simd> for #lhs_simd::#lhs_simd {
            type Output = <#lhs_simd::#lhs_simd as SimdCmpPromote<#rhs_simd::#rhs_simd>>::Output;
            fn _eq(self, rhs: #rhs_simd::#rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _ne(self, rhs: #rhs_simd::#rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _lt(self, rhs: #rhs_simd::#rhs_simd) -> Self::Output {
                unreachable!()
            }

            fn _le(self, rhs: #rhs_simd::#rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _gt(self, rhs: #rhs_simd::#rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _ge(self, rhs: #rhs_simd::#rhs_simd) -> Self::Output {
                unreachable!()
            }
        }
    }
}