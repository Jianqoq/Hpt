use crate::type_utils::type_simd_lanes;
use crate::TokenStream2;
use crate::TypeInfo;
use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;

pub fn impl_simd_bitwise_out() -> TokenStream {
    let mut ret = proc_macro2::TokenStream::new();

    let types = [
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
                    impl BitWiseOut<#rhs_simd_ty::#rhs_simd_ty> for #lhs_simd_ty::#lhs_simd_ty {
                        type Output = #lhs_simd_ty::#lhs_simd_ty;
                        fn _bitand(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                            self & rhs
                        }
                        fn _bitor(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                            self | rhs
                        }
                        fn _bitxor(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                            self ^ rhs
                        }
                        fn _not(self) -> Self::Output {
                            !self
                        }
                        fn _shl(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                            self << rhs
                        }
                        fn _shr(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                            self >> rhs
                        }
                    }
                }
            } else {
                quote! {
                    impl BitWiseOut<#rhs_simd_ty::#rhs_simd_ty> for #lhs_simd_ty::#lhs_simd_ty {
                        type Output = <#lhs_simd_ty::#lhs_simd_ty as NormalOutPromote<#rhs_simd_ty::#rhs_simd_ty>>::Output;
                        fn _bitand(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                            let lhs: Self::Output = self.into_vec();
                            let rhs: Self::Output = rhs.into_vec();
                            lhs & rhs
                        }
                        fn _bitor(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                            let lhs: Self::Output = self.into_vec();
                            let rhs: Self::Output = rhs.into_vec();
                            lhs | rhs
                        }
                        fn _bitxor(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                            let lhs: Self::Output = self.into_vec();
                            let rhs: Self::Output = rhs.into_vec();
                            lhs ^ rhs
                        }
                        fn _not(self) -> Self::Output {
                            let lhs: Self::Output = self.into_vec();
                            !lhs
                        }
                        fn _shl(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                            let lhs: Self::Output = self.into_vec();
                            let rhs: Self::Output = rhs.into_vec();
                            lhs << rhs
                        }
                        fn _shr(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                            let lhs: Self::Output = self.into_vec();
                            let rhs: Self::Output = rhs.into_vec();
                            lhs >> rhs
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
        impl BitWiseOut<#rhs_simd::#rhs_simd> for #lhs_simd::#lhs_simd {
            type Output = <#lhs_simd::#lhs_simd as NormalOutPromote<#rhs_simd::#rhs_simd>>::Output;
            fn _bitand(self, rhs: #rhs_simd::#rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _bitor(self, rhs: #rhs_simd::#rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _bitxor(self, rhs: #rhs_simd::#rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _not(self) -> Self::Output {
                unreachable!()
            }
            fn _shl(self, rhs: #rhs_simd::#rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _shr(self, rhs: #rhs_simd::#rhs_simd) -> Self::Output {
                unreachable!()
            }
        }
    }
}
