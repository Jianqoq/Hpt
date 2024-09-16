use proc_macro::TokenStream;
use crate::TypeInfo;
use quote::quote;
use crate::type_utils::type_simd_lanes;
use proc_macro2::Ident;
use crate::TokenStream2;


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
            let res_type = lhs_type.infer_normal_res_type(&rhs_type);
            let lhs_lanes = type_simd_lanes(lhs);
            let rhs_lanes = type_simd_lanes(rhs);
            let res_lanes = type_simd_lanes(&res_type.to_string());
            let res_simd_ty = Ident::new(&format!("{}x{}", res_type.to_string(), res_lanes), proc_macro2::Span::call_site());
            if lhs_lanes != rhs_lanes || lhs_lanes != res_lanes || rhs_lanes != res_lanes {
                ret.extend(
                    impl_unreachable(
                        lhs_simd_ty,
                        rhs_simd_ty,
                        res_simd_ty
                    )
                );
                continue;
            }

            let shift = if res_type.is_bool() {
                quote! {
                    fn _shl(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                        #res_simd_ty::#res_simd_ty(self.0 || rhs.0)
                    }
                    fn _shr(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                        #res_simd_ty::#res_simd_ty(self.0 && !rhs.0)
                    }
                }
            } else {
                quote! {
                    fn _shl(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                        paste::paste! {
                            #res_simd_ty::#res_simd_ty(self.[<to_ #res_type>]().0 << rhs.[<to_ #res_type>]().0)
                        }
                    }
                    fn _shr(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                        paste::paste! {
                            #res_simd_ty::#res_simd_ty(self.[<to_ #res_type>]().0 >> rhs.[<to_ #res_type>]().0)
                        }
                    }
                }
            };

            let res =
                quote! {
                impl BitWiseOut<#rhs_simd_ty::#rhs_simd_ty> for #lhs_simd_ty::#lhs_simd_ty {
                    type Output = #res_simd_ty::#res_simd_ty;
                    fn _bitand(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                        paste::paste! {
                            #res_simd_ty::#res_simd_ty(self.[<to_ #res_type>]().0 & rhs.[<to_ #res_type>]().0)
                        }
                    }
                    fn _bitor(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                        paste::paste! {
                            #res_simd_ty::#res_simd_ty(self.[<to_ #res_type>]().0 | rhs.[<to_ #res_type>]().0)
                        }
                    }
                    fn _bitxor(self, rhs: #rhs_simd_ty::#rhs_simd_ty) -> Self::Output {
                        paste::paste! {
                            #res_simd_ty::#res_simd_ty(self.[<to_ #res_type>]().0 ^ rhs.[<to_ #res_type>]().0)
                        }
                    }
                    fn _not(self) -> Self::Output {
                        paste::paste! {
                            #res_simd_ty::#res_simd_ty(!self.[<to_ #res_type>]().0)
                        }
                    }
                    #shift
                }
            };
            ret.extend(res);
        }
    }

    ret.into()
}

fn impl_unreachable(lhs_simd: Ident, rhs_simd: Ident, res_type: Ident) -> TokenStream2 {
    quote! {
        impl BitWiseOut<#rhs_simd::#rhs_simd> for #lhs_simd::#lhs_simd {
            type Output = #res_type::#res_type;
            fn _bitand(self, rhs: #rhs_simd::#rhs_simd) -> #res_type::#res_type {
                unreachable!()
            }
            fn _bitor(self, rhs: #rhs_simd::#rhs_simd) -> #res_type::#res_type {
                unreachable!()
            }
            fn _bitxor(self, rhs: #rhs_simd::#rhs_simd) -> #res_type::#res_type {
                unreachable!()
            }
            fn _not(self) -> #res_type::#res_type {
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