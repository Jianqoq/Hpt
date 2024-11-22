use proc_macro::TokenStream;
use crate::TypeInfo;
use quote::quote;
use crate::type_utils::type_simd_lanes;
use proc_macro2::Ident;

pub fn impl_simd_eval() -> TokenStream {
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

    for (simd_ty, lhs) in types.iter() {
        let simd_ty = Ident::new(&simd_ty, proc_macro2::Span::call_site());
        let (mask_ty, mask_meta_ty) = map_mask(lhs);
        let lhs_type = TypeInfo::new(lhs);
        let lhs_dtype = lhs_type.dtype;

        let is_nan = if lhs_dtype.is_float() {
            if lhs_dtype.is_bf16() {
                quote! {
                    fn _is_nan(&self) -> #mask_ty::#mask_ty {
                        self.is_nan()
                    }
                }
            } else if lhs_dtype.is_f16() {
                quote! {
                    fn _is_nan(&self) -> #mask_ty::#mask_ty {
                        self.is_nan()
                    }
                }
            } else {
                quote! {
                    fn _is_nan(&self) -> #mask_ty::#mask_ty {
                        #mask_ty::#mask_ty(unsafe { std::mem::transmute(self.is_nan()) })
                    }
                }
            }
        } else {
            quote! {
                fn _is_nan(&self) -> #mask_ty::#mask_ty {
                    #mask_ty::#mask_ty::splat(#mask_meta_ty::ZERO)
                }
            }
        };

        let is_true = if lhs_dtype.is_bool() {
            quote! {
                fn _is_true(&self) -> #mask_ty::#mask_ty {
                    #mask_ty::#mask_ty(
                       unsafe { std::mem::transmute(self.0) }
                    )
                }
            }
        } else if lhs_dtype.is_float() {
            if lhs_dtype.is_bf16() {
                quote! {
                    fn _is_true(&self) -> #mask_ty::#mask_ty {
                        #[cfg(target_feature = "avx2")]
                        let x: Simd<u16, 16> = unsafe { std::mem::transmute(self.0) };
                        #[cfg(all(
                            any(target_feature = "sse2", target_arch = "aarch64"),
                            not(target_feature = "avx2")
                        ))]
                        let x: Simd<u16, 8> = unsafe { std::mem::transmute(self.0) };
                        #[cfg(target_feature = "avx512f")]
                        let x: Simd<u16, 32> = unsafe { std::mem::transmute(self.0) };
                        #mask_ty::#mask_ty(unsafe { std::mem::transmute(x.simd_ne(#mask_ty::#mask_ty::splat(#mask_meta_ty::ZERO).0)) })
                    }
                }
            } else if lhs_dtype.is_f16() {
                quote! {
                    fn _is_true(&self) -> #mask_ty::#mask_ty {
                        #[cfg(target_feature = "avx2")]
                        let x: Simd<u16, 16> = unsafe { std::mem::transmute(self.0) };
                        #[cfg(all(
                            any(target_feature = "sse2", target_arch = "aarch64"),
                            not(target_feature = "avx2")
                        ))]
                        let x: Simd<u16, 8> = unsafe { std::mem::transmute(self.0) };
                        #[cfg(target_feature = "avx512f")]
                        let x: Simd<u16, 32> = unsafe { std::mem::transmute(self.0) };
                        #mask_ty::#mask_ty(unsafe { std::mem::transmute(x.simd_ne(#mask_ty::#mask_ty::splat(#mask_meta_ty::ZERO).0)) })
                    }
                }
            } else {
                quote! {
                    fn _is_true(&self) -> #mask_ty::#mask_ty {
                        #mask_ty::#mask_ty(unsafe { std::mem::transmute(self.simd_ne(#simd_ty::#simd_ty::splat(#lhs_dtype::ZERO).0)) })
                    }
                }
            }
        } else {
            quote! {
                fn _is_true(&self) -> #mask_ty::#mask_ty {
                    #mask_ty::#mask_ty(unsafe { std::mem::transmute(self.simd_ne(#simd_ty::#simd_ty::splat(#lhs_dtype::ZERO).0)) })
                }
            }
        };

        let is_inf = if lhs_dtype.is_float() {
            quote! {
                fn _is_inf(&self) -> #mask_ty::#mask_ty {
                    unsafe { std::mem::transmute(self.is_infinite()) }
                }
            }
        } else {
            quote! {
                fn _is_inf(&self) -> #mask_ty::#mask_ty {
                    #mask_ty::#mask_ty::splat(#mask_meta_ty::ZERO)
                }
            }
        };

        let res =
            quote! {
                impl Eval for #simd_ty::#simd_ty {
                    type Output = #mask_ty::#mask_ty;
                    #is_nan
                    #is_true
                    #is_inf
                }
            };
        ret.extend(res);
    }

    ret.into()
}

fn map_mask(ty: &str) -> (Ident, Ident) {
    match ty {
        "bool" => (Ident::new(&format!("u8x{}", type_simd_lanes("bool")), proc_macro2::Span::call_site()),
        Ident::new("u8", proc_macro2::Span::call_site())),
        "bf16" => (Ident::new(&format!("u16x{}", type_simd_lanes("bf16")), proc_macro2::Span::call_site()),
        Ident::new("u16", proc_macro2::Span::call_site())),
        "f16" => (Ident::new(&format!("u16x{}", type_simd_lanes("f16")), proc_macro2::Span::call_site()),
        Ident::new("u16", proc_macro2::Span::call_site())),
        "f32" => (Ident::new(&format!("u32x{}", type_simd_lanes("f32")), proc_macro2::Span::call_site()),
        Ident::new("u32", proc_macro2::Span::call_site())),
        "f64" => (Ident::new(&format!("u64x{}", type_simd_lanes("f64")), proc_macro2::Span::call_site()),
        Ident::new("u64", proc_macro2::Span::call_site())),
        "i8" => (Ident::new(&format!("u8x{}", type_simd_lanes("i8")), proc_macro2::Span::call_site()),
        Ident::new("u8", proc_macro2::Span::call_site())),
        "i16" => (Ident::new(&format!("u16x{}", type_simd_lanes("i16")), proc_macro2::Span::call_site()),
        Ident::new("u16", proc_macro2::Span::call_site())),
        "i32" => (Ident::new(&format!("u32x{}", type_simd_lanes("i32")), proc_macro2::Span::call_site()),
        Ident::new("u32", proc_macro2::Span::call_site())),
        "i64" => (Ident::new(&format!("u64x{}", type_simd_lanes("i64")), proc_macro2::Span::call_site()),
        Ident::new("u64", proc_macro2::Span::call_site())),
        "u8" => (Ident::new(&format!("u8x{}", type_simd_lanes("u8")), proc_macro2::Span::call_site()),
        Ident::new("u8", proc_macro2::Span::call_site())),
        "u16" => (Ident::new(&format!("u16x{}", type_simd_lanes("u16")), proc_macro2::Span::call_site()),
        Ident::new("u16", proc_macro2::Span::call_site())),
        "u32" => (Ident::new(&format!("u32x{}", type_simd_lanes("u32")), proc_macro2::Span::call_site()),
        Ident::new("u32", proc_macro2::Span::call_site())),
        "u64" => (Ident::new(&format!("u64x{}", type_simd_lanes("u64")), proc_macro2::Span::call_site()),
        Ident::new("u64", proc_macro2::Span::call_site())),
        "isize" => (Ident::new(&format!("usizex{}", type_simd_lanes("isize")), proc_macro2::Span::call_site()),
        Ident::new("usize", proc_macro2::Span::call_site())),
        "usize" => (Ident::new(&format!("usizex{}", type_simd_lanes("usize")), proc_macro2::Span::call_site()),
        Ident::new("usize", proc_macro2::Span::call_site())),
        _ => panic!("Invalid type"),
    }
}