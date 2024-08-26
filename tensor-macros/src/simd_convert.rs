use proc_macro::TokenStream;
use crate::type_utils::{ type_simd_is_arr, type_simd_lanes, SimdType, TypeInfo };
use quote::quote;
use proc_macro2::Ident;

pub fn __impl_simd_convert() -> TokenStream {
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
        (format!("cplx32x{}", type_simd_lanes("Complex32")), "Complex32"),
        (format!("cplx64x{}", type_simd_lanes("Complex64")), "Complex64"),
    ];

    for (_, lhs) in types.iter() {
        let mut funcs = proc_macro2::TokenStream::new();
        for (res_ty_str, rhs) in types.iter() {
            let lhs_lanes = type_simd_lanes(lhs);
            let rhs_lanes = type_simd_lanes(rhs);
            let lhs_dtype = TypeInfo::new(lhs);
            let func_name = format!("to_{}", rhs.to_lowercase());
            let function_name: Ident = Ident::new(&func_name, proc_macro2::Span::call_site());
            let res_simd: Ident = Ident::new(&res_ty_str, proc_macro2::Span::call_site());
            let func_gen = if lhs_lanes == rhs_lanes {
                if
                    (lhs_dtype.dtype.is_f32() || lhs_dtype.dtype.is_f64()) &&
                    !type_simd_is_arr(rhs) &&
                    !type_simd_is_arr(lhs)
                {
                    quote! {
                        fn #function_name(self) -> #res_simd::#res_simd {
                            #res_simd::#res_simd(self.cast().into())
                        }
                    }
                } else if type_simd_is_arr(rhs) || type_simd_is_arr(lhs) {
                    let unroll = (0..lhs_lanes as usize).map(|i| {
                        quote! {
                            arr[#i] = self_arr[#i].#function_name();
                        }
                    });
                    let rhs_ty: Ident = Ident::new(rhs, proc_macro2::Span::call_site());
                    quote! {
                        fn #function_name(self) -> #res_simd::#res_simd {
                            let mut arr = [#rhs_ty::ZERO; #rhs_lanes as usize];
                            let self_arr = self.0;
                            #(#unroll)*
                            #res_simd::#res_simd(arr.into())
                        }
                    }
                } else {
                    quote! {
                        fn #function_name(self) -> #res_simd::#res_simd {
                            #res_simd::#res_simd(self.cast().into())
                        }
                    }
                }
            } else {
                quote! {
                    fn #function_name(self) -> #res_simd::#res_simd {
                        unreachable!()
                    }
                }
            };
            funcs.extend(func_gen);
        }
        let lhs_simd: SimdType = (*lhs).into();
        ret.extend(
            quote! {
            impl VecConvertor for #lhs_simd {
                #funcs
            }
        }
        );
    }

    ret.into()
}
