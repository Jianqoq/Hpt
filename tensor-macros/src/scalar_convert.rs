use crate::type_utils::Type;
use crate::type_utils::TypeInfo;
use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;

pub fn __impl_scalar_convert() -> TokenStream {
    let mut ret = proc_macro2::TokenStream::new();

    let types = [
        "bool",
        "f16",
        "f32",
        "f64",
        "i8",
        "i16",
        "i32",
        "i64",
        "u8",
        "u16",
        "u32",
        "u64",
        "bf16",
        "isize",
        "usize",
        "Complex32",
        "Complex64",
    ];

    for lhs in types.iter() {
        let mut funcs = proc_macro2::TokenStream::new();
        let lhs_dtype = TypeInfo::new(lhs);
        let lhs_ty = lhs_dtype.dtype;
        for rhs in types.iter() {
            let rhs_dtype = TypeInfo::new(rhs);
            let func_name = format!("to_{}", rhs.to_lowercase());
            let function_name: Ident = Ident::new(&func_name, proc_macro2::Span::call_site());
            let rhs_ty = rhs_dtype.dtype;
            let func_gen = if lhs_dtype.dtype == rhs_dtype.dtype {
                quote! {
                    #[inline(always)]
                    fn #function_name(self) -> #lhs_ty {
                        self
                    }
                }
            } else {
                let body = if rhs_ty.is_f16() || rhs_ty.is_bf16() {
                    let f16_ty = if rhs_ty.is_f16() {
                        quote! {
                            f16
                        }
                    } else {
                        quote! {
                            bf16
                        }
                    };
                    match lhs_ty {
                        Type::Bool => {
                            quote! {
                                #f16_ty::from_bits(self as u16)
                            }
                        }
                        Type::I8 => {
                            quote! {
                                #f16_ty::from_f32(self as f32)
                            }
                        }
                        Type::U8 => {
                            quote! {
                                #f16_ty::from_f32(self as f32)
                            }
                        }
                        Type::I16 => {
                            quote! {
                                #f16_ty::from_f32(self as f32)
                            }
                        }
                        Type::U16 => {
                            quote! {
                                #f16_ty::from_f32(self as f32)
                            }
                        }
                        Type::I32 => {
                            quote! {
                                #f16_ty::from_f32(self as f32)
                            }
                        }
                        Type::U32 => {
                            quote! {
                                #f16_ty::from_f64(self as f64)
                            }
                        }
                        Type::I64 => {
                            quote! {
                                #f16_ty::from_f64(self as f64)
                            }
                        }
                        Type::U64 => {
                            quote! {
                                #f16_ty::from_f64(self as f64)
                            }
                        }
                        Type::BF16 => {
                            if rhs_ty == Type::F16 {
                                quote! {
                                    f16::from_f32(self.to_f32())
                                }
                            } else {
                                quote! {
                                    self
                                }
                            }
                        }
                        Type::F16 => {
                            if rhs_ty == Type::F16 {
                                quote! {
                                    self
                                }
                            } else {
                                quote! {
                                    bf16::from_f32(self.to_f32())
                                }
                            }
                        }
                        Type::F32 => {
                            quote! {
                                #f16_ty::from_f32(self)
                            }
                        }
                        Type::F64 => {
                            quote! {
                                #f16_ty::from_f64(self)
                            }
                        }
                        Type::C32 => {
                            quote! {
                                unimplemented!()
                            }
                        }
                        Type::C64 => {
                            quote! {
                                unimplemented!()
                            }
                        }
                        Type::Isize => {
                            quote! {
                                #f16_ty::from_f64(self as f64)
                            }
                        }
                        Type::Usize => {
                            quote! {
                                #f16_ty::from_f64(self as f64)
                            }
                        }
                        Type::Complex32 => {
                            quote! {
                                unimplemented!()
                            }
                        }
                        Type::Complex64 => {
                            quote! {
                                unimplemented!()
                            }
                        }
                    }
                } else if rhs_ty.is_cplx() {
                    if rhs_ty.is_cplx32() {
                        quote! {
                            Complex32::new(self.to_f32(), 0.0)
                        }
                    } else {
                        quote! {
                            Complex64::new(self.to_f64(), 0.0)
                        }
                    }
                } else if lhs_ty.is_bf16() || lhs_ty.is_f16() {
                    if rhs_ty.is_bool() {
                        quote! {
                            self.to_f32() != 0.0
                        }
                    } else {
                        quote! {
                            self.to_f32() as #rhs_ty
                        }
                    }
                } else if lhs_ty.is_bool() {
                    if rhs_ty.is_float() {
                        match rhs_ty {
                            Type::F32 => {
                                quote! {
                                    self as i32 as f32
                                }
                            }
                            Type::F64 => {
                                quote! {
                                    self as i64 as f64
                                }
                            }
                            _ => {
                                quote! {
                                    self as #rhs_ty
                                }
                            }
                        }
                    } else {
                        quote! {
                            self as #rhs_ty
                        }
                    }
                } else if lhs_ty.is_cplx() {
                    match rhs_ty {
                        Type::C32 | Type::Complex32 => {
                            if lhs_ty.is_cplx32() {
                                quote! {
                                    Complex32::new(self.re, self.im)
                                }
                            } else {
                                quote! {
                                    Complex32::new(self.re as f32, self.im as f32)
                                }
                            }
                        }
                        Type::C64 | Type::Complex64 => {
                            if lhs_ty.is_cplx64() {
                                quote! {
                                    Complex64::new(self.re, self.im)
                                }
                            } else {
                                quote! {
                                    Complex64::new(self.re as f64, self.im as f64)
                                }
                            }
                        }
                        _ => {
                            quote! {
                                unimplemented!()
                            }
                        }
                    }
                } else {
                    if rhs_ty.is_bool() {
                        quote! {
                            self != #lhs_ty::ZERO
                        }
                    } else {
                        quote! {
                            self as #rhs_ty
                        }
                    }
                };
                quote! {
                    #[inline(always)]
                    fn #function_name(self) -> #rhs_ty {
                        #body
                    }
                }
            };
            funcs.extend(func_gen);
        }
        ret.extend(quote! {
            impl Convertor for #lhs_ty {
                #funcs
            }
        });
    }

    ret.into()
}
