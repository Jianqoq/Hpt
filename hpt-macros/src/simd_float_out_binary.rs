use crate::type_utils::{type_simd_lanes, SimdType, Type, TypeInfo};
use crate::TokenStream2;
use proc_macro::TokenStream;
use quote::quote;

pub fn impl_simd_binary_out_float() -> TokenStream {
    let mut ret = proc_macro2::TokenStream::new();

    let types = [
        ("bool", type_simd_lanes("bool"), "bool"),
        ("bf16", type_simd_lanes("bf16"), "bf16"),
        ("f16", type_simd_lanes("f16"), "f16"),
        ("f32", type_simd_lanes("f32"), "f32"),
        ("f64", type_simd_lanes("f64"), "f64"),
        ("i8", type_simd_lanes("i8"), "i8"),
        ("i16", type_simd_lanes("i16"), "i16"),
        ("i32", type_simd_lanes("i32"), "i32"),
        ("i64", type_simd_lanes("i64"), "i64"),
        ("u8", type_simd_lanes("u8"), "u8"),
        ("u16", type_simd_lanes("u16"), "u16"),
        ("u32", type_simd_lanes("u32"), "u32"),
        ("u64", type_simd_lanes("u64"), "u64"),
        ("isize", type_simd_lanes("isize"), "isize"),
        ("usize", type_simd_lanes("usize"), "usize"),
        ("Complex32", type_simd_lanes("complex32"), "complex32"),
        ("Complex64", type_simd_lanes("complex64"), "complex64"),
    ];

    for (lhs_ty, lhs_lanes, lhs) in types.iter() {
        for (rhs_ty, rhs_lanes, rhs) in types.iter() {
            let lhs_lanes = *lhs_lanes;
            let rhs_lanes = *rhs_lanes;
            let lhs_type = TypeInfo::new(&lhs_ty.to_lowercase());
            let rhs_type = TypeInfo::new(&rhs_ty.to_lowercase());
            if lhs_lanes != rhs_lanes {
                ret.extend(impl_unreachable((*lhs).into(), (*rhs).into()));
                continue;
            }
            let lhs_simd: SimdType = (*lhs).into();
            let rhs_simd: SimdType = (*rhs).into();

            let res = if lhs_type.dtype == rhs_type.dtype
                && (lhs_type.dtype.is_float() || lhs_type.dtype.is_cplx())
            {
                quote! {
                    impl FloatOutBinary<#rhs_simd> for #lhs_simd {
                        type Output = <#lhs_simd as FloatOutBinaryPromote<#rhs_simd>>::Output;
                        fn _div(self, rhs: #rhs_simd) -> Self::Output {
                            self / rhs
                        }
                        fn _log(self, base: #rhs_simd) -> Self::Output {
                            self.__log(base)
                        }
                        fn _hypot(self, rhs: #rhs_simd) -> Self::Output {
                            self.__hypot(rhs)
                        }
                    }
                }
            } else {
                quote! {
                    impl FloatOutBinary<#rhs_simd> for #lhs_simd {
                        type Output = <#lhs_simd as FloatOutBinaryPromote<#rhs_simd>>::Output;
                        fn _div(self, rhs: #rhs_simd) -> Self::Output {
                            let lhs: Self::Output = self.into_vec();
                            let rhs: Self::Output = rhs.into_vec();
                            lhs / rhs
                        }
                        fn _log(self, base: #rhs_simd) -> Self::Output {
                            let lhs: Self::Output = self.into_vec();
                            let base: Self::Output = base.into_vec();
                            lhs.__log(base)
                        }
                        fn _hypot(self, rhs: #rhs_simd) -> Self::Output {
                            let lhs: Self::Output = self.into_vec();
                            let rhs: Self::Output = rhs.into_vec();
                            lhs.__hypot(rhs)
                        }
                    }
                }
            };
            ret.extend(res);
        }
    }

    ret.into()
}

fn impl_unreachable(lhs_dtype: SimdType, rhs_simd: SimdType) -> TokenStream2 {
    quote! {
        impl FloatOutBinary<#rhs_simd> for #lhs_dtype {
            type Output = <#lhs_dtype as FloatOutBinaryPromote<#rhs_simd>>::Output;
            fn _div(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _log(self, base: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _hypot(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
        }
    }
}

pub fn impl_simd_binary_out_float_lhs_scalar() -> TokenStream {
    let mut ret = proc_macro2::TokenStream::new();

    let types = [
        ("bool", type_simd_lanes("bool"), "bool"),
        ("bf16", type_simd_lanes("bf16"), "bf16"),
        ("f16", type_simd_lanes("f16"), "f16"),
        ("f32", type_simd_lanes("f32"), "f32"),
        ("f64", type_simd_lanes("f64"), "f64"),
        ("i8", type_simd_lanes("i8"), "i8"),
        ("i16", type_simd_lanes("i16"), "i16"),
        ("i32", type_simd_lanes("i32"), "i32"),
        ("i64", type_simd_lanes("i64"), "i64"),
        ("u8", type_simd_lanes("u8"), "u8"),
        ("u16", type_simd_lanes("u16"), "u16"),
        ("u32", type_simd_lanes("u32"), "u32"),
        ("u64", type_simd_lanes("u64"), "u64"),
        ("isize", type_simd_lanes("isize"), "isize"),
        ("usize", type_simd_lanes("usize"), "usize"),
        ("Complex32", type_simd_lanes("complex32"), "complex32"),
        ("Complex64", type_simd_lanes("complex64"), "complex64"),
    ];

    for (lhs_ty, lhs_lanes, lhs) in types.iter() {
        for (rhs_ty, rhs_lanes, rhs) in types.iter() {
            let lhs_lanes = *lhs_lanes;
            let rhs_lanes = *rhs_lanes;
            let lhs_type = TypeInfo::new(&lhs_ty.to_lowercase());
            let rhs_type = TypeInfo::new(&rhs_ty.to_lowercase());
            let lhs_dtype = lhs_type.dtype;
            if lhs_lanes != rhs_lanes {
                ret.extend(impl_unreachable_lhs_scalar(
                    lhs_dtype,
                    (*rhs).into(),
                    (*lhs).into(),
                ));
                continue;
            }
            let rhs_simd: SimdType = (*rhs).into();
            let lhs_simd: SimdType = (*lhs).into();

            let res = if lhs_type.dtype == rhs_type.dtype
                && (lhs_type.dtype.is_float() || lhs_type.dtype.is_cplx())
            {
                quote! {
                    impl FloatOutBinary<#rhs_simd> for #lhs_dtype {
                        type Output = <#lhs_simd as FloatOutBinaryPromote<#rhs_simd>>::Output;
                        fn _div(self, rhs: #rhs_simd) -> Self::Output {
                            Self::Output::splat(self.cast()) / rhs
                        }
                        fn _log(self, base: #rhs_simd) -> Self::Output {
                            Self::Output::splat(self.cast()).__log(base)
                        }
                        fn _hypot(self, rhs: #rhs_simd) -> Self::Output {
                            Self::Output::splat(self.cast()).__hypot(rhs)
                        }
                    }
                }
            } else {
                quote! {
                    impl FloatOutBinary<#rhs_simd> for #lhs_dtype {
                        type Output = <#lhs_simd as FloatOutBinaryPromote<#rhs_simd>>::Output;
                        fn _div(self, rhs: #rhs_simd) -> Self::Output {
                            let lhs: Self::Output = Self::Output::splat(self.cast());
                            let rhs: Self::Output = rhs.into_vec();
                            lhs / rhs
                        }
                        fn _log(self, base: #rhs_simd) -> Self::Output {
                            let lhs: Self::Output = Self::Output::splat(self.cast());
                            let base: Self::Output = base.into_vec();
                            lhs.__log(base)
                        }
                        fn _hypot(self, rhs: #rhs_simd) -> Self::Output {
                            let lhs: Self::Output = Self::Output::splat(self.cast());
                            let rhs: Self::Output = rhs.into_vec();
                            lhs.__hypot(rhs)
                        }
                    }
                }
            };
            ret.extend(res);
        }
    }

    ret.into()
}

fn impl_unreachable_lhs_scalar(
    lhs_dtype: Type,
    rhs_simd: SimdType,
    lhs_simd: SimdType,
) -> TokenStream2 {
    quote! {
        impl FloatOutBinary<#rhs_simd> for #lhs_dtype {
            type Output = <#lhs_simd as FloatOutBinaryPromote<#rhs_simd>>::Output;
            fn _div(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _log(self, base: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _hypot(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
        }
    }
}

pub fn impl_simd_binary_out_float_rhs_scalar() -> TokenStream {
    let mut ret = proc_macro2::TokenStream::new();

    let types = [
        ("bool", type_simd_lanes("bool"), "bool"),
        ("bf16", type_simd_lanes("bf16"), "bf16"),
        ("f16", type_simd_lanes("f16"), "f16"),
        ("f32", type_simd_lanes("f32"), "f32"),
        ("f64", type_simd_lanes("f64"), "f64"),
        ("i8", type_simd_lanes("i8"), "i8"),
        ("i16", type_simd_lanes("i16"), "i16"),
        ("i32", type_simd_lanes("i32"), "i32"),
        ("i64", type_simd_lanes("i64"), "i64"),
        ("u8", type_simd_lanes("u8"), "u8"),
        ("u16", type_simd_lanes("u16"), "u16"),
        ("u32", type_simd_lanes("u32"), "u32"),
        ("u64", type_simd_lanes("u64"), "u64"),
        ("isize", type_simd_lanes("isize"), "isize"),
        ("usize", type_simd_lanes("usize"), "usize"),
        ("Complex32", type_simd_lanes("complex32"), "complex32"),
        ("Complex64", type_simd_lanes("complex64"), "complex64"),
    ];

    for (lhs_ty, lhs_lanes, lhs) in types.iter() {
        for (rhs_ty, rhs_lanes, rhs) in types.iter() {
            let lhs_lanes = *lhs_lanes;
            let rhs_lanes = *rhs_lanes;
            let lhs_type = TypeInfo::new(&lhs_ty.to_lowercase());
            let rhs_type = TypeInfo::new(&rhs_ty.to_lowercase());
            if lhs_lanes != rhs_lanes {
                ret.extend(impl_unreachable_rhs_scalar(
                    (*lhs).into(),
                    rhs_type.dtype,
                    (*rhs).into(),
                ));
                continue;
            }
            let lhs_simd: SimdType = (*lhs).into();
            let rhs_simd: SimdType = (*rhs).into();
            let rhs_dtype = rhs_type.dtype;

            let res = if lhs_type.dtype == rhs_type.dtype
                && (lhs_type.dtype.is_float() || lhs_type.dtype.is_cplx())
            {
                quote! {
                    impl FloatOutBinary<#rhs_dtype> for #lhs_simd {
                        type Output = <#lhs_simd as FloatOutBinaryPromote<#rhs_simd>>::Output;
                        fn _div(self, rhs: #rhs_dtype) -> Self::Output {
                            self / Self::Output::splat(rhs.cast())
                        }
                        fn _log(self, base: #rhs_dtype) -> Self::Output {
                            self.__log(Self::Output::splat(base.cast()))
                        }
                        fn _hypot(self, rhs: #rhs_dtype) -> Self::Output {
                            self.__hypot(Self::Output::splat(rhs.cast()))
                        }
                    }
                }
            } else {
                quote! {
                    impl FloatOutBinary<#rhs_dtype> for #lhs_simd {
                        type Output = <#lhs_simd as FloatOutBinaryPromote<#rhs_simd>>::Output;
                        fn _div(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs: Self::Output = self.into_vec();
                            let rhs: Self::Output = Self::Output::splat(rhs.cast());
                            lhs / rhs
                        }
                        fn _log(self, base: #rhs_dtype) -> Self::Output {
                            let lhs: Self::Output = self.into_vec();
                            let base: Self::Output = Self::Output::splat(base.cast());
                            lhs.__log(base)
                        }
                        fn _hypot(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs: Self::Output = self.into_vec();
                            let rhs: Self::Output = Self::Output::splat(rhs.cast());
                            lhs.__hypot(rhs)
                        }
                    }
                }
            };
            ret.extend(res);
        }
    }

    ret.into()
}

fn impl_unreachable_rhs_scalar(
    lhs_simd: SimdType,
    rhs_dtype: Type,
    rhs_simd: SimdType,
) -> TokenStream2 {
    quote! {
        impl FloatOutBinary<#rhs_dtype> for #lhs_simd {
            type Output = <#lhs_simd as FloatOutBinaryPromote<#rhs_simd>>::Output;
            fn _div(self, rhs: #rhs_dtype) -> Self::Output {
                unreachable!()
            }
            fn _log(self, base: #rhs_dtype) -> Self::Output {
                unreachable!()
            }
            fn _hypot(self, rhs: #rhs_dtype) -> Self::Output {
                unreachable!()
            }
        }
    }
}
