use crate::type_utils::{type_simd_lanes, SimdType, Type, TypeInfo};
use crate::TokenStream2;
use proc_macro::TokenStream;
use quote::quote;

pub fn impl_simd_normal_out() -> TokenStream {
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

            let res = if lhs_type.dtype == rhs_type.dtype {
                quote! {
                    impl NormalOut<#rhs_simd> for #lhs_simd {
                        type Output = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output;
                        #[inline(always)]
                        fn _add(self, rhs: #rhs_simd) -> Self::Output {
                            self.__add(rhs)
                        }
                        #[inline(always)]
                        fn _sub(self, rhs: #rhs_simd) -> Self::Output {
                            self.__sub(rhs)
                        }
                        #[inline(always)]
                        fn _mul(self, rhs: #rhs_simd) -> Self::Output {
                            self.__mul(rhs)
                        }
                        #[inline(always)]
                        fn _max(self, rhs: #rhs_simd) -> Self::Output {
                            self.__max(rhs)
                        }
                        #[inline(always)]
                        fn _min(self, rhs: #rhs_simd) -> Self::Output {
                            self.__min(rhs)
                        }
                        #[inline(always)]
                        fn _rem(self, rhs: #rhs_simd) -> Self::Output {
                            self.__rem(rhs)
                        }
                        #[inline(always)]
                        fn _clamp(self, min: #rhs_simd, max: #rhs_simd) -> Self::Output {
                            self.__clamp(min, max)
                        }
                        #[inline(always)]
                        fn _mul_add(self, a: #rhs_simd, b: #rhs_simd) -> Self::Output {
                            self.__mul_add(a, b)
                        }
                    }
                }
            } else {
                quote! {
                    impl NormalOut<#rhs_simd> for #lhs_simd {
                        type Output = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output;
                        #[inline(always)]
                        fn _add(self, rhs: #rhs_simd) -> Self::Output {
                            let x: Self::Output = self.into_vec();
                            let y: Self::Output = rhs.into_vec();
                            x.__add(y)
                        }
                        #[inline(always)]
                        fn _sub(self, rhs: #rhs_simd) -> Self::Output {
                            let x: Self::Output = self.into_vec();
                            let y: Self::Output = rhs.into_vec();
                            x.__sub(y)
                        }
                        #[inline(always)]
                        fn _mul(self, rhs: #rhs_simd) -> Self::Output {
                            let x: Self::Output = self.into_vec();
                            let y: Self::Output = rhs.into_vec();
                            x.__mul(y)
                        }
                        #[inline(always)]
                        fn _max(self, rhs: #rhs_simd) -> Self::Output {
                            let x: Self::Output = self.into_vec();
                            let y: Self::Output = rhs.into_vec();
                            x.__max(y)
                        }
                        #[inline(always)]
                        fn _min(self, rhs: #rhs_simd) -> Self::Output {
                            let x: Self::Output = self.into_vec();
                            let y: Self::Output = rhs.into_vec();
                            x.__min(y)
                        }
                        #[inline(always)]
                        fn _rem(self, rhs: #rhs_simd) -> Self::Output {
                            let x: Self::Output = self.into_vec();
                            let y: Self::Output = rhs.into_vec();
                            x.__rem(y)
                        }
                        #[inline(always)]
                        fn _clamp(self, min: #rhs_simd, max: #rhs_simd) -> Self::Output {
                            let x: Self::Output = self.into_vec();
                            let y: Self::Output = min.into_vec();
                            let z: Self::Output = max.into_vec();
                            x.__clamp(y, z)
                        }
                        #[inline(always)]
                        fn _mul_add(self, a: #rhs_simd, b: #rhs_simd) -> Self::Output {
                            let x: Self::Output = self.into_vec();
                            let y: Self::Output = a.into_vec();
                            let z: Self::Output = b.into_vec();
                            x.__mul_add(y, z)
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
        impl NormalOut<#rhs_simd> for #lhs_dtype {
            type Output = <#lhs_dtype as NormalOutPromote<#rhs_simd>>::Output;
            fn _mul_add(self, a: #rhs_simd, b: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _add(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _sub(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _mul(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _rem(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _clamp(self, min: #rhs_simd, max: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _max(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _min(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
        }
    }
}

fn impl_unreachable_with_rhs_scalar(
    lhs_dtype: SimdType,
    rhs_scalar_ty: Type,
    rhs_type: SimdType,
) -> TokenStream2 {
    quote! {
        impl NormalOut<#rhs_scalar_ty> for #lhs_dtype {
            type Output = <#lhs_dtype as NormalOutPromote<#rhs_type>>::Output;
            fn _mul_add(self, a: #rhs_scalar_ty, b: #rhs_scalar_ty) -> Self::Output {
                unreachable!()
            }
            fn _add(self, rhs: #rhs_scalar_ty) -> Self::Output {
                unreachable!()
            }
            fn _sub(self, rhs: #rhs_scalar_ty) -> Self::Output {
                unreachable!()
            }
            fn _mul(self, rhs: #rhs_scalar_ty) -> Self::Output {
                unreachable!()
            }
            fn _rem(self, rhs: #rhs_scalar_ty) -> Self::Output {
                unreachable!()
            }
            fn _clamp(self, min: #rhs_scalar_ty, max: #rhs_scalar_ty) -> Self::Output {
                unreachable!()
            }
            fn _max(self, rhs: #rhs_scalar_ty) -> Self::Output {
                unreachable!()
            }
            fn _min(self, rhs: #rhs_scalar_ty) -> Self::Output {
                unreachable!()
            }
        }
    }
}

pub fn impl_simd_normal_out_with_rhs_scalar() -> TokenStream {
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
            let rhs_dtype = rhs_type.dtype;
            let lhs_simd: SimdType = (*lhs).into();
            let rhs_simd: SimdType = (*rhs).into();
            if lhs_lanes != rhs_lanes {
                ret.extend(impl_unreachable_with_rhs_scalar(
                    lhs_simd, rhs_dtype, rhs_simd,
                ));
                continue;
            }

            let res = if lhs_type.dtype == rhs_type.dtype {
                quote! {
                    impl NormalOut<#rhs_dtype> for #lhs_simd {
                        type Output = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output;
                        #[inline(always)]
                        fn _add(self, rhs: #rhs_dtype) -> Self::Output {
                            let rhs = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(rhs);
                            self.__add(rhs)
                        }
                        #[inline(always)]
                        fn _sub(self, rhs: #rhs_dtype) -> Self::Output {
                            let rhs = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(rhs);
                            self.__sub(rhs)
                        }
                        #[inline(always)]
                        fn _mul(self, rhs: #rhs_dtype) -> Self::Output {
                            let rhs = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(rhs);
                            self.__mul(rhs)
                        }
                        #[inline(always)]
                        fn _max(self, rhs: #rhs_dtype) -> Self::Output {
                            let rhs = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(rhs);
                            self.__max(rhs)
                        }
                        #[inline(always)]
                        fn _min(self, rhs: #rhs_dtype) -> Self::Output {
                            let rhs = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(rhs);
                            self.__min(rhs)
                        }
                        #[inline(always)]
                        fn _rem(self, rhs: #rhs_dtype) -> Self::Output {
                            let rhs = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(rhs);
                            self.__rem(rhs)
                        }
                        #[inline(always)]
                        fn _clamp(self, min: #rhs_dtype, max: #rhs_dtype) -> Self::Output {
                            let min = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(min);
                            let max = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(max);
                            self.__clamp(min, max)
                        }
                        #[inline(always)]
                        fn _mul_add(self, a: #rhs_dtype, b: #rhs_dtype) -> Self::Output {
                            let a = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(a);
                            let b = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(b);
                            self.__mul_add(a, b)
                        }
                    }
                }
            } else {
                quote! {
                    impl NormalOut<#rhs_dtype> for #lhs_simd {
                        type Output = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output;
                        #[inline(always)]
                        fn _add(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs: <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output = self.into_vec();
                            let rhs = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(rhs.cast());
                            lhs.__add(rhs)
                        }
                        #[inline(always)]
                        fn _sub(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs: <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output = self.into_vec();
                            let rhs = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(rhs.cast());
                            lhs.__sub(rhs)
                        }
                        #[inline(always)]
                        fn _mul(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs: <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output = self.into_vec();
                            let rhs = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(rhs.cast());
                            lhs.__mul(rhs)
                        }
                        #[inline(always)]
                        fn _max(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs: <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output = self.into_vec();
                            let rhs = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(rhs.cast());
                            lhs.__max(rhs)
                        }
                        #[inline(always)]
                        fn _min(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs: <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output = self.into_vec();
                            let rhs = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(rhs.cast());
                            lhs.__min(rhs)
                        }
                        #[inline(always)]
                        fn _rem(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs: <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output = self.into_vec();
                            let rhs = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(rhs.cast());
                            lhs.__rem(rhs)
                        }
                        #[inline(always)]
                        fn _clamp(self, min: #rhs_dtype, max: #rhs_dtype) -> Self::Output {
                            let min = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(min.cast());
                            let max = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(max.cast());
                            let lhs: <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output = self.into_vec();
                            lhs.__clamp(min, max)
                        }
                        #[inline(always)]
                        fn _mul_add(self, a: #rhs_dtype, b: #rhs_dtype) -> Self::Output {
                            let a = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(a.cast());
                            let b = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(b.cast());
                            let lhs: <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output = self.into_vec();
                            lhs.__mul_add(a, b)
                        }
                    }
                }
            };
            ret.extend(res);
        }
    }

    ret.into()
}

pub fn impl_simd_normal_out_with_lhs_scalar() -> TokenStream {
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
            let rhs_simd = (*rhs).into();
            let lhs_simd = (*lhs).into();
            if lhs_lanes != rhs_lanes {
                ret.extend(impl_unreachable_lhs_scalar(lhs_dtype, rhs_simd, lhs_simd));
                continue;
            }

            let res = if lhs_type.dtype == rhs_type.dtype {
                quote! {
                    impl NormalOut<#rhs_simd> for #lhs_dtype {
                        type Output = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output;
                        #[inline(always)]
                        fn _add(self, rhs: #rhs_simd) -> Self::Output {
                            let lhs = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(self);
                            lhs.__add(rhs)
                        }
                        #[inline(always)]
                        fn _sub(self, rhs: #rhs_simd) -> Self::Output {
                            let lhs = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(self);
                            lhs.__sub(rhs)
                        }
                        #[inline(always)]
                        fn _mul(self, rhs: #rhs_simd) -> Self::Output {
                            let lhs = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(self);
                            lhs.__mul(rhs)
                        }
                        #[inline(always)]
                        fn _max(self, rhs: #rhs_simd) -> Self::Output {
                            let lhs = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(self);
                            lhs.__max(rhs)
                        }
                        #[inline(always)]
                        fn _min(self, rhs: #rhs_simd) -> Self::Output {
                            let lhs = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(self);
                            lhs.__min(rhs)
                        }
                        #[inline(always)]
                        fn _rem(self, rhs: #rhs_simd) -> Self::Output {
                            let lhs = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(self);
                            lhs.__rem(rhs)
                        }
                        #[inline(always)]
                        fn _clamp(self, min: #rhs_simd, max: #rhs_simd) -> Self::Output {
                            let lhs = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(self);
                            lhs.__clamp(min, max)
                        }
                        #[inline(always)]
                        fn _mul_add(self, a: #rhs_simd, b: #rhs_simd) -> Self::Output {
                            let lhs = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(self);
                            lhs.__mul_add(a, b)
                        }
                    }
                }
            } else {
                quote! {
                    impl NormalOut<#rhs_simd> for #lhs_dtype {
                        type Output = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output;
                        #[inline(always)]
                        fn _add(self, rhs: #rhs_simd) -> Self::Output {
                            let lhs = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(self.cast());
                            lhs.__add(rhs.into_vec())
                        }
                        #[inline(always)]
                        fn _sub(self, rhs: #rhs_simd) -> Self::Output {
                            let lhs = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(self.cast());
                            lhs.__sub(rhs.into_vec())
                        }
                        #[inline(always)]
                        fn _mul(self, rhs: #rhs_simd) -> Self::Output {
                            let lhs = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(self.cast());
                            lhs.__mul(rhs.into_vec())
                        }
                        #[inline(always)]
                        fn _max(self, rhs: #rhs_simd) -> Self::Output {
                            let lhs = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(self.cast());
                            lhs.__max(rhs.into_vec())
                        }
                        #[inline(always)]
                        fn _min(self, rhs: #rhs_simd) -> Self::Output {
                            let lhs = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(self.cast());
                            lhs.__min(rhs.into_vec())
                        }
                        #[inline(always)]
                        fn _rem(self, rhs: #rhs_simd) -> Self::Output {
                            let lhs = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(self.cast());
                            lhs.__rem(rhs.into_vec())
                        }
                        #[inline(always)]
                        fn _clamp(self, min: #rhs_simd, max: #rhs_simd) -> Self::Output {
                            let lhs = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(self.cast());
                            lhs.__clamp(min.into_vec(), max.into_vec())
                        }
                        #[inline(always)]
                        fn _mul_add(self, a: #rhs_simd, b: #rhs_simd) -> Self::Output {
                            let lhs = <#lhs_simd as NormalOutPromote<#rhs_simd>>::Output::splat(self.cast());
                            lhs.__mul_add(a.into_vec(), b.into_vec())
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
    lhs_type: SimdType,
) -> TokenStream2 {
    quote! {
        impl NormalOut<#rhs_simd> for #lhs_dtype {
            type Output = <#lhs_type as NormalOutPromote<#rhs_simd>>::Output;
            fn _mul_add(self, a: #rhs_simd, b: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _add(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _sub(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _mul(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _rem(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _clamp(self, min: #rhs_simd, max: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _max(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
            fn _min(self, rhs: #rhs_simd) -> Self::Output {
                unreachable!()
            }
        }
    }
}
