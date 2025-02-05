use crate::type_utils::TypeInfo;
use proc_macro::TokenStream;
use quote::quote;

pub fn impl_float_out_binary() -> TokenStream {
    let mut ret = proc_macro2::TokenStream::new();

    let types = [
        "bool",
        "f16",
        "bf16",
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
        "isize",
        "usize",
        "Complex32",
        "Complex64",
    ];

    for lhs in types.iter() {
        for rhs in types.iter() {
            let lhs_type = TypeInfo::new(lhs);
            let rhs_type = TypeInfo::new(rhs);
            let lhs_dtype = lhs_type.dtype;
            let rhs_dtype = rhs_type.dtype;
            let res = if lhs_dtype == rhs_dtype
                && ((lhs_dtype.is_float() || lhs_dtype.is_cplx())
                    || (rhs_dtype.is_float() || rhs_dtype.is_cplx()))
            {
                quote! {
                    impl FloatOutBinary<#rhs_dtype> for #lhs_dtype {
                        type Output = <#lhs_dtype as FloatOutBinaryPromote<#rhs_dtype>>::Output;

                        fn _div(self, rhs: #rhs_dtype) -> Self::Output {
                            self.__div(rhs)
                        }
                        fn _log(self, base: #rhs_dtype) -> Self::Output {
                            self.__log(base)
                        }
                    }
                }
            } else {
                quote! {
                    impl FloatOutBinary<#rhs_dtype> for #lhs_dtype {
                        type Output = <#lhs_dtype as FloatOutBinaryPromote<#rhs_dtype>>::Output;

                        fn _div(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs_scalar: Self::Output = self.cast();
                            let rhs_scalar: Self::Output = rhs.cast();
                            lhs_scalar / rhs_scalar
                        }
                        fn _log(self, base: #rhs_dtype) -> Self::Output {
                            let lhs_scalar: Self::Output = self.cast();
                            let base_scalar: Self::Output = base.cast();
                            lhs_scalar.__log(base_scalar)
                        }
                    }
                }
            };
            ret.extend(res);
        }
    }

    ret.into()
}

#[cfg(feature = "cuda")]
pub fn impl_cuda_float_out_binary() -> TokenStream {
    let mut ret = proc_macro2::TokenStream::new();

    let types = [
        "bool",
        "f16",
        "bf16",
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
        "isize",
        "usize",
        "Complex32",
        "Complex64",
    ];

    for lhs in types.iter() {
        for rhs in types.iter() {
            let lhs_type = TypeInfo::new(lhs);
            let rhs_type = TypeInfo::new(rhs);
            let lhs_dtype = lhs_type.dtype;
            let rhs_dtype = rhs_type.dtype;
            let res = if lhs_dtype == rhs_dtype
                && ((lhs_dtype.is_float() || lhs_dtype.is_cplx())
                    || (rhs_dtype.is_float() || rhs_dtype.is_cplx()))
            {
                quote! {
                    impl FloatOutBinary<Scalar<#rhs_dtype>> for Scalar<#lhs_dtype> {
                        type Output = <Scalar<#lhs_dtype> as FloatOutBinaryPromote<Scalar<#rhs_dtype>>>::Output;

                        fn _div(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            self.__div(rhs)
                        }
                        fn _log(self, base: Scalar<#rhs_dtype>) -> Self::Output {
                            self.__log(base)
                        }
                    }
                }
            } else {
                quote! {
                    impl FloatOutBinary<Scalar<#rhs_dtype>> for Scalar<#lhs_dtype> {
                        type Output = <Scalar<#lhs_dtype> as FloatOutBinaryPromote<Scalar<#rhs_dtype>>>::Output;

                        fn _div(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            let lhs_scalar: Self::Output = self.cast();
                            let rhs_scalar: Self::Output = rhs.cast();
                            lhs_scalar.__div(rhs_scalar)
                        }
                        fn _log(self, base: Scalar<#rhs_dtype>) -> Self::Output {
                            let lhs_scalar: Self::Output = self.cast();
                            let base_scalar: Self::Output = base.cast();
                            lhs_scalar.__log(base_scalar)
                        }
                    }
                }
            };
            ret.extend(res);
        }
    }

    ret.into()
}
