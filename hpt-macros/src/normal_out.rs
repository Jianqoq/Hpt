use crate::type_utils::TypeInfo;
use proc_macro::TokenStream;
use quote::quote;

pub(crate) fn __impl_normal_out_binary() -> TokenStream {
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
        for rhs in types.iter() {
            let lhs_type = TypeInfo::new(lhs);
            let rhs_type = TypeInfo::new(rhs);
            let lhs_dtype = lhs_type.dtype;
            let rhs_dtype = rhs_type.dtype;
            let res = if lhs_dtype == rhs_dtype {
                quote! {
                    impl NormalOut<#rhs_dtype> for #lhs_dtype {
                        type Output = <Self as NormalOutPromote<#rhs_dtype>>::Output;
                        #[inline(always)]
                        fn _clamp(self, min: #rhs_dtype, max: #rhs_dtype) -> Self::Output {
                            self.__clamp(min, max)
                        }
                        #[inline(always)]
                        fn _mul_add(self, a: #rhs_dtype, b: #rhs_dtype) -> Self::Output {
                            self.__mul_add(a, b)
                        }
                        #[inline(always)]
                        fn _add(self, rhs: #rhs_dtype) -> Self::Output {
                            self.__add(rhs)
                        }
                        #[inline(always)]
                        fn _sub(self, rhs: #rhs_dtype) -> Self::Output {
                            self.__sub(rhs)
                        }
                        #[inline(always)]
                        fn _mul(self, rhs: #rhs_dtype) -> Self::Output {
                            self.__mul(rhs)
                        }
                        #[inline(always)]
                        fn _rem(self, rhs: #rhs_dtype) -> Self::Output {
                            self.__rem(rhs)
                        }
                        #[inline(always)]
                        fn _max(self, rhs: #rhs_dtype) -> Self::Output {
                            self.__max(rhs)
                        }
                        #[inline(always)]
                        fn _min(self, rhs: #rhs_dtype) -> Self::Output {
                            self.__min(rhs)
                        }
                    }
                }
            } else {
                quote! {
                    impl NormalOut<#rhs_dtype> for #lhs_dtype {
                        type Output = <Self as NormalOutPromote<#rhs_dtype>>::Output;
                        #[inline(always)]
                        fn _clamp(self, min: #rhs_dtype, max: #rhs_dtype) -> Self::Output {
                            let lhs_scalar: Self::Output = self.cast();
                            let min_scalar: Self::Output = min.cast();
                            let max_scalar: Self::Output = max.cast();
                            lhs_scalar.__clamp(min_scalar, max_scalar)
                        }
                        #[inline(always)]
                        fn _mul_add(self, a: #rhs_dtype, b: #rhs_dtype) -> Self::Output {
                            let lhs_scalar: Self::Output = self.cast();
                            let a_scalar: Self::Output = a.cast();
                            let b_scalar: Self::Output = b.cast();
                            lhs_scalar.__mul_add(a_scalar, b_scalar)
                        }
                        #[inline(always)]
                        fn _add(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs_scalar: Self::Output = self.cast();
                            let rhs_scalar: Self::Output = rhs.cast();
                            lhs_scalar.__add(rhs_scalar)
                        }
                        #[inline(always)]
                        fn _sub(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs_scalar: Self::Output = self.cast();
                            let rhs_scalar: Self::Output = rhs.cast();
                            lhs_scalar.__sub(rhs_scalar)
                        }
                        #[inline(always)]
                        fn _mul(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs_scalar: Self::Output = self.cast();
                            let rhs_scalar: Self::Output = rhs.cast();
                            lhs_scalar.__mul(rhs_scalar)
                        }
                        #[inline(always)]
                        fn _rem(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs_scalar: Self::Output = self.cast();
                            let rhs_scalar: Self::Output = rhs.cast();
                            lhs_scalar.__rem(rhs_scalar)
                        }
                        #[inline(always)]
                        fn _max(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs_scalar: Self::Output = self.cast();
                            let rhs_scalar: Self::Output = rhs.cast();
                            lhs_scalar.__max(rhs_scalar)
                        }
                        #[inline(always)]
                        fn _min(self, rhs: #rhs_dtype) -> Self::Output {
                            let lhs_scalar: Self::Output = self.cast();
                            let rhs_scalar: Self::Output = rhs.cast();
                            lhs_scalar.__min(rhs_scalar)
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
pub(crate) fn __impl_cuda_normal_out_binary() -> TokenStream {
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
        for rhs in types.iter() {
            let lhs_type = TypeInfo::new(lhs);
            let rhs_type = TypeInfo::new(rhs);
            let lhs_dtype = lhs_type.dtype;
            let rhs_dtype = rhs_type.dtype;
            let res = if lhs_dtype == rhs_dtype {
                quote! {
                    impl NormalOut<Scalar<#rhs_dtype>> for Scalar<#lhs_dtype> {
                        type Output = <Self as NormalOutPromote<Scalar<#rhs_dtype>>>::Output;
                        #[inline(always)]
                        fn _clamp(self, min: Scalar<#rhs_dtype>, max: Scalar<#rhs_dtype>) -> Self::Output {
                            self.__clamp(min, max)
                        }
                        #[inline(always)]
                        fn _mul_add(self, a: Scalar<#rhs_dtype>, b: Scalar<#rhs_dtype>) -> Self::Output {
                            self.__mul_add(a, b)
                        }
                        #[inline(always)]
                        fn _add(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            self.__add(rhs)
                        }
                        #[inline(always)]
                        fn _sub(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            self.__sub(rhs)
                        }
                        #[inline(always)]
                        fn _mul(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            self.__mul(rhs)
                        }
                        #[inline(always)]
                        fn _rem(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            self.__rem(rhs)
                        }
                        #[inline(always)]
                        fn _max(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            self.__max(rhs)
                        }
                        #[inline(always)]
                        fn _min(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            self.__min(rhs)
                        }
                    }
                }
            } else {
                quote! {
                    impl NormalOut<Scalar<#rhs_dtype>> for Scalar<#lhs_dtype> {
                        type Output = <Self as NormalOutPromote<Scalar<#rhs_dtype>>>::Output;
                        #[inline(always)]
                        fn _clamp(self, min: Scalar<#rhs_dtype>, max: Scalar<#rhs_dtype>) -> Self::Output {
                            let lhs_scalar: Self::Output = self.cast();
                            let min_scalar: Self::Output = min.cast();
                            let max_scalar: Self::Output = max.cast();
                            lhs_scalar.__clamp(min_scalar, max_scalar)
                        }
                        #[inline(always)]
                        fn _mul_add(self, a: Scalar<#rhs_dtype>, b: Scalar<#rhs_dtype>) -> Self::Output {
                            let lhs_scalar: Self::Output = self.cast();
                            let a_scalar: Self::Output = a.cast();
                            let b_scalar: Self::Output = b.cast();
                            lhs_scalar.__mul_add(a_scalar, b_scalar)
                        }
                        #[inline(always)]
                        fn _add(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            let lhs_scalar: Self::Output = self.cast();
                            let rhs_scalar: Self::Output = rhs.cast();
                            lhs_scalar.__add(rhs_scalar)
                        }
                        #[inline(always)]
                        fn _sub(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            let lhs_scalar: Self::Output = self.cast();
                            let rhs_scalar: Self::Output = rhs.cast();
                            lhs_scalar.__sub(rhs_scalar)
                        }
                        #[inline(always)]
                        fn _mul(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            let lhs_scalar: Self::Output = self.cast();
                            let rhs_scalar: Self::Output = rhs.cast();
                            lhs_scalar.__mul(rhs_scalar)
                        }
                        #[inline(always)]
                        fn _rem(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            let lhs_scalar: Self::Output = self.cast();
                            let rhs_scalar: Self::Output = rhs.cast();
                            lhs_scalar.__rem(rhs_scalar)
                        }
                        #[inline(always)]
                        fn _max(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            let lhs_scalar: Self::Output = self.cast();
                            let rhs_scalar: Self::Output = rhs.cast();
                            lhs_scalar.__max(rhs_scalar)
                        }
                        #[inline(always)]
                        fn _min(self, rhs: Scalar<#rhs_dtype>) -> Self::Output {
                            let lhs_scalar: Self::Output = self.cast();
                            let rhs_scalar: Self::Output = rhs.cast();
                            lhs_scalar.__min(rhs_scalar)
                        }
                    }
                }
            };
            ret.extend(res);
        }
    }

    ret.into()
}
