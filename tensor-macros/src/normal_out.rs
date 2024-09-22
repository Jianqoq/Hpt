use crate::type_utils::{Type, TypeInfo};
use proc_macro::TokenStream;
use proc_macro2::Ident;
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
            let res_type = lhs_type.infer_normal_res_type(&rhs_type);
            let res_type_ident = Ident::new(
                &res_type.to_string().to_lowercase(),
                proc_macro2::Span::call_site(),
            );
            let to_res_type = Ident::new(
                &format!("to_{}", res_type.to_string().to_lowercase()),
                proc_macro2::Span::call_site(),
            );
            let mul_add_method = mul_add(rhs_dtype, res_type, to_res_type.clone());
            let pow_method = pow(rhs_dtype, res_type, to_res_type.clone());
            let cmp_method = cmp(rhs_dtype, res_type, to_res_type.clone());
            let std_ops = std_ops(rhs_dtype, res_type, to_res_type.clone());

            let clamp = if res_type.is_cplx() {
                quote! {
                    #[inline(always)]
                    fn _clip(self, min: Self::Output, max: Self::Output) -> Self::Output {
                        paste::paste! {
                            let c = self.[<to_ #res_type_ident>]();
                            let min = min.[<to_ #res_type_ident>]();
                            let max = max.[<to_ #res_type_ident>]();
                            let clamped_re = c.re.clamp(min.re, max.re);
                            let clamped_im = c.im.clamp(min.im, max.im);
                            #res_type::new(clamped_re, clamped_im)
                        }
                    }
                }
            } else {
                quote! {
                    #[inline(always)]
                    fn _clip(self, min: Self::Output, max: Self::Output) -> Self::Output {
                        paste::paste! {
                            let a = self.[<to_ #res_type_ident>]();
                            let min = min.[<to_ #res_type_ident>]();
                            let max = max.[<to_ #res_type_ident>]();
                            if a < min { min } else if a > max { max } else { a }
                        }
                    }
                }
            };

            let res = quote! {
                impl NormalOut<#rhs_dtype> for #lhs_dtype {
                    type Output = #res_type;
                    #pow_method
                    #clamp
                    #mul_add_method
                    #std_ops
                    #cmp_method
                }
            };
            ret.extend(res);
        }
    }

    ret.into()
}

fn mul_add(rhs_type: Type, res_type: Type, to_res_type: Ident) -> proc_macro2::TokenStream {
    let mul_add_body = if res_type.is_float() {
        quote! {
            self.#to_res_type() * a.#to_res_type() + b.#to_res_type()
        }
    } else if res_type.is_bool() {
        quote! {
            self || a && b
        }
    } else if res_type.is_cplx() {
        quote! {
            self.#to_res_type() * a.#to_res_type() + b.#to_res_type()
        }
    } else {
        quote! {
                self.#to_res_type().wrapping_mul(a.#to_res_type()) + b.#to_res_type()
        }
    };
    quote! {
        #[inline(always)]
        fn _mul_add(self, a: #rhs_type, b: #rhs_type) -> Self::Output {
            #mul_add_body
        }
    }
}

fn pow(rhs_dtype: Type, res_type: Type, to_res_type: Ident) -> proc_macro2::TokenStream {
    let pow_body = if res_type.is_float() {
        quote! {
            self.#to_res_type().powf(rhs.#to_res_type())
        }
    } else if res_type.is_cplx() {
        quote! {
            self.#to_res_type().powc(rhs.#to_res_type())
        }
    } else {
        if res_type.is_bool() {
            quote! {
                self || rhs
            }
        } else {
            quote! {
                self.#to_res_type().pow(rhs.to_u32())
            }
        }
    };
    quote! {
        #[inline(always)]
        fn _pow(self, rhs: #rhs_dtype) -> Self::Output {
            #pow_body
        }
    }
}

fn cmp(rhs_dtype: Type, res_type: Type, to_res_type: Ident) -> proc_macro2::TokenStream {
    let (max_body, min_body) = if res_type.is_bool() {
        let max_body = quote! {
            self | rhs
        };
        let min_body = quote! {
            self & rhs
        };
        (max_body, min_body)
    } else if res_type.is_cplx() {
        let max_body = quote! {
            panic!("max method is not supported for complex number")
        };
        let min_body = quote! {
            panic!("min method is not supported for complex number")
        };
        (max_body, min_body)
    } else {
        let max_body = quote! {
            self.#to_res_type().max(rhs.#to_res_type())
        };
        let min_body = quote! {
            self.#to_res_type().min(rhs.#to_res_type())
        };
        (max_body, min_body)
    };
    quote! {
        #[inline(always)]
        fn _max(self, rhs: #rhs_dtype) -> Self::Output {
            #max_body
        }
        #[inline(always)]
        fn _min(self, rhs: #rhs_dtype) -> Self::Output {
            #min_body
        }
    }
}

fn std_ops(rhs_dtype: Type, res_type: Type, to_res_type: Ident) -> proc_macro2::TokenStream {
    let (add, sub, mul, rem) = if res_type.is_bool() {
        let add = quote! {
            let res = self.to_i8() + rhs.to_i8();
            res != 0
        };
        let sub = quote! {
            let res = self.to_i8() - rhs.to_i8();
            res != 0
        };
        let mul = quote! {
            let res = self.to_i8() * rhs.to_i8();
            res != 0
        };
        let rem = quote! {
            let res = self.to_i8() % rhs.to_i8();
            res != 0
        };
        (add, sub, mul, rem)
    } else {
        let add = quote! {
            self.#to_res_type() + rhs.#to_res_type()
        };
        let sub = quote! {
            self.#to_res_type() - rhs.#to_res_type()
        };
        let mul = quote! {
            self.#to_res_type() * rhs.#to_res_type()
        };
        let rem = quote! {
            self.#to_res_type() % rhs.#to_res_type()
        };
        (add, sub, mul, rem)
    };
    quote! {
        #[inline(always)]
        fn _add(self, rhs: #rhs_dtype) -> Self::Output {
            #add
        }
        #[inline(always)]
        fn _sub(self, rhs: #rhs_dtype) -> Self::Output {
            #sub
        }
        #[inline(always)]
        fn _mul(self, rhs: #rhs_dtype) -> Self::Output {
            #mul
        }
        #[inline(always)]
        fn _rem(self, rhs: #rhs_dtype) -> Self::Output {
            #rem
        }
    }
}
