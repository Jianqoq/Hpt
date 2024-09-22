use crate::type_utils::{Type, TypeInfo};
use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;

pub(crate) fn __impl_normal_out_unary() -> TokenStream {
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
        let lhs_type = TypeInfo::new(lhs);
        let lhs_dtype = lhs_type.dtype;
        let neg_method = neg(lhs_dtype);
        let abs_method = abs(lhs_dtype);
        let ceil_method = ceil_floor_round(lhs_dtype, 1);
        let floor_method = ceil_floor_round(lhs_dtype, 2);
        let sign_method = sign(lhs_dtype);
        let round_method = ceil_floor_round(lhs_dtype, 0);

        let res = quote! {
            impl NormalOutUnary for #lhs_dtype {
                #[inline(always)]
                fn _square(self) -> Self {
                    self._mul(self)
                }
                #neg_method
                #abs_method
                #ceil_method
                #floor_method
                #sign_method
                #round_method
            }
        };
        ret.extend(res);
    }

    ret.into()
}

fn neg(lhs_dtype: Type) -> proc_macro2::TokenStream {
    let neg_body = if lhs_dtype.is_float() {
        quote! {
            -self
        }
    } else if lhs_dtype.is_bool() {
        quote! {
            !self
        }
    } else if lhs_dtype.is_unsigned() {
        quote! {
            !self + 1
        }
    } else {
        quote! {
            -self
        }
    };
    quote! {
        #[inline(always)]
        fn _neg(self) -> Self {
            #neg_body
        }
    }
}

fn abs(lhs_dtype: Type) -> proc_macro2::TokenStream {
    let abs_body = if lhs_dtype.is_unsigned() {
        quote! {
            self
        }
    } else if lhs_dtype.is_cplx() {
        quote! {
                panic!("abs method is not supported for complex number")
        }
    } else {
        quote! {
            self.abs()
        }
    };
    quote! {
        #[inline(always)]
        fn _abs(self) -> Self {
            #abs_body
        }
    }
}

fn ceil_floor_round(lhs_dtype: Type, mode: u8) -> proc_macro2::TokenStream {
    let ceil_body = if lhs_dtype.is_float() {
        match mode {
            0 => quote! {
                self.round()
            },
            1 => quote! {
                self.ceil()
            },
            2 => quote! {
                self.floor()
            },
            _ => unreachable!(),
        }
    } else {
        quote! {
            self
        }
    };
    let method = match mode {
        0 => "_round",
        1 => "_ceil",
        2 => "_floor",
        _ => unreachable!(),
    };
    let method_ident = Ident::new(method, proc_macro2::Span::call_site());
    quote! {
        #[inline(always)]
        fn #method_ident(self) -> Self {
            #ceil_body
        }
    }
}

fn sign(res_type: Type) -> proc_macro2::TokenStream {
    let sign_body = if res_type.is_float() {
        quote! {
            self.signum()
        }
    } else if res_type.is_cplx() {
        quote! {
            panic!("sign method is not supported for complex number")
        }
    } else if res_type.is_unsigned() {
        quote! {
            #res_type::ZERO
        }
    } else {
        quote! {
            self.signum()
        }
    };
    quote! {
        #[inline(always)]
        fn _sign(self) -> Self {
            #sign_body
        }
    }
}