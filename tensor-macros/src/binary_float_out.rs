use proc_macro::TokenStream;
use quote::quote;
use crate::type_utils::TypeInfo;

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
            let res_type = lhs_type.infer_float_res_type(&rhs_type);
            let to_res_type = proc_macro2::Ident::new(
                &format!("to_{}", res_type.to_string().to_lowercase()),
                proc_macro2::Span::call_site(),
            );
            let log_body = if rhs_dtype.is_cplx() {
                quote! {
                    panic!("Cannot take the log of a complex number")
                }
            } else {
                if res_type.is_cplx32() {
                    quote! {
                        self.#to_res_type().log(base.to_f32())
                    }
                } else if res_type.is_cplx64() {
                    quote! {
                        self.#to_res_type().log(base.to_f64())
                    }
                } else {
                    quote! {
                        self.#to_res_type().log(base.#to_res_type())
                    }
                }
            };
            let res =
                quote! {
                impl FloatOutBinary<#rhs_dtype> for #lhs_dtype {
                    type Output = #res_type;
                
                    fn _div(self, rhs: #rhs_dtype) -> Self::Output {
                            self.#to_res_type() / rhs.#to_res_type()
                    }
                    fn _log(self, base: #rhs_dtype) -> Self::Output {
                        #log_body
                    }
                }
            };
            ret.extend(res);
        }
    }

    ret.into()
}
