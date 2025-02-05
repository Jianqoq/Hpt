use proc_macro2::TokenStream;

use crate::type_utils::TypeInfo;

pub(crate) fn __impl_into_scalar() -> TokenStream {
    let mut ret = proc_macro2::TokenStream::new();

    let types = [
        "bool",
        "bf16",
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
        "isize",
        "usize",
        "complex32",
        "complex64",
    ];

    for lhs in types.iter() {
        for rhs in types.iter() {
            let lhs_type = TypeInfo::new(lhs);
            let lhs_dtype = lhs_type.dtype;
            let rhs_type = TypeInfo::new(rhs);
            let rhs_dtype = rhs_type.dtype;
            let into_method =
                syn::Ident::new(&format!("to_{}", rhs), proc_macro2::Span::call_site());
            ret.extend(quote::quote! {
                impl Cast<#rhs_dtype> for #lhs_dtype {
                    fn cast(self) -> #rhs_dtype {
                        self.#into_method()
                    }
                }
            });
        }
    }

    ret.into()
}
