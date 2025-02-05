use crate::type_utils::TypeInfo;
use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;

pub fn __impl_from_scalar() -> TokenStream {
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
        let lhs_dtype = TypeInfo::new(lhs);
        let lhs_ty = lhs_dtype.dtype;
        for rhs in types.iter() {
            let rhs_dtype = TypeInfo::new(rhs);
            let func_name = format!("to_{}", lhs.to_lowercase());
            let function_name: Ident = Ident::new(&func_name, proc_macro2::Span::call_site());
            let rhs_ty = rhs_dtype.dtype;
            let func_gen = quote! {
                impl FromScalar<#rhs_ty> for #lhs_ty {
                    #[inline(always)]
                    fn _from(a: #rhs_ty) -> Self {
                        a.#function_name()
                    }
                }
            };
            ret.extend(func_gen);
        }
    }

    ret.into()
}
