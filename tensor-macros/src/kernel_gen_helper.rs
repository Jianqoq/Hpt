use proc_macro::TokenStream;
use syn::parse_macro_input;
use quote::quote;
use syn::Ident;

pub fn __gen_fast_reduce_simd_helper(stream: TokenStream) -> TokenStream {
    let input = parse_macro_input!(stream as Ident);

    #[cfg(target_feature = "avx2")]
    let num_registers = 16;
    #[cfg(all(target_feature = "sse", not(target_feature = "avx2")))]
    let num_registers = 8;
    #[cfg(target_feature = "avx512f")]
    let num_registers = 32;

    let mut body = proc_macro2::TokenStream::new();
    for i in 0..num_registers as isize {
        let i = i + 1;
        let elements = 1..=i;
        let arr = quote! {
            [ #(#elements),* ]
        };
        let i_u32 = i as u32;
        body.extend(
            quote! {
            #i_u32 => {
                gen_kernel!(1, #i, inp_ptr, res_ptr, vec_size, outer_loop_size, vec_op, inp_strides, inp_shape, prg, #arr);
            }
        }
        );
    }
    (quote! {
        match #input {
            #body
            _ => unreachable!()
        }
    }).into()
}
