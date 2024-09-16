use proc_macro::TokenStream;
use syn::parse_macro_input;
use quote::quote;
use syn::Ident;

pub fn __gen_fast_reduce_simd_helper(stream: TokenStream) -> TokenStream {
    let input = parse_macro_input!(stream as Ident);

    #[cfg(target_feature = "avx2")]
    let num_registers = 16;
    #[cfg(all(any(target_feature = "sse", target_feature = "neon"), not(target_feature = "avx2")))]
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
                gen_kernel!(1, #i, inp_ptr, res_ptr, vec_size, outer_loop_size, vec_op, inp_strides, inp_shape, prg, vec_op_post, #arr);
            }
        }
        );
    }
    quote! {
        match #input {
            #body
            0 => {}
            _ => unreachable!()
        }
    }.into()
}

pub fn __gen_reduce_dim_not_include_simd_helper(stream: TokenStream) -> TokenStream {
    let input = parse_macro_input!(stream as Ident);

    #[cfg(target_feature = "avx2")]
    let num_registers = 16;
    #[cfg(all(any(target_feature = "sse", target_feature = "neon"), not(target_feature = "avx2")))]
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
                gen_kernel3!(
                    1,
                    #i,
                    outer_loop_size,
                    inp_ptr,
                    res_ptr,
                    <O as TypeCommon>::Vec::SIZE as isize,
                    intermediate_size,
                    vec_op,
                    inp_strides,
                    inp_shape,
                    prg1,
                    prg2,
                    shape_len,
                    inner_loop_size,
                    #arr
                );
            }
        }
        );
    }
    quote! {
        match #input {
            #body,
            0 => {}
            _ => unreachable!()
        }
    }.into()
}
