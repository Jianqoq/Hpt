use proc_macro::TokenStream;
use syn::{ parse, parse_macro_input, Ident, Token };
use quote::quote;

struct Params {
    vec_fn: Ident,
    scalar_fn: Ident,
}
    
impl parse::Parse for Params {
    fn parse(input: parse::ParseStream) -> syn::Result<Self> {
        let vec_fn = input.parse::<Ident>().expect("expect a fn");
        input.parse::<Token![,]>()?;
        let scalar_fn = input.parse::<Ident>().expect("expect a fn");
        Ok(Self { vec_fn, scalar_fn })
    }
}

pub(crate) fn conv2d_helper(input: TokenStream) -> TokenStream {
    let inputs = parse_macro_input!(input as Params);
    let vec_fn = &inputs.vec_fn;
    let scalar_fn = &inputs.scalar_fn;
    (
        quote! {
        for j in (0..out_channels).step_by(T::Vec::SIZE * OC_NVEC) {
            if j + (T::Vec::SIZE * OC_NVEC) as i64 > out_channels {
                continue;
            }
            for l in ll..l_end {
                #vec_fn::<T, OC_NVEC>(
                    [ii, i_end],
                    [kernel_height, kernel_width],
                    [b, l, k, j],
                    [osb, osh, osw],
                    [step_height, step_width],
                    [isb, ish, isw],
                    [ks0, ks1, ks2],
                    &mut out,
                    &inp,
                    &kernel
                );
            }
        }
        let mut remain = oc_remain;
        if oc_remain > (T::Vec::SIZE as i64) {
            let j_start = out_channels - oc_remain;
            for j in (j_start..out_channels).step_by(T::Vec::SIZE) {
                for l in ll..l_end {
                    #vec_fn::<T, OC_NVEC>(
                        [ii, i_end],
                        [kernel_height, kernel_width],
                        [b, l, k, j],
                        [osb, osh, osw],
                        [step_height, step_width],
                        [isb, ish, isw],
                        [ks0, ks1, ks2],
                        &mut out,
                        &inp,
                        &kernel
                    );
                }
            }
            remain = oc_remain % (T::Vec::SIZE as i64);
        }
        let j_start = out_channels - remain;
        for j in j_start..out_channels {
            for l in ll..l_end {
                #scalar_fn::<T>(
                    [ii, i_end],
                    [kernel_height, kernel_width],
                    [b, l, k, j],
                    [osb, osh, osw],
                    [step_height, step_width],
                    [isb, ish, isw],
                    [ks0, ks1, ks2],
                    remain,
                    &mut out,
                    &inp,
                    &kernel
                );
            }
        }
    }
    ).into()
}
