use std::collections::HashSet;
use petgraph::graph::NodeIndex;
use proc_macro2::TokenStream as TokenStream2;
use super::{ build_graph::CmpNode, fuse::{ FusionGroup, Input } };

pub(crate) fn cmp_gen_fuse(
    cfg: &crate::fuse::cfg::CFG,
    graph: &petgraph::stable_graph::StableGraph<CmpNode, ()>,
    groups: &FusionGroup
) -> Vec<(TokenStream2, TokenStream2)> {
    _cmp_gen_fuse(cfg, &graph, &groups)
}

fn cmp_gen_body(
    sorted: Vec<NodeIndex>,
    inputs: &HashSet<Input>,
    graph: &petgraph::stable_graph::StableGraph<CmpNode, ()>,
    cfg: &crate::fuse::cfg::CFG
) -> proc_macro2::TokenStream {
    let mut comp_tokens = proc_macro2::TokenStream::new();
    for idx in sorted {
        let mut node = graph[idx].clone();
        if !inputs.iter().any(|input| input.comp_graph_idx == idx) {
            let origin_ident = cfg.graph[NodeIndex::new(node.block_idx)].origin_var_map
                .get(&node.ident)
                .expect("gen_fuse::out");
            node.ident = origin_ident.clone();
            for (idx, out) in node.outputs_ident.clone().into_iter().enumerate() {
                let origin_out = cfg.graph[NodeIndex::new(node.block_idx)].origin_var_map
                    .get(&out)
                    .expect("gen_fuse::out");
                node.outputs_ident[idx] = origin_out.clone();
            }
            for (idx, inp) in node.args_ident.clone().into_iter().enumerate() {
                let origin_inp = cfg.graph[NodeIndex::new(node.block_idx)].origin_var_map
                    .get(&inp)
                    .expect("gen_fuse::out");
                node.args_ident[idx] = origin_inp.clone();
            }
            comp_tokens.extend(quote::quote!(
                #node
            ));
        }
    }
    // println!("comp_tokens: {:#?}", comp_tokens.to_token_stream().to_string());
    comp_tokens
}

pub(crate) fn _cmp_gen_fuse(
    cfg: &crate::fuse::cfg::CFG,
    graph: &petgraph::stable_graph::StableGraph<CmpNode, ()>,
    groups: &FusionGroup
) -> Vec<(TokenStream2, TokenStream2)> {
    // println!("graph: {:#?}", graph);
    let sorteds = petgraph::algo::toposort(graph, None).expect("gen_fuse::topological_sort");
    // println!("sorteds: {:#?}", sorteds);
    let mut tuple_vec = vec![];
    let inputs = groups.inputs
        .iter()
        .map(|inputs| {
            let mut v = vec![];
            for input in inputs {
                v.push(
                    cfg.graph[NodeIndex::new(input.block_idx)].origin_var_map
                        .get(&input.var)
                        .expect("gen_fuse::origin_ident")
                        .clone()
                );
            }
            v
        })
        .collect::<Vec<_>>();
    for mut iters in inputs.clone().iter_mut() {
        let tokens = gen_tuple(&mut iters);
        tuple_vec.push(quote::quote!(
            (res, #tokens)
        ));
    }

    let mut sorted_groups = Vec::new();

    for group in groups.groups.iter() {
        let mut v = vec![];
        for sorted in sorteds.iter() {
            if group.contains(sorted) {
                v.push(sorted.clone());
            }
        }
        sorted_groups.push(v);
    }
    let mut fused_vec = Vec::new();
    for (i, (sorted, (inputs, outputs))) in sorted_groups
        .into_iter()
        .zip(inputs.iter().zip(groups.outputs.iter()))
        .enumerate() {
        if outputs.len() != 1 {
            panic!("gen_fuse::output_len: {:?}", outputs.len());
        }
        let mut scalar_comp = TokenStream2::new();
        let mut vec_comp = TokenStream2::new();
        let output = &outputs.iter().next().expect("gen_fuse::output");
        let origin_output = cfg.graph[NodeIndex::new(output.block_idx)].origin_var_map
            .get(&output.var)
            .expect("gen_fuse::origin_output");
        // println!("sorted: {:#?}", sorted);
        let tokens = cmp_gen_body(sorted, &groups.inputs[i], graph, cfg);
        scalar_comp.extend(tokens.clone());
        scalar_comp.extend(quote::quote!(
            #origin_output
        ));
        vec_comp.extend(tokens);
        vec_comp.extend(quote::quote!(
            #origin_output
        ));
        let tuple = &tuple_vec[i];
        let func_name = quote::format_ident!("__fuse_group_{}", i);
        let func_args = inputs.iter().map(|input| {
            let ident = quote::format_ident!("{}", input);
            quote::quote!(&#ident)
        });
        let fused =
            quote::quote!(
                #func_name(#(#func_args,)*|#(#inputs),*| {
                    #scalar_comp
                },|#(#inputs),*| {
                    #vec_comp
                })?
            );
        let input_generics = inputs
            .iter()
            .map(|input| { quote::format_ident!("{}", input.to_string().to_uppercase()) });
        let closure_bounds = input_generics.clone();
        let closure_simd_bounds = inputs.iter().map(|input| {
            let generic_ident = quote::format_ident!("{}", input.to_string().to_uppercase());
            quote::quote!(<#generic_ident as TypeCommon>::Vec)
        });
        let input_args = inputs.iter().map(|input| {
            let arg_ident = quote::format_ident!("{}_arg", input);
            let generic_ident = quote::format_ident!("{}", input.to_string().to_uppercase());
            quote::quote! { #arg_ident: &Tensor<#generic_ident> }
        });
        let input_bounds = inputs.iter().map(|input| {
            let generic_ident = quote::format_ident!("{}", input.to_string().to_uppercase());
            quote::quote! { #generic_ident: CommonBounds }
        });
        let check_contiguous = inputs.iter().map(|input| {
            let lhs_ident = quote::format_ident!("{}_arg", input);
            quote::quote!(#lhs_ident.is_contiguous())
        });
        let check_shape_eq = inputs.iter().flat_map(|lhs| {
            inputs
                .iter()
                .filter(move |&rhs| lhs != rhs)
                .map(move |rhs| {
                    let lhs_ident = quote::format_ident!("{}_arg", lhs);
                    let rhs_ident = quote::format_ident!("{}_arg", rhs);
                    quote::quote!(#lhs_ident.shape() == #rhs_ident.shape())
                })
        });
        let vec_size_eq = inputs
            .iter()
            .map(|input| {
                let ident = quote::format_ident!("{}", input.to_string().to_uppercase());
                quote::quote!(<#ident as TypeCommon>::Vec::SIZE == <__HPTRES as TypeCommon>::Vec::SIZE)
            })
            .collect::<Vec<_>>();
        let zip_chunks = inputs
            .iter()
            .enumerate()
            .map(|(idx, input)| {
                let ident = quote::format_ident!("{}_arg", input);
                let generic_ident = quote::format_ident!("{}", input.to_string().to_uppercase());
                if idx == 0 {
                    quote::quote! { #ident.as_raw().par_chunks_exact(<#generic_ident as TypeCommon>::Vec::SIZE) }
                } else {
                    quote::quote! { .zip(#ident.as_raw().par_chunks_exact(<#generic_ident as TypeCommon>::Vec::SIZE)) }
                }
            });
        let zip_remain_scalars = inputs
            .iter()
            .enumerate()
            .map(|(idx, input)| {
                let ident = quote::format_ident!("{}_arg", input);
                if idx == 0 {
                    quote::quote! { #ident.as_raw()[ret_size - remain..].iter() }
                } else {
                    quote::quote! { .zip(#ident.as_raw()[ret_size - remain..].iter()) }
                }
            });
        let zip_min_len = inputs
            .iter()
            .enumerate()
            .map(|(idx, input)| {
                let ident = quote::format_ident!("{}_arg", input);
                if idx == 0 {
                    quote::quote! { #ident.as_raw().par_iter().with_min_len(min_len) }
                } else {
                    quote::quote! { .zip(#ident.as_raw().par_iter().with_min_len(min_len)) }
                }
            });
        let zip_scalar_par = inputs
            .iter()
            .enumerate()
            .map(|(idx, input)| {
                let ident = quote::format_ident!("{}_arg", input);
                if idx == 0 {
                    quote::quote! { #ident.par_iter() }
                } else {
                    quote::quote! { .zip(#ident.par_iter()) }
                }
            });
        let layout_broadcast = inputs
            .iter()
            .enumerate()
            .map(|(idx, input)| {
                let ident = quote::format_ident!("{}_arg", input);
                if idx == 0 {
                    quote::quote! { let mut layout = #ident.layout().clone(); }
                } else {
                    quote::quote! { layout = layout.broadcast(#ident.layout())?; }
                }
            });
        let from_ptr = inputs.iter().map(|input| {
            let ident = quote::format_ident!("{}", input);
            let generic_ident = quote::format_ident!("{}", input.to_string().to_uppercase());
            quote::quote!(let #ident = unsafe { <#generic_ident as TypeCommon>::Vec::from_ptr(#ident.as_ptr()) };)
        });
        let f2_args = inputs
            .iter()
            .map(|input| { quote::format_ident!("{}", input) })
            .collect::<Vec<_>>();
        let first = quote::format_ident!("{}_arg", inputs[0]);
        let func =
            quote::quote!(
            fn #func_name<#(#input_generics),*, __HPTRES, F, F2>(
                #(#input_args),*,
                f: F,
                f2: F2,
            ) -> anyhow::Result<Tensor<__HPTRES>>
            where
                #(#input_bounds),*,
                __HPTRES: CommonBounds,
                F: Fn(#(#closure_bounds),*) -> __HPTRES + Sync + Send + Copy,
                F2: Fn(#(#closure_simd_bounds),*) -> <__HPTRES as TypeCommon>::Vec
                    + Sync
                    + Send
                    + Copy,
            {
                if #(#check_shape_eq &&)* #(#check_contiguous)&&* {
                    let mut ret = Tensor::<__HPTRES, Cpu>::empty(#first.shape())?;
                    if #(#vec_size_eq) && * {
                        let remain = ret.size() % <__HPTRES as TypeCommon>::Vec::SIZE;
                        ret.as_raw_mut()
                            .par_chunks_exact_mut(<__HPTRES as TypeCommon>::Vec::SIZE)
                            .zip(#(#zip_chunks)*)
                            .for_each(|#tuple| {
                                #(#from_ptr)*
                                let ret = f2(#(#f2_args),*);
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        ret.as_ptr(),
                                        res.as_mut_ptr(),
                                        <__HPTRES as TypeCommon>::Vec::SIZE,
                                    );
                                }
                            });
                        if remain > 0 {
                            let ret_size = ret.size();
                            ret.as_raw_mut()[ret_size - remain..]
                                .iter_mut()
                                .zip(#(#zip_remain_scalars)*)
                                .for_each(|#tuple| {
                                    *res = f(#(*#f2_args),*);
                                });
                        }
                    } else {
                        let min_len: usize =
                        ret.size() / (((rayon::current_num_threads() as f64) * 1.3) as usize);
                        ret.as_raw_mut()
                            .par_iter_mut()
                            .with_min_len(min_len)
                            .zip(#(#zip_min_len)*)
                            .for_each(|#tuple| {
                                *res = f(#(*#f2_args),*);
                            });
                    }
                    Ok(ret)
                } else {
                    #(#layout_broadcast)*
                    let ret = Tensor::<__HPTRES, Cpu>::empty(layout.shape())?;
                    ret.par_iter_mut()
                        .zip(#(#zip_scalar_par)*)
                        .for_each(|#tuple| {
                            *res = f(#(#f2_args),*);
                        });
                    Ok(ret)
                }
            }            
        );
        fused_vec.push((fused, func));
    }
    fused_vec
}

fn gen_tuple(iters: &mut Vec<syn::Ident>) -> TokenStream2 {
    if iters.is_empty() {
        quote::quote!()
    } else if iters.len() == 1 {
        let first = iters.remove(0);
        quote::quote!(#first)
    } else {
        let first = iters.remove(0);
        let rest = gen_tuple(iters);
        quote::quote!(
            (#first, #rest)
        )
    }
}
