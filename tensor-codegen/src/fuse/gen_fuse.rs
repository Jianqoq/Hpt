use std::collections::HashSet;
use petgraph::graph::NodeIndex;
use proc_macro2::TokenStream as TokenStream2;
use super::{ build_graph::CmpNode, fuse::{ FusionGroup, Input } };

pub(crate) fn cmp_gen_fuse(
    cfg: &crate::fuse::cfg::CFG,
    graph: &petgraph::stable_graph::StableGraph<CmpNode, ()>,
    groups: &FusionGroup
) -> Vec<TokenStream2> {
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
        if
            !inputs
                .iter()
                .any(|input| input.stmt_index == node.stmt_idx && input.block_idx == node.block_idx)
        {
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
    comp_tokens
}

pub(crate) fn _cmp_gen_fuse(
    cfg: &crate::fuse::cfg::CFG,
    graph: &petgraph::stable_graph::StableGraph<CmpNode, ()>,
    groups: &FusionGroup
) -> Vec<TokenStream2> {
    let sorteds = petgraph::algo::toposort(graph, None).expect("gen_fuse::topological_sort");

    let mut iters_vec = Vec::new();
    for inputs in groups.inputs.iter() {
        let mut iters = Vec::new();
        for input in inputs {
            let ident = &input.var;
            let origin_ident = cfg.graph[NodeIndex::new(input.block_idx)].origin_var_map
                .get(&ident)
                .expect("gen_fuse::origin_ident");
            iters.push(
                quote::quote!(
                    #origin_ident.par_iter_simd()
                )
            );
        }
        iters_vec.push(iters);
    }
    let mut zipped_vec = vec![];
    for mut iters in iters_vec.clone() {
        let tokens = gen_zip(&mut iters);
        zipped_vec.push(tokens);
    }
    let mut tuple_vec = vec![];
    for mut iters in groups.inputs
        .iter()
        .map(|inputs|
            inputs
                .iter()
                .map(|input| {
                    cfg.graph[NodeIndex::new(input.block_idx)].origin_var_map
                        .get(&input.var)
                        .expect("gen_fuse::origin_ident")
                        .clone()
                })
                .collect::<Vec<_>>()
        )
        .collect::<Vec<_>>() {
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
        .zip(groups.inputs.iter().zip(groups.outputs.iter()))
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
        let tokens = cmp_gen_body(sorted, inputs, graph, cfg);
        scalar_comp.extend(tokens.clone());
        scalar_comp.extend(quote::quote!(
            *res = #origin_output
        ));
        vec_comp.extend(tokens);
        vec_comp.extend(quote::quote!(
            res.write_unaligned(#origin_output)
        ));
        let zipped = &zipped_vec[i];
        let tuple = &tuple_vec[i];
        let fused =
            quote::quote!(
                #zipped.strided_map_simd(|#tuple| {
                    #scalar_comp
                },|#tuple| {
                    #vec_comp
                }).collect::<Tensor<_>>()
            );
        fused_vec.push(fused);
    }
    fused_vec
}

fn gen_zip(iters: &mut Vec<TokenStream2>) -> TokenStream2 {
    if iters.is_empty() {
        quote::quote!()
    } else if iters.len() == 1 {
        let first = iters.remove(0);
        quote::quote!(#first)
    } else {
        let first = iters.remove(0);
        let rest = gen_zip(iters);
        quote::quote!(
            #first.zip(#rest)
        )
    }
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
