use std::collections::{ HashMap, HashSet };
use petgraph::graph::NodeIndex;

use crate::{ fuse::node::Node, TokenStream2 };
use super::fuse::FusionGroup;

pub(crate) fn gen_fuse(
    cfg: &crate::fuse::cfg::CFG,
    graph: &petgraph::stable_graph::StableGraph<&(crate::fuse::node::Node<'_>, i64, usize), ()>,
    groups: &FusionGroup
) -> (Vec<TokenStream2>, Vec<(Vec<(syn::Ident, i64)>, Vec<(syn::Ident, i64)>)>) {
    let (fused_codes, inp_outs) = _gen_fuse(cfg, &graph, &groups.vars);
    (fused_codes, inp_outs)
}

fn fill_indegree(node: &Node, in_degrees: &mut HashMap<String, (usize, usize)>) {
    match node {
        Node::Unary(unary) => {
            in_degrees.entry(unary.output.to_string()).or_insert((0, 0)).0 += 1;
        }
        Node::Binary(binary) => {
            in_degrees.entry(binary.output.to_string()).or_insert((0, 0)).0 += 2;
        }
        Node::Input(_) => {}
    }
}

fn init_degrees(idx: usize, node: &Node, degrees: &mut HashMap<String, (usize, usize)>) {
    match node {
        Node::Unary(unary) => {
            degrees.insert(unary.operand.to_string(), (0, idx));
            degrees.insert(unary.output.to_string(), (0, idx));
        }
        Node::Binary(binary) => {
            degrees.insert(binary.left.to_string(), (0, idx));
            degrees.insert(binary.right.to_string(), (0, idx));
            degrees.insert(binary.output.to_string(), (0, idx));
        }
        Node::Input(ident) => {
            degrees.insert(ident.to_string(), (0, idx));
        }
    }
}

fn fill_outdegree(node: &Node, out_degrees: &mut HashMap<String, (usize, usize)>) {
    match node {
        Node::Unary(unary) => {
            out_degrees.entry(unary.operand.to_string()).or_insert((0, 0)).0 += 1;
        }
        Node::Binary(binary) => {
            out_degrees.entry(binary.left.to_string()).or_insert((0, 0)).0 += 1;
            out_degrees.entry(binary.right.to_string()).or_insert((0, 0)).0 += 1;
        }
        Node::Input(_) => {}
    }
}

fn gen_body(
    sorted: Vec<NodeIndex>,
    graph: &petgraph::stable_graph::StableGraph<&(crate::fuse::node::Node<'_>, i64, usize), ()>,
    cfg: &crate::fuse::cfg::CFG
) -> crate::TokenStream2 {
    let mut comp_tokens = TokenStream2::new();
    for idx in sorted {
        let mut node = graph[idx].clone();
        match &mut node {
            (Node::Unary(unary), _, block_idx) => {
                let origin_out = cfg.graph[NodeIndex::new(*block_idx)].origin_var_map
                    .get(&unary.output.to_string())
                    .expect("gen_fuse::out");
                let origin_operand = cfg.graph[NodeIndex::new(*block_idx)].origin_var_map
                    .get(&unary.operand.to_string())
                    .expect("gen_fuse::origin_operand");
                unary.operand = syn::Ident::new(&origin_operand.to_string(), unary.operand.span());
                unary.output = syn::Ident::new(origin_out, unary.output.span());
                comp_tokens.extend(
                    quote::quote!(
                        #unary
                    )
                );
            }
            (Node::Binary(binary), _, block_idx) => {
                let origin_out = cfg.graph[NodeIndex::new(*block_idx)].origin_var_map
                    .get(&binary.output.to_string())
                    .expect("gen_fuse::out");
                let origin_left = cfg.graph[NodeIndex::new(*block_idx)].origin_var_map
                    .get(&binary.left.to_string())
                    .expect("gen_fuse::origin_left");
                binary.left = syn::Ident::new(&origin_left.to_string(), binary.left.span());
                let origin_right = cfg.graph[NodeIndex::new(*block_idx)].origin_var_map
                    .get(&binary.right.to_string())
                    .expect("gen_fuse::origin_right");
                binary.right = syn::Ident::new(&origin_right.to_string(), binary.right.span());
                binary.output = syn::Ident::new(origin_out, binary.output.span());
                comp_tokens.extend(
                    quote::quote!(
                            #binary
                        )
                );
            }
            (Node::Input(_), _, _) => {}
        }
    }
    println!("comp_tokens: {:#?}", comp_tokens.to_string());
    comp_tokens
}

pub(crate) fn _gen_fuse(
    cfg: &crate::fuse::cfg::CFG,
    graph: &petgraph::stable_graph::StableGraph<&(crate::fuse::node::Node<'_>, i64, usize), ()>,
    groups: &Vec<HashSet<NodeIndex>>
) -> (Vec<TokenStream2>, Vec<(Vec<(syn::Ident, i64)>, Vec<(syn::Ident, i64)>)>) {
    println!("graph: {:#?}", graph);
    let sorteds = petgraph::algo::toposort(graph, None).expect("gen_fuse::topological_sort");

    let results = groups
        .iter()
        .map(|group| {
            let mut inputs_ident = Vec::new();
            let mut outputs_ident = Vec::new();
            let mut in_degrees = HashMap::new();
            let mut out_degrees = HashMap::new();
            for &idx in group {
                init_degrees(idx.index(), &graph[idx].0, &mut in_degrees);
                init_degrees(idx.index(), &graph[idx].0, &mut out_degrees);
            }
            for &idx in group {
                fill_indegree(&graph[idx].0, &mut in_degrees);
                fill_outdegree(&graph[idx].0, &mut out_degrees);
            }
            for (string, (cnt, idx)) in in_degrees.into_iter() {
                if cnt == 0 {
                    let origin_ident = cfg.graph[
                        NodeIndex::new(graph[NodeIndex::new(idx)].2)
                    ].origin_var_map
                        .get(&string)
                        .expect("gen_fuse::origin_ident");
                    inputs_ident.push((
                        syn::Ident::new(&origin_ident.to_string(), proc_macro2::Span::call_site()),
                        graph[NodeIndex::new(idx)].1,
                    ));
                }
            }
            for (string, (cnt, idx)) in out_degrees.into_iter() {
                if cnt == 0 {
                    let origin_ident = cfg.graph[
                        NodeIndex::new(graph[NodeIndex::new(idx)].2)
                    ].origin_var_map
                        .get(&string)
                        .expect("gen_fuse::origin_ident");
                    outputs_ident.push((
                        syn::Ident::new(&origin_ident.to_string(), proc_macro2::Span::call_site()),
                        graph[NodeIndex::new(idx)].1,
                    ));
                }
            }
            // println!("inputs_ident: {:#?}", inputs_ident);
            // println!("outputs_ident: {:#?}", outputs_ident);
            (inputs_ident, outputs_ident)
        })
        .collect::<Vec<_>>();
    let mut iters_vec = Vec::new();
    for (inputs_ident, _) in results.iter() {
        let mut iters = Vec::new();
        for (input, _) in inputs_ident {
            iters.push(quote::quote!(
                    #input.par_iter_simd()
                ));
        }
        iters_vec.push(iters);
    }
    let mut zipped_vec = vec![];
    for mut iters in iters_vec.clone() {
        let tokens = gen_zip(&mut iters);
        zipped_vec.push(tokens);
    }
    let mut tuple_vec = vec![];
    for mut iters in results
        .iter()
        .map(|(inputs_ident, _)|
            inputs_ident
                .iter()
                .map(|(ident, _)| ident.clone())
                .collect::<Vec<_>>()
        )
        .collect::<Vec<_>>() {
        let tokens = gen_tuple(&mut iters);
        tuple_vec.push(quote::quote!(
            (res, #tokens)
        ));
    }

    let mut sorted_groups = Vec::new();

    for group in groups.iter() {
        let mut v = vec![];
        for sorted in sorteds.iter() {
            if group.contains(sorted) {
                v.push(sorted.clone());
            }
        }
        sorted_groups.push(v);
    }
    let mut fused_vec = Vec::new();
    for (i, (sorted, (_, outputs))) in sorted_groups.into_iter().zip(results.iter()).enumerate() {
        assert_eq!(outputs.len(), 1);
        let mut scalar_comp = TokenStream2::new();
        let mut vec_comp = TokenStream2::new();
        let output = &outputs.iter().next().expect("gen_fuse::output").0;
        println!("sorted: {:?}", sorted);
        let tokens = gen_body(sorted, graph, cfg);
        scalar_comp.extend(tokens.clone());
        scalar_comp.extend(quote::quote!(
            *res = #output
        ));
        vec_comp.extend(tokens);
        vec_comp.extend(quote::quote!(
            res.write_unaligned(#output)
        ));
        let zipped = &zipped_vec[i];
        let tuple = &tuple_vec[i];
        let fused =
            quote::quote!(
                #zipped.strided_map_simd(|#tuple| {
                    #scalar_comp
                },|#tuple| {
                    #vec_comp
                }).collect::<_Tensor<_>>()
            );
        println!("fused: {:#?}", fused.to_string());
        fused_vec.push(fused);
    }
    (fused_vec, results)
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
