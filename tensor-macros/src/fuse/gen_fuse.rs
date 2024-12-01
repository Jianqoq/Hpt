use std::collections::{ HashMap, HashSet };
use petgraph::graph::NodeIndex;

use crate::{ fuse::node::Node, TokenStream2 };
use super::fuse::FusionGroup;

pub(crate) fn gen_fuse(
    cfg: &crate::fuse::cfg::CFG,
    graph: &petgraph::stable_graph::StableGraph<&(crate::fuse::node::Node, i64, usize), ()>,
    groups: &FusionGroup
) -> (Vec<TokenStream2>, Vec<(Vec<(syn::Ident, i64, usize, NodeIndex)>, Vec<(syn::Ident, i64, usize, NodeIndex)>)>) {
    let (fused_codes, inp_outs) = _gen_fuse(cfg, &graph, &groups.vars);
    (fused_codes, inp_outs)
}

fn fill_indegree(node: &Node, in_degrees: &mut HashMap<syn::Ident, (usize, i64, usize, NodeIndex)>) {
    match node {
        Node::Unary(unary) => {
            in_degrees.entry(unary.output.clone()).or_insert((0, 0, 0, NodeIndex::new(0))).0 += 1;
        }
        Node::Binary(binary) => {
            in_degrees.entry(binary.output.clone()).or_insert((0, 0, 0, NodeIndex::new(0))).0 += 2;
        }
        Node::Input(_) => {}
    }
}

fn init_degrees(
    comp_graph_idx: NodeIndex,
    stmt_index: i64,
    block_idx: usize,
    node: &Node,
    degrees: &mut HashMap<syn::Ident, (usize, i64, usize, NodeIndex)>
) {
    match node {
        Node::Unary(unary) => {
            degrees.insert(unary.output.clone(), (0, stmt_index, block_idx, comp_graph_idx));
        }
        Node::Binary(binary) => {
            degrees.insert(binary.output.clone(), (0, stmt_index, block_idx, comp_graph_idx));
        }
        Node::Input(ident) => {
            degrees.insert(ident.clone(), (0, stmt_index, block_idx, comp_graph_idx));
        }
    }
}

// handle node inputs, they may not be initilized after init_degrees
fn init_degrees_remain(
    comp_graph_idx: NodeIndex,
    block_idx: usize,
    node: &Node,
    degrees: &mut HashMap<syn::Ident, (usize, i64, usize, NodeIndex)>
) {
    match node {
        Node::Unary(unary) => {
            if let None = degrees.get_mut(&unary.operand) {
                degrees.insert(unary.operand.clone(), (0, -1, block_idx, comp_graph_idx));
            }
        }
        Node::Binary(binary) => {
            if let None = degrees.get_mut(&binary.left) {
                degrees.insert(binary.left.clone(), (0, -1, block_idx, comp_graph_idx));
            }
            if let None = degrees.get_mut(&binary.right) {
                degrees.insert(binary.right.clone(), (0, -1, block_idx, comp_graph_idx));
            }
        }
        Node::Input(_) => {}
    }
}

fn fill_outdegree(node: &Node, out_degrees: &mut HashMap<syn::Ident, (usize, i64, usize, NodeIndex)>) {
    match node {
        Node::Unary(unary) => {
            out_degrees.entry(unary.operand.clone()).or_insert((0, 0, 0, NodeIndex::new(0))).0 += 1;
        }
        Node::Binary(binary) => {
            out_degrees.entry(binary.left.clone()).or_insert((0, 0, 0, NodeIndex::new(0))).0 += 1;
            out_degrees.entry(binary.right.clone()).or_insert((0, 0, 0, NodeIndex::new(0))).0 += 1;
        }
        Node::Input(_) => {}
    }
}

fn gen_body(
    sorted: Vec<NodeIndex>,
    graph: &petgraph::stable_graph::StableGraph<&(crate::fuse::node::Node, i64, usize), ()>,
    cfg: &crate::fuse::cfg::CFG
) -> crate::TokenStream2 {
    let mut comp_tokens = TokenStream2::new();
    for idx in sorted {
        let mut node = graph[idx].clone();
        match &mut node {
            (Node::Unary(unary), _, block_idx) => {
                let origin_out = cfg.graph[NodeIndex::new(*block_idx)].origin_var_map
                    .get(&unary.output)
                    .expect("gen_fuse::out");
                let origin_operand = cfg.graph[NodeIndex::new(*block_idx)].origin_var_map
                    .get(&unary.operand)
                    .expect("gen_fuse::origin_operand");
                unary.operand = origin_operand.clone();
                unary.output = origin_out.clone();
                comp_tokens.extend(
                    quote::quote!(
                        #unary
                    )
                );
            }
            (Node::Binary(binary), _, block_idx) => {
                let origin_out = cfg.graph[NodeIndex::new(*block_idx)].origin_var_map
                    .get(&binary.output)
                    .expect("gen_fuse::out");
                let origin_left = cfg.graph[NodeIndex::new(*block_idx)].origin_var_map
                    .get(&binary.left)
                    .expect("gen_fuse::origin_left");
                binary.left = origin_left.clone();
                let origin_right = cfg.graph[NodeIndex::new(*block_idx)].origin_var_map
                    .get(&binary.right)
                    .expect("gen_fuse::origin_right");
                binary.right = origin_right.clone();
                binary.output = origin_out.clone();
                comp_tokens.extend(
                    quote::quote!(
                            #binary
                        )
                );
            }
            (Node::Input(_), _, _) => {}
        }
    }
    comp_tokens
}

pub(crate) fn _gen_fuse(
    cfg: &crate::fuse::cfg::CFG,
    graph: &petgraph::stable_graph::StableGraph<&(crate::fuse::node::Node, i64, usize), ()>,
    groups: &Vec<HashSet<NodeIndex>>
) -> (Vec<TokenStream2>, Vec<(Vec<(syn::Ident, i64, usize, NodeIndex)>, Vec<(syn::Ident, i64, usize, NodeIndex)>)>) {
    let sorteds = petgraph::algo::toposort(graph, None).expect("gen_fuse::topological_sort");

    let results = groups
        .iter()
        .map(|group| {
            let mut inputs_ident = Vec::new();
            let mut outputs_ident = Vec::new();
            let mut in_degrees = HashMap::new();
            let mut out_degrees = HashMap::new();
            for &idx in group {
                init_degrees(idx, graph[idx].1, graph[idx].2, &graph[idx].0, &mut in_degrees);
                init_degrees(idx, graph[idx].1, graph[idx].2, &graph[idx].0, &mut out_degrees);
            }
            for &idx in group {
                init_degrees_remain(idx, graph[idx].2, &graph[idx].0, &mut in_degrees);
                init_degrees_remain(idx, graph[idx].2, &graph[idx].0, &mut out_degrees);
            }

            for &idx in group {
                fill_indegree(&graph[idx].0, &mut in_degrees);
                fill_outdegree(&graph[idx].0, &mut out_degrees);
            }
            for (string, (cnt, stmt_index, block_idx, comp_graph_idx)) in in_degrees.into_iter() {
                if cnt == 0 {
                    let origin_ident = cfg.graph[NodeIndex::new(block_idx)].origin_var_map
                        .get(&string)
                        .expect("gen_fuse::origin_ident");
                    inputs_ident.push((
                        syn::Ident::new(&origin_ident.to_string(), proc_macro2::Span::call_site()),
                        stmt_index,
                        block_idx,
                        comp_graph_idx,
                    ));
                }
            }
            for (string, (cnt, stmt_index, block_idx, comp_graph_idx)) in out_degrees.into_iter() {
                if cnt == 0 {
                    let origin_ident = cfg.graph[NodeIndex::new(block_idx)].origin_var_map
                        .get(&string)
                        .expect("gen_fuse::origin_ident");
                    outputs_ident.push((
                        syn::Ident::new(&origin_ident.to_string(), proc_macro2::Span::call_site()),
                        stmt_index,
                        block_idx,
                        comp_graph_idx,
                    ));
                }
            }
            (inputs_ident, outputs_ident)
        })
        .collect::<Vec<_>>();
    let mut iters_vec = Vec::new();
    for (inputs_ident, _) in results.iter() {
        let mut iters = Vec::new();
        for (input, _, _, _) in inputs_ident {
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
                .map(|(ident, _, _, _)| ident.clone())
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
