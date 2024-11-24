use std::collections::{ HashMap, HashSet };
use petgraph::graph::NodeIndex;

use crate::{ fuse::node::Node, TokenStream2 };
use super::fuse::FusionGroup;

pub(crate) struct GenFuse {
    pub(crate) fused_outs: Vec<(NodeIndex, i64)>,
    pub(crate) fused_inputs: Vec<HashSet<(NodeIndex, i64)>>,
    pub(crate) codes: HashMap<(NodeIndex, i64), TokenStream2>,
}

pub(crate) fn gen_fuse(
    graph: &petgraph::Graph<&(crate::fuse::node::Node<'_>, i64), ()>,
    groups: &FusionGroup
) -> GenFuse {
    let (fused_codes, fused_outs, fused_inputs) = _gen_fuse(&graph, &groups.vars);
    let mut codes = HashMap::new();
    for (i, code) in fused_codes.iter().enumerate() {
        let out = graph
            .node_weight(fused_outs.get(i).expect("gen_fuse::fused_outs::i").0)
            .expect("gen_fuse::graph.node_weight");
        match out {
            (Node::Unary(unary), _) => {
                let out_ident = &unary.output;
                codes.insert(
                    *fused_outs.get(i).expect("gen_fuse::fused_outs::i2"),
                    quote::quote!(
                #out_ident = #code;
            )
                );
            }
            (Node::Binary(binary), _) => {
                let out_ident = &binary.output;
                codes.insert(
                    *fused_outs.get(i).expect("gen_fuse::fused_outs::i3"),
                    quote::quote!(
                #out_ident = #code;
            )
                );
            }
            (Node::Input(_), _) => unreachable!(),
        }
    }
    GenFuse { fused_outs, fused_inputs, codes }
}

pub(crate) fn _gen_fuse(
    graph: &petgraph::Graph<&(crate::fuse::node::Node<'_>, i64), ()>,
    groups: &Vec<HashSet<NodeIndex>>
) -> (Vec<TokenStream2>, Vec<(NodeIndex, i64)>, Vec<HashSet<(NodeIndex, i64)>>) {
    let id_to_ident = graph
        .node_indices()
        .map(|idx| (
            idx,
            {
                match &graph[idx] {
                    (Node::Unary(unary), _) => unary.output.clone(),
                    (Node::Binary(binary), _) => binary.output.clone(),
                    (Node::Input(ident), _) => ident.clone(),
                }
            },
        ))
        .collect::<HashMap<_, _>>();
    let ident_to_id = graph
        .node_indices()
        .map(|idx| (id_to_ident.get(&idx).expect("gen_fuse::id_to_ident::idx"), idx))
        .collect::<HashMap<_, _>>();

    let inputs_vec = groups
        .iter()
        .map(|group| {
            let mut inputs_ident = Vec::new();
            let mut inputs_idx = Vec::new();
            for &idx in group {
                if let Some(node) = graph.node_weight(idx) {
                    match node {
                        (Node::Unary(unary), stmt_idx) => {
                            if !group.contains(&ident_to_id.get(&unary.operand).expect("gen_fuse::ident_to_id::unary.operand")) {
                                inputs_ident.push(unary.output.clone());
                                inputs_idx.push((idx, *stmt_idx));
                            }
                        }
                        (Node::Binary(binary), stmt_idx) => {
                            if
                                !group.contains(&ident_to_id.get(&binary.left).expect("gen_fuse::ident_to_id::binary.left")) &&
                                !group.contains(&ident_to_id.get(&binary.right).expect("gen_fuse::ident_to_id::binary.right"))
                            {
                                inputs_ident.push(binary.output.clone());
                                inputs_idx.push((idx, *stmt_idx));
                            }
                        }
                        (Node::Input(ident), stmt_idx) => {
                            inputs_ident.push(ident.clone());
                            inputs_idx.push((idx, *stmt_idx));
                        }
                    }
                }
            }
            (inputs_ident, inputs_idx)
        })
        .collect::<Vec<_>>();
    let mut iters_vec = Vec::new();
    for (inputs_ident, _) in inputs_vec.iter() {
        let mut iters = Vec::new();
        for input in inputs_ident {
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
    for (mut iters, _) in inputs_vec.clone() {
        let tokens = gen_tuple(&mut iters);
        tuple_vec.push(quote::quote!(
            (res, #tokens)
        ));
    }

    let sorteds = petgraph::algo::toposort(graph, None).expect("gen_fuse::topological_sort");
    // println!("graph: {:#?}", graph);
    // println!("sorteds: {:#?}", sorteds);

    let mut sorted_groups = Vec::new();

    for group in groups.iter() {
        sorted_groups.push(vec![]);
        for sorted in sorteds.iter() {
            if group.contains(sorted) {
                sorted_groups.last_mut().expect("gen_fuse::last_mut").push(sorted.clone());
            }
        }
    }

    let mut fused_vec = vec![];
    let mut fused_outs = vec![];
    for (i, sorted) in sorted_groups.iter().enumerate() {
        let (mut output_idx, mut output) = (
            (NodeIndex::new(0), -1),
            syn::Ident::new("__out0", proc_macro2::Span::call_site()),
        );
        let mut comp_tokens = TokenStream2::new();
        for &idx in sorted {
            let node = &graph[idx];
            match node {
                (Node::Unary(unary), stmt_idx) => {
                    if sorted.contains(&ident_to_id.get(&unary.operand).expect("gen_fuse::ident_to_id::unary.operand")) {
                        comp_tokens.extend(
                            quote::quote!(
                                #unary
                            )
                        );
                    }
                    output = unary.output.clone();
                    output_idx = (idx, *stmt_idx);
                }
                (Node::Binary(binary), stmt_idx) => {
                    if
                        !(
                            !sorted.contains(&ident_to_id.get(&binary.left).expect("gen_fuse::ident_to_id::binary.left")) &&
                            !sorted.contains(&ident_to_id.get(&binary.right).expect("gen_fuse::ident_to_id::binary.right"))
                        )
                    {
                        comp_tokens.extend(
                            quote::quote!(
                                    #binary
                                )
                        );
                    }
                    output = binary.output.clone();
                    output_idx = (idx, *stmt_idx);
                }
                (Node::Input(_), _) => {}
            }
        }
        let mut vec_comp_tokens = comp_tokens.clone();
        comp_tokens.extend(quote::quote!(
            *res = #output
        ));
        vec_comp_tokens.extend(quote::quote!(
            res.write_unaligned(#output)
        ));
        fused_outs.push(output_idx);
        let zipped = &zipped_vec[i];
        let tuple = &tuple_vec[i];
        let fused =
            quote::quote!(
                #zipped.strided_map_simd(|#tuple| {
                    #comp_tokens
                },|#tuple| {
                    #vec_comp_tokens
                }).collect::<_Tensor<_>>()
            );
        fused_vec.push(fused);
    }

    let inputs_vec = inputs_vec
        .iter()
        .map(|(_, idx)|
            idx
                .iter()
                .map(|i| i.clone())
                .collect::<HashSet<_>>()
        )
        .collect::<Vec<_>>();
    // println!("inputs_vec: {:#?}", inputs_vec);
    // println!("fused_outs: {:#?}", fused_outs);
    (fused_vec, fused_outs, inputs_vec)
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
