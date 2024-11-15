use std::collections::{ HashMap, HashSet };
use crate::{ fuse::node::Node, TokenStream2 };
use super::{ dag::_Graph, fuse::FusionGroup };

pub(crate) struct GenFuse {
    pub(crate) fused_outs: Vec<syn::Ident>,
    pub(crate) fused_inputs: Vec<HashSet<syn::Ident>>,
    pub(crate) codes: HashMap<syn::Ident, TokenStream2>,
    pub(crate) next_gen_fuse: Option<Box<GenFuse>>,
}

pub(crate) fn gen_fuse(graph: &_Graph, groups: &FusionGroup) -> GenFuse {
    let (fused_codes, fused_outs, fused_inputs) = _gen_fuse(&graph, &groups.vars);
    let mut codes = HashMap::new();
    for (i, code) in fused_codes.iter().enumerate() {
        let out = &fused_outs[i];
        codes.insert(out.clone(), quote::quote!(
                #out = #code;
            ));
    }
    let mut ret = GenFuse { fused_outs, fused_inputs, next_gen_fuse: None, codes };
    match (&graph.next_graph, &groups._next_group) {
        (Some(next_graph), Some(next_group)) => {
            let next_gen_fuse = Some(Box::new(gen_fuse(&next_graph, &next_group)));
            ret.next_gen_fuse = next_gen_fuse;
        }
        _ => {}
    }
    ret
}

pub(crate) fn _gen_fuse(
    graph: &_Graph,
    groups: &Vec<HashSet<syn::Ident>>
) -> (Vec<TokenStream2>, Vec<syn::Ident>, Vec<HashSet<syn::Ident>>) {
    let inputs_vec = groups
        .iter()
        .map(|group| {
            let mut inputs = Vec::new();
            for ident in group {
                if let Some(node) = graph.map.get(&ident) {
                    match node {
                        Node::Unary(unary) => {
                            if !group.contains(&unary.operand) {
                                inputs.push(unary.output.clone());
                            }
                        }
                        Node::Binary(binary) => {
                            if !group.contains(&binary.left) && !group.contains(&binary.right) {
                                inputs.push(binary.output.clone());
                            }
                        }
                        Node::Input(ident) => inputs.push(ident.clone()),
                    }
                }
            }
            inputs
        })
        .collect::<Vec<_>>();
    let mut iters_vec = Vec::new();
    for inputs in inputs_vec.iter() {
        let mut iters = Vec::new();
        for input in inputs {
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
    for mut iters in inputs_vec.clone() {
        let tokens = gen_tuple(&mut iters);
        tuple_vec.push(quote::quote!(
            (res, #tokens)
        ));
    }

    let sorteds = graph.topological_sort().expect("gen_fuse::topological_sort");

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
        let mut output = syn::Ident::new("__out0", proc_macro2::Span::call_site());
        let mut comp_tokens = TokenStream2::new();
        for ident in sorted {
            if let Some(node) = graph.map.get(&ident) {
                match node {
                    Node::Unary(unary) => {
                        if sorted.contains(&unary.operand) {
                            comp_tokens.extend(
                                quote::quote!(
                                #unary
                            )
                            );
                        }
                        output = unary.output.clone();
                    }
                    Node::Binary(binary) => {
                        if !(!sorted.contains(&binary.left) && !sorted.contains(&binary.right)) {
                            comp_tokens.extend(
                                quote::quote!(
                                    #binary
                                )
                            );
                        }
                        output = binary.output.clone();
                    }
                    Node::Input(_) => {}
                }
            }
        }
        let mut vec_comp_tokens = comp_tokens.clone();
        comp_tokens.extend(quote::quote!(
            *res = #output
        ));
        vec_comp_tokens.extend(quote::quote!(
            res.write_unaligned(#output)
        ));
        fused_outs.push(output);
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
        .map(|input|
            input
                .iter()
                .map(|i| i.clone())
                .collect::<HashSet<_>>()
        )
        .collect::<Vec<_>>();

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
