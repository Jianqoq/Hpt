use std::collections::HashSet;
use crate::{ fuse::{ dag::Var, node::Node }, TokenStream2 };
use super::dag::{ Graph, Var2 };

pub(crate) fn gen_fuse(graph: &Graph, groups: &Vec<HashSet<Var2>>) -> (Vec<TokenStream2>, Vec<syn::Ident>, Vec<HashSet<syn::Ident>>) {
    let inputs_vec = groups
        .iter()
        .map(|group| {
            let mut inputs = Vec::new();
            for var in group {
                if let Some(node) = graph.map.get(&(Var { ident: &var.ident })) {
                    if let Node::Input(input) = node {
                        inputs.push(input.ident.clone());
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
                    #input.par_iter()
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
        // println!("{:#?}", tokens.to_string());
        tuple_vec.push(tokens);
    }

    let sorteds = graph.topological_sort().unwrap();

    let mut sorted_groups = Vec::new();

    for group in groups.iter() {
        sorted_groups.push(vec![]);
        for sorted in sorteds.iter() {
            if group.contains(&(Var2 { ident: sorted.clone() })) {
                sorted_groups.last_mut().unwrap().push(sorted.clone());
            }
        }
    }

    // println!("{:#?}", sorted_groups);
    let mut fused_vec = vec![];
    let mut fused_outs = vec![];
    for (i, sorted) in sorted_groups.iter().enumerate() {
        let mut output = syn::Ident::new("__out0", proc_macro2::Span::call_site());
        let mut comp_tokens = TokenStream2::new();
        for ident in sorted {
            if let Some(node) = graph.map.get(&(Var { ident: &ident })) {
                match node {
                    Node::Unary(unary, _) => {
                        comp_tokens.extend(
                            quote::quote!(
                        #unary
                    )
                        );
                        output = unary.output.clone();
                    }
                    Node::Binary(binary, _) => {
                        comp_tokens.extend(
                            quote::quote!(
                        #binary
                    )
                        );
                        output = binary.output.clone();
                    }
                    Node::Input(_) => {}
                }
            }
        }
        comp_tokens.extend(quote::quote!(
        #output
    ));
    fused_outs.push(output);
    // println!("{:#?}", comp_tokens.to_string());
        let zipped = &zipped_vec[i];
        let tuple = &tuple_vec[i];
        let fused =
            quote::quote!(
            #zipped.strided_map(|#tuple| {
                #comp_tokens
            }
            ).collect::<_Tensor<_>>()
        );
        fused_vec.push(fused);
    }

    let inputs_vec = inputs_vec
        .iter()
        .map(|input| input.iter().map(|i| i.clone()).collect::<HashSet<_>>())
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
