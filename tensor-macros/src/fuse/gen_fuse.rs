use std::collections::HashSet;
use crate::{ fuse::{ dag::Var, node::Node }, TokenStream2 };
use super::dag::{ Graph, Var2 };

pub(crate) fn gen_fuse(graph: &Graph, groups: &Vec<HashSet<Var2>>) -> Vec<TokenStream2> {
    let inputs_vec = groups
        .iter()
        .map(|group| {
            let mut inputs = Vec::new();
            for var in group {
                if let Some(node) = graph.map.get(&(Var { ident: &var.ident })) {
                    if let Node::Input(input) = node {
                        inputs.push(input);
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
        tuple_vec.push(tokens);
    }

    let sorteds = vec![graph.topological_sort().unwrap()];
    let mut comp_tokens = TokenStream2::new();
    let mut output = proc_macro2::TokenStream::new();
    let mut fused_vec = vec![];
    for (i, sorted) in sorteds.iter().enumerate() {
        for ident in sorted {
            if let Some(node) = graph.map.get(&(Var { ident: &ident })) {
                match node {
                    Node::Unary(unary, _) => {
                        comp_tokens.extend(
                            quote::quote!(
                        #unary
                    )
                        );
                        let out = unary.output.clone();
                        output = quote::quote!(
                        #out
                    );
                    }
                    Node::Binary(binary, _) => {
                        comp_tokens.extend(
                            quote::quote!(
                        #binary
                    )
                        );
                        let out = binary.output.clone();
                        output = quote::quote!(
                        #out
                    );
                    }
                    Node::Input(_) => {}
                }
            }
        }
        comp_tokens.extend(quote::quote!(
        #output
    ));
        let zipped = &zipped_vec[i];
        let tuple = &tuple_vec[i];
        let fused = quote::quote!(
            #zipped.strided_map(|#tuple| {
                #comp_tokens
            }
            ).collect()
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

fn gen_tuple(iters: &mut Vec<&Var>) -> TokenStream2 {
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
