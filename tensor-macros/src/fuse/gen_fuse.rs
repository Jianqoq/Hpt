use std::collections::HashSet;
use crate::{fuse::{dag::Var, node::Node}, TokenStream2};
use super::dag::{Graph, Var2};


pub(crate) fn gen_fuse(graph: &Graph, groups: &Vec<HashSet<Var2>>) -> TokenStream2 {
    let inputs_vec = groups.iter().map(|group| {
        let mut inputs = Vec::new();
        for var in group {
            if let Some(node) = graph.map.get(&Var { ident: &var.ident }) {
                if let Node::Input(input) = node {
                    inputs.push(input);
                }
            }
        }
        inputs
    }).collect::<Vec<_>>();
    let mut iters_vec = Vec::new();
    for inputs in inputs_vec {
        let mut iters = Vec::new();
        for input in inputs {
            iters.push(
                quote::quote!(
                    #input.par_iter()
                )
            );
        }
        iters_vec.push(iters);
    }
    let mut zipped_vec = vec![];
    for iters in iters_vec {
        let mut tokens = TokenStream2::new();
        for (first, sec) in iters.iter().zip(iters.iter().skip(1)) {
            tokens.extend(quote::quote! {
                #first.zip(#sec)
            });
        }
        println!("{:#?}", tokens.to_string());
        zipped_vec.push(tokens);
    }

    todo!()
}
