use std::collections::HashSet;

use crate::{ fuse::codegen::Codegen, TokenStream2 };
use quote::ToTokens;
use syn::visit::Visit;

use crate::fuse::{ dag::Graph, fuse::fuse, gen_fuse::gen_fuse };

use super::{ dag::Var, node::Node, visitor::Visitor };

pub(crate) fn fuse_impl(item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let func = syn::parse_macro_input!(item as syn::ItemFn);
    let mut visitor = Visitor::new();
    visitor.visit_item_fn(&func);
    for arg in func.sig.inputs.iter() {
        if let syn::FnArg::Typed(pat_type) = arg {
            let string = pat_type.ty.to_token_stream().to_string();
            if string.contains("Tensor") || string.contains("_Tensor") {
                visitor.nodes.push(
                    Node::Input(Var {
                        ident: {
                            if let syn::Pat::Ident(ident) = &pat_type.pat.as_ref() {
                                &ident.ident
                            } else {
                                panic!("not an ident")
                            }
                        },
                    })
                );
            }
        }
    }
    // println!("{:#?}", visitor.nodes);
    let graph = Graph::from_nodes(&visitor.nodes);
    // println!("{:#?}", graph);
    let fused = fuse(&graph);
    // println!("{:#?}", fused);
    let (fused_codes, fused_outs, fused_inputs) = gen_fuse(&graph, &fused);
    let mut to_remove = vec![];
    // for ((code, input), out) in fused_codes.iter().zip(fused_inputs.iter()).zip(fused_outs.iter()) {
    //     println!(
    //         "input: {:#?}",
    //         input
    //             .iter()
    //             .map(|i| i.to_string())
    //             .collect::<Vec<String>>()
    //     );
    //     println!("code: {:#?}", code.to_string());
    //     println!("out: {:#?}", out.to_string());
    // }

    for (input, total) in fused_inputs.iter().zip(fused.iter()) {
        let mut intermediate = total
            .iter()
            .map(|i| i.ident.clone())
            .collect::<HashSet<_>>();
        for input in input {
            intermediate.remove(input);
        }
        to_remove.push(intermediate);
    }

    let mut codes = Vec::new();
    for (i, code) in fused_codes.iter().enumerate() {
        let out = &fused_outs[i];
        codes.push(quote::quote!(
                let #out = #code;
            ));
    }
    // println!(
    //     "intermediates: {:#?}",
    //     to_remove
    //         .iter()
    //         .map(|i|
    //             i
    //                 .iter()
    //                 .map(|i| i.to_string())
    //                 .collect::<Vec<String>>()
    //         )
    //         .collect::<Vec<Vec<String>>>()
    // );
    // println!("fused_outs: {:#?}", fused_outs);
    let mut codegen = Codegen {
        fused_codes: codes,
        to_remove,
        current_tokens: Vec::new(),
        current_idx: 0,
    };
    codegen.visit_item_fn(&func);
    let code = codegen.get_code();

    let vis = func.vis.clone();
    let sig = func.sig.clone();
    let ret = quote::quote!(
        #vis #sig {
            #code
        }
    );
    ret.into()
}

pub(crate) fn fuse_proc_macro(item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    fuse_impl(item)
}
