use quote::ToTokens;
use syn::visit::Visit;

use crate::fuse::{ dag::Graph, fuse::fuse, gen_fuse::gen_fuse };

use super::{ dag::Var, node::Node, visitor::Visitor };

pub(crate) fn fuse_impl(
    item: proc_macro::TokenStream
) -> proc_macro::TokenStream {
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
    let code = if visitor.nodes.len() > 1 {
        let graph = Graph::from_nodes(&visitor.nodes);
        println!("{:#?}", graph);
        let fused = fuse(&graph);
        println!("{:#?}", fused);
        let fused_codes = gen_fuse(&graph, &fused);
        fused_codes[0].clone()
    } else {
        visitor.code.clone()
    };

    let vis = func.vis.clone();
    let sig = func.sig.clone();
    let ret = quote::quote!(
        #vis #sig {
            Ok(#code)
        }
    );
    println!("{:#?}", ret.to_string());
    (ret).into()
}

pub(crate) fn fuse_proc_macro(
    item: proc_macro::TokenStream
) -> proc_macro::TokenStream {
    fuse_impl(item)
}