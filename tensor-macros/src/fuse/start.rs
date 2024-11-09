use quote::ToTokens;
use syn::visit::Visit;

use crate::fuse::{ dag::Graph, fuse::fuse, gen_fuse::gen_fuse };

use super::{ dag::Var, node::Node, visitor::Visitor };

pub(crate) fn fuse_impl(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream
) -> proc_macro::TokenStream {
    let mut func = syn::parse_macro_input!(item as syn::ItemFn);
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
        gen_fuse(&graph, &fused);
        visitor.code.clone()
    } else {
        visitor.code.clone()
    };

    // 创建新的函数名
    let new_name = syn::Ident::new("test", func.sig.ident.span());
    func.sig.ident = new_name;
    (quote::quote! {
        #func
    }).into()
}
