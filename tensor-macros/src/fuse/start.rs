use syn::visit::Visit;

use crate::fuse::{ dag::Graph, fuse::fuse };

use super::visitor::Visitor;

pub(crate) fn fuse_impl(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream
) -> proc_macro::TokenStream {
    let mut func = syn::parse_macro_input!(item as syn::ItemFn);
    let mut visitor = Visitor::new();
    visitor.visit_item_fn(&func);
    let code = if visitor.nodes.len() > 1 {
        let graph = Graph::from_nodes(&visitor.nodes);
        println!("{:#?}", graph);
        let fused = fuse(&graph);
        println!("{:#?}", fused);
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
