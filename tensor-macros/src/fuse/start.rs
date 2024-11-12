use crate::fuse::{
    codegen::{ Codegen, _Codegen },
    dag::Graph,
    fuse::fuse_graph,
    gen_fuse::gen_fuse,
    rcmut::RCMut,
    ssa::SSAContext,
    to_remove::gen_to_remove,
    visitor::Visitor,
};
use syn::visit::Visit;

pub(crate) fn fuse_impl(item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let func = syn::parse_macro_input!(item as syn::ItemFn);
    let mut visitor = Visitor::new();
    visitor.visit_item_fn(&func);
    if !visitor.visitor.errors.is_empty() {
        // 合并所有错误
        let combined_error = visitor.visitor.errors
            .into_iter()
            .reduce(|mut acc, e| {
                acc.combine(e);
                acc
            })
            .unwrap();
        return combined_error.to_compile_error().into();
    }
    visitor.remove_unused();
    let graph = Graph::from_visitor(&visitor.visitor);
    let fused = fuse_graph(&graph);
    let gen_fuse = gen_fuse(&graph._graph, &fused);
    let to_remove = gen_to_remove(&gen_fuse, &fused);

    let mut codegen = Codegen {
        _codegen: _Codegen {
            fused_codes: &gen_fuse,
            to_remove: &to_remove,
            current_tokens: Vec::new(),
            ssa_ctx: RCMut::new(SSAContext::new()),
            _visitor: &visitor.visitor,
            next_codegen: None,
            pat_ident_need_remove: false,
            pat_ident_is_ret: false,
        },
    };
    codegen.visit_item_fn(&func);
    let code = codegen.get_code();

    let ret = quote::quote!(
            #code
    );
    ret.into()
}

pub(crate) fn fuse_proc_macro(item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    fuse_impl(item)
}
