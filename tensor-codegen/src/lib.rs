use proc_macro::TokenStream;

pub(crate) mod fuse {
    pub mod start;
    pub(crate) mod node;
    pub(crate) mod fuse;
    pub(crate) mod kernel_type;
    pub(crate) mod gen_fuse;
    pub(crate) mod codegen;
    pub(crate) mod cfg;
    pub(crate) mod ty_infer;
    pub(crate) mod expr_ty;
    pub(crate) mod build_graph;
    pub(crate) mod use_define_visitor;
    pub(crate) mod variable_collector;
    pub(crate) mod phi_function;
    pub(crate) mod var_recover;
    pub(crate) mod expr_call_use_visitor;
    pub(crate) mod expr_expand;
    pub(crate) mod controlflow_detector;
    pub(crate) mod cfg_builder;
}

#[proc_macro]
pub fn fuse_proc_macro(item: TokenStream) -> TokenStream {
    let func = syn::parse_macro_input!(item as syn::ItemFn);
    fuse::start::fuse_impl(func).into()
}

