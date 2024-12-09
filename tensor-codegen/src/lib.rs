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
    pub(crate) mod cfg_builder;
    pub(crate) mod errors;
    pub(crate) mod operator_lists;
    pub(crate) mod dead_node_elimination;
    pub(crate) mod var_coalescer;
}

#[proc_macro]
pub fn fuse_proc_macro(item: TokenStream) -> TokenStream {
    let func = syn::parse_macro_input!(item as syn::ItemFn);
    match fuse::start::fuse_impl(func) {
        Ok(ret) => ret.into(),
        Err(e) => e.downcast::<syn::Error>().unwrap().to_compile_error().into(),
    }
}

#[proc_macro_attribute]
pub fn compile(_: TokenStream, item: TokenStream) -> TokenStream {
    fuse_proc_macro(item)
}

