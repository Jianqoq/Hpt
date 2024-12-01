use std::collections::HashMap;

pub(crate) struct VarRecover<'ast> {
    pub(crate) origin_var_map: &'ast HashMap<syn::Ident, syn::Ident>,
}

impl<'ast> VarRecover<'ast> {
    pub(crate) fn new(origin_var_map: &'ast HashMap<syn::Ident, syn::Ident>) -> Self {
        Self { origin_var_map }
    }
}

impl<'ast> syn::visit_mut::VisitMut for VarRecover<'ast> {
    fn visit_ident_mut(&mut self, i: &mut syn::Ident) {
        if let Some(origin_var) = self.origin_var_map.get(i) {
            *i = origin_var.clone();
        }
    }
    fn visit_macro_mut(&mut self, i: &mut syn::Macro) {
        let mut new_tokens = proc_macro2::TokenStream::new();
        let copy = i.tokens.clone();
        for token in copy {
            if let proc_macro2::TokenTree::Ident(mut ident) = token {
                if let Some(origin_var) = self.origin_var_map.get(&ident) {
                    ident = origin_var.clone();
                }
                new_tokens.extend(quote::quote!(#ident));
            } else {
                new_tokens.extend(quote::quote!(#token));
            }
        }
        i.tokens = new_tokens;
    }
}
