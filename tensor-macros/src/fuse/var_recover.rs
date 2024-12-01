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
}
