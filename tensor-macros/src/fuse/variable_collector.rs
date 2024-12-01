use std::collections::HashSet;

use syn::visit::Visit;

pub(crate) struct VariableCollector {
    pub(crate) vars: HashSet<String>,
}

impl VariableCollector {
    pub(crate) fn new() -> Self {
        Self { vars: HashSet::new() }
    }
}

impl<'ast> Visit<'ast> for VariableCollector {
    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        self.visit_expr(&node.receiver);
        for arg in node.args.iter() {
            self.visit_expr(arg);
        }
    }
    fn visit_ident(&mut self, i: &'ast proc_macro2::Ident) {
        self.vars.insert(i.to_string());
    }
    fn visit_expr_path(&mut self, i: &'ast syn::ExprPath) {
        if let Some(ident) = i.path.get_ident() {
            self.vars.insert(ident.to_string());
        }
    }
    fn visit_generic_argument(&mut self, _: &'ast syn::GenericArgument) {}
    fn visit_type_path(&mut self, _: &'ast syn::TypePath) {}
    fn visit_pat(&mut self, i: &'ast syn::Pat) {
        match i {
            syn::Pat::Ident(i) => self.visit_ident(&i.ident),
            syn::Pat::Path(path) => {
                if let Some(ident) = path.path.get_ident() {
                    self.vars.insert(ident.to_string());
                }
            }
            syn::Pat::Type(ty) => {
                self.visit_pat(&ty.pat);
            }
            _ => {}
        }
    }
}
