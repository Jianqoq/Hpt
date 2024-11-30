use std::collections::HashSet;
use syn::visit::Visit;

use super::variable_collector::VariableCollector;

pub(crate) struct UseDefineVisitor {
    pub(crate) used_vars: HashSet<String>,
    pub(crate) define_vars: HashSet<String>,
    pub(crate) assigned_vars: HashSet<String>,
}

impl UseDefineVisitor {
    pub(crate) fn new() -> Self {
        Self {
            used_vars: HashSet::new(),
            define_vars: HashSet::new(),
            assigned_vars: HashSet::new(),
        }
    }
}

impl<'ast> Visit<'ast> for UseDefineVisitor {
    fn visit_expr_assign(&mut self, node: &'ast syn::ExprAssign) {
        let mut collector = VariableCollector::new();
        if let syn::Expr::Path(left) = node.left.as_ref() {
            if let Some(ident) = left.path.get_ident() {
                self.assigned_vars.insert(ident.to_string());
            }
        }
        collector.visit_expr(node.right.as_ref());
        self.used_vars.extend(collector.vars);
    }
    fn visit_local(&mut self, i: &'ast syn::Local) {
        let mut collector = VariableCollector::new();
        collector.visit_pat(&i.pat);
        self.define_vars.extend(collector.vars);
        if let Some(init) = &i.init {
            let mut collector = VariableCollector::new();
            collector.visit_local_init(init);
            self.used_vars.extend(collector.vars);
        }
    }
    fn visit_expr_binary(&mut self, i: &'ast syn::ExprBinary) {
        if let syn::Expr::Path(left) = i.left.as_ref() {
            if let Some(ident) = left.path.get_ident() {
                self.used_vars.insert(ident.to_string());
            }
        } else {
            self.visit_expr(i.left.as_ref());
        }
        if let syn::Expr::Path(right) = i.right.as_ref() {
            if let Some(ident) = right.path.get_ident() {
                self.used_vars.insert(ident.to_string());
            }
        } else {
            self.visit_expr(i.right.as_ref());
        }
    }
}
