use std::collections::HashSet;
use syn::visit::Visit;

use super::variable_collector::VariableCollector;

pub(crate) struct UseDefineVisitor {
    pub(crate) used_vars: HashSet<String>,
    pub(crate) define_vars: HashSet<String>,
}

impl UseDefineVisitor {
    pub(crate) fn new() -> Self {
        Self { used_vars: HashSet::new(), define_vars: HashSet::new() }
    }
}

impl<'ast> Visit<'ast> for UseDefineVisitor {
    fn visit_expr_assign(&mut self, node: &'ast syn::ExprAssign) {
        let mut collector = VariableCollector::new();
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
}
