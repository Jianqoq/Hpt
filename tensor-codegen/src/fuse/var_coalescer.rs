use petgraph::graph::NodeIndex;

use super::cfg::CFG;

pub(crate) struct VarCoalescer<'ast> {
    pub(crate) cfg: &'ast mut CFG,
    pub(crate) current_block: NodeIndex,
}

impl<'ast> VarCoalescer<'ast> {
    pub(crate) fn new(cfg: &'ast mut CFG) -> Self {
        let entry = cfg.entry;
        Self { cfg, current_block: entry }
    }

    pub(crate) fn run(&mut self) {
        for node in self.cfg.graph.node_indices() {
            self.current_block = node;
            let block = &mut self.cfg.graph[node];
            let mut rhs = syn::Expr::Verbatim(quote::quote! {});
            let mut lhs = syn::Ident::new("a", proc_macro2::Span::call_site());
            let mut to_remove = Vec::new();
            for (idx, stmt) in block.statements.iter_mut().enumerate() {
                match &mut stmt.stmt {
                    syn::Stmt::Local(local) => {
                        if let Some(init) = &mut local.init {
                            if let syn::Expr::Try(expr_try) = &mut *init.expr {
                                if let syn::Expr::Path(path) = &mut *expr_try.expr {
                                    if let Some(ident) = path.path.get_ident() {
                                        if ident == &lhs {
                                            *expr_try.expr = rhs.clone();
                                            to_remove.push(idx - 1);
                                        }
                                    }
                                }
                            }
                        }
                        if let syn::Pat::Ident(pat) = &local.pat {
                            lhs = pat.ident.clone();
                        }
                        if let Some(init) = &local.init {
                            rhs = *init.expr.clone();
                        }
                    }
                    syn::Stmt::Item(_) => {}
                    syn::Stmt::Expr(_, _) => {}
                    syn::Stmt::Macro(_) => {}
                }
            }
            let mut new_statements = Vec::new();
            for (idx, stmt) in block.statements.drain(..).enumerate() {
                if !to_remove.contains(&idx) {
                    new_statements.push(stmt.clone());
                }
            }
            block.statements = new_statements;
        }
    }
}

impl<'ast> syn::visit_mut::VisitMut for VarCoalescer<'ast> {}
