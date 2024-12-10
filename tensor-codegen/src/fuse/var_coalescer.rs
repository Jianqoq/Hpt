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
            let mut lhs = syn::Ident::new("_____________", proc_macro2::Span::call_site());
            let mut to_remove = Vec::new();
            for (idx, stmt) in block.statements.iter_mut().enumerate() {
                match &mut stmt.stmt {
                    syn::Stmt::Local(local) => {
                        if let Some(init) = &mut local.init {
                            match &mut *init.expr {
                                syn::Expr::Try(expr_try) => {
                                    if let syn::Expr::Path(path) = &mut *expr_try.expr {
                                        if let Some(ident) = path.path.get_ident() {
                                            if ident == &lhs {
                                                *expr_try.expr = rhs.clone();
                                                to_remove.push(idx - 1);
                                                block.defined_vars.remove(&block.origin_var_map[&lhs]);
                                                block.used_vars.remove(&block.origin_var_map[&lhs]);
                                                block.origin_var_map.remove(&lhs);
                                            }
                                        }
                                    }
                                }
                                syn::Expr::Path(path) => {
                                    if let Some(ident) = path.path.get_ident() {
                                        if ident == &lhs {
                                            *init.expr = rhs.clone();
                                            to_remove.push(idx - 1);
                                            block.defined_vars.remove(&block.origin_var_map[&lhs]);
                                            block.used_vars.remove(&block.origin_var_map[&lhs]);
                                            block.origin_var_map.remove(&lhs);
                                        }
                                    }
                                }
                                syn::Expr::Reference(reference) => {
                                    if let syn::Expr::Path(path) = &mut *reference.expr {
                                        if let Some(ident) = path.path.get_ident() {
                                            if ident == &lhs {
                                                *reference.expr = rhs.clone();
                                                to_remove.push(idx - 1);
                                                block.defined_vars.remove(&block.origin_var_map[&lhs]);
                                                block.used_vars.remove(&block.origin_var_map[&lhs]);
                                                block.origin_var_map.remove(&lhs);
                                            }
                                        }
                                    }
                                }
                                _ => {}
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
                    syn::Stmt::Expr(expr, _) => {
                        if let syn::Expr::Path(path) = &mut *expr {
                            if let Some(ident) = path.path.get_ident() {
                                if ident == &lhs {
                                    *expr = rhs.clone();
                                    to_remove.push(idx - 1);
                                    block.defined_vars.remove(&block.origin_var_map[&lhs]);
                                    block.used_vars.remove(&block.origin_var_map[&lhs]);
                                    block.origin_var_map.remove(&lhs);
                                }
                            }
                        }
                    }
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
