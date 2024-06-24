use crate::halide::{
    expr::Expr,
    exprs::Load,
    ir_cmp::expr_equal,
    stmt::Stmt,
    traits::{ mutate_expr, IRMutateVisitor, MutatorGetSet },
};

pub struct SubstituteLoadPartialIdx {
    load_var: Expr,
    find: Expr,
    replace: Expr,
    stmt: Stmt,
    expr: Expr,
}

impl SubstituteLoadPartialIdx {
    pub fn new<T: Into<Expr>>(load_var: T) -> Self {
        SubstituteLoadPartialIdx {
            load_var: load_var.into(),
            find: Expr::None,
            replace: Expr::None,
            stmt: Stmt::None,
            expr: Expr::None,
        }
    }

    pub fn find(&self) -> &Expr {
        &self.find
    }

    pub fn replace(&self) -> &Expr {
        &self.replace
    }

    pub fn set_find<T: Into<Expr>>(&mut self, find: T) {
        self.find = find.into();
    }

    pub fn set_replace<T: Into<Expr>>(&mut self, replace: T) {
        self.replace = replace.into();
    }
}

impl MutatorGetSet for SubstituteLoadPartialIdx {
    fn set_expr<T: Into<Expr>>(&mut self, expr: T) {
        self.expr = expr.into();
    }

    fn set_stmt<T: Into<Stmt>>(&mut self, stmt: T) {
        self.stmt = stmt.into();
    }

    fn expr(&self) -> &Expr {
        &self.expr
    }

    fn stmt(&self) -> &Stmt {
        &self.stmt
    }
}

impl IRMutateVisitor for SubstituteLoadPartialIdx {
    fn mutate_expr(&mut self, expr: &Expr) -> Expr {
        if expr_equal(&self.find, expr) {
            return self.replace.clone();
        } else {
            return mutate_expr(self, expr);
        }
    }

    fn visit_load(&mut self, load: &Load) {
        if expr_equal(load.name(), &self.load_var) {
            let indices = self.mutate_expr(load.indices());
            if indices.same_as(load.indices()) {
                self.set_expr(load);
            } else {
                self.set_expr(Expr::Load(Load::make(load.name(), &indices)));
            }
        } else {
            self.set_expr(load);
        }
    }
}
