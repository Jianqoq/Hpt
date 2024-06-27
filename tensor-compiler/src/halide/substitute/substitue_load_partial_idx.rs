use crate::halide::{
    prime_expr::PrimeExpr,
    exprs::Load,
    ir_cmp::expr_equal,
    stmt::Stmt,
    traits::{ mutate_expr, IRMutateVisitor, MutatorGetSet },
};

pub struct SubstituteLoadPartialIdx {
    load_var: PrimeExpr,
    find: PrimeExpr,
    replace: PrimeExpr,
    stmt: Stmt,
    expr: PrimeExpr,
}

impl SubstituteLoadPartialIdx {
    pub fn new<T: Into<PrimeExpr>>(load_var: T) -> Self {
        SubstituteLoadPartialIdx {
            load_var: load_var.into(),
            find: PrimeExpr::None,
            replace: PrimeExpr::None,
            stmt: Stmt::None,
            expr: PrimeExpr::None,
        }
    }

    pub fn find(&self) -> &PrimeExpr {
        &self.find
    }

    pub fn replace(&self) -> &PrimeExpr {
        &self.replace
    }

    pub fn set_find<T: Into<PrimeExpr>>(&mut self, find: T) {
        self.find = find.into();
    }

    pub fn set_replace<T: Into<PrimeExpr>>(&mut self, replace: T) {
        self.replace = replace.into();
    }
}

impl MutatorGetSet for SubstituteLoadPartialIdx {
    fn set_expr<T: Into<PrimeExpr>>(&mut self, expr: T) {
        self.expr = expr.into();
    }

    fn set_stmt<T: Into<Stmt>>(&mut self, stmt: T) {
        self.stmt = stmt.into();
    }

    fn expr(&self) -> &PrimeExpr {
        &self.expr
    }

    fn stmt(&self) -> &Stmt {
        &self.stmt
    }
}

impl IRMutateVisitor for SubstituteLoadPartialIdx {
    fn mutate_expr(&mut self, expr: &PrimeExpr) -> PrimeExpr {
        if expr_equal(&self.find, expr) {
            return self.replace.clone();
        } else {
            return mutate_expr(self, expr);
        }
    }

    fn visit_load(&mut self, load: &Load) {
        if expr_equal(load.name(), &self.load_var) {
            let indices = load.indices().iter().map(|idx| self.mutate_expr(idx)).collect::<Vec<_>>();
            let has_new = indices.iter().any(|idx| expr_equal(idx, &self.replace));
            if has_new {
                self.set_expr(Load::make(load.name(), &indices));
            } else {
                self.set_expr(load.clone());
            }
        } else {
            self.set_expr(load);
        }
    }
}
