use crate::halide::{
    expr::Expr,
    ir_cmp::expr_equal,
    let_stmt::LetStmt,
    stmt::Stmt,
    traits::{ mutate_expr, IRMutateVisitor, MutatorGetSet },
};

#[derive(Clone, Debug)]
pub struct SubstituteExpr {
    find: Expr,
    replace: Expr,
    stmt: Stmt,
    expr: Expr,
}

impl SubstituteExpr {
    pub fn new() -> Self {
        SubstituteExpr {
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

impl MutatorGetSet for SubstituteExpr {
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

impl IRMutateVisitor for SubstituteExpr {
    fn mutate_expr(&mut self, expr: &Expr) -> Expr {
        if expr_equal(&self.find, expr) {
            return self.replace.clone();
        } else {
            return mutate_expr(self, expr);
        }
    }

    fn visit_let_stmt(&mut self, let_stmt: &LetStmt) {
        let body = self.mutate_expr(let_stmt.body());
        if body.same_as(let_stmt.body()) {
            self.set_stmt(let_stmt);
        } else {
            self.set_stmt(LetStmt::make(let_stmt.var(), body));
        }
    }
}
