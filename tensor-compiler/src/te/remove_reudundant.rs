use std::collections::HashSet;

use crate::halide::{
    prime_expr::PrimeExpr,
    stmt::Stmt,
    traits::{ IRMutateVisitor, MutatorGetSet },
    variable::Variable,
};

pub struct RemoveRedundant {
    pub(crate) set: HashSet<Variable>,
    pub(crate) expr: PrimeExpr,
    pub(crate) stmt: Stmt,
}

impl RemoveRedundant {
    pub fn new() -> Self {
        Self {
            set: HashSet::new(),
            expr: PrimeExpr::None,
            stmt: Stmt::None,
        }
    }
}

impl MutatorGetSet for RemoveRedundant {
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

impl IRMutateVisitor for RemoveRedundant {
    fn visit_let_stmt(&mut self, let_stmt: &crate::halide::let_stmt::LetStmt) {
        if let None = self.set.get(let_stmt.var()) {
            self.set.insert(let_stmt.var().clone());
            self.set_stmt(let_stmt.clone());
        }
    }
}
