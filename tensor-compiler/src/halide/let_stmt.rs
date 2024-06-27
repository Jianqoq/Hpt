use std::sync::Arc;

use super::{
    prime_expr::PrimeExpr,
    stmt::Stmt,
    traits::{ IRMutVisitor, IRMutateVisitor, IRVisitor },
    variable::Variable,
};

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct LetStmt {
    var: Variable,
    body: Arc<PrimeExpr>,
}

impl LetStmt {
    pub fn var(&self) -> &Variable {
        &self.var
    }

    pub fn body(&self) -> &PrimeExpr {
        &self.body
    }

    pub fn make<T: Into<PrimeExpr>>(var: &Variable, body: T) -> Self {
        LetStmt {
            var: var.clone(),
            body: body.into().into(),
        }
    }

    pub fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_let_stmt(self);
    }

    pub fn accept_mutate<V: IRMutateVisitor>(&self, visitor: &mut V) {
        visitor.visit_let_stmt(self);
    }

    pub fn accept_mut<V: IRMutVisitor>(&self, visitor: &mut V) {
        visitor.visit_let_stmt(self);
    }
}

impl std::fmt::Display for LetStmt {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "let {} = {};", self.var, self.body)
    }
}

impl Into<Stmt> for LetStmt {
    fn into(self) -> Stmt {
        Stmt::LetStmt(self)
    }
}

impl Into<Stmt> for &LetStmt {
    fn into(self) -> Stmt {
        Stmt::LetStmt(self.clone())
    }
}
