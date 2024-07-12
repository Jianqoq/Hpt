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
    value: Arc<PrimeExpr>,
    body: Arc<Stmt>,
}

impl LetStmt {
    pub fn var(&self) -> &Variable {
        &self.var
    }
    pub fn value(&self) -> &PrimeExpr {
        &self.value
    }
    pub fn body(&self) -> &Stmt {
        &self.body
    }

    pub fn make<A: Into<PrimeExpr>, T: Into<Stmt>>(var: &Variable, value: A, body: T) -> Self {
        LetStmt {
            var: var.clone(),
            body: body.into().into(),
            value: value.into().into(),
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
