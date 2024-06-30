

use std::fmt::Display;

use super::{prime_expr::PrimeExpr, stmt::Stmt, traits::{IRMutVisitor, IRMutateVisitor, IRVisitor}, variable::Variable};


#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct AssignStmt {
    pub lhs: Variable,
    pub rhs: PrimeExpr,
}

impl AssignStmt {
    pub fn make<T: Into<Variable>, U: Into<PrimeExpr>>(lhs: T, rhs: U) -> Self {
        AssignStmt {
            lhs: lhs.into(),
            rhs: rhs.into(),
        }
    }

    pub fn new(lhs: Variable, rhs: PrimeExpr) -> Self {
        AssignStmt { lhs, rhs }
    }

    pub fn lhs(&self) -> &Variable {
        &self.lhs
    }

    pub fn rhs(&self) -> &PrimeExpr {
        &self.rhs
    }

    pub fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_assign(self);
    }
    pub fn accept_mut<V: IRMutVisitor>(&self, visitor: &mut V) {
        visitor.visit_assign(self);
    }
    pub fn accept_mutate<V: IRMutateVisitor>(&self, visitor: &mut V) {
        visitor.visit_assign(self);
    }
}

impl Display for AssignStmt {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} = {};", self.lhs, self.rhs)
    }
}

impl Into<Stmt> for AssignStmt {
    fn into(self) -> Stmt {
        Stmt::AssignStmt(self)
    }
}

impl Into<Stmt> for &AssignStmt {
    fn into(self) -> Stmt {
        Stmt::AssignStmt(self.clone())
    }
}