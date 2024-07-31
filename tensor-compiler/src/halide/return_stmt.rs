use std::{ fmt::Display, sync::Arc };

use colored::Colorize;

use super::{
    prime_expr::PrimeExpr,
    stmt::Stmt,
    traits::{ Accepter, AccepterMut, AccepterMutate, IRMutVisitor, IRMutateVisitor, IRVisitor },
};

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct ReturnStmt {
    exprs: Arc<Vec<PrimeExpr>>,
}

impl ReturnStmt {
    pub fn exprs(&self) -> &Vec<PrimeExpr> {
        &self.exprs
    }

    pub fn make<A: Into<Vec<PrimeExpr>>>(exprs: A) -> Self {
        ReturnStmt {
            exprs: Arc::new(exprs.into()),
        }
    }

    pub fn accept<V: IRVisitor>(&self, visitor: &V) {
        for e in self.exprs.iter() {
            e.accept(visitor);
        }
    }

    pub fn accept_mut<V: IRMutVisitor>(&self, visitor: &mut V) {
        for e in self.exprs.iter() {
            e.accept_mut(visitor);
        }
    }

    pub fn accept_mutate<V: IRMutateVisitor>(&self, visitor: &mut V) {
        for e in self.exprs.iter() {
            e.accept_mutate(visitor);
        }
    }
}

impl Display for ReturnStmt {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.exprs.is_empty() {
            return write!(f, "{};", "return".purple());
        }
        write!(f, "{} ", "return".purple())?;
        for (i, e) in self.exprs.iter().enumerate() {
            write!(f, "{}", e)?;
            if i != self.exprs.len() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, ";")
    }
}

impl Into<Stmt> for ReturnStmt {
    fn into(self) -> Stmt {
        Stmt::Return(self)
    }
}

impl Into<Stmt> for &ReturnStmt {
    fn into(self) -> Stmt {
        Stmt::Return(self.clone())
    }
}
