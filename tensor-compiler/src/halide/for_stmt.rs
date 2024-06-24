use std::sync::Arc;

use super::{
    expr::Expr,
    stmt::Stmt,
    traits::{ IRMutVisitor, IRMutateVisitor, IRVisitor },
    variable::Variable,
};

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct For {
    var: Variable,
    start: Arc<Expr>,
    end: Arc<Expr>,
    stmt: Arc<Stmt>,
}

impl For {
    pub fn var(&self) -> &Variable {
        &self.var
    }

    pub fn start(&self) -> &Expr {
        &self.start
    }

    pub fn end(&self) -> &Expr {
        &self.end
    }

    pub fn stmt(&self) -> &Stmt {
        &self.stmt
    }

    pub fn set_stmt<T: Into<Stmt>>(&mut self, stmt: T) {
        self.stmt = stmt.into().into();
    }

    pub fn make<T: Into<Expr> + Clone, S: Into<Stmt> + Clone>(
        var: &Variable,
        start: T,
        end: T,
        stmt: S
    ) -> Self {
        For {
            var: var.clone(),
            start: Arc::new(start.clone().into()),
            end: Arc::new(end.clone().into()),
            stmt: Arc::new(stmt.into()),
        }
    }

    pub fn accept_mut<V: IRMutVisitor>(&self, visitor: &mut V) {
        visitor.visit_for(self);
    }

    pub fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_for(self);
    }

    pub fn accept_mutate<V: IRMutateVisitor>(&self, visitor: &mut V) {
        visitor.visit_for(self);
    }
}

impl std::fmt::Display for For {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "for {} in {}..{} {{\n{}\n}}", self.var, self.start, self.end, self.stmt)
    }
}

impl Into<Stmt> for For {
    fn into(self) -> Stmt {
        Stmt::For(self)
    }
}

impl Into<Stmt> for &For {
    fn into(self) -> Stmt {
        Stmt::For(self.clone())
    }
}
