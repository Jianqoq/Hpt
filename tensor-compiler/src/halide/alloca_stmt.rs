use std::sync::Arc;

use super::{prime_expr::PrimeExpr, primitive_type::PrimitiveType, stmt::Stmt, traits::{IRMutVisitor, IRMutateVisitor, IRVisitor}, variable::Variable};

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct AllocaStmt {
    var: Variable,
    dtype: PrimitiveType,
    size: PrimeExpr,
    body: Arc<Stmt>,
}

impl AllocaStmt {
    pub fn var(&self) -> &Variable {
        &self.var
    }
    pub fn size(&self) -> &PrimeExpr {
        &self.size
    }
    pub fn body(&self) -> &Stmt {
        &self.body
    }
    pub fn dtype(&self) -> &PrimitiveType {
        &self.dtype
    }

    pub fn make<A: Into<PrimeExpr>, B: Into<Stmt>>(var: &Variable, dtype: PrimitiveType, size: A, body: B) -> Self {
        AllocaStmt {
            var: var.clone(),
            dtype,
            size: size.into(),
            body: body.into().into(),
        }
    }

    pub fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_alloca(self);
    }

    pub fn accept_mutate<V: IRMutateVisitor>(&self, visitor: &mut V) {
        visitor.visit_alloca(self);
    }

    pub fn accept_mut<V: IRMutVisitor>(&self, visitor: &mut V) {
        visitor.visit_alloca(self);
    }
}

impl std::fmt::Display for AllocaStmt {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "let {} = alloca<{}>({});", self.var, self.dtype, self.size)
    }
}



impl Into<Stmt> for AllocaStmt {
    fn into(self) -> Stmt {
        Stmt::AllocaStmt(self)
    }
}

impl Into<Stmt> for &AllocaStmt {
    fn into(self) -> Stmt {
        Stmt::AllocaStmt(self.clone())
    }
}