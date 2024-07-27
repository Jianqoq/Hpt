use std::sync::Arc;

use tensor_types::dtype::Dtype;

use super::{
    prime_expr::PrimeExpr,
    exprs::Int,
    stmt::Stmt,
    traits::{ IRMutVisitor, IRMutateVisitor, IRVisitor },
    variable::Variable,
};
use crate::halide::traits::{ Accepter, AccepterMut };

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct StoreStmt {
    pub(crate) var: Variable,
    pub(crate) indices: Arc<PrimeExpr>,
    pub(crate) val: Arc<PrimeExpr>,
}

impl StoreStmt {
    pub fn var(&self) -> &Variable {
        &self.var
    }

    pub fn indices(&self) -> &PrimeExpr {
        &self.indices
    }

    pub fn val(&self) -> &PrimeExpr {
        &self.val
    }

    pub fn make_from_strides<T: Into<PrimeExpr>>(
        var: &Variable,
        indices: &[&Variable],
        strides: &[i64],
        e2: T
    ) -> Self {
        if indices.len() != strides.len() {
            panic!("Indices and strides must have the same length");
        }
        let sum = strides
            .iter()
            .zip(indices.iter())
            .map(|(stride, index)| {
                *index * Int::make(Dtype::I64, *stride)
            })
            .reduce(|acc, e| acc + e)
            .expect("Failed to reduce");
        StoreStmt {
            var: var.clone(),
            indices: Arc::new(sum),
            val: Arc::new(e2.into()),
        }
    }

    pub fn make<A: Into<PrimeExpr>, B: Into<PrimeExpr>>(var: &Variable, indices: A, e2: B) -> Self {
        StoreStmt {
            var: var.clone(),
            indices: indices.into().into(),
            val: e2.into().into(),
        }
    }

    pub fn accept<V: IRVisitor>(&self, visitor: &V) {
        self.var.accept(visitor);
        self.indices.accept(visitor);
        self.val.accept(visitor);
    }

    pub fn accept_mut<V: IRMutVisitor>(&self, visitor: &mut V) {
        self.var.accept_mut(visitor);
        self.indices.accept_mut(visitor);
        self.val.accept_mut(visitor);
    }

    pub fn accept_mutate<V: IRMutateVisitor>(&self, visitor: &mut V) {
        visitor.visit_store(self);
    }
}

impl std::fmt::Display for StoreStmt {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}[{}] = {};", self.var, self.indices, self.val)
    }
}

impl Into<Stmt> for StoreStmt {
    fn into(self) -> Stmt {
        Stmt::StoreStmt(self)
    }
}

impl Into<Stmt> for &StoreStmt {
    fn into(self) -> Stmt {
        Stmt::StoreStmt(self.clone())
    }
}
