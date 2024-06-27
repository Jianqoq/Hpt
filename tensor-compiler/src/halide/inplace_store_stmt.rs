use std::{ fmt::Display, sync::Arc };

use super::{ prime_expr::PrimeExpr, stmt::Stmt, traits::{ IRMutVisitor, IRMutateVisitor, IRVisitor } };

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct InplaceStore {
    to_store: Arc<PrimeExpr>,
    val: Arc<PrimeExpr>,
}

impl InplaceStore {
    pub fn make<T: Into<PrimeExpr>>(to_store: T, val: T) -> Self {
        InplaceStore {
            to_store: to_store.into().into(),
            val: val.into().into(),
        }
    }

    pub fn new<T: Into<PrimeExpr>>(to_store: T, val: T) -> Self {
        InplaceStore {
            to_store: to_store.into().into(),
            val: val.into().into(),
        }
    }

    pub fn to_store(&self) -> &PrimeExpr {
        &self.to_store
    }

    pub fn val(&self) -> &PrimeExpr {
        &self.val
    }

    pub fn accept_mut<V: IRMutVisitor>(&self, visitor: &mut V) {
        visitor.visit_inplace_store(self);
    }

    pub fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_inplace_store(self);
    }

    pub fn accept_mutate<V: IRMutateVisitor>(&self, visitor: &mut V) {
        visitor.visit_inplace_store(self);
    }
}

impl Into<Stmt> for InplaceStore {
    fn into(self) -> Stmt {
        Stmt::InplaceStore(self)
    }
}

impl Into<Stmt> for &InplaceStore {
    fn into(self) -> Stmt {
        Stmt::InplaceStore(self.clone())
    }
}

impl Display for InplaceStore {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} = {};", self.to_store, self.val)
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct InplaceAdd {
    to_store: Arc<PrimeExpr>,
    val: Arc<PrimeExpr>,
}

impl InplaceAdd {
    pub fn make<T: Into<PrimeExpr>>(to_store: T, val: T) -> Self {
        InplaceAdd {
            to_store: to_store.into().into(),
            val: val.into().into(),
        }
    }

    pub fn new<T: Into<PrimeExpr>>(to_store: T, val: T) -> Self {
        InplaceAdd {
            to_store: to_store.into().into(),
            val: val.into().into(),
        }
    }

    pub fn to_store(&self) -> &PrimeExpr {
        &self.to_store
    }

    pub fn val(&self) -> &PrimeExpr {
        &self.val
    }

    pub fn accept_mut<V: IRMutVisitor>(&self, visitor: &mut V) {
        visitor.visit_inplace_add(self);
    }

    pub fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_inplace_add(self);
    }

    pub fn accept_mutate<V: IRMutateVisitor>(&self, visitor: &mut V) {
        visitor.visit_inplace_add(self);
    }
}

impl Into<Stmt> for InplaceAdd {
    fn into(self) -> Stmt {
        Stmt::InplaceAdd(self)
    }
}

impl Into<Stmt> for &InplaceAdd {
    fn into(self) -> Stmt {
        Stmt::InplaceAdd(self.clone())
    }
}

impl Display for InplaceAdd {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} += {};", self.to_store, self.val)
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct InplaceSub {
    to_store: Arc<PrimeExpr>,
    val: Arc<PrimeExpr>,
}

impl InplaceSub {
    pub fn make<T: Into<PrimeExpr>>(to_store: T, val: T) -> Self {
        InplaceSub {
            to_store: to_store.into().into(),
            val: val.into().into(),
        }
    }

    pub fn to_store(&self) -> &PrimeExpr {
        &self.to_store
    }

    pub fn val(&self) -> &PrimeExpr {
        &self.val
    }

    pub fn accept_mut<V: IRMutVisitor>(&self, visitor: &mut V) {
        visitor.visit_inplace_sub(self);
    }

    pub fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_inplace_sub(self);
    }

    pub fn accept_mutate<V: IRMutateVisitor>(&self, visitor: &mut V) {
        visitor.visit_inplace_sub(self);
    }
}

impl Into<Stmt> for InplaceSub {
    fn into(self) -> Stmt {
        Stmt::InplaceSub(self)
    }
}

impl Into<Stmt> for &InplaceSub {
    fn into(self) -> Stmt {
        Stmt::InplaceSub(self.clone())
    }
}

impl Display for InplaceSub {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} -= {};", self.to_store, self.val)
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct InplaceMul {
    to_store: Arc<PrimeExpr>,
    val: Arc<PrimeExpr>,
}

impl InplaceMul {
    pub fn make<T: Into<PrimeExpr>>(to_store: T, val: T) -> Self {
        InplaceMul {
            to_store: to_store.into().into(),
            val: val.into().into(),
        }
    }

    pub fn to_store(&self) -> &PrimeExpr {
        &self.to_store
    }

    pub fn val(&self) -> &PrimeExpr {
        &self.val
    }

    pub fn accept_mut<V: IRMutVisitor>(&self, visitor: &mut V) {
        visitor.visit_inplace_mul(self);
    }

    pub fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_inplace_mul(self);
    }

    pub fn accept_mutate<V: IRMutateVisitor>(&self, visitor: &mut V) {
        visitor.visit_inplace_mul(self);
    }
}

impl Into<Stmt> for InplaceMul {
    fn into(self) -> Stmt {
        Stmt::InplaceMul(self)
    }
}

impl Into<Stmt> for &InplaceMul {
    fn into(self) -> Stmt {
        Stmt::InplaceMul(self.clone())
    }
}

impl Display for InplaceMul {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} *= {};", self.to_store, self.val)
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct InplaceDiv {
    to_store: Arc<PrimeExpr>,
    val: Arc<PrimeExpr>,
}

impl InplaceDiv {
    pub fn make<A: Into<PrimeExpr>, B: Into<PrimeExpr>>(to_store: A, val: B) -> Self {
        InplaceDiv {
            to_store: to_store.into().into(),
            val: val.into().into(),
        }
    }

    pub fn to_store(&self) -> &PrimeExpr {
        &self.to_store
    }

    pub fn val(&self) -> &PrimeExpr {
        &self.val
    }

    pub fn accept_mut<V: IRMutVisitor>(&self, visitor: &mut V) {
        visitor.visit_inplace_div(self);
    }

    pub fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_inplace_div(self);
    }

    pub fn accept_mutate<V: IRMutateVisitor>(&self, visitor: &mut V) {
        visitor.visit_inplace_div(self);
    }
}

impl Into<Stmt> for InplaceDiv {
    fn into(self) -> Stmt {
        Stmt::InplaceDiv(self)
    }
}

impl Into<Stmt> for &InplaceDiv {
    fn into(self) -> Stmt {
        Stmt::InplaceDiv(self.clone())
    }
}

impl Display for InplaceDiv {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "*{} /= {};", self.to_store, self.val)
    }
}
