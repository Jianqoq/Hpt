use std::{ fmt::{ Display, Formatter }, sync::Arc };

use super::{
    prime_expr::PrimeExpr,
    stmt::Stmt,
    traits::{ IRMutVisitor, IRMutateVisitor, IRVisitor },
};

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct IfThenElse {
    cond: PrimeExpr,
    then_case: Arc<Stmt>,
    else_case: Arc<Stmt>,
}

impl IfThenElse {
    pub fn make<T: Into<PrimeExpr>, U: Into<Stmt>, V: Into<Stmt>>(
        cond: T,
        then_case: U,
        else_case: V
    ) -> Self {
        IfThenElse {
            cond: cond.into(),
            then_case: Arc::new(then_case.into()),
            else_case: Arc::new(else_case.into()),
        }
    }

    pub fn accept_mut<V: IRMutVisitor>(&self, visitor: &mut V) {
        visitor.visit_if_then_else(self);
    }

    pub fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_if_then_else(self);
    }

    pub fn accept_mutate<V: IRMutateVisitor>(&self, visitor: &mut V) {
        visitor.visit_if_then_else(self);
    }

    pub fn cond(&self) -> &PrimeExpr {
        &self.cond
    }

    pub fn then_case(&self) -> &Stmt {
        &self.then_case
    }

    pub fn else_case(&self) -> &Stmt {
        &self.else_case
    }
}

impl Into<Stmt> for IfThenElse {
    fn into(self) -> Stmt {
        Stmt::IfThenElse(self)
    }
}

impl Into<Stmt> for &IfThenElse {
    fn into(self) -> Stmt {
        Stmt::IfThenElse(self.clone())
    }
}

impl Display for IfThenElse {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match self.else_case.as_ref() {
            Stmt::None => write!(f, "if {} {{\n{}}}", self.cond, self.then_case),
            _ =>
                write!(
                    f,
                    "if {} {{\n{}}} else {{\n{}}}",
                    self.cond,
                    self.then_case,
                    self.else_case
                ),
        }
    }
}
