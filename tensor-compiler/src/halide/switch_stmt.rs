use std::{fmt::Display, sync::Arc};

use super::{prime_expr::PrimeExpr, stmt::Stmt, traits::{IRMutVisitor, IRMutateVisitor, IRVisitor}};

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct SwitchCase {
    pub case_value: PrimeExpr,
    pub action: Stmt,
}

impl SwitchCase {
    pub fn make<A: Into<PrimeExpr>, B: Into<Stmt>>(case_value: A, action: B) -> Self {
        SwitchCase {
            case_value: case_value.into().into(),
            action: action.into(),
        }
    }

    pub fn case_value(&self) -> &PrimeExpr {
        &self.case_value
    }

    pub fn action(&self) -> &Stmt {
        &self.action
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct SwitchStmt {
    pub cond: PrimeExpr,
    pub actions: Vec<SwitchCase>,
    pub default: Arc<Stmt>,
}

impl SwitchStmt {
    pub fn make<A: Into<PrimeExpr>>(cond: A, actions: Vec<SwitchCase>) -> Self {
        SwitchStmt {
            cond: cond.into().into(),
            actions,
            default: Arc::new(Stmt::None),
        }
    }

    pub fn cond(&self) -> &PrimeExpr {
        &self.cond
    }

    pub fn actions(&self) -> &Vec<SwitchCase> {
        &self.actions
    }

    pub fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_switch(self);
    }

    pub fn accept_mutate<V: IRMutateVisitor>(&self, visitor: &mut V) {
        visitor.visit_switch(self);
    }

    pub fn accept_mut<V: IRMutVisitor>(&self, visitor: &mut V) {
        visitor.visit_switch(self);
    }
}

impl Display for SwitchStmt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "switch {} {{\n", self.cond)?;
        for action in &self.actions {
            write!(f, "case {}:\n{}\n", action.case_value, action.action)?;
        }
        write!(f, "}}")
    }
}

impl Into<Stmt> for SwitchStmt {
    fn into(self) -> Stmt {
        Stmt::SwitchStmt(self)
    }
}

impl Into<Stmt> for &SwitchStmt {
    fn into(self) -> Stmt {
        Stmt::SwitchStmt(self.clone())
    }
}