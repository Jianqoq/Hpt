use std::{fmt::Display, sync::Arc};

use super::{
    expr::Expr, exprs::{Add, Int, Mul}, traits::{IRMutVisitor, IRMutateVisitor, IRVisitor}
};

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub struct Variable {
    name: Arc<String>,
}

impl Variable {
    pub fn new(name: String) -> Self {
        Variable {
            name: Arc::new(name),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn make(name: &str) -> Self {
        Variable::new(name.to_string())
    }

    pub fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_variable(self);
    }

    pub fn accept_mutate<V: IRMutateVisitor>(&self, visitor: &mut V) {
        visitor.visit_variable(self);
    }

    pub fn accept_mut<V: IRMutVisitor>(&self, visitor: &mut V) {
        visitor.visit_variable(self);
    }
}

impl Into<Variable> for &Variable {
    fn into(self) -> Variable {
        self.clone()
    }
}

impl Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl std::ops::Mul for Variable {
    type Output = Expr;

    fn mul(self, rhs: Variable) -> Self::Output {
        Expr::Mul(Mul::new(
            Expr::Variable(self).into(),
            Expr::Variable(rhs).into(),
        ))
    }
}

impl std::ops::Mul<&Int> for &Variable {
    type Output = Expr;

    fn mul(self, rhs: &Int) -> Self::Output {
        Expr::Mul(Mul::new(
            Expr::Variable(self.clone()).into(),
            Expr::Int(rhs.clone()).into(),
        ))
    }
}

impl std::ops::Mul<Int> for &Variable {
    type Output = Expr;

    fn mul(self, rhs: Int) -> Self::Output {
        Expr::Mul(Mul::new(
            Expr::Variable(self.clone()).into(),
            Expr::Int(rhs).into(),
        ))
    }
}

impl std::ops::Mul<&Int> for Variable {
    type Output = Expr;

    fn mul(self, rhs: &Int) -> Self::Output {
        Expr::Mul(Mul::new(
            Expr::Variable(self).into(),
            Expr::Int(rhs.clone()).into(),
        ))
    }
}

impl std::ops::Mul<Int> for Variable {
    type Output = Expr;

    fn mul(self, rhs: Int) -> Self::Output {
        Expr::Mul(Mul::new(Expr::Variable(self).into(), Expr::Int(rhs).into()))
    }
}

impl std::ops::Mul<Variable> for Int {
    type Output = Expr;

    fn mul(self, rhs: Variable) -> Self::Output {
        Expr::Mul(Mul::new(Expr::Int(self).into(), Expr::Variable(rhs).into()))
    }
}

impl std::ops::Mul<&Variable> for Int {
    type Output = Expr;

    fn mul(self, rhs: &Variable) -> Self::Output {
        Expr::Mul(Mul::new(
            Expr::Int(self).into(),
            Expr::Variable(rhs.clone()).into(),
        ))
    }
}

impl std::ops::Mul<Variable> for &Int {
    type Output = Expr;

    fn mul(self, rhs: Variable) -> Self::Output {
        Expr::Mul(Mul::new(
            Expr::Int(self.clone()).into(),
            Expr::Variable(rhs).into(),
        ))
    }
}

impl std::ops::Mul<&Variable> for &Int {
    type Output = Expr;

    fn mul(self, rhs: &Variable) -> Self::Output {
        Expr::Mul(Mul::new(
            Expr::Int(self.clone()).into(),
            Expr::Variable(rhs.clone()).into(),
        ))
    }
}

impl std::ops::Add for Variable {
    type Output = Expr;

    fn add(self, rhs: Variable) -> Self::Output {
        Expr::Add(Add::new(
            Expr::Variable(self).into(),
            Expr::Variable(rhs).into(),
        ))
    }
}

impl std::ops::Add<&Int> for &Variable {
    type Output = Expr;

    fn add(self, rhs: &Int) -> Self::Output {
        Expr::Add(Add::new(
            Expr::Variable(self.clone()).into(),
            Expr::Int(rhs.clone()).into(),
        ))
    }
}

impl std::ops::Add<Int> for &Variable {
    type Output = Expr;

    fn add(self, rhs: Int) -> Self::Output {
        Expr::Add(Add::new(
            Expr::Variable(self.clone()).into(),
            Expr::Int(rhs).into(),
        ))
    }
}

impl std::ops::Add<&Int> for Variable {
    type Output = Expr;

    fn add(self, rhs: &Int) -> Self::Output {
        Expr::Add(Add::new(
            Expr::Variable(self).into(),
            Expr::Int(rhs.clone()).into(),
        ))
    }
}

impl std::ops::Add<Int> for Variable {
    type Output = Expr;

    fn add(self, rhs: Int) -> Self::Output {
        Expr::Add(Add::new(Expr::Variable(self).into(), Expr::Int(rhs).into()))
    }
}

impl std::ops::Add<Variable> for Int {
    type Output = Expr;

    fn add(self, rhs: Variable) -> Self::Output {
        Expr::Add(Add::new(Expr::Int(self).into(), Expr::Variable(rhs).into()))
    }
}

impl std::ops::Add<&Variable> for Int {
    type Output = Expr;

    fn add(self, rhs: &Variable) -> Self::Output {
        Expr::Add(Add::new(
            Expr::Int(self).into(),
            Expr::Variable(rhs.clone()).into(),
        ))
    }
}

impl std::ops::Add<Variable> for &Int {
    type Output = Expr;

    fn add(self, rhs: Variable) -> Self::Output {
        Expr::Add(Add::new(
            Expr::Int(self.clone()).into(),
            Expr::Variable(rhs).into(),
        ))
    }
}

impl std::ops::Add<&Variable> for &Int {
    type Output = Expr;

    fn add(self, rhs: &Variable) -> Self::Output {
        Expr::Add(Add::new(
            Expr::Int(self.clone()).into(),
            Expr::Variable(rhs.clone()).into(),
        ))
    }
}

impl Into<Expr> for Variable {
    fn into(self) -> Expr {
        Expr::Variable(self)
    }
}

impl Into<Expr> for &Variable {
    fn into(self) -> Expr {
        Expr::Variable(self.clone())
    }
}