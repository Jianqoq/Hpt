use std::{ fmt::Display, sync::Arc };

use super::{
    exprs::*,
    traits::{
        HlirAccepterMut,
        HlirAccepterMutate,
        HlirAcceptor,
        HlirMutVisitor,
        HlirMutateVisitor,
        HlirVisitor,
    },
};

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Expr {
    Value(Value),
    Str(Str),
    Variable(Variable),
    Tensor(Tensor),
    Cast(Cast),
    Add(Add),
    Sub(Sub),
    Mul(Mul),
    Div(Div),
    Mod(Mod),
    Min(Min),
    Max(Max),
    Eq(Eq),
    Ne(Ne),
    Lt(Lt),
    Le(Le),
    Gt(Gt),
    Ge(Ge),
    And(And),
    Or(Or),
    Xor(Xor),
    Not(Not),
    Call(Call),
    Select(Select),
    Let(Let),
    Alloc(Alloc),
    If(If),
    For(For),
    While(While),
    Function(Function),
    None,
}

impl Expr {
    const fn precedence(&self) -> i32 {
        match self {
            Expr::Add(_) | Expr::Sub(_) => 1,
            Expr::Mul(_) | Expr::Div(_) | Expr::Mod(_) => 2,
            _ => 3,
        }
    }

    fn print(&self, parent_prec: i32) -> String {
        let prec = self.precedence();
        let s = match self {
            Expr::Value(a) => a.to_string(),
            Expr::Str(a) => a.to_string(),
            Expr::Variable(a) => a.to_string(),
            Expr::Cast(a) => a.to_string(),
            Expr::Add(a) => format!("{} + {}", a.lhs().print(prec), a.rhs().print(prec + 1)),
            Expr::Sub(a) => format!("{} - {}", a.lhs().print(prec), a.rhs().print(prec + 1)),
            Expr::Mul(a) => format!("{} * {}", a.lhs().print(prec), a.rhs().print(prec + 1)),
            Expr::Div(a) => format!("{} / {}", a.lhs().print(prec), a.rhs().print(prec + 1)),
            Expr::Mod(a) => format!("{} % {}", a.lhs().print(prec), a.rhs().print(prec + 1)),
            Expr::Min(a) => a.to_string(),
            Expr::Max(a) => a.to_string(),
            Expr::Eq(a) => a.to_string(),
            Expr::Ne(a) => a.to_string(),
            Expr::Lt(a) => a.to_string(),
            Expr::Le(a) => a.to_string(),
            Expr::Gt(a) => a.to_string(),
            Expr::Ge(a) => a.to_string(),
            Expr::And(a) => a.to_string(),
            Expr::Xor(a) => a.to_string(),
            Expr::Or(a) => a.to_string(),
            Expr::Not(a) => a.to_string(),
            Expr::Call(a) => a.to_string(),
            Expr::Select(a) => a.to_string(),
            Expr::Let(a) => a.to_string(),
            Expr::Alloc(a) => a.to_string(),
            Expr::If(a) => a.to_string(),
            Expr::For(a) => a.to_string(),
            Expr::While(a) => a.to_string(),
            Expr::Function(a) => a.to_string(),
            Expr::Tensor(a) => a.to_string(),
            Expr::None => "".to_string(),
        };
        if prec < parent_prec {
            format!("({})", s)
        } else {
            s
        }
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Expr::Value(v) => write!(f, "{}", v),
            Expr::Str(v) => write!(f, "{}", v),
            Expr::Variable(v) => write!(f, "{}", v),
            Expr::Tensor(v) => write!(f, "{}", v),
            Expr::Cast(v) => write!(f, "{}", v),
            Expr::Add(v) => write!(f, "{}", v),
            Expr::Sub(v) => write!(f, "{}", v),
            Expr::Mul(v) => write!(f, "{}", v),
            Expr::Div(v) => write!(f, "{}", v),
            Expr::Mod(v) => write!(f, "{}", v),
            Expr::Min(v) => write!(f, "{}", v),
            Expr::Max(v) => write!(f, "{}", v),
            Expr::Eq(v) => write!(f, "{}", v),
            Expr::Ne(v) => write!(f, "{}", v),
            Expr::Lt(v) => write!(f, "{}", v),
            Expr::Le(v) => write!(f, "{}", v),
            Expr::Gt(v) => write!(f, "{}", v),
            Expr::Ge(v) => write!(f, "{}", v),
            Expr::And(v) => write!(f, "{}", v),
            Expr::Or(v) => write!(f, "{}", v),
            Expr::Xor(v) => write!(f, "{}", v),
            Expr::Not(v) => write!(f, "{}", v),
            Expr::Call(v) => write!(f, "{}", v),
            Expr::Select(v) => write!(f, "{}", v),
            Expr::Let(v) => write!(f, "{}", v),
            Expr::Alloc(v) => write!(f, "{}", v),
            Expr::If(v) => write!(f, "{}", v),
            Expr::For(v) => write!(f, "{}", v),
            Expr::While(v) => write!(f, "{}", v),
            Expr::Function(v) => write!(f, "{}", v),
            Expr::None => write!(f, ""),
        }
    }
}

impl HlirAcceptor for Expr {
    fn accept<V: HlirVisitor>(&self, visitor: &V) {
        visitor.visit_expr(self);
    }
}

impl HlirAccepterMut for Expr {
    fn accept_mut<V: HlirMutVisitor>(&self, visitor: &mut V) {
        visitor.visit_expr(self);
    }
}

impl HlirAccepterMutate for Expr {
    fn accept_mutate<V: HlirMutateVisitor>(&self, visitor: &mut V) {
        visitor.visit_expr(self);
    }
}
