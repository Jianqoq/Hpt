use std::fmt::Display;

use super::exprs::*;


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
    None,
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
            Expr::None => write!(f, ""),
        }
    }
}