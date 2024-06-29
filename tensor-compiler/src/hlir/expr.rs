use std::fmt::Display;

use crate::halide::variable::Variable;

use super::{
    exprs::*, func_type::Type, traits::{
        HlirAccepterMut,
        HlirAccepterMutate,
        HlirAcceptor,
        HlirMutVisitor,
        HlirMutateVisitor,
        HlirVisitor,
    }
};

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Expr {
    Value(Value),
    Str(Str),
    Variable(Variable),
    Tuple(Tuple),
    Type(Type),
    TensorType(TensorType),
    OpNode(OpNode),
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
    Slice(Slice),
    Return(Return),
    None,
}

macro_rules! cast_expr {
    ($fn_name:ident, $t:ident) => {
        pub fn $fn_name(&self) -> Option<&$t> {
            match self {
                Expr::$t(e) => Some(e),
                _ => None,
            }
        }
    };
}

impl Expr {
    pub const fn is_none(&self) -> bool {
        matches!(self, Expr::None)
    }

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
            Expr::Tuple(a) => a.to_string(),
            Expr::Type(a) => a.to_string(),
            Expr::TensorType(a) => a.to_string(),
            Expr::Slice(a) => a.to_string(),
            Expr::OpNode(a) => a.to_string(),
            Expr::Tensor(a) => a.to_string(),
            Expr::Return(a) => a.to_string(),
            Expr::None => "".to_string(),
        };
        if prec < parent_prec {
            format!("({})", s)
        } else {
            s
        }
    }

    cast_expr!(to_variable, Variable);
    cast_expr!(to_value, Value);
    cast_expr!(to_type, Type);
}

impl Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.print(0))
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

impl Into<Expr> for &Expr {
    fn into(self) -> Expr {
        self.clone()
    }
}
