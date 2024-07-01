use std::fmt::Display;

use crate::halide::{ prime_expr::PrimeExpr, variable::Variable };

use super::{
    exprs::*, func_type::Type, tensor::Tensor, traits::{
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

    fn print(&self) -> String {
        let s = match self {
            Expr::Value(a) => a.to_string(),
            Expr::Str(a) => a.to_string(),
            Expr::Variable(a) => a.to_string(),
            Expr::Cast(a) => a.to_string(),
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
        s
    }

    cast_expr!(to_variable, Variable);
    cast_expr!(to_value, Value);
    cast_expr!(to_type, Type);
    cast_expr!(to_tensor, Tensor);
    cast_expr!(to_tuple, Tuple);

    pub fn to_primexpr(&self) -> Option<PrimeExpr> {
        match self {
            Expr::Value(a) => { Some(a.to_primexpr()) }
            Expr::Str(a) => { Some(a.to_primexpr()) }
            Expr::Variable(a) => { Some(a.into()) }
            Expr::Cast(a) => {
                crate::halide::prime_expr::PrimeExpr
                    ::Cast(crate::halide::exprs::Cast::make(a.value().to_primexpr(), a.dtype()))
                    .into()
            }
            _ => None,
        }
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.print())
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

impl Into<Expr> for &&Expr {
    fn into(self) -> Expr {
        (*self).clone()
    }
}