use std::{ fmt::Display, sync::Arc };

use super::{
    exprs::*,
    traits::{ Accepter, AccepterMut, AccepterMutate, IRMutVisitor, IRMutateVisitor, IRVisitor },
    variable::Variable,
};

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub enum Expr {
    Int(Int),
    Float(Float),
    UInt(UInt),
    Str(Str),
    Variable(Variable),
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
    Load(Load),
    None,
}

#[derive(Clone, Copy, PartialEq, Hash, Eq, PartialOrd, Ord)]
pub enum ExprType {
    Int,
    Float,
    UInt,
    Str,
    Variable,
    Cast,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Min,
    Max,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    Xor,
    Not,
    Call,
    Select,
    Let,
    Load,
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

    pub fn same_as(&self, other: &Expr) -> bool {
        match (self, other) {
            (Expr::Int(i1), Expr::Int(i2)) => i1.value() == i2.value(),
            (Expr::Float(f1), Expr::Float(f2)) => f1.value() == f2.value(),
            (Expr::UInt(u1), Expr::UInt(u2)) => u1.value() == u2.value(),
            (Expr::Str(s1), Expr::Str(s2)) => s1.value() == s2.value(),
            (Expr::Variable(v1), Expr::Variable(v2)) => v1.name() == v2.name(),
            (Expr::Cast(c1), Expr::Cast(c2)) => {
                Arc::ptr_eq(c1.expr_(), c2.expr_()) && c1.dtype() == c2.dtype()
            }
            (Expr::Add(a1), Expr::Add(a2)) => {
                (Arc::ptr_eq(a1.e1_(), a2.e1_()) && Arc::ptr_eq(a1.e2_(), a2.e2_())) ||
                    (Arc::ptr_eq(a1.e1_(), a2.e2_()) && Arc::ptr_eq(a1.e2_(), a2.e1_()))
            }
            (Expr::Sub(s1), Expr::Sub(s2)) => {
                (Arc::ptr_eq(s1.e1_(), s2.e1_()) && Arc::ptr_eq(s1.e2_(), s2.e2_())) ||
                    (Arc::ptr_eq(s1.e1_(), s2.e2_()) && Arc::ptr_eq(s1.e2_(), s2.e1_()))
            }
            (Expr::Mul(m1), Expr::Mul(m2)) => {
                (Arc::ptr_eq(m1.e1_(), m2.e1_()) && Arc::ptr_eq(m1.e2_(), m2.e2_())) ||
                    (Arc::ptr_eq(m1.e1_(), m2.e2_()) && Arc::ptr_eq(m1.e2_(), m2.e1_()))
            }
            (Expr::Div(d1), Expr::Div(d2)) => {
                (Arc::ptr_eq(d1.e1_(), d2.e1_()) && Arc::ptr_eq(d1.e2_(), d2.e2_())) ||
                    (Arc::ptr_eq(d1.e1_(), d2.e2_()) && Arc::ptr_eq(d1.e2_(), d2.e1_()))
            }
            (Expr::Mod(m1), Expr::Mod(m2)) => {
                (Arc::ptr_eq(m1.e1_(), m2.e1_()) && Arc::ptr_eq(m1.e2_(), m2.e2_())) ||
                    (Arc::ptr_eq(m1.e1_(), m2.e2_()) && Arc::ptr_eq(m1.e2_(), m2.e1_()))
            }
            (Expr::Min(m1), Expr::Min(m2)) => {
                (Arc::ptr_eq(m1.e1_(), m2.e1_()) && Arc::ptr_eq(m1.e2_(), m2.e2_())) ||
                    (Arc::ptr_eq(m1.e1_(), m2.e2_()) && Arc::ptr_eq(m1.e2_(), m2.e1_()))
            }
            (Expr::Max(m1), Expr::Max(m2)) => {
                (Arc::ptr_eq(m1.e1_(), m2.e1_()) && Arc::ptr_eq(m1.e2_(), m2.e2_())) ||
                    (Arc::ptr_eq(m1.e1_(), m2.e2_()) && Arc::ptr_eq(m1.e2_(), m2.e1_()))
            }
            (Expr::Eq(e1), Expr::Eq(e2)) => {
                (Arc::ptr_eq(e1.e1_(), e2.e1_()) && Arc::ptr_eq(e1.e2_(), e2.e2_())) ||
                    (Arc::ptr_eq(e1.e1_(), e2.e2_()) && Arc::ptr_eq(e1.e2_(), e2.e1_()))
            }
            (Expr::Ne(n1), Expr::Ne(n2)) => {
                (Arc::ptr_eq(n1.e1_(), n2.e1_()) && Arc::ptr_eq(n1.e2_(), n2.e2_())) ||
                    (Arc::ptr_eq(n1.e1_(), n2.e2_()) && Arc::ptr_eq(n1.e2_(), n2.e1_()))
            }
            (Expr::Lt(l1), Expr::Lt(l2)) => {
                (Arc::ptr_eq(l1.e1_(), l2.e1_()) && Arc::ptr_eq(l1.e2_(), l2.e2_())) ||
                    (Arc::ptr_eq(l1.e1_(), l2.e2_()) && Arc::ptr_eq(l1.e2_(), l2.e1_()))
            }
            (Expr::Le(l1), Expr::Le(l2)) => {
                (Arc::ptr_eq(l1.e1_(), l2.e1_()) && Arc::ptr_eq(l1.e2_(), l2.e2_())) ||
                    (Arc::ptr_eq(l1.e1_(), l2.e2_()) && Arc::ptr_eq(l1.e2_(), l2.e1_()))
            }
            (Expr::Gt(g1), Expr::Gt(g2)) => {
                (Arc::ptr_eq(g1.e1_(), g2.e1_()) && Arc::ptr_eq(g1.e2_(), g2.e2_())) ||
                    (Arc::ptr_eq(g1.e1_(), g2.e2_()) && Arc::ptr_eq(g1.e2_(), g2.e1_()))
            }
            (Expr::Ge(g1), Expr::Ge(g2)) => {
                (Arc::ptr_eq(g1.e1_(), g2.e1_()) && Arc::ptr_eq(g1.e2_(), g2.e2_())) ||
                    (Arc::ptr_eq(g1.e1_(), g2.e2_()) && Arc::ptr_eq(g1.e2_(), g2.e1_()))
            }
            (Expr::And(a1), Expr::And(a2)) => {
                (Arc::ptr_eq(a1.e1_(), a2.e1_()) && Arc::ptr_eq(a1.e2_(), a2.e2_())) ||
                    (Arc::ptr_eq(a1.e1_(), a2.e2_()) && Arc::ptr_eq(a1.e2_(), a2.e1_()))
            }
            (Expr::Xor(x1), Expr::Xor(x2)) => {
                (Arc::ptr_eq(x1.e1_(), x2.e1_()) && Arc::ptr_eq(x1.e2_(), x2.e2_())) ||
                    (Arc::ptr_eq(x1.e1_(), x2.e2_()) && Arc::ptr_eq(x1.e2_(), x2.e1_()))
            }
            (Expr::Or(o1), Expr::Or(o2)) => {
                (Arc::ptr_eq(o1.e1_(), o2.e1_()) && Arc::ptr_eq(o1.e2_(), o2.e2_())) ||
                    (Arc::ptr_eq(o1.e1_(), o2.e2_()) && Arc::ptr_eq(o1.e2_(), o2.e1_()))
            }
            (Expr::Not(n1), Expr::Not(n2)) => Arc::ptr_eq(n1.e_(), n2.e_()),
            (Expr::Call(c1), Expr::Call(c2)) => {
                c1.name() == c2.name() &&
                    c1
                        .args()
                        .iter()
                        .zip(c2.args().iter())
                        .all(|(a1, a2)| Arc::ptr_eq(a1, a2))
            }
            (Expr::Select(s1), Expr::Select(s2)) => {
                Arc::ptr_eq(s1.cond_(), s2.cond_()) &&
                    Arc::ptr_eq(s1.true_expr_(), s2.true_expr_()) &&
                    Arc::ptr_eq(s1.false_expr_(), s2.false_expr_())
            }
            (Expr::Let(l1), Expr::Let(l2)) => {
                Arc::ptr_eq(l1.e1_(), l2.e1_()) && l1.name() == l2.name()
            }
            (Expr::Load(l1), Expr::Load(l2)) => {
                Arc::ptr_eq(l1.name_(), l2.name_()) && Arc::ptr_eq(l1.indices_(), l2.indices_())
            }
            (Expr::None, Expr::None) => true,
            _ => false,
        }
    }

    pub const fn type_info(&self) -> ExprType {
        match self {
            Expr::Int(_) => ExprType::Int,
            Expr::Float(_) => ExprType::Float,
            Expr::UInt(_) => ExprType::UInt,
            Expr::Str(_) => ExprType::Str,
            Expr::Variable(_) => ExprType::Variable,
            Expr::Cast(_) => ExprType::Cast,
            Expr::Add(_) => ExprType::Add,
            Expr::Sub(_) => ExprType::Sub,
            Expr::Mul(_) => ExprType::Mul,
            Expr::Div(_) => ExprType::Div,
            Expr::Mod(_) => ExprType::Mod,
            Expr::Min(_) => ExprType::Min,
            Expr::Max(_) => ExprType::Max,
            Expr::Eq(_) => ExprType::Eq,
            Expr::Ne(_) => ExprType::Ne,
            Expr::Lt(_) => ExprType::Lt,
            Expr::Le(_) => ExprType::Le,
            Expr::Gt(_) => ExprType::Gt,
            Expr::Ge(_) => ExprType::Ge,
            Expr::And(_) => ExprType::And,
            Expr::Xor(_) => ExprType::Xor,
            Expr::Or(_) => ExprType::Or,
            Expr::Not(_) => ExprType::Not,
            Expr::Call(_) => ExprType::Call,
            Expr::Select(_) => ExprType::Select,
            Expr::Let(_) => ExprType::Let,
            Expr::Load(_) => ExprType::Load,
            Expr::None => ExprType::None,
        }
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
            Expr::Int(a) => a.to_string(),
            Expr::Float(a) => a.to_string(),
            Expr::UInt(a) => a.to_string(),
            Expr::Str(a) => a.to_string(),
            Expr::Variable(a) => a.to_string(),
            Expr::Cast(a) => a.to_string(),
            Expr::Add(a) => format!("{} + {}", a.e1().print(prec), a.e2().print(prec + 1)),
            Expr::Sub(a) => format!("{} - {}", a.e1().print(prec), a.e2().print(prec + 1)),
            Expr::Mul(a) => format!("{} * {}", a.e1().print(prec), a.e2().print(prec + 1)),
            Expr::Div(a) => format!("{} / {}", a.e1().print(prec), a.e2().print(prec + 1)),
            Expr::Mod(a) => format!("{} % {}", a.e1().print(prec), a.e2().print(prec + 1)),
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
            Expr::Load(a) => a.to_string(),
            Expr::None => "".to_string(),
        };
        if prec < parent_prec {
            format!("({})", s)
        } else {
            s
        }
    }

    cast_expr!(to_variable, Variable);
    cast_expr!(to_add, Add);
    cast_expr!(to_sub, Sub);
    cast_expr!(to_mul, Mul);
    cast_expr!(to_div, Div);
    cast_expr!(to_mod, Mod);
    cast_expr!(to_min, Min);
    cast_expr!(to_max, Max);
    cast_expr!(to_eq, Eq);
    cast_expr!(to_ne, Ne);
    cast_expr!(to_lt, Lt);
    cast_expr!(to_le, Le);
    cast_expr!(to_gt, Gt);
    cast_expr!(to_ge, Ge);
    cast_expr!(to_and, And);
    cast_expr!(to_or, Or);
    cast_expr!(to_not, Not);
    cast_expr!(to_call, Call);
    cast_expr!(to_select, Select);
    cast_expr!(to_let, Let);
    cast_expr!(to_load, Load);
    cast_expr!(to_int, Int);
    cast_expr!(to_float, Float);
    cast_expr!(to_uint, UInt);
    cast_expr!(to_str, Str);
    cast_expr!(to_cast, Cast);
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

impl Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.print(0))
    }
}

impl Accepter for Expr {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_expr(self);
    }
}

impl AccepterMut for Expr {
    fn accept_mut<V: IRMutVisitor>(&self, visitor: &mut V) {
        visitor.visit_expr(self);
    }
}

impl AccepterMutate for Expr {
    fn accept_mutate<V: IRMutateVisitor>(&self, visitor: &mut V) {
        visitor.visit_expr(self);
    }
}

impl std::ops::Add for Expr {
    type Output = Expr;

    fn add(self, rhs: Expr) -> Self::Output {
        match (&self, &rhs) {
            (Expr::Int(i1), Expr::Int(i2)) => Expr::Int(i1 + i2),
            (Expr::Float(f1), Expr::Float(f2)) => Expr::Float(Float::new(f1.value() + f2.value())),
            (Expr::Int(i), Expr::Float(f)) =>
                Expr::Float(Float::new((i.value() as f64) + f.value())),
            (Expr::Float(f), Expr::Int(i)) =>
                Expr::Float(Float::new(f.value() + (i.value() as f64))),
            (Expr::UInt(u1), Expr::UInt(u2)) => Expr::UInt(u1 + u2),
            (Expr::Mul(m1), Expr::Mul(m2)) => Expr::Add(Add::new(m1.into(), m2.into())),
            (Expr::Add(a1), Expr::Add(a2)) => Expr::Add(Add::new(a1.into(), a2.into())),
            (Expr::Sub(s1), Expr::Sub(s2)) => Expr::Add(Add::new(s1.into(), s2.into())),
            (Expr::Div(d1), Expr::Div(d2)) => Expr::Add(Add::new(d1.into(), d2.into())),
            (Expr::Mod(m1), Expr::Mod(m2)) => Expr::Add(Add::new(m1.into(), m2.into())),
            (Expr::Add(a), Expr::Mul(m)) => Expr::Add(Add::new(a.into(), m.into())),
            (Expr::Add(a), Expr::Sub(s)) => Expr::Add(Add::new(a.into(), s.into())),
            (Expr::Add(a), Expr::Div(d)) => Expr::Add(Add::new(a.into(), d.into())),
            (Expr::Add(a), Expr::Mod(m)) => Expr::Add(Add::new(a.into(), m.into())),
            (Expr::Mul(m), Expr::Add(a)) => Expr::Add(Add::new(m.into(), a.into())),
            (Expr::Sub(s), Expr::Add(a)) => Expr::Add(Add::new(s.into(), a.into())),
            (Expr::Div(d), Expr::Add(a)) => Expr::Add(Add::new(d.into(), a.into())),
            (Expr::Mod(m), Expr::Add(a)) => Expr::Add(Add::new(m.into(), a.into())),
            (Expr::Add(a), Expr::Int(i)) => Expr::Add(Add::new(a.into(), i.into())),
            (Expr::Sub(s), Expr::Int(i)) => Expr::Add(Add::new(s.into(), i.into())),
            (Expr::Mul(m), Expr::Int(i)) => Expr::Add(Add::new(m.into(), i.into())),
            (Expr::Div(d), Expr::Int(i)) => Expr::Add(Add::new(d.into(), i.into())),
            (Expr::Mod(m), Expr::Int(i)) => Expr::Add(Add::new(m.into(), i.into())),
            _ => panic!("{}", &format!("Failed to add {} and {}", self, rhs)),
        }
    }
}
