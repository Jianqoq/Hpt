use std::fmt::Display;

use tensor_types::dtype::Dtype;

use super::{
    exprs::*, tensor_load::TensorLoad, traits::{ Accepter, AccepterMut, AccepterMutate, IRMutVisitor, IRMutateVisitor, IRVisitor }, variable::Variable
};

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub enum PrimeExpr {
    Int(Int),
    Float(Float),
    UInt(UInt),
    Str(Str),
    Variable(Variable),
    Reduce(Reduce),
    Cast(Cast),
    BitCast(BitCast),
    Add(Add),
    Sub(Sub),
    Mul(Mul),
    Div(Div),
    Neg(Neg),
    Rem(Rem),
    Min(Min),
    Max(Max),
    FloorDiv(FloorDiv),
    Eq(Eq),
    Ne(Ne),
    Lt(Lt),
    Le(Le),
    Gt(Gt),
    Ge(Ge),
    BitAnd(BitAnd),
    BitOr(BitOr),
    BitXor(BitXor),
    Shl(Shl),
    Shr(Shr),
    Not(Not),
    Call(Call),
    Select(Select),
    Let(Let),
    Load(Load),
    Malloc(Malloc),
    Layout(Layout),
    TensorLoad(TensorLoad),
    Null,
    None,
}

#[derive(Clone, Copy, PartialEq, Hash, Eq, PartialOrd, Ord)]
pub enum PrimeType {
    Int,
    Float,
    UInt,
    Str,
    Variable,
    TensorSlice,
    Reduce,
    Cast,
    BitCast,
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    FloorDiv,
    Shl,
    Shr,
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
    Malloc,
    Layout,
    Alloca,
    TensorLoad,
    Null,
    None,
}

macro_rules! cast_expr {
    ($fn_name:ident, $t:ident) => {
        pub fn $fn_name(&self) -> Option<&$t> {
            match self {
                PrimeExpr::$t(e) => Some(e),
                _ => None,
            }
        }
    };
}

impl PrimeExpr {
    pub const fn is_none(&self) -> bool {
        matches!(self, PrimeExpr::None)
    }

    pub const fn is_add(&self) -> bool {
        matches!(self, PrimeExpr::Add(_))
    }

    pub const fn is_mod(&self) -> bool {
        matches!(self, PrimeExpr::Rem(_))
    }

    pub const fn is_int(&self) -> bool {
        matches!(self, PrimeExpr::Int(_))
    }

    pub const fn type_info(&self) -> PrimeType {
        match self {
            PrimeExpr::Int(_) => PrimeType::Int,
            PrimeExpr::Float(_) => PrimeType::Float,
            PrimeExpr::UInt(_) => PrimeType::UInt,
            PrimeExpr::Str(_) => PrimeType::Str,
            PrimeExpr::Variable(_) => PrimeType::Variable,
            PrimeExpr::Cast(_) => PrimeType::Cast,
            PrimeExpr::BitCast(_) => PrimeType::BitCast,
            PrimeExpr::Add(_) => PrimeType::Add,
            PrimeExpr::Sub(_) => PrimeType::Sub,
            PrimeExpr::Mul(_) => PrimeType::Mul,
            PrimeExpr::Div(_) => PrimeType::Div,
            PrimeExpr::Neg(_) => PrimeType::Neg,
            PrimeExpr::FloorDiv(_) => PrimeType::FloorDiv,
            PrimeExpr::Rem(_) => PrimeType::Mod,
            PrimeExpr::Min(_) => PrimeType::Min,
            PrimeExpr::Max(_) => PrimeType::Max,
            PrimeExpr::Eq(_) => PrimeType::Eq,
            PrimeExpr::Ne(_) => PrimeType::Ne,
            PrimeExpr::Lt(_) => PrimeType::Lt,
            PrimeExpr::Le(_) => PrimeType::Le,
            PrimeExpr::Gt(_) => PrimeType::Gt,
            PrimeExpr::Ge(_) => PrimeType::Ge,
            PrimeExpr::Shl(_) => PrimeType::Shl,
            PrimeExpr::Shr(_) => PrimeType::Shr,
            PrimeExpr::BitAnd(_) => PrimeType::And,
            PrimeExpr::BitXor(_) => PrimeType::Xor,
            PrimeExpr::BitOr(_) => PrimeType::Or,
            PrimeExpr::Not(_) => PrimeType::Not,
            PrimeExpr::Call(_) => PrimeType::Call,
            PrimeExpr::Select(_) => PrimeType::Select,
            PrimeExpr::Let(_) => PrimeType::Let,
            PrimeExpr::Load(_) => PrimeType::Load,
            PrimeExpr::Reduce(_) => PrimeType::Reduce,
            PrimeExpr::Malloc(_) => PrimeType::Malloc,
            PrimeExpr::Layout(_) => PrimeType::Layout,
            PrimeExpr::TensorLoad(_) => PrimeType::TensorLoad,
            PrimeExpr::Null => PrimeType::Null,
            PrimeExpr::None => PrimeType::None,
        }
    }

    const fn precedence(&self) -> i32 {
        match self {
            PrimeExpr::Add(_) | PrimeExpr::Sub(_) => 1,
            PrimeExpr::Mul(_) | PrimeExpr::Div(_) | PrimeExpr::Rem(_) => 2,
            _ => 3,
        }
    }

    fn print(&self, parent_prec: i32) -> String {
        let prec = self.precedence();
        let s = match self {
            PrimeExpr::Int(a) => a.to_string(),
            PrimeExpr::Float(a) => a.to_string(),
            PrimeExpr::UInt(a) => a.to_string(),
            PrimeExpr::Str(a) => a.to_string(),
            PrimeExpr::Variable(a) => a.to_string(),
            PrimeExpr::Cast(a) => a.to_string(),
            PrimeExpr::BitCast(a) => a.to_string(),
            PrimeExpr::Add(a) => format!("{} + {}", a.e1().print(prec), a.e2().print(prec + 1)),
            PrimeExpr::Sub(a) => format!("{} - {}", a.e1().print(prec), a.e2().print(prec + 1)),
            PrimeExpr::Mul(a) => format!("{} * {}", a.e1().print(prec), a.e2().print(prec + 1)),
            PrimeExpr::Div(a) => format!("{} / {}", a.e1().print(prec), a.e2().print(prec + 1)),
            PrimeExpr::Neg(a) => format!("-{}", a.e().print(prec + 1)),
            PrimeExpr::FloorDiv(a) =>
                format!("{} // {}", a.e1().print(prec), a.e2().print(prec + 1)),
            PrimeExpr::Rem(a) => format!("{} % {}", a.e1().print(prec), a.e2().print(prec + 1)),
            PrimeExpr::Shl(a) => format!("{} << {}", a.e1().print(prec), a.e2().print(prec + 1)),
            PrimeExpr::Shr(a) => format!("{} >> {}", a.e1().print(prec), a.e2().print(prec + 1)),
            PrimeExpr::Min(a) => a.to_string(),
            PrimeExpr::Max(a) => a.to_string(),
            PrimeExpr::Eq(a) => a.to_string(),
            PrimeExpr::Ne(a) => a.to_string(),
            PrimeExpr::Lt(a) => a.to_string(),
            PrimeExpr::Le(a) => a.to_string(),
            PrimeExpr::Gt(a) => a.to_string(),
            PrimeExpr::Ge(a) => a.to_string(),
            PrimeExpr::BitAnd(a) => a.to_string(),
            PrimeExpr::BitXor(a) => a.to_string(),
            PrimeExpr::BitOr(a) => a.to_string(),
            PrimeExpr::Not(a) => a.to_string(),
            PrimeExpr::Call(a) => a.to_string(),
            PrimeExpr::Select(a) => a.to_string(),
            PrimeExpr::Let(a) => a.to_string(),
            PrimeExpr::Load(a) => a.to_string(),
            PrimeExpr::Reduce(a) => a.to_string(),
            PrimeExpr::Malloc(a) => a.to_string(),
            PrimeExpr::Layout(a) => a.to_string(),
            PrimeExpr::TensorLoad(a) => a.to_string(),
            PrimeExpr::Null => "null".to_string(),
            PrimeExpr::None => "".to_string(),
        };
        if prec < parent_prec {
            format!("({})", s)
        } else {
            s
        }
    }

    pub fn floor_div(&self, e2: &PrimeExpr) -> PrimeExpr {
        PrimeExpr::FloorDiv(FloorDiv::make(self, e2))
    }

    pub fn evaluate_i64(&self) -> i64 {
        match self {
            PrimeExpr::Int(a) => a.value(),
            PrimeExpr::Float(a) => a.value() as i64,
            PrimeExpr::UInt(a) => a.value() as i64,
            _ => panic!("Cannot evaluate non-integer expression"),
        }
    }

    cast_expr!(to_variable, Variable);
    cast_expr!(to_add, Add);
    cast_expr!(to_sub, Sub);
    cast_expr!(to_mul, Mul);
    cast_expr!(to_div, Div);
    cast_expr!(to_mod, Rem);
    cast_expr!(to_min, Min);
    cast_expr!(to_max, Max);
    cast_expr!(to_eq, Eq);
    cast_expr!(to_ne, Ne);
    cast_expr!(to_lt, Lt);
    cast_expr!(to_le, Le);
    cast_expr!(to_gt, Gt);
    cast_expr!(to_ge, Ge);
    cast_expr!(to_and, BitAnd);
    cast_expr!(to_or, BitOr);
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

impl Into<PrimeExpr> for &PrimeExpr {
    fn into(self) -> PrimeExpr {
        self.clone()
    }
}

impl Into<PrimeExpr> for &&PrimeExpr {
    fn into(self) -> PrimeExpr {
        (*self).clone()
    }
}

impl Display for PrimeExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.print(0))
    }
}

impl Accepter for PrimeExpr {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_expr(self);
    }
}

impl AccepterMut for PrimeExpr {
    fn accept_mut<V: IRMutVisitor>(&self, visitor: &mut V) {
        visitor.visit_expr(self);
    }
}

impl AccepterMutate for PrimeExpr {
    fn accept_mutate<V: IRMutateVisitor>(&self, visitor: &mut V) {
        visitor.visit_expr(self);
    }
}

impl std::ops::Add for PrimeExpr {
    type Output = PrimeExpr;

    fn add(self, rhs: PrimeExpr) -> Self::Output {
        PrimeExpr::Add(Add::make(self, rhs))
    }
}

impl std::ops::Add<&PrimeExpr> for &PrimeExpr {
    type Output = PrimeExpr;

    fn add(self, rhs: &PrimeExpr) -> Self::Output {
        PrimeExpr::Add(Add::make(self, rhs))
    }
}

impl std::ops::Add<PrimeExpr> for &PrimeExpr {
    type Output = PrimeExpr;

    fn add(self, rhs: PrimeExpr) -> Self::Output {
        PrimeExpr::Add(Add::make(self, rhs))
    }
}

impl std::ops::Add<&PrimeExpr> for PrimeExpr {
    type Output = PrimeExpr;

    fn add(self, rhs: &PrimeExpr) -> Self::Output {
        PrimeExpr::Add(Add::make(self, rhs))
    }
}

impl std::ops::Add<i32> for &PrimeExpr {
    type Output = PrimeExpr;

    fn add(self, rhs: i32) -> Self::Output {
        PrimeExpr::Add(Add::make(self, rhs))
    }
}

impl std::ops::Add<i32> for PrimeExpr {
    type Output = PrimeExpr;

    fn add(self, rhs: i32) -> Self::Output {
        PrimeExpr::Add(Add::make(self, rhs))
    }
}

impl std::ops::Sub for PrimeExpr {
    type Output = PrimeExpr;

    fn sub(self, rhs: PrimeExpr) -> Self::Output {
        PrimeExpr::Sub(Sub::make(self, rhs))
    }
}

impl std::ops::Sub<i32> for &PrimeExpr {
    type Output = PrimeExpr;

    fn sub(self, rhs: i32) -> Self::Output {
        PrimeExpr::Sub(Sub::make(self, rhs))
    }
}

impl std::ops::Sub<i32> for PrimeExpr {
    type Output = PrimeExpr;

    fn sub(self, rhs: i32) -> Self::Output {
        PrimeExpr::Sub(Sub::make(self, rhs))
    }
}

impl std::ops::Sub<&PrimeExpr> for &PrimeExpr {
    type Output = PrimeExpr;

    fn sub(self, rhs: &PrimeExpr) -> Self::Output {
        PrimeExpr::Sub(Sub::make(self, rhs))
    }
}

impl std::ops::Mul for PrimeExpr {
    type Output = PrimeExpr;

    fn mul(self, rhs: PrimeExpr) -> Self::Output {
        PrimeExpr::Mul(Mul::make(self, rhs))
    }
}

impl std::ops::Mul for &PrimeExpr {
    type Output = PrimeExpr;

    fn mul(self, rhs: &PrimeExpr) -> Self::Output {
        PrimeExpr::Mul(Mul::make(self, rhs))
    }
}

impl std::ops::Div for PrimeExpr {
    type Output = PrimeExpr;

    fn div(self, rhs: PrimeExpr) -> Self::Output {
        PrimeExpr::Div(Div::make(self, rhs))
    }
}

impl std::ops::Rem for PrimeExpr {
    type Output = PrimeExpr;

    fn rem(self, rhs: PrimeExpr) -> Self::Output {
        PrimeExpr::Rem(Rem::make(self, rhs))
    }
}

impl std::ops::Rem<&PrimeExpr> for &PrimeExpr {
    type Output = PrimeExpr;

    fn rem(self, rhs: &PrimeExpr) -> Self::Output {
        PrimeExpr::Rem(Rem::make(self, rhs))
    }
}

impl Into<PrimeExpr> for bool {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Int(Int::make(Dtype::Bool, self as i64))
    }
}

impl Into<PrimeExpr> for &bool {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Int(Int::make(Dtype::Bool, *self as i64))
    }
}

impl Into<PrimeExpr> for half::f16 {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Float(Float::make(Dtype::F16, self.to_f64()))
    }
}

impl Into<PrimeExpr> for &half::f16 {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Float(Float::make(Dtype::F16, self.to_f64()))
    }
}

impl Into<PrimeExpr> for half::bf16 {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Float(Float::make(Dtype::BF16, self.to_f64()))
    }
}

impl Into<PrimeExpr> for &half::bf16 {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Float(Float::make(Dtype::BF16, self.to_f64()))
    }
}

impl Into<PrimeExpr> for isize {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Int(Int::make(Dtype::Isize, self as i64))
    }
}

impl Into<PrimeExpr> for &isize {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Int(Int::make(Dtype::Isize, *self as i64))
    }
}

impl Into<PrimeExpr> for usize {
    fn into(self) -> PrimeExpr {
        PrimeExpr::UInt(UInt::make(Dtype::Usize, self as u64))
    }
}

impl Into<PrimeExpr> for &usize {
    fn into(self) -> PrimeExpr {
        PrimeExpr::UInt(UInt::make(Dtype::Usize, *self as u64))
    }
}

impl Into<PrimeExpr> for i8 {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Int(Int::make(Dtype::I8, self as i64))
    }
}

impl Into<PrimeExpr> for &i8 {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Int(Int::make(Dtype::I8, *self as i64))
    }
}

impl Into<PrimeExpr> for i16 {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Int(Int::make(Dtype::I16, self as i64))
    }
}

impl Into<PrimeExpr> for &i16 {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Int(Int::make(Dtype::I16, *self as i64))
    }
}

impl Into<PrimeExpr> for i32 {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Int(Int::make(Dtype::I32, self as i64))
    }
}

impl Into<PrimeExpr> for &i32 {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Int(Int::make(Dtype::I32, *self as i64))
    }
}

impl Into<PrimeExpr> for i64 {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Int(Int::make(Dtype::I64, self))
    }
}

impl Into<PrimeExpr> for &i64 {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Int(Int::make(Dtype::I64, *self))
    }
}

impl Into<PrimeExpr> for f32 {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Float(Float::make(Dtype::F32, self as f64))
    }
}

impl Into<PrimeExpr> for &f32 {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Float(Float::make(Dtype::F32, *self as f64))
    }
}

impl Into<PrimeExpr> for f64 {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Float(Float::make(Dtype::F64, self))
    }
}

impl Into<PrimeExpr> for &f64 {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Float(Float::make(Dtype::F64, *self))
    }
}

impl Into<PrimeExpr> for u8 {
    fn into(self) -> PrimeExpr {
        PrimeExpr::UInt(UInt::make(Dtype::U8, self as u64))
    }
}

impl Into<PrimeExpr> for &u8 {
    fn into(self) -> PrimeExpr {
        PrimeExpr::UInt(UInt::make(Dtype::U8, *self as u64))
    }
}

impl Into<PrimeExpr> for u16 {
    fn into(self) -> PrimeExpr {
        PrimeExpr::UInt(UInt::make(Dtype::U16, self as u64))
    }
}

impl Into<PrimeExpr> for &u16 {
    fn into(self) -> PrimeExpr {
        PrimeExpr::UInt(UInt::make(Dtype::U16, *self as u64))
    }
}

impl Into<PrimeExpr> for u32 {
    fn into(self) -> PrimeExpr {
        PrimeExpr::UInt(UInt::make(Dtype::U32, self as u64))
    }
}

impl Into<PrimeExpr> for &u32 {
    fn into(self) -> PrimeExpr {
        PrimeExpr::UInt(UInt::make(Dtype::U32, *self as u64))
    }
}

impl Into<PrimeExpr> for u64 {
    fn into(self) -> PrimeExpr {
        PrimeExpr::UInt(UInt::make(Dtype::U64, self))
    }
}

impl Into<PrimeExpr> for &u64 {
    fn into(self) -> PrimeExpr {
        PrimeExpr::UInt(UInt::make(Dtype::U64, *self))
    }
}

impl Into<PrimeExpr> for &str {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Str(Str::make(self))
    }
}

impl Into<PrimeExpr> for String {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Str(Str::make(&self))
    }
}