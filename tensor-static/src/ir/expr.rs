use std::sync::Arc;


pub enum Expr {
    Int(i64),
    Float(f64),
    Add(Arc<Expr>, Arc<Expr>),
    Sub(Arc<Expr>, Arc<Expr>),
    Mul(Arc<Expr>, Arc<Expr>),
    Div(Arc<Expr>, Arc<Expr>),
    Mod(Arc<Expr>, Arc<Expr>),
    Pow(Arc<Expr>, Arc<Expr>),
    Cast(Arc<Expr>, Arc<String>),
    Neg(Arc<Expr>),
    Var(Arc<String>),

}