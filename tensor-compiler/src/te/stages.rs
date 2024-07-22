use crate::{ halide::{prime_expr::PrimeExpr, stmt::Stmt}, iter_var::IterVar };

#[derive(Clone)]
pub enum Body {
    PrimeExpr(PrimeExpr),
    Stmt(Stmt),
    Stage(usize),
}

pub struct Stage {
    pub(crate) dims: Vec<IterVar>,
    pub(crate) bodys: Vec<Body>,
    pub(crate) id: usize,
}
