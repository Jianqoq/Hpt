use crate::{ halide::prime_expr::PrimeExpr, iter_var::IterVar };

pub enum Body {
    PrimeExpr(PrimeExpr),
    Stage(usize),
}

pub struct Stage {
    pub(crate) dims: Vec<IterVar>,
    pub(crate) bodys: Vec<Body>,
    pub(crate) id: usize,
}
