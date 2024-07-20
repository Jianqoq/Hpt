use std::sync::Arc;

use crate::halide::prime_expr::PrimeExpr;


#[derive(Clone)]
pub enum Operation {
    Reshape(Arc<Vec<PrimeExpr>>),
    Transpose(Arc<Vec<usize>>),
    Slice(Arc<Vec<(PrimeExpr, PrimeExpr, PrimeExpr)>>),
    Sum(Arc<Vec<usize>>),
    Sin,
    Add,
    None,
}