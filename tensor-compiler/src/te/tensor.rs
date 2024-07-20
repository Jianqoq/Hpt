use std::sync::Arc;

use crate::halide::prime_expr::PrimeExpr;

use super::operation::Operation;

#[derive(Clone)]
pub struct Tensor {
    pub(crate) shape: Arc<Vec<PrimeExpr>>,
    pub(crate) body: PrimeExpr,
    pub(crate) inputs: Arc<Vec<Tensor>>,
    pub(crate) op: Operation,
    pub(crate) id: usize,
}
