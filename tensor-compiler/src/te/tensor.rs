use std::sync::Arc;

use crate::halide::prime_expr::PrimeExpr;

use super::{context::Context, rc_mut::RcMut};

#[derive(Clone)]
pub struct Tensor {
    pub(crate) shape: Vec<PrimeExpr>,
    pub(crate) ctx: Context,
    pub(crate) body: PrimeExpr,
    pub(crate) inputs: Arc<Vec<Tensor>>,
    pub(crate) id: usize,
}
