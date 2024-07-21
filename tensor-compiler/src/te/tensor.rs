use std::{panic::Location, sync::Arc};

use tensor_types::dtype::Dtype;

use crate::halide::prime_expr::PrimeExpr;

use super::operation::Operation;

#[derive(Clone)]
pub struct Tensor {
    pub(crate) shape: Arc<Vec<PrimeExpr>>,
    pub(crate) dtype: Dtype,
    pub(crate) body: PrimeExpr,
    pub(crate) inputs: Arc<Vec<usize>>,
    pub(crate) op: Operation,
    pub(crate) span: &'static Location<'static>,
    pub(crate) id: usize,
}
