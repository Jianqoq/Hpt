use std::{ panic::Location, sync::Arc };

use tensor_types::dtype::Dtype;

use crate::halide::prime_expr::PrimeExpr;

use super::tensor::StridesCal;

#[derive(Clone)]
pub struct SrgNode {
    pub(crate) id: usize,
    pub(crate) dtype: Dtype,
    pub(crate) shape: Arc<Vec<PrimeExpr>>,
    pub(crate) inputs: Arc<Vec<usize>>,
    pub(crate) outputs: Arc<Vec<usize>>,
    pub(crate) strides_cal: StridesCal,
    pub(crate) span: &'static Location<'static>,
}

impl SrgNode {
    pub fn is_output(&self) -> bool {
        self.outputs.is_empty()
    }
    pub fn is_input(&self) -> bool {
        self.inputs.is_empty()
    }
}
