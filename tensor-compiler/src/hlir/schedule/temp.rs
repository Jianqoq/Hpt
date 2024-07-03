use std::sync::Arc;

use tensor_types::dtype::Dtype;

use crate::{ halide::prime_expr::PrimeExpr, hlir::tensor::Tensor, iter_val::IterVar };

#[derive(Clone)]
pub struct Temp {
    shape: Vec<IterVar>,
    op: Arc<dyn Fn(Arc<Vec<Tensor>>, Vec<PrimeExpr>) -> PrimeExpr>,
    name: Arc<String>,
    inputs: Vec<Tensor>,
    dtype: Dtype,
}

impl From<Tensor> for Temp {
    fn from(tensor: Tensor) -> Self {
        let dtype = tensor.dtype();
        Self {
            shape: tensor.shape().clone(),
            op: tensor.op().clone(),
            name: Arc::new(tensor.name().to_string()),
            inputs: vec![tensor],
            dtype,
        }
    }
}

impl From<&Tensor> for Temp {
    fn from(tensor: &Tensor) -> Self {
        let dtype = tensor.dtype();
        Self {
            shape: tensor.shape().clone(),
            op: tensor.op().clone(),
            name: Arc::new(tensor.name().to_string()),
            inputs: vec![tensor.clone()],
            dtype,
        }
    }
}