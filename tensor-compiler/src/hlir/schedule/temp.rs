use std::{ collections::VecDeque, sync::Arc };

use tensor_types::dtype::Dtype;

use crate::{ halide::prime_expr::PrimeExpr, hlir::{tensor::Tensor, tensor_slice::TensorSlice}, iter_var::IterVar };

use super::transforms::Transforms;

#[derive(Clone)]
pub struct Temp {
    pub(crate) shape: Vec<IterVar>,
    pub(crate) strides: Vec<usize>,
    pub(crate) body: PrimeExpr,
    pub(crate) name: Arc<String>,
    pub(crate) inputs: Vec<TensorSlice>,
    pub(crate) original: Arc<Tensor>,
    pub(crate) dtype: Dtype,
    pub(crate) transforms: VecDeque<Transforms>,
}

impl From<Tensor> for Temp {
    fn from(tensor: Tensor) -> Self {
        let dtype = tensor.dtype();
        let original = Arc::new(tensor.clone());
        Self {
            shape: original.shape().clone(),
            strides: original.strides().clone(),
            body: original.body().clone(),
            name: Arc::new(tensor.name().to_string()),
            inputs: tensor.inputs().clone(),
            dtype,
            original,
            transforms: VecDeque::new(),
        }
    }
}

impl From<&Tensor> for Temp {
    fn from(tensor: &Tensor) -> Self {
        let dtype = tensor.dtype();
        let original = Arc::new(tensor.clone());
        Self {
            shape: original.shape().clone(),
            strides: original.strides().clone(),
            body: original.body().clone(),
            name: Arc::new(tensor.name().to_string()),
            inputs: tensor.inputs().clone(),
            dtype,
            original,
            transforms: VecDeque::new(),
        }
    }
}
