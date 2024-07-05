use std::{ cell::RefCell, collections::VecDeque, rc::Rc, sync::Arc };

use tensor_types::dtype::Dtype;

use crate::{ halide::prime_expr::PrimeExpr, hlir::{ tensor::Tensor, tensor_slice::TensorSlice } };

use super::{ iter::{ Iter, IterVar }, transforms::Transforms };

#[derive(Clone)]
pub struct Temp {
    pub(crate) shape: Vec<Rc<RefCell<Iter>>>,
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
            shape: original
                .shape()
                .iter()
                .map(|x| {
                    let x = x.to_iter_var().unwrap();
                    Rc::new(RefCell::new(Iter::IterVar(IterVar::make(x.var(), x.start(), x.end(), x.step()))))
                })
                .collect(),
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
            shape: original
                .shape()
                .iter()
                .map(|x| {
                    let x = x.to_iter_var().unwrap();
                    Rc::new(RefCell::new(Iter::IterVar(IterVar::make(x.var(), x.start(), x.end(), x.step()))))
                })
                .collect(),
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
