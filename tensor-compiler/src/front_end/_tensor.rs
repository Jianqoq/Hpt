#![allow(dead_code)]
use std::rc::Rc;

use tensor_common::{ err_handler::ErrHandler, layout::Layout };
use tensor_types::dtype::Dtype;

use crate::op::Op;

use super::tensor::Tensor;

/// unchangeable tensor
#[derive(Clone)]
pub(crate) struct _Tensor {
    pub(crate) inputs: Rc<Vec<usize>>,
    pub(crate) dtype: Dtype,
    pub(crate) op: Op,
    pub(crate) layout: Layout,
    pub(crate) name: Option<Rc<String>>,
    pub(crate) error_msg: Rc<Vec<ErrHandler>>,
    pub(crate) block_id: usize,
}

impl From<Tensor> for _Tensor {
    fn from(tensor: Tensor) -> Self {
        _Tensor {
            inputs: tensor.inputs,
            dtype: tensor.dtype,
            op: tensor.op,
            layout: tensor.layout,
            name: tensor.name,
            error_msg: tensor.error_msg,
            block_id: tensor.block_id,
        }
    }
}