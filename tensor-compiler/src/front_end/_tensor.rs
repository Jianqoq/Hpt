#![allow(dead_code)]
use std::rc::Rc;

use serde::Serialize;
use serde::ser::SerializeStruct;
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

impl Serialize for _Tensor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: serde::Serializer {
        let mut state = serializer.serialize_struct("Tensor", 7)?;
        state.serialize_field("inputs", &*self.inputs)?;
        state.serialize_field("dtype", &self.dtype)?;
        state.serialize_field("op", &self.op)?;
        state.serialize_field("layout", &self.layout)?;
        state.serialize_field("name", &self.name.as_ref().map(|x| x.as_ref()))?;
        state.serialize_field("error_msg", &*self.error_msg)?;
        state.serialize_field("block_id", &self.block_id)?;
        todo!()
    }
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
