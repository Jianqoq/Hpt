use std::ffi::c_void;
use tensor_common::shape::Shape;
use tensor_traits::tensor::TensorInfo;
use tensor_types::dtype::{ Dtype, TypeCommon };

pub struct Array {
    pub shape: Shape,
    pub dtype: Dtype,
    pub data: *mut c_void,
    pub id: usize,
}

impl<T: TypeCommon> From<tensor_dyn::tensor::Tensor<T>> for Array {
    fn from(value: tensor_dyn::tensor::Tensor<T>) -> Self {
        Self {
            shape: value.shape().clone(),
            dtype: T::ID,
            data: value.ptr().ptr as *mut c_void,
            id: 0,
        }
    }
}
