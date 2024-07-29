use std::ffi::c_void;
use tensor_common::shape::Shape;
use tensor_dyn::tensor::Tensor;
use tensor_traits::tensor::TensorInfo;
use tensor_types::dtype::Dtype;

pub struct Array {
    pub shape: Shape,
    pub dtype: Dtype,
    pub data: *mut c_void,
}

impl Array {
    pub fn ptr(&self) -> *mut c_void {
        self.data
    }
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }
}

impl<T> From<Tensor<T>> for Array {
    fn from(tensor: Tensor<T>) -> Self {
        Array {
            shape: TensorInfo::shape(&tensor).clone(),
            dtype: Dtype::F32,
            data: tensor.ptr().ptr as *mut c_void,
        }
    }
}
