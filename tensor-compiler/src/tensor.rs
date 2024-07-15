use std::ffi::c_void;
use tensor_traits::tensor::TensorInfo;
use tensor_types::dtype::{ Dtype, TypeCommon };

#[derive(Clone)]
#[repr(C)]
pub struct Tensor {
    pub(crate) name: String,
    pub(crate) ptr: *mut c_void,
    pub(crate) dtype: Dtype,
    pub(crate) shape: *mut i64,
    pub(crate) strides: *mut i64,
}

impl Tensor {
    pub fn new<T: TypeCommon>(tensor: tensor_dyn::tensor::Tensor<T>, name: &str) -> Self {
        let ptr = tensor.ptr().ptr as *mut c_void;
        unsafe {
            let layout = std::alloc::Layout
                ::from_size_align(tensor.shape().len() * std::mem::size_of::<i64>(), 8)
                .unwrap();
            let shape = std::alloc::alloc(layout) as *mut i64;
            std::ptr::copy_nonoverlapping(tensor.shape().as_ptr(), shape, tensor.shape().len());
            let strides = std::alloc::alloc(layout) as *mut i64;
            std::ptr::copy_nonoverlapping(
                tensor.strides().as_ptr(),
                strides,
                tensor.strides().len()
            );
            Self {
                name: name.to_string(),
                ptr,
                dtype: T::ID,
                shape: tensor.shape().as_ptr() as *mut i64,
                strides: tensor.strides().as_ptr() as *mut i64,
            }
        }
    }
}
