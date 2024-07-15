use std::{ alloc::Layout, ffi::c_void };
use tensor_traits::tensor::TensorInfo;
use tensor_types::dtype::{ Dtype, TypeCommon };

#[repr(C)]
pub struct Tensor {
    pub(crate) ptr: *mut c_void,
    pub(crate) dtype: Dtype,
    pub(crate) shape: *mut i64,
    pub(crate) strides: *mut i64,
}

impl Tensor {
    pub fn raw_new<T: TypeCommon>(tensor: tensor_dyn::tensor::Tensor<T>) -> *mut Self {
        let ptr = tensor.ptr().ptr as *mut c_void;
        let layout = Layout::from_size_align(
            std::mem::size_of::<i64>() * tensor.shape().len(),
            8
        ).expect("Failed to create layout");
        unsafe {
            let shape = std::alloc::alloc(layout) as *mut i64;
            std::ptr::copy_nonoverlapping(tensor.shape().as_ptr(), shape, tensor.shape().len());
            let strides = std::alloc::alloc(layout) as *mut i64;
            std::ptr::copy_nonoverlapping(
                tensor.strides().as_ptr(),
                strides,
                tensor.strides().len()
            );
            println!("shape {:p}", shape);
            let tensor = std::alloc::alloc(Layout::new::<Self>()) as *mut Self;
            std::ptr::write(tensor, Self {
                ptr,
                dtype: T::ID,
                shape,
                strides,
            });
            tensor
        }
    }
}

impl<T: TypeCommon> From<tensor_dyn::tensor::Tensor<T>> for Tensor {
    fn from(value: tensor_dyn::tensor::Tensor<T>) -> Self {
        let ptr = value.ptr().ptr as *mut c_void;
        let layout = Layout::from_size_align(
            std::mem::size_of::<i64>() * value.shape().len(),
            8
        ).expect("Failed to create layout");

        unsafe {
            let shape = std::alloc::alloc(layout) as *mut i64;
            std::ptr::copy_nonoverlapping(value.shape().as_ptr(), shape, value.shape().len());
            let strides = std::alloc::alloc(layout) as *mut i64;
            std::ptr::copy_nonoverlapping(value.strides().as_ptr(), strides, value.strides().len());
            Tensor {
                ptr,
                dtype: T::ID,
                shape,
                strides,
            }
        }
    }
}
