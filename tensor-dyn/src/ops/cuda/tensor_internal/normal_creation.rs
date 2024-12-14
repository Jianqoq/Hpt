use std::{
    ops::{Div, Sub},
    sync::Arc,
};

use crate::{
    backend::Backend,
    ops::cuda::{
        cuda_utils::compile_kernel,
        kernel_constants::{ARANGE_KERNELS, FILL_KERNELS},
    },
    tensor_base::_Tensor,
    BoolVector, Cuda, ALIGN,
};
use anyhow::Result;
use cudarc::driver::{DeviceRepr, LaunchAsync, LaunchConfig};
use tensor_allocator::CUDA_CACHE;
use tensor_common::{layout::Layout, pointer::Pointer, shape::Shape};
use tensor_traits::{CommonBounds, TensorCreator, TensorInfo};
use tensor_types::{
    convertion::{Convertor, FromScalar},
    dtype::Dtype,
    into_scalar::IntoScalar,
    type_promote::{FloatOutUnary, NormalOut},
};

impl<T: CommonBounds + DeviceRepr, const DEVICE_ID: usize> TensorCreator<T>
    for _Tensor<T, Cuda, DEVICE_ID>
{
    fn empty<S: Into<Shape>>(shape: S) -> Result<Self> {
        let _shape = shape.into();
        let res_shape = Shape::from(_shape);
        let size = res_shape
            .iter()
            .try_fold(1i64, |acc, &num| acc.checked_mul(num).or(Some(i64::MAX)))
            .unwrap_or(i64::MAX) as usize;
        let layout = std::alloc::Layout::from_size_align(
            size.checked_mul(size_of::<T>())
                .unwrap_or((isize::MAX as usize) - (ALIGN - 1)), // when overflow happened, we use max memory `from_size_align` accept
            ALIGN,
        )?;
        let (ptr, device) = if let Ok(mut cache) = CUDA_CACHE.lock() {
            cache.allocate(layout, DEVICE_ID)?
        } else {
            return Err(anyhow::anyhow!("failed to lock CUDA_CACHE"));
        };
        Ok(_Tensor {
            #[cfg(feature = "bound_check")]
            data: Pointer::new(ptr as *mut T, size as i64),
            #[cfg(not(feature = "bound_check"))]
            data: Pointer::new(ptr as *mut T),
            parent: None,
            layout: Layout::from(res_shape.clone()),
            mem_layout: Arc::new(layout),
            _backend: Backend::<Cuda>::new(ptr as u64, device),
        })
    }

    fn zeros<S: Into<Shape>>(shape: S) -> Result<Self> {
        let empty = Self::empty(shape)?;
        if let Ok(mut cache) = CUDA_CACHE.lock() {
            cache.memset_zeros(empty.ptr().ptr as *mut u8, &empty.mem_layout, DEVICE_ID)
        } else {
            return Err(anyhow::anyhow!("failed to lock CUDA_CACHE"));
        };
        Ok(empty)
    }

    fn ones<S: Into<Shape>>(shape: S) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        Self::full(T::ONE, shape)
    }

    fn empty_like(&self) -> Result<Self> {
        Self::empty(self.shape())
    }

    fn zeros_like(&self) -> Result<Self> {
        // Self::zeros(self.shape())
        unimplemented!()
    }

    fn ones_like(&self) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        // Self::ones(self.shape())
        unimplemented!()
    }

    fn full<S: Into<Shape>>(val: T, shape: S) -> Result<Self> {
        let ret = Self::empty(shape)?;
        compile_kernel(
            "fill",
            include_str!("../kernels/fill.cu"),
            ret.device(),
            &FILL_KERNELS,
        )?;
        let func_name = if T::ID == Dtype::F16 {
            format!("fill_{}_vec2", T::ID)
        } else {
            format!("fill_{}_vec4", T::ID)
        };
        let fill_kernel = ret.device().get_func("fill", &func_name).unwrap();
        let cfg = LaunchConfig::for_num_elems(ret.size() as u32);
        let mut slice = unsafe {
            ret.device()
                .upgrade_device_ptr::<T>(ret.ptr().ptr as u64, ret.size())
        };
        unsafe { fill_kernel.launch(cfg, (&mut slice, val, ret.size())) }?;
        slice.leak();
        Ok(ret)
    }

    fn full_like(&self, val: T) -> Result<Self> {
        _Tensor::full(val, self.shape())
    }

    fn arange<U>(start: U, end: U) -> Result<Self>
    where
        T: Convertor + FromScalar<U>,
        usize: IntoScalar<T>,
        U: Convertor + IntoScalar<T> + Copy,
    {
        let start: T = start.into_scalar();
        let end: T = end.into_scalar();
        let size = end.to_i64() - start.to_i64();
        if size <= 0 {
            return _Tensor::<T, Cuda, DEVICE_ID>::empty(Arc::new(vec![0]));
        }
        let ret = Self::empty(Arc::new(vec![size]))?;
        compile_kernel(
            "arange",
            include_str!("../kernels/arange.cu"),
            ret.device(),
            &ARANGE_KERNELS,
        )?;
        let func_name = if T::ID == Dtype::F16 {
            format!("arange_{}_vec2", T::ID)
        } else {
            format!("arange_{}_vec4", T::ID)
        };
        let arange_kernel = ret.device().get_func("arange", &func_name).unwrap();
        let cfg = LaunchConfig::for_num_elems(ret.size() as u32);
        let mut slice = unsafe {
            ret.device()
                .upgrade_device_ptr::<T>(ret.ptr().ptr as u64, ret.size())
        };
        unsafe { arange_kernel.launch(cfg, (&mut slice, start, T::ONE, ret.size())) }?;
        slice.leak();
        Ok(ret)
    }

    fn arange_step(start: T, end: T, step: T) -> Result<Self>
    where
        T: Convertor + FromScalar<usize>,
    {
        let step_float = step.to_f64();
        let end_usize = end.to_i64();
        let start_usize = start.to_i64();
        let start: T = start.into_scalar();
        let step: T = step.into_scalar();
        let size = ((end_usize - start_usize) as usize) / (step_float.abs() as usize);
        if size <= 0 {
            return _Tensor::<T, Cuda, DEVICE_ID>::empty(Arc::new(vec![0]));
        }
        let ret = Self::empty(Arc::new(vec![size as i64]))?;
        compile_kernel(
            "arange",
            include_str!("../kernels/arange.cu"),
            ret.device(),
            &ARANGE_KERNELS,
        )?;
        let func_name = if T::ID == Dtype::F16 {
            format!("arange_{}_vec2", T::ID)
        } else {
            format!("arange_{}_vec4", T::ID)
        };
        let arange_kernel = ret.device().get_func("arange", &func_name).unwrap();
        let cfg = LaunchConfig::for_num_elems(ret.size() as u32);
        let mut slice = unsafe {
            ret.device()
                .upgrade_device_ptr::<T>(ret.ptr().ptr as u64, ret.size())
        };
        unsafe { arange_kernel.launch(cfg, (&mut slice, start, step, ret.size())) }?;
        slice.leak();
        Ok(ret)
    }

    fn eye(n: usize, m: usize, k: usize) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        unimplemented!()
    }

    fn linspace(start: T, end: T, num: usize, include_end: bool) -> Result<Self>
    where
        T: Convertor + num::Float,
        usize: IntoScalar<T>,
        f64: IntoScalar<T>,
    {
        unimplemented!()
    }

    fn logspace(start: T, end: T, num: usize, include_end: bool, base: T) -> Result<Self>
    where
        T: Convertor + num::Float + FromScalar<usize> + FromScalar<f64>,
    {
        unimplemented!()
    }

    fn geomspace(start: T, end: T, n: usize, include_end: bool) -> Result<Self>
    where
        T: PartialOrd + FromScalar<<T as FloatOutUnary>::Output> + std::ops::Neg<Output = T>,
        <T as FloatOutUnary>::Output: Sub<Output = <T as FloatOutUnary>::Output>
            + FromScalar<usize>
            + FromScalar<f64>
            + Div<Output = <T as FloatOutUnary>::Output>
            + CommonBounds,
    {
        unimplemented!()
    }

    fn tri(n: usize, m: usize, k: i64, low_triangle: bool) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        unimplemented!()
    }

    fn tril(&self, k: i64) -> Result<Self>
    where
        T: NormalOut<bool, Output = T> + IntoScalar<T>,
        T::Vec: NormalOut<BoolVector, Output = T::Vec>,
    {
        unimplemented!()
    }

    fn triu(&self, k: i64) -> Result<Self>
    where
        T: NormalOut<bool, Output = T> + IntoScalar<T>,
        T::Vec: NormalOut<BoolVector, Output = T::Vec>,
    {
        unimplemented!()
    }

    fn identity(n: usize) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        _Tensor::eye(n, n, 0)
    }
}
