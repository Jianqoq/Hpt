use crate::{
    ops::{
        common::creation::geomspace_preprocess_start_step,
        cuda::cuda_utils::{compute_kernel_launch_config, load_ptx_and_get_data},
    },
    tensor_base::_Tensor,
    Backend, BoolVector, Cuda, ALIGN,
};
use cudarc::driver::{DeviceRepr, LaunchAsync, LaunchConfig};
use hpt_allocator::CUDA_CACHE;
use hpt_common::{
    error::{base::TensorError, memory::MemoryError},
    layout::layout::Layout,
    shape::shape::Shape,
    Pointer,
};
use hpt_cudakernels::CREATION;
use hpt_traits::{CommonBounds, TensorCreator, TensorInfo};
use hpt_types::{dtype::CudaType, into_scalar::Cast, type_promote::NormalOut};
use std::{panic::Location, sync::Arc};

impl<T: CommonBounds + DeviceRepr + CudaType, const DEVICE: usize> TensorCreator<T>
    for _Tensor<T, Cuda, DEVICE>
{
    type Output = Self;
    fn empty<S: Into<Shape>>(shape: S) -> std::result::Result<Self, TensorError> {
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
        )
        .map_err(|e| {
            TensorError::Memory(MemoryError::AllocationFailed {
                device: "cuda".to_string(),
                id: DEVICE,
                size,
                source: Some(Box::new(e)),
                location: Location::caller(),
            })
        })?;
        let (ptr, device) = CUDA_CACHE
            .lock()
            .expect("CUDA_CACHE is poisoned")
            .allocate(layout, DEVICE)?;
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

    fn zeros<S: Into<Shape>>(shape: S) -> std::result::Result<Self, TensorError> {
        let empty = Self::empty(shape)?;
        CUDA_CACHE
            .lock()
            .expect("CUDA_CACHE is poisoned")
            .memset_zeros(empty.ptr().ptr as *mut u8, &empty.mem_layout, DEVICE);
        Ok(empty)
    }

    fn ones<S: Into<Shape>>(shape: S) -> std::result::Result<Self, TensorError>
    where
        u8: Cast<T>,
    {
        Self::full(T::ONE, shape)
    }

    fn empty_like(&self) -> std::result::Result<Self, TensorError> {
        Self::empty(self.shape())
    }

    fn zeros_like(&self) -> std::result::Result<Self, TensorError> {
        Self::zeros(self.shape())
    }

    fn ones_like(&self) -> std::result::Result<Self, TensorError>
    where
        u8: Cast<T>,
    {
        Self::ones(self.shape())
    }

    fn full<S: Into<Shape>>(val: T, shape: S) -> std::result::Result<Self, TensorError> {
        let ret = Self::empty(shape)?;
        let (fill_kernel, _) = load_ptx_and_get_data(
            "creation",
            &format!("fill_{}", T::STR),
            ret.device(),
            ret.device_cap(),
            &CREATION,
        )?;
        let cfg = LaunchConfig::for_num_elems(ret.size() as u32);
        let mut slice = unsafe {
            ret.device()
                .upgrade_device_ptr::<T>(ret.ptr().ptr as u64, ret.size())
        };
        unsafe { fill_kernel.launch(cfg, (&mut slice, val, ret.size())) }?;
        slice.leak();
        Ok(ret)
    }

    fn full_like(&self, val: T) -> std::result::Result<Self, TensorError> {
        Self::full(val, self.shape())
    }

    fn arange<U>(start: U, end: U) -> std::result::Result<Self, TensorError>
    where
        usize: Cast<T>,
        U: Cast<i64> + Cast<T> + Copy,
    {
        let end_i64: i64 = end.cast();
        let start_i64: i64 = start.cast();
        let size: i64 = end_i64 - start_i64;
        let start: T = start.cast();
        let ret = Self::empty(Arc::new(vec![size]))?;
        let (arange_kernel, _) = load_ptx_and_get_data(
            "creation",
            &format!("arange_{}", T::STR),
            ret.device(),
            ret.device_cap(),
            &CREATION,
        )?;
        let cfg = LaunchConfig::for_num_elems(ret.size() as u32);
        let slice = ret.cuda_slice();
        unsafe { arange_kernel.launch(cfg, (slice, start, T::ONE, ret.size())) }?;
        Ok(ret)
    }

    fn arange_step(start: T, end: T, step: T) -> std::result::Result<Self, TensorError>
    where
        T: Cast<f64> + Cast<usize>,
        usize: Cast<T>,
    {
        let step_float: f64 = step.cast();
        let end_usize: usize = end.cast();
        let start_usize: usize = start.cast();
        let size = ((end_usize - start_usize) as usize) / (step_float.abs() as usize);
        let ret = Self::empty(Arc::new(vec![size as i64]))?;
        let (arange_kernel, reg_info) = load_ptx_and_get_data(
            "creation",
            &format!("arange_{}", T::STR),
            ret.device(),
            ret.device_cap(),
            &CREATION,
        )?;
        let mut slice = unsafe {
            ret.device()
                .upgrade_device_ptr::<T>(ret.ptr().ptr as u64, ret.size())
        };
        let cfg = compute_kernel_launch_config(ret.device(), &reg_info, ret.size());
        unsafe { arange_kernel.launch(cfg, (&mut slice, start, step, ret.size())) }?;
        slice.leak();
        Ok(ret)
    }

    fn eye(n: usize, m: usize, k: usize) -> std::result::Result<Self, TensorError> {
        let shape = vec![n as i64, m as i64];
        let ret = Self::empty(Arc::new(shape))?;
        let (eye_kernel, reg_info) = load_ptx_and_get_data(
            "creation",
            &format!("eye_{}", T::STR),
            ret.device(),
            ret.device_cap(),
            &CREATION,
        )?;
        let cfg = compute_kernel_launch_config(ret.device(), &reg_info, ret.size());
        unsafe { eye_kernel.launch(cfg, (ret.cuda_slice(), n, m, k)) }?;
        Ok(ret)
    }

    fn linspace<U>(
        start: U,
        end: U,
        num: usize,
        include_end: bool,
    ) -> std::result::Result<Self, TensorError>
    where
        U: Cast<f64> + Cast<T> + Copy,
        usize: Cast<T>,
        f64: Cast<T>,
    {
        let _start: f64 = start.cast();
        let _end: f64 = end.cast();
        let n: f64 = num as f64;
        let step: f64 = if include_end {
            (_end - _start) / (n - 1.0)
        } else {
            (_end - _start) / n
        };
        let step_t: T = step.cast();
        let start_t: T = start.cast();
        let end_t: T = end.cast();
        let ret = _Tensor::<T, Cuda, DEVICE>::empty(Arc::new(vec![num as i64]))?;

        let (linspace_kernel, reg_info) = load_ptx_and_get_data(
            "creation",
            &format!("linspace_{}", T::STR),
            ret.device(),
            ret.device_cap(),
            &CREATION,
        )?;
        let cfg = compute_kernel_launch_config(ret.device(), &reg_info, ret.size());
        unsafe {
            linspace_kernel.launch(
                cfg,
                (ret.cuda_slice(), start_t, step_t, end_t, include_end, num),
            )
        }?;
        Ok(ret)
    }

    fn logspace(
        start: T,
        end: T,
        num: usize,
        include_end: bool,
        base: T,
    ) -> std::result::Result<Self, TensorError>
    where
        T: Cast<f64> + num::Float + NormalOut<T, Output = T>,
        usize: Cast<T>,
        f64: Cast<T>,
    {
        let _start: f64 = start.cast();
        let _end: f64 = end.cast();
        let n: f64 = num as f64;
        let step: f64 = if include_end {
            (_end - _start) / (n - 1.0)
        } else {
            (_end - _start) / n
        };
        let step_t: T = step.cast();
        let ret = _Tensor::<T, Cuda, DEVICE>::empty(Arc::new(vec![num as i64]))?;

        let (logspace_kernel, reg_info) = load_ptx_and_get_data(
            "creation",
            &format!("logspace_{}", T::STR),
            ret.device(),
            ret.device_cap(),
            &CREATION,
        )?;
        let cfg = compute_kernel_launch_config(ret.device(), &reg_info, ret.size());
        unsafe { logspace_kernel.launch(cfg, (ret.cuda_slice(), base, start, step_t, num)) }?;
        Ok(ret)
    }

    fn geomspace(
        start: T,
        end: T,
        n: usize,
        include_end: bool,
    ) -> std::result::Result<Self, TensorError>
    where
        f64: Cast<T>,
        usize: Cast<T>,
        T: Cast<f64>,
    {
        let start_f64: f64 = start.cast();
        let end_f64: f64 = end.cast();
        let both_negative = start_f64 < 0.0 && end_f64 < 0.0;
        let (new_start, step) =
            geomspace_preprocess_start_step(start_f64, end_f64, n, include_end)?;
        let start_t: T = new_start.cast();
        let step_t: T = step.cast();
        let ret = Self::empty(Arc::new(vec![n as i64]))?;
        let (geomspace_kernel, reg_info) = load_ptx_and_get_data(
            "creation",
            &format!("geomspace_{}", T::STR),
            ret.device(),
            ret.device_cap(),
            &CREATION,
        )?;
        let cfg = compute_kernel_launch_config(ret.device(), &reg_info, ret.size());
        unsafe {
            geomspace_kernel.launch(cfg, (ret.cuda_slice(), start_t, step_t, both_negative, n))
        }?;
        Ok(ret)
    }

    fn tri(n: usize, m: usize, k: i64, low_triangle: bool) -> std::result::Result<Self, TensorError>
    where
        u8: Cast<T>,
    {
        let shape = vec![n as i64, m as i64];
        let ret = Self::empty(Arc::new(shape))?;
        let (tri_kernel, reg_info) = load_ptx_and_get_data(
            "creation",
            &format!("tri_{}", T::STR),
            ret.device(),
            ret.device_cap(),
            &CREATION,
        )?;
        let cfg = compute_kernel_launch_config(ret.device(), &reg_info, ret.size());
        unsafe { tri_kernel.launch(cfg, (ret.cuda_slice(), n, m, k, low_triangle)) }?;
        Ok(ret)
    }

    fn tril(&self, _: i64) -> std::result::Result<Self, TensorError>
    where
        T: NormalOut<bool, Output = T> + Cast<T>,
        T::Vec: NormalOut<BoolVector, Output = T::Vec>,
    {
        unimplemented!()
    }

    fn triu(&self, _: i64) -> std::result::Result<Self, TensorError>
    where
        T: NormalOut<bool, Output = T> + Cast<T>,
        T::Vec: NormalOut<BoolVector, Output = T::Vec>,
    {
        unimplemented!()
    }

    fn identity(n: usize) -> std::result::Result<Self, TensorError>
    where
        u8: Cast<T>,
    {
        Self::eye(n, n, 0)
    }
}
