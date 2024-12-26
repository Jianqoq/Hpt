use crate::{
    backend::Backend,
    ops::{
        common::creation::geomspace_preprocess_start_step,
        cuda::cuda_utils::{compute_kernel_launch_config, load_ptx_and_get_data},
    },
    tensor_base::_Tensor,
    BoolVector, Cuda, ALIGN,
};
use cudarc::driver::{DeviceRepr, LaunchAsync, LaunchConfig};
use std::{panic::Location, sync::Arc};
use tensor_allocator::CUDA_CACHE;
use tensor_common::{err_handler::ErrHandler, layout::Layout, pointer::Pointer, shape::Shape};
use tensor_cudakernels::CREATION;
use tensor_traits::{CommonBounds, TensorCreator, TensorInfo};
use tensor_types::{
    convertion::{Convertor, FromScalar},
    into_scalar::IntoScalar,
    type_promote::NormalOut,
};

impl<T: CommonBounds + DeviceRepr, const DEVICE_ID: usize> TensorCreator<T>
    for _Tensor<T, Cuda, DEVICE_ID>
{
    type Output = Self;
    fn empty<S: Into<Shape>>(shape: S) -> std::result::Result<Self, ErrHandler> {
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
        .map_err(|e| ErrHandler::StdMemLayoutError(ALIGN, size, Location::caller(), e))?;
        let (ptr, device) = if let Ok(mut cache) = CUDA_CACHE.lock() {
            cache.allocate(layout, DEVICE_ID)?
        } else {
            return Err(ErrHandler::LockFailed("CUDA_CACHE", Location::caller()));
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

    fn zeros<S: Into<Shape>>(shape: S) -> std::result::Result<Self, ErrHandler> {
        let empty = Self::empty(shape)?;
        if let Ok(mut cache) = CUDA_CACHE.lock() {
            cache.memset_zeros(empty.ptr().ptr as *mut u8, &empty.mem_layout, DEVICE_ID)
        } else {
            return Err(ErrHandler::LockFailed("CUDA_CACHE", Location::caller()));
        };
        Ok(empty)
    }

    fn ones<S: Into<Shape>>(shape: S) -> std::result::Result<Self, ErrHandler>
    where
        u8: IntoScalar<T>,
    {
        Self::full(T::ONE, shape)
    }

    fn empty_like(&self) -> std::result::Result<Self, ErrHandler> {
        Self::empty(self.shape())
    }

    fn zeros_like(&self) -> std::result::Result<Self, ErrHandler> {
        Self::zeros(self.shape())
    }

    fn ones_like(&self) -> std::result::Result<Self, ErrHandler>
    where
        u8: IntoScalar<T>,
    {
        Self::ones(self.shape())
    }

    fn full<S: Into<Shape>>(val: T, shape: S) -> std::result::Result<Self, ErrHandler> {
        let ret = Self::empty(shape)?;
        let (fill_kernel, _) = load_ptx_and_get_data(
            "creation",
            &format!("fill_{}", T::ID),
            ret.device(),
            ret.device_cap(),
            &CREATION,
        )?;
        let cfg = LaunchConfig::for_num_elems(ret.size() as u32);
        let mut slice = unsafe {
            ret.device()
                .upgrade_device_ptr::<T>(ret.ptr().ptr as u64, ret.size())
        };
        unsafe { fill_kernel.launch(cfg, (&mut slice, val, ret.size())) }.map_err(|e| {
            ErrHandler::CudaKernelLaunchingError(
                "creation".to_string(),
                format!("fill_{}", T::ID),
                Location::caller(),
                e,
            )
        })?;
        slice.leak();
        Ok(ret)
    }

    fn full_like(&self, val: T) -> std::result::Result<Self, ErrHandler> {
        Self::full(val, self.shape())
    }

    fn arange<U>(start: U, end: U) -> std::result::Result<Self, ErrHandler>
    where
        T: Convertor + FromScalar<U>,
        usize: IntoScalar<T>,
        U: Convertor + IntoScalar<T> + Copy,
    {
        let size = end.to_i64() - start.to_i64();
        if size <= 0 {
            return _Tensor::<T, Cuda, DEVICE_ID>::empty(Arc::new(vec![0]));
        }
        let start: T = start.into_scalar();
        let ret = Self::empty(Arc::new(vec![size]))?;
        let (arange_kernel, _) = load_ptx_and_get_data(
            "creation",
            &format!("arange_{}", T::ID),
            ret.device(),
            ret.device_cap(),
            &CREATION,
        )?;
        let cfg = LaunchConfig::for_num_elems(ret.size() as u32);
        let slice = ret.cuda_slice();
        unsafe { arange_kernel.launch(cfg, (slice, start, T::ONE, ret.size())) }.map_err(|e| {
            ErrHandler::CudaKernelLaunchingError(
                "creation".to_string(),
                format!("arange_{}", T::ID),
                Location::caller(),
                e,
            )
        })?;
        Ok(ret)
    }

    fn arange_step(start: T, end: T, step: T) -> std::result::Result<Self, ErrHandler>
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
            return Self::empty(Arc::new(vec![0]));
        }
        let ret = Self::empty(Arc::new(vec![size as i64]))?;
        let (arange_kernel, reg_info) = load_ptx_and_get_data(
            "creation",
            &format!("arange_{}", T::ID),
            ret.device(),
            ret.device_cap(),
            &CREATION,
        )?;
        let mut slice = unsafe {
            ret.device()
                .upgrade_device_ptr::<T>(ret.ptr().ptr as u64, ret.size())
        };
        let cfg = compute_kernel_launch_config(ret.device(), &reg_info, ret.size());
        unsafe { arange_kernel.launch(cfg, (&mut slice, start, step, ret.size())) }.map_err(
            |e| {
                ErrHandler::CudaKernelLaunchingError(
                    "creation".to_string(),
                    format!("arange_{}", T::ID),
                    Location::caller(),
                    e,
                )
            },
        )?;
        slice.leak();
        Ok(ret)
    }

    fn eye(n: usize, m: usize, k: usize) -> std::result::Result<Self, ErrHandler>
    where
        u8: IntoScalar<T>,
    {
        let shape = vec![n as i64, m as i64];
        let ret = Self::empty(Arc::new(shape))?;
        let (eye_kernel, reg_info) = load_ptx_and_get_data(
            "creation",
            &format!("eye_{}", T::ID),
            ret.device(),
            ret.device_cap(),
            &CREATION,
        )?;
        let cfg = compute_kernel_launch_config(ret.device(), &reg_info, ret.size());
        unsafe { eye_kernel.launch(cfg, (ret.cuda_slice(), n, m, k)) }.map_err(|e| {
            ErrHandler::CudaKernelLaunchingError(
                "creation".to_string(),
                format!("eye_{}", T::ID),
                Location::caller(),
                e,
            )
        })?;
        Ok(ret)
    }

    fn linspace<U>(
        start: U,
        end: U,
        num: usize,
        include_end: bool,
    ) -> std::result::Result<Self, ErrHandler>
    where
        T: Convertor,
        U: Convertor + IntoScalar<T> + Copy,
        usize: IntoScalar<T>,
        f64: IntoScalar<T>,
    {
        let _start = start.to_f64();
        let _end = end.to_f64();
        let n = num as f64;
        let step = if include_end {
            (_end - _start) / (n - 1.0)
        } else {
            (_end - _start) / n
        };
        let step_t: T = step.into_scalar();
        let start_t: T = start.into_scalar();
        let end_t: T = end.into_scalar();
        let ret = _Tensor::<T, Cuda, DEVICE_ID>::empty(Arc::new(vec![num as i64]))?;

        let (linspace_kernel, reg_info) = load_ptx_and_get_data(
            "creation",
            &format!("linspace_{}", T::ID),
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
        }
        .map_err(|e| {
            ErrHandler::CudaKernelLaunchingError(
                "creation".to_string(),
                format!("linspace_{}", T::ID),
                Location::caller(),
                e,
            )
        })?;
        Ok(ret)
    }

    fn logspace(
        start: T,
        end: T,
        num: usize,
        include_end: bool,
        base: T,
    ) -> std::result::Result<Self, ErrHandler>
    where
        T: Convertor + num::Float + FromScalar<usize> + FromScalar<f64>,
    {
        let _start = start.to_f64();
        let _end = end.to_f64();
        let n = num as f64;
        let step = if include_end {
            (_end - _start) / (n - 1.0)
        } else {
            (_end - _start) / n
        };
        let step_t = T::_from(step);
        let ret = _Tensor::<T, Cuda, DEVICE_ID>::empty(Arc::new(vec![num as i64]))?;

        let (logspace_kernel, reg_info) = load_ptx_and_get_data(
            "creation",
            &format!("logspace_{}", T::ID),
            ret.device(),
            ret.device_cap(),
            &CREATION,
        )?;
        let cfg = compute_kernel_launch_config(ret.device(), &reg_info, ret.size());
        unsafe { logspace_kernel.launch(cfg, (ret.cuda_slice(), base, start, step_t, num)) }
            .map_err(|e| {
                ErrHandler::CudaKernelLaunchingError(
                    "creation".to_string(),
                    format!("logspace_{}", T::ID),
                    Location::caller(),
                    e,
                )
            })?;
        Ok(ret)
    }

    fn geomspace(
        start: T,
        end: T,
        n: usize,
        include_end: bool,
    ) -> std::result::Result<Self, ErrHandler>
    where
        f64: IntoScalar<T>,
        usize: IntoScalar<T>,
    {
        let start_f64 = start.to_f64();
        let end_f64 = end.to_f64();
        let both_negative = start_f64 < 0.0 && end_f64 < 0.0;
        let (new_start, step) =
            geomspace_preprocess_start_step(start_f64, end_f64, n, include_end)?;
        let start_t: T = new_start.into_scalar();
        let step_t: T = step.into_scalar();
        let ret = Self::empty(Arc::new(vec![n as i64]))?;
        let (geomspace_kernel, reg_info) = load_ptx_and_get_data(
            "creation",
            &format!("geomspace_{}", T::ID),
            ret.device(),
            ret.device_cap(),
            &CREATION,
        )?;
        let cfg = compute_kernel_launch_config(ret.device(), &reg_info, ret.size());
        unsafe {
            geomspace_kernel.launch(cfg, (ret.cuda_slice(), start_t, step_t, both_negative, n))
        }
        .map_err(|e| {
            ErrHandler::CudaKernelLaunchingError(
                "creation".to_string(),
                format!("geomspace_{}", T::ID),
                Location::caller(),
                e,
            )
        })?;
        Ok(ret)
    }

    fn tri(n: usize, m: usize, k: i64, low_triangle: bool) -> std::result::Result<Self, ErrHandler>
    where
        u8: IntoScalar<T>,
    {
        let shape = vec![n as i64, m as i64];
        let ret = Self::empty(Arc::new(shape))?;
        let (tri_kernel, reg_info) = load_ptx_and_get_data(
            "creation",
            &format!("tri_{}", T::ID),
            ret.device(),
            ret.device_cap(),
            &CREATION,
        )?;
        let cfg = compute_kernel_launch_config(ret.device(), &reg_info, ret.size());
        unsafe { tri_kernel.launch(cfg, (ret.cuda_slice(), n, m, k, low_triangle)) }.map_err(
            |e| {
                ErrHandler::CudaKernelLaunchingError(
                    "creation".to_string(),
                    format!("tri_{}", T::ID),
                    Location::caller(),
                    e,
                )
            },
        )?;
        Ok(ret)
    }

    fn tril(&self, _: i64) -> std::result::Result<Self, ErrHandler>
    where
        T: NormalOut<bool, Output = T> + IntoScalar<T>,
        T::Vec: NormalOut<BoolVector, Output = T::Vec>,
    {
        unimplemented!()
    }

    fn triu(&self, _: i64) -> std::result::Result<Self, ErrHandler>
    where
        T: NormalOut<bool, Output = T> + IntoScalar<T>,
        T::Vec: NormalOut<BoolVector, Output = T::Vec>,
    {
        unimplemented!()
    }

    fn identity(n: usize) -> std::result::Result<Self, ErrHandler>
    where
        u8: IntoScalar<T>,
    {
        Self::eye(n, n, 0)
    }
}
