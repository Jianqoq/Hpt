use crate::{
    backend::Backend,
    ops::cuda::{
        cuda_utils::{compile_kernel, compute_kernel_launch_config},
        kernel_constants::{
            ARANGE_KERNELS, EYE_KERNELS, FILL_KERNELS, GEOMSPACE_KERNELS, LINSPACE_KERNELS,
            LOGSPACE_KERNELS, TRIU_KERNELS,
        },
    },
    tensor_base::_Tensor,
    BoolVector, Cuda, ALIGN,
};
use anyhow::Result;
use cudarc::driver::{DeviceRepr, LaunchAsync, LaunchConfig};
use std::sync::Arc;
use tensor_allocator::CUDA_CACHE;
use tensor_common::{layout::Layout, pointer::Pointer, shape::Shape};
use tensor_traits::{CommonBounds, TensorCreator, TensorInfo};
use tensor_types::{
    convertion::{Convertor, FromScalar},
    dtype::Dtype,
    into_scalar::IntoScalar,
    type_promote::NormalOut,
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
        Self::zeros(self.shape())
    }

    fn ones_like(&self) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        Self::ones(self.shape())
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
        let slice = ret.cuda_slice();
        unsafe { arange_kernel.launch(cfg, (slice, start, T::ONE, ret.size())) }?;
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
        let map = compile_kernel(
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
        let mut slice = unsafe {
            ret.device()
                .upgrade_device_ptr::<T>(ret.ptr().ptr as u64, ret.size())
        };
        let reg_info = map.get(&func_name).expect("func_name not found");
        let cfg = compute_kernel_launch_config(ret.device(), reg_info, ret.size());
        unsafe { arange_kernel.launch(cfg, (&mut slice, start, step, ret.size())) }?;
        slice.leak();
        Ok(ret)
    }

    fn eye(n: usize, m: usize, k: usize) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        let shape = vec![n as i64, m as i64];
        let ret = Self::empty(Arc::new(shape))?;
        let map = compile_kernel(
            "eye",
            include_str!("../kernels/eye.cu"),
            ret.device(),
            &EYE_KERNELS,
        )?;
        let func_name = format!("eye_{}", T::ID);
        let arange_kernel = ret.device().get_func("eye", &func_name).unwrap();
        let mut slice = unsafe {
            ret.device()
                .upgrade_device_ptr::<T>(ret.ptr().ptr as u64, ret.size())
        };
        let reg_info = map.get(&func_name).expect("func_name not found");
        let cfg = compute_kernel_launch_config(ret.device(), reg_info, ret.size());
        unsafe { arange_kernel.launch(cfg, (&mut slice, n, m, k)) }?;
        slice.leak();
        Ok(ret)
    }

    fn linspace(start: T, end: T, num: usize, include_end: bool) -> Result<Self>
    where
        T: Convertor + num::Float,
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
        let ret = _Tensor::<T, Cuda, DEVICE_ID>::empty(Arc::new(vec![num as i64]))?;

        let map = compile_kernel(
            "linspace",
            include_str!("../kernels/linspace.cu"),
            ret.device(),
            &LINSPACE_KERNELS,
        )?;
        let func_name = if T::ID == Dtype::F16 {
            format!("linspace_{}_vec2", T::ID)
        } else {
            format!("linspace_{}_vec4", T::ID)
        };
        let kernel = ret.device().get_func("linspace", &func_name).unwrap();
        let mut slice = unsafe {
            ret.device()
                .upgrade_device_ptr::<T>(ret.ptr().ptr as u64, ret.size())
        };
        let reg_info = map.get(&func_name).expect("func_name not found");
        let cfg = compute_kernel_launch_config(ret.device(), reg_info, ret.size());
        unsafe {
            kernel.launch(cfg, (&mut slice, start, step_t, num))?;
        }
        slice.leak();
        Ok(ret)
    }

    fn logspace(start: T, end: T, num: usize, include_end: bool, base: T) -> Result<Self>
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

        let map = compile_kernel(
            "logspace",
            include_str!("../kernels/logspace.cu"),
            ret.device(),
            &LOGSPACE_KERNELS,
        )?;

        let func_name = if T::ID == Dtype::F16 {
            format!("logspace_{}_vec2", T::ID)
        } else {
            format!("logspace_{}_vec4", T::ID)
        };
        let kernel = ret.device().get_func("logspace", &func_name).unwrap();
        let mut slice = unsafe {
            ret.device()
                .upgrade_device_ptr::<T>(ret.ptr().ptr as u64, ret.size())
        };
        let reg_info = map.get(&func_name).expect("func_name not found");
        let cfg = compute_kernel_launch_config(ret.device(), reg_info, ret.size());
        unsafe {
            kernel.launch(cfg, (&mut slice, base, start, step_t, num))?;
        }
        slice.leak();
        Ok(ret)
    }

    fn geomspace(start: T, end: T, n: usize, include_end: bool) -> Result<Self>
    where
        f64: IntoScalar<T>,
        usize: IntoScalar<T>,
    {
        let start_f64 = start.to_f64();
        let end_f64 = end.to_f64();
        let both_negative = start_f64 < 0.0 && end_f64 < 0.0;
        let float_n = n.to_f64();
        let step = if include_end {
            if start_f64 >= 0.0 && end_f64 > 0.0 {
                (end_f64.log10() - start_f64.log10()) / (float_n - 1.0)
            } else if start_f64 < 0.0 && end_f64 < 0.0 {
                (end_f64.abs().log10() - start_f64.abs().log10()) / (float_n - 1.0)
            } else {
                return Err(anyhow::Error::msg("start and end must have the same sign"));
            }
        } else if start_f64 >= 0.0 && end_f64 > 0.0 {
            (end_f64.log10() - start_f64.log10()) / float_n
        } else if start_f64 < 0.0 && end_f64 < 0.0 {
            (end_f64.abs().log10() - start_f64.abs().log10()) / float_n
        } else {
            return Err(anyhow::Error::msg("start and end must have the same sign"));
        };
        let ret = Self::empty(Arc::new(vec![n as i64]))?;
        let start = if start_f64 > 0.0 {
            start_f64.log10()
        } else {
            start_f64.abs().log10()
        };
        let start_t: T = start.into_scalar();
        let step_t: T = step.into_scalar();
        let map = compile_kernel(
            "geomspace",
            include_str!("../kernels/geomspace.cu"),
            ret.device(),
            &GEOMSPACE_KERNELS,
        )?;

        let func_name = if T::ID == Dtype::F16 {
            format!("geomspace_{}_vec2", T::ID)
        } else {
            format!("geomspace_{}_vec4", T::ID)
        };
        let kernel = ret.device().get_func("geomspace", &func_name).unwrap();
        let mut slice = unsafe {
            ret.device()
                .upgrade_device_ptr::<T>(ret.ptr().ptr as u64, ret.size())
        };
        let reg_info = map.get(&func_name).expect("func_name not found");
        let cfg = compute_kernel_launch_config(ret.device(), reg_info, ret.size());
        unsafe {
            kernel.launch(cfg, (&mut slice, start_t, step_t, both_negative, n))?;
        }
        slice.leak();
        Ok(ret)
    }

    fn tri(n: usize, m: usize, k: i64, low_triangle: bool) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        let shape = vec![n as i64, m as i64];
        let ret = Self::empty(Arc::new(shape))?;
        let map = compile_kernel(
            "triu",
            include_str!("../kernels/triu.cu"),
            ret.device(),
            &TRIU_KERNELS,
        )?;
        let func_name = format!("triu_{}", T::ID);
        let arange_kernel = ret.device().get_func("triu", &func_name).unwrap();
        let mut slice = unsafe {
            ret.device()
                .upgrade_device_ptr::<T>(ret.ptr().ptr as u64, ret.size())
        };
        let reg_info = map.get(&func_name).expect("func_name not found");
        let cfg = compute_kernel_launch_config(ret.device(), reg_info, ret.size());
        unsafe { arange_kernel.launch(cfg, (&mut slice, n, m, k, low_triangle)) }?;
        slice.leak();
        Ok(ret)
    }

    fn tril(&self, _: i64) -> Result<Self>
    where
        T: NormalOut<bool, Output = T> + IntoScalar<T>,
        T::Vec: NormalOut<BoolVector, Output = T::Vec>,
    {
        unimplemented!()
    }

    fn triu(&self, _: i64) -> Result<Self>
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
