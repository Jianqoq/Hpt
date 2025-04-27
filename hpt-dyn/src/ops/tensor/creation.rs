use std::panic::Location;

use hpt_common::error::base::TensorError;
use hpt_common::error::memory::MemoryError;
use hpt_common::error::shape::ShapeError;
use hpt_common::layout::layout::Layout;
use hpt_common::shape::shape::Shape;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::utils::allocator::Allocator;
use crate::utils::backend::Backend;
use crate::utils::index_cal::{
    dispatch_loop_progress_update, dispatch_map_global_idx, dispatch_map_gp,
};
use crate::{DType, Device, Tensor};
use hpt_types::scalar::*;

impl Tensor {
    #[duplicate::duplicate_item(
        func_name    alloc_method;
        [empty]      [allocate];
        [zeros]      [allocate_zeroed];
    )]
    pub fn func_name(shape: &[i64], dtype: DType, mut device: Device) -> Result<Self, TensorError> {
        let mut allocator = Allocator::new(&device)?;
        let shape: Shape = shape.into();
        let layout = Layout::from(shape);
        let mem_layout = std::alloc::Layout::from_size_align(
            (layout.size() as usize)
                .checked_mul(dtype.sizeof())
                .unwrap_or((isize::MAX as usize) - (64 - 1)), // when overflow happened, we use max memory `from_size_align` accept,
            64,
        );
        match mem_layout {
            Ok(mem_layout) => {
                let prg_update = dispatch_loop_progress_update(&layout, dtype.sizeof());
                let map_global_idx = dispatch_map_global_idx(&layout);
                let map_gp = dispatch_map_gp(&layout);
                let ptr = allocator.alloc_method(mem_layout, &mut device)?;
                let backend = match &device {
                    Device::Cpu => Backend::new_cpu(ptr, 0, true),
                    Device::CudaWithDevice(cuda_device) => {
                        Backend::new_cuda(ptr, cuda_device.clone(), true)
                    }
                    Device::Cuda(_) => unreachable!(),
                };
                Ok(Tensor {
                    data: ptr,
                    layout,
                    dtype,
                    device,
                    parent: None,
                    prg_update,
                    map_global_idx,
                    map_gp,
                    mem_layout,
                    backend,
                })
            }
            Err(e) => Err(TensorError::Memory(MemoryError::AllocationFailed {
                device: device.to_string(),
                id: device.id(),
                size: layout.size() as usize,
                source: Some(Box::new(e)),
                location: std::panic::Location::caller(),
            })),
        }
    }

    pub fn full(
        shape: &[i64],
        val: f64,
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        let empty = Tensor::empty(shape, dtype, device)?;

        let fill_fn = dispatch_fill(dtype, val);

        let fill_fn_ref = fill_fn.as_ref();

        let sizeof = dtype.sizeof() as i64;
        let ptr = empty.data;
        (0..empty.layout.size()).into_par_iter().for_each(|i| {
            fill_fn_ref(ptr.offset_addr(i * sizeof));
        });

        Ok(empty)
    }

    pub fn fill(&mut self, val: f64) {
        let fill_fn = dispatch_fill(self.dtype, val);

        let fill_fn_ref = fill_fn.as_ref();

        let sizeof = self.dtype.sizeof() as i64;
        let ptr = self.data;
        (0..self.layout.size()).into_par_iter().for_each(|i| {
            fill_fn_ref(ptr.offset_addr(i * sizeof));
        });
    }

    pub fn ones(shape: &[i64], dtype: DType, device: Device) -> Result<Self, TensorError> {
        Self::full(shape, 1.0, dtype, device)
    }

    pub fn empty_like(&self) -> Result<Self, TensorError> {
        Self::empty(&self.layout.shape(), self.dtype, self.device.clone())
    }

    pub fn zeros_like(&self) -> Result<Self, TensorError> {
        Self::zeros(&self.layout.shape(), self.dtype, self.device.clone())
    }

    pub fn ones_like(&self) -> Result<Self, TensorError> {
        Self::ones(&self.layout.shape(), self.dtype, self.device.clone())
    }

    pub fn full_like(&self, val: f64) -> Result<Self, TensorError> {
        Self::full(&self.layout.shape(), val, self.dtype, self.device.clone())
    }

    pub fn arange(start: f64, end: f64, dtype: DType, device: Device) -> Result<Self, TensorError> {
        let end_i64: i64 = end as i64;
        let start_i64: i64 = start as i64;
        let size: i64 = end_i64 - start_i64;
        let start: f64 = start;
        if size <= 0 {
            return Self::empty(&[0], dtype, device);
        }
        let ret = Self::empty(&[size], dtype, device)?;

        let arange_fn = dispatch_arange(dtype, start);

        let arange_fn_ref = arange_fn.as_ref();

        let ptr = ret.data;
        let sizeof = dtype.sizeof();
        (0..size as usize).into_par_iter().for_each(|i| {
            arange_fn_ref(ptr.add_addr(i * sizeof), i);
        });

        Ok(ret)
    }

    pub fn arange_step(
        start: f64,
        end: f64,
        step: f64,
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        let size = if step > 0.0 {
            ((end - start) / step).floor() as i64 + 1
        } else {
            ((start - end) / (-step)).floor() as i64 + 1
        };
        if size <= 0 {
            return Self::empty(&[0], dtype, device);
        }
        let ret = Self::empty(&[size], dtype, device)?;

        let arange_fn = dispatch_arange_step(dtype, start, step);

        let arange_fn_ref = arange_fn.as_ref();

        let ptr = ret.data;
        let sizeof = dtype.sizeof();
        (0..size as usize).into_par_iter().for_each(|i| {
            arange_fn_ref(ptr.add_addr(i * sizeof), i);
        });

        Ok(ret)
    }

    pub fn eye(
        n: usize,
        m: usize,
        k: usize,
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        let shape = vec![n as i64, m as i64];
        let res = Self::empty(&shape, dtype, device)?;
        let eye_fn = dispatch_eye(dtype, m, k);

        let eye_fn_ref = eye_fn.as_ref();
        let ptr = res.data;
        let sizeof = dtype.sizeof();
        (0..res.layout.size() as usize)
            .into_par_iter()
            .for_each(|i| {
                eye_fn_ref(ptr.add_addr(i * sizeof), i);
            });
        Ok(res)
    }

    pub fn linspace(
        start: f64,
        end: f64,
        num: usize,
        include_end: bool,
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        let n: f64 = num as f64;
        let step: f64 = if include_end {
            (end - start) / (n - 1.0)
        } else {
            (end - start) / n
        };
        let res = Self::empty(&[n as i64], dtype, device)?;
        let linspace_fn = dispatch_linspace(dtype, start, end, num, step, include_end);
        let linspace_fn_ref = linspace_fn.as_ref();
        let ptr = res.data;
        let sizeof = dtype.sizeof();
        (0..res.layout.size() as usize)
            .into_par_iter()
            .for_each(|i| {
                linspace_fn_ref(ptr.add_addr(i * sizeof), i);
            });
        Ok(res)
    }

    pub fn logspace(
        start: f64,
        end: f64,
        num: usize,
        include_end: bool,
        base: f64,
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        let n: f64 = num as f64;
        let step: f64 = if include_end {
            (end - start) / (n - 1.0)
        } else {
            (end - start) / n
        };
        let res = Self::empty(&[n as i64], dtype, device)?;
        let logspace_fn = dispatch_logspace(dtype, base, start, step);
        let logspace_fn_ref = logspace_fn.as_ref();
        let ptr = res.data;
        let sizeof = dtype.sizeof();
        (0..res.layout.size() as usize)
            .into_par_iter()
            .for_each(|i| {
                logspace_fn_ref(ptr.add_addr(i * sizeof), i);
            });
        Ok(res)
    }

    pub fn geomspace(
        start: f64,
        end: f64,
        n: usize,
        include_end: bool,
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        let both_negative = start < 0.0 && end < 0.0;
        let (new_start, step) = geomspace_preprocess_start_step(start, end, n, include_end)?;
        let res = Self::empty(&[n as i64], dtype, device)?;
        let geomspace_fn = dispatch_geomspace(dtype, new_start, step, both_negative);
        let geomspace_fn_ref = geomspace_fn.as_ref();
        let ptr = res.data;
        let sizeof = dtype.sizeof();
        (0..res.layout.size() as usize)
            .into_par_iter()
            .for_each(|i| {
                geomspace_fn_ref(ptr.add_addr(i * sizeof), i);
            });
        Ok(res)
    }

    pub fn identity(n: usize, dtype: DType, device: Device) -> Result<Self, TensorError> {
        Self::eye(n, n, 0, dtype, device)
    }
}

pub(crate) fn geomspace_preprocess_start_step(
    start: f64,
    end: f64,
    n: usize,
    include_end: bool,
) -> std::result::Result<(f64, f64), TensorError> {
    let float_n = n as f64;
    let step = if include_end {
        if start >= 0.0 && end > 0.0 {
            (end.log10() - start.log10()) / (float_n - 1.0)
        } else if start < 0.0 && end < 0.0 {
            (end.abs().log10() - start.abs().log10()) / (float_n - 1.0)
        } else {
            return Err(ShapeError::GeomSpaceError {
                start,
                end,
                location: Location::caller(),
            }
            .into());
        }
    } else if start >= 0.0 && end > 0.0 {
        (end.log10() - start.log10()) / float_n
    } else if start < 0.0 && end < 0.0 {
        (end.abs().log10() - start.abs().log10()) / float_n
    } else {
        return Err(ShapeError::GeomSpaceError {
            start,
            end,
            location: Location::caller(),
        }
        .into());
    };
    let start = if start > 0.0 {
        start.log10()
    } else {
        start.abs().log10()
    };
    Ok((start, step))
}
