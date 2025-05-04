use hpt_common::error::base::TensorError;
use hpt_common::{Pointer, layout::layout::Layout, shape::shape::Shape, strides::strides::Strides};
use hpt_traits::tensor::{CommonBounds, TensorInfo};
use hpt_types::dtype::ToDType;
use std::sync::Arc;

use crate::onnx::TensorProto;
use crate::utils::index_cal::{
    dispatch_loop_progress_update, dispatch_map_global_idx, dispatch_map_gp,
};
use crate::utils::onnx::map_dtype::to_dtype;
use crate::utils::{backend::Backend, device::Device};
use crate::{ALIGN, DType};

use hpt_iterator::TensorIterator;

#[derive(Clone)]
pub struct Tensor {
    pub(crate) data: Pointer<u8>,
    pub(crate) layout: Layout,
    pub(crate) dtype: DType,
    pub(crate) device: Device,
    pub(crate) parent: Option<Pointer<u8>>,
    /// update loop progress
    ///
    /// return # of element need to jump
    pub(crate) prg_update: Arc<dyn Fn(&mut [i64]) -> i64 + Send + Sync>,
    /// map global index to physical index
    ///
    /// return physical index
    pub(crate) map_global_idx: Arc<dyn Fn(i64) -> i64 + Send + Sync>,
    /// map global index to physical index and provide loop progress
    ///
    /// return (physical index, loop progress)
    pub(crate) map_gp: Arc<dyn Fn(i64) -> (i64, Vec<i64>) + Send + Sync>,
    pub(crate) mem_layout: std::alloc::Layout,
    pub(crate) backend: Backend,
}

impl Tensor {
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn as_slice<T: Sized>(&self) -> &[T] {
        if !self.is_contiguous() {
            panic!("uncontiguous tensor cannot be converted to slice");
        }
        unsafe {
            std::slice::from_raw_parts(
                self.data.ptr as *const T,
                (self.mem_layout.size() as usize) / std::mem::size_of::<T>(),
            )
        }
    }
    pub fn as_slice_mut<T: Sized>(&mut self) -> &mut [T] {
        if !self.is_contiguous() {
            panic!("uncontiguous tensor cannot be converted to slice");
        }
        unsafe {
            std::slice::from_raw_parts_mut(
                self.data.ptr as *mut T,
                (self.mem_layout.size() as usize) / std::mem::size_of::<T>(),
            )
        }
    }
    pub unsafe fn from_raw(
        data: *mut u8,
        layout: Layout,
        dtype: DType,
        device: Device,
        take_ownership: bool,
    ) -> Result<Self, TensorError> {
        let len = layout.size() as usize;
        match device {
            Device::Cpu => {
                let ptr = Pointer::new(data, len as i64 * dtype.sizeof() as i64);
                if (data as usize) % ALIGN != 0 {
                    assert_eq!(take_ownership, false);
                } else {
                    assert_eq!((data as usize) % ALIGN, 0);
                }
                let prg_update = dispatch_loop_progress_update(&layout);
                let map_global_idx = dispatch_map_global_idx(&layout);
                let map_gp = dispatch_map_gp(&layout);
                let mem_layout = std::alloc::Layout::from_size_align(len * dtype.sizeof(), ALIGN)
                    .expect("failed to create memory layout");
                Ok(Self {
                    data: ptr,
                    layout,
                    dtype,
                    device,
                    parent: None,
                    prg_update,
                    map_global_idx,
                    map_gp,
                    mem_layout,
                    backend: Backend::new_cpu(ptr, 0, take_ownership).clone(),
                })
            }
            #[cfg(feature = "cuda")]
            _ => {
                unimplemented!()
            }
        }
    }

    pub fn from_onnx_tensor(
        tensor: &mut TensorProto,
        permute: &Option<Vec<i64>>,
    ) -> Result<Self, TensorError> {
        let shape = Shape::from(&tensor.dims);
        let layout = Layout::from(shape);
        let dtype = to_dtype(tensor.data_type());
        let device = Device::Cpu;

        macro_rules! from_raw_data {
            () => {
                if let Some(raw_data) = tensor.raw_data.take() {
                    let raw_ptr = raw_data.as_ptr() as *mut u8;
                    if raw_data.len() == 0 {
                        unimplemented!()
                    }
                    assert_eq!(raw_data.len(), layout.size() as usize * dtype.sizeof());
                    if let Some(permute) = permute {
                        unsafe {
                            Tensor::from_raw(raw_ptr, layout, dtype, device, false)?
                                .permute(&permute)?
                                .contiguous()
                        }
                    } else {
                        let empty = Self::empty(layout.shape(), dtype, device)?;
                        let ptr = empty.data.ptr as *mut u8;
                        unsafe {
                            std::ptr::copy(raw_ptr, ptr, raw_data.len());
                        }
                        Ok(empty)
                    }
                } else {
                    unimplemented!()
                }
            };
        }
        macro_rules! from_specific_data {
            ($specific_data:ident) => {
                if !tensor.$specific_data.is_empty() {
                    let raw_ptr = tensor.$specific_data.as_ptr() as *mut u8;
                    if tensor.$specific_data.len() == 0 {
                        unimplemented!()
                    }
                    assert_eq!(tensor.$specific_data.len(), layout.size() as usize);
                    if let Some(permute) = permute {
                        unsafe {
                            Tensor::from_raw(raw_ptr, layout, dtype, device, false)?
                                .permute(&permute)?
                                .contiguous()
                        }
                    } else {
                        let empty = Self::empty(layout.shape(), dtype, device)?;
                        let ptr = empty.data.ptr as *mut u8;
                        unsafe {
                            std::ptr::copy(raw_ptr, ptr, tensor.$specific_data.len());
                        }
                        Ok(empty)
                    }
                } else {
                    from_raw_data!()
                }
            };
        }
        match dtype {
            DType::Bool => unimplemented!(),
            DType::I8
            | DType::U8
            | DType::I16
            | DType::U16
            | DType::U32
            | DType::F16
            | DType::BF16 => from_raw_data!(),
            DType::I32 => from_specific_data!(int32_data),
            DType::I64 => from_specific_data!(int64_data),
            DType::F32 => from_specific_data!(float_data),
        }
    }

    pub fn get<T: ToDType + CommonBounds>(&self, index: &[i64]) -> Result<T, TensorError> {
        assert_eq!(self.dtype(), T::to_dtype());
        let ptr = self.data.cast::<T>();
        let strides = self.layout.strides();
        let mut idx = 0;
        for (i, &val) in index.iter().zip(strides.iter()) {
            idx += val * i;
        }
        Ok(ptr[idx])
    }

    pub fn from_vec<T: ToDType + CommonBounds>(
        vec: Vec<T>,
        shape: &[i64],
    ) -> Result<Self, TensorError> {
        let size = shape.iter().product::<i64>();
        if size != vec.len() as i64 {
            panic!("vec size mismatch");
        }
        let dtype = T::to_dtype();
        let tensor = Self::empty(shape, dtype, Device::Cpu)?;
        let ptr = tensor.data.cast::<T>();
        unsafe {
            std::ptr::copy(vec.as_ptr(), ptr.ptr, vec.len());
        }
        Ok(tensor)
    }
}

macro_rules! impl_tensor_info {
    ($t:ty) => {
        impl TensorInfo for $t {
            fn ptr<T>(&self) -> Pointer<T> {
                self.data.cast::<T>()
            }

            fn size(&self) -> usize {
                self.layout.size() as usize
            }

            fn shape(&self) -> &Shape {
                self.layout.shape()
            }

            fn strides(&self) -> &Strides {
                self.layout.strides()
            }

            fn layout(&self) -> &Layout {
                &self.layout
            }

            fn parent<T>(&self) -> Option<Pointer<T>> {
                self.parent.map(|p| p.cast::<T>())
            }

            fn ndim(&self) -> usize {
                self.layout.ndim()
            }

            fn is_contiguous(&self) -> bool {
                self.layout.is_contiguous()
            }
        }
    };
}

impl_tensor_info!(Tensor);
impl_tensor_info!(&Tensor);
impl_tensor_info!(&mut Tensor);

impl<T: CommonBounds> TensorIterator<'_, T> for Tensor {}

impl Drop for Tensor {
    fn drop(&mut self) {
        self.backend.dealloc(self.mem_layout);
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("data", &self.data)
            .field("layout", &self.layout)
            .field("dtype", &self.dtype)
            .field("parent", &self.parent)
            .field("align", &self.mem_layout.align())
            .field("backend", &self.backend)
            .finish()
    }
}

unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}
