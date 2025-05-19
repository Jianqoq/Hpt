use std::ops::Deref;
use std::sync::Arc;

use hpt_common::error::base::TensorError;
use hpt_common::{ Pointer, layout::layout::Layout, shape::shape::Shape, strides::strides::Strides };
use hpt_dataloader::FromSafeTensors;
use hpt_traits::tensor::{ CommonBounds, TensorInfo };
use hpt_types::dtype::ToDType;

use crate::onnx::TensorProto;
use crate::utils::onnx::map_dtype::to_dtype;
use crate::utils::{ backend::Backend, device::Device };
use crate::{ ALIGN, DType };

use hpt_iterator::TensorIterator;

#[derive(Clone)]
pub struct _Tensor {
    pub(crate) data: Pointer<u8>,
    pub(crate) layout: Layout,
    pub(crate) dtype: DType,
    pub(crate) device: Device,
    pub(crate) parent: Option<Pointer<u8>>,
    pub(crate) mem_layout: std::alloc::Layout,
    pub(crate) backend: Backend,
}

#[derive(Clone)]
pub struct Tensor {
    pub(crate) inner: Arc<_Tensor>,
}

impl Tensor {
    pub fn dtype(&self) -> DType {
        self.inner.dtype
    }

    pub fn as_slice<T: Sized>(&self) -> &[T] {
        if !self.is_contiguous() {
            panic!("uncontiguous tensor cannot be converted to slice");
        }
        unsafe { std::slice::from_raw_parts(self.inner.data.ptr as *const T, self.size()) }
    }
    pub fn as_slice_mut<T: Sized>(&mut self) -> &mut [T] {
        if !self.is_contiguous() {
            panic!("uncontiguous tensor cannot be converted to slice");
        }
        unsafe {
            std::slice::from_raw_parts_mut(
                self.inner.data.ptr as *mut T,
                (self.inner.mem_layout.size() as usize) / std::mem::size_of::<T>()
            )
        }
    }
    pub unsafe fn from_raw(
        data: *mut u8,
        layout: Layout,
        dtype: DType,
        device: Device,
        take_ownership: bool
    ) -> Result<Self, TensorError> {
        let len = layout.size() as usize;
        let res = match device {
            Device::Cpu => {
                let ptr = Pointer::new(data, (len as i64) * (dtype.sizeof() as i64));
                if (data as usize) % ALIGN != 0 {
                    assert_eq!(take_ownership, false);
                } else {
                    assert_eq!((data as usize) % ALIGN, 0);
                }
                let mem_layout = std::alloc::Layout
                    ::from_size_align(len * dtype.sizeof(), ALIGN)
                    .expect("failed to create memory layout");
                _Tensor {
                    data: ptr,
                    layout,
                    dtype,
                    device,
                    parent: None,
                    mem_layout,
                    backend: Backend::new_cpu(ptr, 0, take_ownership).clone(),
                }
            }
            #[cfg(feature = "cuda")]
            _ => { unimplemented!() }
        };
        Ok(Self {
            inner: Arc::new(res),
        })
    }

    pub fn from_onnx_tensor(
        tensor: &mut TensorProto,
        permute: &Option<Vec<i64>>
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
                        let ptr = empty.inner.data.ptr as *mut u8;
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
                        let ptr = empty.inner.data.ptr as *mut u8;
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
            #[cfg(feature = "bool")]
            DType::Bool => unimplemented!(),
            #[cfg(feature = "i8")]
            DType::I8 => from_raw_data!(),
            #[cfg(feature = "u8")]
            DType::U8 => from_raw_data!(),
            #[cfg(feature = "i16")]
            DType::I16 => from_raw_data!(),
            #[cfg(feature = "u16")]
            DType::U16 => from_raw_data!(),
            #[cfg(feature = "u32")]
            DType::U32 => from_raw_data!(),
            #[cfg(feature = "f16")]
            DType::F16 => from_raw_data!(),
            #[cfg(feature = "bf16")]
            DType::BF16 => from_raw_data!(),
            #[cfg(feature = "u64")]
            DType::U64 => from_raw_data!(),
            #[cfg(feature = "i32")]
            DType::I32 => from_specific_data!(int32_data),
            #[cfg(feature = "i64")]
            DType::I64 => from_specific_data!(int64_data),
            #[cfg(feature = "f32")]
            DType::F32 => from_specific_data!(float_data),
            #[cfg(feature = "f64")]
            DType::F64 => from_specific_data!(double_data),
            _ => unimplemented!("unsupported dtype {:?}", dtype),
        }
    }

    pub fn get<T: ToDType + CommonBounds>(&self, index: &[i64]) -> Result<T, TensorError> {
        assert_eq!(self.dtype(), T::to_dtype());
        let ptr = self.inner.data.cast::<T>();
        let strides = self.inner.layout.strides();
        let mut idx = 0;
        for (i, &val) in index.iter().zip(strides.iter()) {
            idx += val * i;
        }
        Ok(ptr[idx])
    }

    pub fn from_vec<T: ToDType + CommonBounds>(
        vec: Vec<T>,
        shape: &[i64]
    ) -> Result<Self, TensorError> {
        let size = shape.iter().product::<i64>();
        if size != (vec.len() as i64) {
            panic!("vec size mismatch");
        }
        let dtype = T::to_dtype();
        let tensor = Self::empty(shape, dtype, Device::Cpu)?;
        let ptr = tensor.inner.data.cast::<T>();
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
                self.inner.data.cast::<T>()
            }

            fn size(&self) -> usize {
                self.inner.layout.size() as usize
            }

            fn shape(&self) -> &Shape {
                self.inner.layout.shape()
            }

            fn strides(&self) -> &Strides {
                self.inner.layout.strides()
            }

            fn layout(&self) -> &Layout {
                &self.inner.layout
            }

            fn parent<T>(&self) -> Option<Pointer<T>> {
                self.inner.parent.map(|p| p.cast::<T>())
            }

            fn ndim(&self) -> usize {
                self.inner.layout.ndim()
            }

            fn is_contiguous(&self) -> bool {
                self.inner.layout.is_contiguous()
            }
        }
    };
}

impl_tensor_info!(Tensor);
impl_tensor_info!(&Tensor);
impl_tensor_info!(&mut Tensor);

impl FromSafeTensors for Tensor {
    fn from_safe_tensors(data: &safetensors::SafeTensors, tensor_name: &str) -> Self {
        let tensor = data.tensor(tensor_name);
        match tensor {
            Ok(view) => {
                let shape = Shape::from(view.shape());
                let dtype = match view.dtype() {
                    safetensors::Dtype::F32 => DType::F32,
                    _ => todo!(),
                };
                let mut ret = Self::empty(&shape, dtype, Device::Cpu).expect(
                    "failed to create tensor"
                );
                let size = ret.size();
                let slice = ret.as_slice_mut::<u8>();
                let view_slice = unsafe {
                    std::slice::from_raw_parts(
                        view.data().as_ptr() as *const u8,
                        size * (dtype.sizeof() as usize)
                    )
                };
                slice.copy_from_slice(view_slice);
                ret
            }
            Err(e) => {
                panic!("tensor not found: {}", e);
            }
        }
    }
}

impl<T: CommonBounds> TensorIterator<'_, T> for Tensor {}

impl Drop for _Tensor {
    fn drop(&mut self) {
        self.backend.dealloc(self.mem_layout);
    }
}

impl Deref for Tensor {
    type Target = _Tensor;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("data", &self.inner.data)
            .field("layout", &self.inner.layout)
            .field("dtype", &self.inner.dtype)
            .field("parent", &self.inner.parent)
            .field("align", &self.inner.mem_layout.align())
            .field("backend", &self.inner.backend)
            .finish()
    }
}

unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

impl Into<Tensor> for _Tensor {
    fn into(self) -> Tensor {
        Tensor {
            inner: Arc::new(self),
        }
    }
}