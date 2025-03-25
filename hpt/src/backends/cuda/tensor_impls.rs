use std::marker::PhantomData;
use std::sync::Arc;

use crate::backends::common::divmod::FastDivmod;
use crate::backends::cuda::cuda_utils::get_module_name_1;
use crate::ALIGN;
use crate::{tensor_base::_Tensor, Tensor};
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr};
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_allocator::{Backend, Cpu, Cuda};
use hpt_common::error::base::TensorError;
use hpt_common::error::common::CommonError;
use hpt_common::{layout::layout::Layout, shape::shape::Shape, utils::pointer::Pointer};
use hpt_dataloader::data_loader::TensorMeta;
use hpt_dataloader::utils::ToDataLoader;
use hpt_dataloader::{CompressionAlgo, DataLoader, Endian, Save};
use hpt_traits::ops::unary::Contiguous;
use hpt_traits::tensor::{CommonBounds, TensorInfo, TensorLike};
use hpt_types::cuda_types::scalar::Scalar;
use hpt_types::dtype::CudaType;
use hpt_types::into_scalar::Cast;
use hpt_types::type_promote::{Cmp, Eval};

use crate::backends::cuda::utils::unary::unary::uary_fn_with_out_simd;
use hpt_traits::ops::creation::TensorCreator;

use super::cuda_utils::{get_fast_divmod, get_slice_i32};

impl<T, const DEVICE_ID: usize, Al> TensorInfo<T> for _Tensor<T, Cuda, DEVICE_ID, Al>
where
    T: CommonBounds,
    Al: Allocator,
{
    fn ptr(&self) -> Pointer<T> {
        self.data.clone()
    }
    fn size(&self) -> usize {
        self.layout.size() as usize
    }
    fn shape(&self) -> &Shape {
        self.layout.shape()
    }
    fn strides(&self) -> &hpt_common::strides::strides::Strides {
        self.layout.strides()
    }
    fn layout(&self) -> &Layout {
        &self.layout
    }
    fn parent(&self) -> Option<Pointer<T>> {
        self.parent.clone()
    }
    fn ndim(&self) -> usize {
        self.layout.ndim()
    }
    fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }
}

impl<T, const DEVICE_ID: usize, Al> TensorInfo<T> for &_Tensor<T, Cuda, DEVICE_ID, Al>
where
    T: CommonBounds,
    Al: Allocator,
{
    fn ptr(&self) -> Pointer<T> {
        self.data.clone()
    }

    fn size(&self) -> usize {
        self.layout.size() as usize
    }

    fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    fn strides(&self) -> &hpt_common::strides::strides::Strides {
        self.layout.strides()
    }

    fn layout(&self) -> &Layout {
        &self.layout
    }

    fn parent(&self) -> Option<Pointer<T>> {
        self.parent.clone()
    }

    fn ndim(&self) -> usize {
        self.layout.ndim()
    }

    fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }
}

impl<T: CommonBounds + DeviceRepr + CudaType, const DEVICE_ID: usize, Al>
    _Tensor<T, Cuda, DEVICE_ID, Al>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    pub unsafe fn from_raw<S: Into<Shape>>(
        data: CudaSlice<T>,
        shape: S,
    ) -> Result<Self, TensorError> {
        let shape = shape.into();
        let ptr = data.leak();
        let device = cudarc::driver::CudaDevice::new(DEVICE_ID)?;
        let backend = Backend::<Cuda>::new(ptr, device, false).clone();
        Ok(Self {
            #[cfg(feature = "bound_check")]
            data: Pointer::new(ptr as *mut T, shape.size()),
            #[cfg(not(feature = "bound_check"))]
            data: Pointer::new(ptr as *mut T),
            parent: None,
            layout: Layout::from(&shape),
            mem_layout: Arc::new(
                std::alloc::Layout::from_size_align(
                    (shape.size() as usize) * std::mem::size_of::<T>(),
                    ALIGN,
                )
                .unwrap(),
            ),
            backend,
            phantom: PhantomData,
        })
    }

    /// cast the tensor to the new type
    pub fn astype<U>(&self) -> std::result::Result<_Tensor<U, Cuda, DEVICE_ID, Al>, TensorError>
    where
        U: CommonBounds + DeviceRepr + CudaType,
        Scalar<T>: Cast<Scalar<U>>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1(&format!("astype_{}_{}", T::STR, U::STR), self),
            |out, x| out.assign(x.cast()),
            None::<_Tensor<U, Cuda, DEVICE_ID, Al>>,
        )
    }

    pub fn to_cpu<const CPU_DEVICE: usize>(
        &self,
    ) -> std::result::Result<Tensor<T, Cpu, CPU_DEVICE, <Al as Allocator>::CpuAllocator>, TensorError>
    where
        T: DeviceRepr,
        <Al as Allocator>::CpuAllocator: Allocator,
        <<Al as Allocator>::CpuAllocator as Allocator>::Output: AllocatorOutputRetrive,
    {
        let mut data = _Tensor::<T, Cpu, CPU_DEVICE, <Al as Allocator>::CpuAllocator>::empty(
            self.layout.shape().clone(),
        )?;
        let device = self.device();
        if !self.is_contiguous() || self.parent().is_some() {
            let a = self.contiguous()?;
            let ptr = unsafe { device.upgrade_device_ptr(a.data.ptr as u64, a.size()) };
            self.device()
                .dtoh_sync_copy_into(&ptr, data.as_raw_mut())
                .expect("failed to copy data from cuda to cpu");
            ptr.leak();
        } else {
            let ptr = unsafe { device.upgrade_device_ptr(self.data.ptr as u64, self.size()) };
            self.device()
                .dtoh_sync_copy_into(&ptr, data.as_raw_mut())
                .expect("failed to copy data from cuda to cpu");
            ptr.leak();
        }
        Ok(data.into())
    }
    pub(crate) fn device(&self) -> Arc<CudaDevice> {
        self.backend.inner.device.clone()
    }
    pub(crate) fn cuda_slice(&self) -> super::cuda_slice::CudaSlice {
        super::cuda_slice::CudaSlice {
            inner: self.data.ptr as u64,
        }
    }
    pub(crate) fn cuda_shape(
        &self,
    ) -> std::result::Result<cudarc::driver::CudaSlice<i64>, TensorError> {
        let res = self.device().htod_sync_copy(self.shape())?;
        Ok(res)
    }
    pub(crate) fn cuda_strides(
        &self,
    ) -> std::result::Result<cudarc::driver::CudaSlice<i64>, TensorError> {
        let res = self.device().htod_sync_copy(self.strides())?;
        Ok(res)
    }
    pub(crate) fn cuda_divmod(
        &self,
    ) -> std::result::Result<cudarc::driver::CudaSlice<FastDivmod>, TensorError> {
        get_fast_divmod(self.shape(), self.device())
    }
    #[allow(unused)]
    pub(crate) fn cuda_shape_i32(
        &self,
    ) -> std::result::Result<cudarc::driver::CudaSlice<i32>, TensorError> {
        get_slice_i32(self.shape(), self.device())
    }
    pub(crate) fn cuda_strides_i32(
        &self,
    ) -> std::result::Result<cudarc::driver::CudaSlice<i32>, TensorError> {
        get_slice_i32(self.strides(), self.device())
    }
    pub(crate) fn device_cap(&self) -> usize {
        self.backend.inner.cap
    }
}

impl<T: CommonBounds + DeviceRepr + CudaType, const DEVICE_ID: usize, Al>
    Tensor<T, Cuda, DEVICE_ID, Al>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    /// create a new tensor from a raw pointer and a shape
    ///
    /// # Note
    ///
    /// It is safer to call forget to get the pointer back because the forget method will track the reference count
    ///
    /// # Safety
    ///
    /// - The pointer must be valid for the lifetime of the tensor.
    /// - The pointer must be aligned and properly sized.
    /// - The shape must be valid.
    ///
    /// # Note
    ///
    /// It is the user's responsibility to manage the lifetime of the data. Hpt won't drop the data even if the tensor is dropped.
    pub unsafe fn from_raw<S: Into<Shape>>(
        data: CudaSlice<T>,
        shape: S,
    ) -> Result<Self, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID, Al>::from_raw(data, shape)?.into())
    }
    /// copy the data from the cuda tensor to the cpu tensor
    pub fn to_cpu<const CPU_DEVICE: usize>(
        &self,
    ) -> Result<Tensor<T, Cpu, CPU_DEVICE, <Al as Allocator>::CpuAllocator>, TensorError> {
        Ok(self.inner.as_ref().to_cpu()?.into())
    }
    /// get the device of the tensor
    pub fn device(&self) -> Arc<CudaDevice> {
        self.inner.as_ref().device()
    }

    /// cast the tensor to the new type
    pub fn astype<U>(&self) -> Result<Tensor<U, Cuda, DEVICE_ID, Al>, TensorError>
    where
        U: CommonBounds + DeviceRepr + CudaType,
        Scalar<T>: Cast<Scalar<U>>,
    {
        Ok(self.inner.astype()?.into())
    }

    /// check if two tensors are close to each other
    pub fn allclose(&self, other: &Tensor<T, Cuda, DEVICE_ID, Al>, rtol: T, atol: T) -> bool
    where
        T: Eval<Output = bool> + Cmp<Output = bool>,
    {
        let cpu_lhs = self
            .to_cpu::<0>()
            .expect("failed to convert cuda tensor to cpu tensor");
        let cpu_rhs = other
            .to_cpu::<0>()
            .expect("failed to convert cuda tensor to cpu tensor");
        cpu_lhs.allclose(&cpu_rhs, rtol, atol)
    }

    /// Forget the tensor, return the raw pointer of the data
    ///
    /// # Safety
    ///
    /// - The user must ensure the tensor is not used after forgetting
    pub unsafe fn forget(
        self,
    ) -> Result<(cudarc::driver::CudaSlice<u8>, std::alloc::Layout), TensorError> {
        match Arc::try_unwrap(self.inner) {
            Ok(mut inner) => {
                if inner.parent.is_some() {
                    return Err(CommonError::CantForgetTensor {
                        msg: "tensor is a view, cannot forget".to_string(),
                        location: std::panic::Location::caller(),
                    }
                    .into());
                }
                let mut allocator = Al::new();
                use hpt_allocator::Buffer;
                let ret = inner.backend.inner.get_ptr() as *mut u8;
                allocator.forget(ret, DEVICE_ID);
                inner.backend.forget();
                let ret = inner.device().upgrade_device_ptr(ret as u64, inner.size());
                Ok((ret, *inner.mem_layout.as_ref()))
            }
            Err(inner) => {
                let ref_count = Arc::strong_count(&inner);
                Err(CommonError::CantForgetTensor {
                    msg: format!("ref_count: {}", ref_count),
                    location: std::panic::Location::caller(),
                }
                .into())
            }
        }
    }

    /// clone the tensor and return the cloned tensor data
    pub unsafe fn forget_copy(
        &self,
    ) -> Result<(cudarc::driver::CudaSlice<u8>, std::alloc::Layout), TensorError> {
        let to_forget = self.contiguous()?;
        let ptr = to_forget.forget()?;
        Ok(ptr)
    }
}

impl<T, const DEVICE_ID: usize, Al> std::fmt::Display for _Tensor<T, Cuda, DEVICE_ID, Al>
where
    T: CommonBounds + DeviceRepr + Cast<f64> + CudaType,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cpu_data = self
            .to_cpu::<0>()
            .expect("failed to convert cuda tensor to cpu tensor");
        write!(f, "{}", cpu_data)
    }
}

impl<T, const DEVICE_ID: usize, Al> std::fmt::Display for Tensor<T, Cuda, DEVICE_ID, Al>
where
    T: CommonBounds + DeviceRepr + Cast<f64> + CudaType,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner.as_ref())
    }
}

impl<T, const DEVICE_ID: usize, Al> Into<Tensor<T, Cuda, DEVICE_ID, Al>>
    for _Tensor<T, Cuda, DEVICE_ID, Al>
where
    Al: Allocator,
{
    fn into(self) -> Tensor<T, Cuda, DEVICE_ID, Al> {
        Tensor { inner: self.into() }
    }
}

impl<T, const DEVICE_ID: usize, Al> Into<Tensor<T, Cuda, DEVICE_ID, Al>>
    for &Tensor<T, Cuda, DEVICE_ID, Al>
where
    Al: Allocator,
{
    fn into(self) -> Tensor<T, Cuda, DEVICE_ID, Al> {
        Tensor {
            inner: self.inner.clone(),
        }
    }
}
impl<T, const DEVICE_ID: usize, Al> Into<_Tensor<T, Cuda, DEVICE_ID, Al>>
    for &_Tensor<T, Cuda, DEVICE_ID, Al>
where
    Al: Allocator,
{
    fn into(self) -> _Tensor<T, Cuda, DEVICE_ID, Al> {
        _Tensor {
            data: self.data.clone(),
            parent: self.parent.clone(),
            layout: self.layout.clone(),
            mem_layout: self.mem_layout.clone(),
            backend: self.backend.clone(),
            phantom: std::marker::PhantomData,
        }
    }
}
impl<T, const DEVICE: usize, Al> Save for Tensor<T, Cuda, DEVICE, Al>
where
    T: CommonBounds + bytemuck::NoUninit + DeviceRepr + CudaType + bytemuck::Pod,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
    <Al as Allocator>::CpuAllocator: 'static,
{
    type Meta = TensorMeta<T, Self>;
    fn __save(
        data: &Self,
        file: &mut std::fs::File,
        len_so_far: &mut usize,
        global_cnt: &mut usize,
        compression_algo: CompressionAlgo,
        level: u32,
    ) -> std::io::Result<Self::Meta> {
        let cpu_data: Tensor<T, Cpu, 0, <Al as Allocator>::CpuAllocator> = data
            .to_cpu::<0>()
            .expect("failed to convert cuda tensor to cpu tensor");
        let meta = Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::__save(
            &cpu_data,
            file,
            len_so_far,
            global_cnt,
            compression_algo,
            level,
        )?;
        Ok(TensorMeta {
            begin: meta.begin,
            shape: meta.shape,
            strides: meta.strides,
            size: meta.size,
            dtype: meta.dtype,
            compression_algo,
            endian: Endian::Native,
            indices: meta.indices,
            phantom: std::marker::PhantomData,
        })
    }
}

impl<T, const DEVICE_ID: usize, A> ToDataLoader for Tensor<T, Cuda, DEVICE_ID, A>
where
    T: CommonBounds + DeviceRepr + CudaType,
    A: Allocator,
{
    type Output = DataLoader<T, Tensor<T, Cpu, 0, A::CpuAllocator>>;
    fn to_dataloader(self) -> Self::Output {
        let shape = self.inner.layout.shape().clone();
        let strides = self.inner.layout.strides().clone();
        DataLoader::new(
            shape,
            strides,
            self.to_cpu::<0>()
                .expect("failed to convert cuda tensor to cpu tensor"),
        )
    }
}
