use std::sync::Arc;

use crate::ops::common::divmod::FastDivmod;
use crate::ops::cuda::cuda_utils::get_module_name_1;
use crate::Cpu;
use crate::{tensor_base::_Tensor, Cuda, Tensor};
use cudarc::driver::{CudaDevice, DeviceRepr, LaunchAsync};
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_common::error::base::TensorError;
use hpt_common::{layout::layout::Layout, shape::shape::Shape, utils::pointer::Pointer};
use hpt_dataloader::data_loader::TensorMeta;
use hpt_dataloader::{CompressionAlgo, Endian, Save};
use hpt_traits::TensorCreator;
use hpt_traits::{CommonBounds, TensorAlloc, TensorInfo, TensorLike};
use hpt_types::cuda_types::scalar::Scalar;
use hpt_types::dtype::CudaType;
use hpt_types::into_scalar::Cast;
use num::traits::ToBytes;

use crate::ops::cuda::cuda_utils::compute_kernel_launch_config;
use crate::ops::cuda::utils::unary::unary::uary_fn_with_out_simd;

use super::cuda_utils::load_ptx_and_get_data;

impl<T, const DEVICE_ID: usize, A> TensorLike<T> for _Tensor<T, Cuda, DEVICE_ID, A>
where
    T: CommonBounds + DeviceRepr + CudaType,
    A: Allocator,
    A::Output: AllocatorOutputRetrive,
{
    fn as_raw(&self) -> &[T] {
        unimplemented!()
    }

    fn as_raw_mut(&mut self) -> &mut [T] {
        unimplemented!()
    }

    fn contiguous(&self) -> std::result::Result<Self, TensorError> {
        let res = Self::empty(self.shape().clone())?;
        let (kernel, reg_info) = load_ptx_and_get_data(
            "strided_copy",
            &format!("strided_copy_{}", T::STR),
            res.device(),
            self.device_cap(),
            &hpt_cudakernels::STRIDED_COPY,
        )?;
        let cfg = compute_kernel_launch_config(res.device(), &reg_info, res.size());
        let out_slice = res.cuda_slice();
        let inp_slice = self.cuda_slice();
        let shape = self.cuda_divmod()?;
        let strides = self.cuda_strides()?;
        let ndim = self.ndim();
        let size = self.size();
        unsafe {
            kernel.launch(
                cfg,
                (out_slice, inp_slice, &shape, &strides, ndim as i32, size),
            )
        }?;
        Ok(res)
    }
}

impl<T, const DEVICE_ID: usize, Al> TensorLike<T> for Tensor<T, Cuda, DEVICE_ID, Al>
where
    T: CommonBounds + DeviceRepr + CudaType,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    fn as_raw(&self) -> &[T] {
        unimplemented!()
    }

    fn as_raw_mut(&mut self) -> &mut [T] {
        unimplemented!()
    }

    fn contiguous(&self) -> std::result::Result<Self, TensorError> {
        Ok(self.inner.as_ref().contiguous()?.into())
    }
}

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

impl<T: CommonBounds + DeviceRepr + CudaType, const DEVICE_ID: usize, Al> TensorAlloc
    for _Tensor<T, Cuda, DEVICE_ID, Al>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Meta = T;
    fn _empty<S: Into<Shape>>(shape: S) -> std::result::Result<Self, TensorError>
    where
        Self: Sized,
    {
        Self::empty(shape)
    }
}

impl<T: CommonBounds + DeviceRepr + CudaType, const DEVICE_ID: usize, Al>
    _Tensor<T, Cuda, DEVICE_ID, Al>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    /// cast the tensor to the new type
    pub fn astype<U>(&self) -> std::result::Result<_Tensor<U, Cuda, DEVICE_ID, Al>, TensorError>
    where
        U: CommonBounds + DeviceRepr + CudaType,
        Scalar<T>: Cast<Scalar<U>>,
    {
        uary_fn_with_out_simd(
            self,
            &get_module_name_1("astype", self),
            |out, x| out.assign(x.cast()),
            None::<_Tensor<U, Cuda, DEVICE_ID, Al>>,
        )
    }

    // /// check if two tensors are close to each other
    // pub fn allclose<U: CommonBounds>(&self, other: &_Tensor<U, Cuda>) -> bool
    // where
    //     T: Convertor,
    //     U: Convertor,
    // {
    //     if self.shape() != other.shape() {
    //         return false;
    //     }
    //     let folder = self.par_iter().zip(other.par_iter()).fold(
    //         || true,
    //         |acc, (a, b)| {
    //             let a_val: f64 = a.to_f64();
    //             let b_val: f64 = b.to_f64();
    //             let abs_diff: f64 = (a_val - b_val).abs();
    //             let torlerance: f64 = 1.0e-8 + 1.0e-5 * b_val.abs();
    //             acc && abs_diff <= torlerance
    //         },
    //     );
    //     folder.reduce(|| true, |a, b| a && b)
    // }

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
        self._backend._backend.device.clone()
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
        let mut res = vec![];
        for i in self.shape().iter() {
            let fast_divmod = FastDivmod::new(*i as i32);
            res.push(fast_divmod);
        }
        let res = self.device().htod_sync_copy(&res)?;
        Ok(res)
    }
    #[allow(unused)]
    pub(crate) fn cuda_shape_i32(
        &self,
    ) -> std::result::Result<cudarc::driver::CudaSlice<i32>, TensorError> {
        let strides = self.shape().iter().map(|x| *x as i32).collect::<Vec<_>>();
        let res = self.device().htod_sync_copy(&strides)?;
        Ok(res)
    }
    pub(crate) fn cuda_strides_i32(
        &self,
    ) -> std::result::Result<cudarc::driver::CudaSlice<i32>, TensorError> {
        let strides = self.strides().iter().map(|x| *x as i32).collect::<Vec<_>>();
        let res = self.device().htod_sync_copy(&strides)?;
        Ok(res)
    }
    pub(crate) fn device_cap(&self) -> usize {
        self._backend._backend.cap
    }
}

impl<T: CommonBounds + DeviceRepr + CudaType, const DEVICE_ID: usize, Al>
    Tensor<T, Cuda, DEVICE_ID, Al>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
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

    // /// copy the data from the other tensor to this tensor
    // pub fn assign(&mut self, other: &Tensor<T, Cuda>) {
    //     let mut mut_self = self.inner.as_ref().clone();
    //     mut_self.assign(&other.inner.as_ref());
    // }

    /// cast the tensor to the new type
    pub fn astype<U>(&self) -> Result<Tensor<U, Cuda, DEVICE_ID, Al>, TensorError>
    where
        U: CommonBounds + DeviceRepr + CudaType,
        Scalar<T>: Cast<Scalar<U>>,
    {
        Ok(self.inner.astype()?.into())
    }

    // /// try to cast the tensor to the new type, if the type is the same, return the tensor itself, otherwise return the new tensor
    // pub fn try_astype<U>(&self) -> anyhow::Result<Tensor<U, Cuda>>
    // where
    //     U: CommonBounds,
    //     T: Cast<U>,
    // {
    //     Ok(self.inner.try_astype()?.into())
    // }

    // /// bitcast the tensor to the new type, the user must ensure the size of the new type is the same as the old type
    // pub fn static_cast<Dst>(&self) -> anyhow::Result<Tensor<Dst, Cuda>>
    // where
    //     Dst: CommonBounds,
    // {
    //     Ok(self.inner.static_cast()?.into())
    // }

    // /// check if two tensors are close to each other
    // pub fn allclose<U: CommonBounds>(&self, other: &Tensor<U, Cuda>) -> bool
    // where
    //     T: Convertor,
    //     U: Convertor,
    // {
    //     self.inner.allclose(&other.inner)
    // }
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
impl<
        const N: usize,
        T: CommonBounds + ToBytes<Bytes = [u8; N]> + DeviceRepr + CudaType,
        const DEVICE: usize,
        Al,
    > Save for Tensor<T, Cuda, DEVICE, Al>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Meta = TensorMeta<T, Self>;
    fn __save(
        data: &Self,
        file: &mut std::fs::File,
        len_so_far: &mut usize,
        global_cnt: &mut usize,
        compression_algo: CompressionAlgo,
        endian: Endian,
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
            endian,
            level,
        )?;
        Ok(TensorMeta {
            begin: meta.begin,
            shape: meta.shape,
            strides: meta.strides,
            size: meta.size,
            dtype: meta.dtype,
            compression_algo,
            endian,
            indices: meta.indices,
            phantom: std::marker::PhantomData,
        })
    }
}
