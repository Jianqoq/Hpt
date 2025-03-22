use std::marker::PhantomData;
use std::sync::Arc;
use std::{fmt::Display, sync::atomic::Ordering};

use crate::tensor::DiffTensor;
use crate::ALIGN;
use crate::{tensor_base::_Tensor, Tensor, DISPLAY_LR_ELEMENTS, DISPLAY_PRECISION};
#[cfg(feature = "cuda")]
use cudarc::driver::DeviceRepr;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
#[cfg(feature = "cuda")]
use hpt_allocator::Cuda;
use hpt_allocator::{Backend, Cpu};
use hpt_common::error::base::TensorError;
use hpt_common::{layout::layout::Layout, shape::shape::Shape, utils::pointer::Pointer};
use hpt_dataloader::data_loader::TensorMeta;
use hpt_dataloader::utils::ToDataLoader;
use hpt_dataloader::{CompressionAlgo, DataLoader, Endian, FromSafeTensors, Meta, Save};
use hpt_display::display;
use hpt_iterator::iterator_traits::ParStridedIteratorZip;
use hpt_iterator::TensorIterator;
use hpt_traits::ops::creation::TensorCreator;
use hpt_traits::tensor::TensorInfo;
use hpt_traits::tensor::{CommonBounds, TensorLike};
#[cfg(feature = "cuda")]
use hpt_types::dtype::CudaType;
use hpt_types::into_scalar::Cast;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

impl<T, const DEVICE: usize, A> TensorLike<T> for _Tensor<T, Cpu, DEVICE, A>
where
    T: CommonBounds,
    A: Allocator,
{
    fn as_raw(&self) -> &[T] {
        let ptr = self.data.ptr;
        let size;
        if !self.is_contiguous() {
            size = self.layout.real_size();
        } else {
            size = self.size();
        }
        let slice = unsafe { std::slice::from_raw_parts(ptr, size) };
        slice
    }

    fn as_raw_mut(&mut self) -> &mut [T] {
        let ptr = self.data.ptr;
        let size;
        if !self.is_contiguous() {
            size = self.layout.real_size();
        } else {
            size = self.size();
        }
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, size) };
        slice
    }
}

macro_rules! impl_tensor_info {
    ($tensor:ty) => {
        impl<T, const DEVICE: usize, A> TensorInfo<T> for $tensor
        where
            A: Allocator,
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
    };
}

impl_tensor_info!(_Tensor<T, Cpu, DEVICE, A>);
impl_tensor_info!(&_Tensor<T, Cpu, DEVICE, A>);
impl_tensor_info!(&mut _Tensor<T, Cpu, DEVICE, A>);

impl<'a, T: CommonBounds, const DEVICE: usize, A> TensorIterator<'a, T>
    for _Tensor<T, Cpu, DEVICE, A>
where
    A: Allocator + 'a,
    A::Output: AllocatorOutputRetrive,
{
}

impl<T: CommonBounds, const DEVICE: usize, A> _Tensor<T, Cpu, DEVICE, A>
where
    A: Allocator,
    A::Output: AllocatorOutputRetrive,
{
    /// create a new tensor from a raw pointer and a shape
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
    pub unsafe fn from_raw<S: Into<Shape>>(data: *mut T, shape: S) -> Result<Self, TensorError> {
        let shape = shape.into();
        assert_ne!(data, std::ptr::null_mut(), "data is null");
        assert_eq!(
            data as usize % ALIGN,
            0,
            "data is not aligned, it must be aligned to {}",
            ALIGN
        );
        Ok(Self {
            #[cfg(feature = "bound_check")]
            data: Pointer::new(data, shape.size()),
            #[cfg(not(feature = "bound_check"))]
            data: Pointer::new(data),
            parent: None,
            layout: Layout::from(&shape),
            mem_layout: Arc::new(
                std::alloc::Layout::from_size_align(
                    (shape.size() as usize) * std::mem::size_of::<T>(),
                    ALIGN,
                )
                .unwrap(),
            ),
            backend: Backend::<Cpu>::new(data as u64, DEVICE, false),
            phantom: PhantomData,
        })
    }

    /// cast the tensor to the new type
    pub(crate) fn astype<U>(&self) -> std::result::Result<_Tensor<U, Cpu, DEVICE, A>, TensorError>
    where
        U: CommonBounds,
        T: Cast<U>,
    {
        // Create an empty tensor of the new type with the same shape.
        let mut ret = _Tensor::<U, Cpu, DEVICE, A>::empty(self.layout.shape().clone())?;

        // Parallel iteration to convert and copy each element to the new tensor.
        ret.as_raw_mut()
            .par_iter_mut()
            .zip(self.as_raw().par_iter())
            .for_each(|(a, &b)| {
                *a = b.cast();
            });
        Ok(ret)
    }

    /// try to cast the tensor to the new type, if the type is the same, return the tensor itself, otherwise return the new tensor
    pub(crate) fn try_astype<U>(&self) -> Result<_Tensor<U, Cpu, DEVICE, A>, TensorError>
    where
        U: CommonBounds,
        T: Cast<U>,
    {
        if U::STR == T::STR {
            Ok(self.static_cast()?)
        } else {
            Ok(self.astype::<U>()?)
        }
    }

    /// bitcast the tensor to the new type, the user must ensure the size of the new type is the same as the old type
    pub(crate) fn static_cast<Dst>(
        &self,
    ) -> std::result::Result<_Tensor<Dst, Cpu, DEVICE, A>, TensorError>
    where
        Dst: CommonBounds,
    {
        if T::STR == Dst::STR {
            match self.parent.clone() {
                Some(parent) => {
                    #[cfg(feature = "bound_check")]
                    let new_parent = Pointer::new(parent.ptr as *mut Dst, parent.len);
                    #[cfg(not(feature = "bound_check"))]
                    let new_parent = Pointer::new(parent.ptr as *mut Dst);
                    Ok(_Tensor {
                        #[cfg(feature = "bound_check")]
                        data: Pointer::new(self.data.ptr as *mut Dst, self.ptr().len),
                        #[cfg(not(feature = "bound_check"))]
                        data: Pointer::new(self.data.ptr as *mut Dst),
                        parent: Some(new_parent),
                        mem_layout: self.mem_layout.clone(),
                        layout: self.layout.clone(),
                        backend: self.backend.clone(),
                        phantom: PhantomData,
                    })
                }
                None => Ok(_Tensor {
                    #[cfg(feature = "bound_check")]
                    data: Pointer::new(self.data.ptr as *mut Dst, self.ptr().len),
                    #[cfg(not(feature = "bound_check"))]
                    data: Pointer::new(self.data.ptr as *mut Dst),
                    parent: None,
                    mem_layout: self.mem_layout.clone(),
                    layout: self.layout.clone(),
                    backend: self.backend.clone(),
                    phantom: PhantomData,
                }),
            }
        } else {
            panic!("Cannot cast tensor to different type")
        }
    }

    /// check if two tensors are close to each other
    pub fn allclose<U: CommonBounds>(
        &self,
        other: &_Tensor<U, Cpu, DEVICE, A>,
        rtol: f64,
        atol: f64,
    ) -> bool
    where
        T: Cast<f64>,
        U: Cast<f64>,
    {
        if self.shape() != other.shape() {
            return false;
        }
        let folder = self.par_iter().zip(other.par_iter()).fold(
            || true,
            |acc, (a, b)| {
                let a_val: f64 = a.cast();
                let b_val: f64 = b.cast();
                if a_val.is_nan() && b_val.is_nan() {
                    return acc;
                }
                if a_val.is_infinite() && b_val.is_infinite() {
                    return acc && a_val.is_sign_positive() == b_val.is_sign_positive();
                }
                let tolerance = atol + rtol * b_val.abs();
                let abs_diff = (a_val - b_val).abs();
                acc && abs_diff <= tolerance
            },
        );
        folder.reduce(|| true, |a, b| a && b)
    }
}

impl<T: CommonBounds, const DEVICE: usize, A> Tensor<T, Cpu, DEVICE, A>
where
    A: Allocator,
    A::Output: AllocatorOutputRetrive,
{
    /// create a new tensor from a raw pointer and a shape
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
    pub unsafe fn from_raw<S: Into<Shape>>(data: *mut T, shape: S) -> Result<Self, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE, A>::from_raw(data, shape)?.into())
    }
    /// cast the tensor to the new type
    pub fn astype<U>(&self) -> Result<Tensor<U, Cpu, DEVICE, A>, TensorError>
    where
        U: CommonBounds,
        T: Cast<U>,
    {
        Ok(self.inner.astype()?.into())
    }

    /// try to cast the tensor to the new type, if the type is the same, return the tensor itself, otherwise return the new tensor
    pub fn try_astype<U>(&self) -> Result<Tensor<U, Cpu, DEVICE, A>, TensorError>
    where
        U: CommonBounds,
        T: Cast<U>,
    {
        Ok(self.inner.try_astype()?.into())
    }

    /// bitcast the tensor to the new type, the user must ensure the size of the new type is the same as the old type
    pub fn static_cast<Dst>(&self) -> Result<Tensor<Dst, Cpu, DEVICE, A>, TensorError>
    where
        Dst: CommonBounds,
    {
        Ok(self.inner.static_cast()?.into())
    }

    /// check if two tensors are close to each other
    pub fn allclose<U: CommonBounds>(
        &self,
        other: &Tensor<U, Cpu, DEVICE, A>,
        rtol: f64,
        atol: f64,
    ) -> bool
    where
        T: Cast<f64>,
        U: Cast<f64>,
    {
        self.inner.allclose(&other.inner, rtol, atol)
    }

    /// convert the tensor from cpu to the cuda tensor
    #[cfg(feature = "cuda")]
    pub fn to_cuda<const CUDA_DEVICE: usize>(
        &self,
    ) -> Result<Tensor<T, Cuda, CUDA_DEVICE, <A as Allocator>::CudaAllocator>, TensorError>
    where
        T: DeviceRepr + CudaType,
    {
        let data =
            _Tensor::<T, Cuda, CUDA_DEVICE, <A as Allocator>::CudaAllocator>::empty(self.shape())
                .unwrap();
        let device = data.device();
        let mut ptr = unsafe { device.upgrade_device_ptr(data.ptr().ptr as u64, data.size()) };
        data.device()
            .htod_sync_copy_into(self.as_raw(), &mut ptr)
            .unwrap();
        ptr.leak();
        Ok(data.into())
    }
}

impl<T, const DEVICE: usize, A> Save for Tensor<T, Cpu, DEVICE, A>
where
    T: CommonBounds + bytemuck::NoUninit + bytemuck::Pod,
    A: Allocator + 'static,
    Tensor<T, Cpu, DEVICE, A>:
        hpt_traits::ops::creation::TensorCreator<Output = Tensor<T, Cpu, DEVICE, A>>,
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
        let data_loader: DataLoader<T, Self> = data.clone().to_dataloader();
        let meta = Meta {
            name: "".to_string(),
            compression_algo,
            endian: Endian::Native,
            data_saver: Box::new(data_loader),
            compression_level: level,
        };
        let info = crate::save_load::save(file, meta, len_so_far, *global_cnt)?;
        *global_cnt += 1;
        Ok(TensorMeta {
            begin: info.0,
            shape: info.2,
            strides: info.3,
            size: info.4,
            dtype: info.5,
            compression_algo: info.6,
            endian: info.7,
            indices: info.8,
            phantom: PhantomData,
        })
    }
}

impl<T, const DEVICE: usize> Display for _Tensor<T, Cpu, DEVICE>
where
    T: CommonBounds + Cast<f64>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let precision = DISPLAY_PRECISION.load(Ordering::Relaxed);
        let lr_element_size = DISPLAY_LR_ELEMENTS.load(Ordering::Relaxed);
        display(self, f, lr_element_size, precision, false)
    }
}

impl<T, const DEVICE: usize> std::fmt::Debug for _Tensor<T, Cpu, DEVICE>
where
    T: CommonBounds + Cast<f64>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let precision = DISPLAY_PRECISION.load(Ordering::Relaxed);
        let lr_element_size = DISPLAY_LR_ELEMENTS.load(Ordering::Relaxed);
        display(self, f, lr_element_size, precision, false)
    }
}

impl<T, const DEVICE: usize, A> Into<Tensor<T, Cpu, DEVICE, A>> for _Tensor<T, Cpu, DEVICE, A>
where
    A: Allocator,
{
    fn into(self) -> Tensor<T, Cpu, DEVICE, A> {
        Tensor { inner: self.into() }
    }
}

impl<T, const DEVICE: usize, A> Into<_Tensor<T, Cpu, DEVICE, A>> for &_Tensor<T, Cpu, DEVICE, A>
where
    T: CommonBounds,
    A: Allocator,
{
    fn into(self) -> _Tensor<T, Cpu, DEVICE, A> {
        _Tensor {
            data: self.data.clone(),
            parent: self.parent.clone(),
            layout: self.layout.clone(),
            mem_layout: self.mem_layout.clone(),
            backend: self.backend.clone(),
            phantom: PhantomData,
        }
    }
}

impl<T, const DEVICE: usize, A> Into<Tensor<T, Cpu, DEVICE, A>> for &Tensor<T, Cpu, DEVICE, A>
where
    T: CommonBounds,
    A: Allocator,
{
    fn into(self) -> Tensor<T, Cpu, DEVICE, A> {
        Tensor {
            inner: self.inner.clone(),
        }
    }
}

impl<'a, T: CommonBounds, const DEVICE: usize, A> Into<_Tensor<T, Cpu, DEVICE, A>> for &'a [T]
where
    A: Allocator,
    A::Output: AllocatorOutputRetrive,
{
    fn into(self) -> _Tensor<T, Cpu, DEVICE, A> {
        let mut ret = _Tensor::<T, Cpu, DEVICE, A>::empty(vec![self.len() as i64]).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(self.as_ptr(), ret.as_raw_mut().as_mut_ptr(), self.len());
        }
        ret
    }
}

impl<T: CommonBounds, const DEVICE: usize> FromSafeTensors for Tensor<T, Cpu, DEVICE> {
    fn from_safe_tensors(data: &safetensors::SafeTensors, tensor_name: &str) -> Self {
        let tensor = data.tensor(tensor_name);
        match tensor {
            Ok(view) => {
                let shape = Shape::from(view.shape());
                let mut ret = Self::empty(shape).expect("failed to create tensor");
                let size = ret.size();
                let slice = ret.as_raw_mut();
                let view_slice =
                    unsafe { std::slice::from_raw_parts(view.data().as_ptr() as *const T, size) };
                slice.copy_from_slice(view_slice);
                ret
            }
            Err(e) => {
                panic!("tensor not found: {}", e);
            }
        }
    }
}

impl<T, const DEVICE: usize, Al> Display for Tensor<T, Cpu, DEVICE, Al>
where
    T: CommonBounds + Cast<f64>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let precision = DISPLAY_PRECISION.load(Ordering::Relaxed);
        let lr_element_size = DISPLAY_LR_ELEMENTS.load(Ordering::Relaxed);
        display(self, f, lr_element_size, precision, false)
    }
}

impl<T: Clone, const DEVICE: usize> DiffTensor<T, Cpu, DEVICE> {
    /// Backward the gradient of the tensor
    pub fn backward(&mut self, grad: Tensor<T, Cpu, DEVICE>) -> Result<(), TensorError> {
        if let Ok(true) = self.backward.borrow_mut()(grad.clone()) {
            self.grad.borrow_mut().replace(grad);
        }
        Ok(())
    }

    /// Get the gradient of the tensor
    pub fn grad(&self) -> Option<Tensor<T, Cpu, DEVICE>> {
        self.grad.borrow().as_ref().cloned()
    }
}

#[cfg(feature = "cuda")]
impl<T: CommonBounds, const CPU_DEVICE: usize, const CUDA_DEVICE: usize, Al>
    Into<Tensor<T, Cuda, CUDA_DEVICE, <Al as Allocator>::CudaAllocator>>
    for Tensor<T, Cpu, CPU_DEVICE, Al>
where
    T: DeviceRepr + CudaType,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    fn into(self) -> Tensor<T, Cuda, CUDA_DEVICE, <Al as Allocator>::CudaAllocator> {
        self.to_cuda::<CUDA_DEVICE>()
            .expect("failed to convert cpu tensor to cuda tensor")
    }
}

impl<T, const DEVICE_ID: usize, A> ToDataLoader for Tensor<T, Cpu, DEVICE_ID, A>
where
    T: CommonBounds,
    A: Allocator,
{
    type Output = DataLoader<T, Tensor<T, Cpu, DEVICE_ID, A>>;
    fn to_dataloader(self) -> Self::Output {
        DataLoader::new(
            self.inner.layout.shape().clone(),
            self.inner.layout.strides().clone(),
            self,
        )
    }
}
