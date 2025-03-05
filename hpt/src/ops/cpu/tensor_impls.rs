use std::marker::PhantomData;
use std::{fmt::Display, sync::atomic::Ordering};

use crate::tensor::DiffTensor;
use crate::CompressionAlgo;
use crate::Cpu;
#[cfg(feature = "cuda")]
use crate::Cuda;
use crate::{save, Save};
use crate::{tensor_base::_Tensor, Tensor, DISPLAY_LR_ELEMENTS, DISPLAY_PRECISION};
#[cfg(feature = "cuda")]
use cudarc::driver::DeviceRepr;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_common::error::base::TensorError;
use hpt_common::{layout::layout::Layout, shape::shape::Shape, utils::pointer::Pointer};
use hpt_dataloader::data_loader::TensorMeta;
use hpt_dataloader::{DataLoader, Endian, FromSafeTensors, Meta};
use hpt_display::display;
use hpt_iterator::iterator_traits::ParStridedIteratorZip;
use hpt_iterator::TensorIterator;
use hpt_traits::TensorCreator;
use hpt_traits::{CommonBounds, TensorAlloc, TensorInfo, TensorLike};
#[cfg(feature = "cuda")]
use hpt_types::dtype::CudaType;
use hpt_types::into_scalar::Cast;
use num::traits::ToBytes;
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

    fn contiguous(&self) -> std::result::Result<Self, TensorError> {
        Ok(self.par_iter().strided_map(|(res, x)| *res = x).collect())
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

impl<T: CommonBounds, const DEVICE: usize, A> TensorAlloc for _Tensor<T, Cpu, DEVICE, A>
where
    A: Allocator,
    A::Output: AllocatorOutputRetrive,
{
    type Meta = T;
    fn _empty<S: Into<Shape>>(shape: S) -> std::result::Result<Self, TensorError>
    where
        Self: Sized,
    {
        Self::empty(shape)
    }
}

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
    /// cast the tensor to the new type
    pub fn astype<U>(&self) -> std::result::Result<_Tensor<U, Cpu, DEVICE, A>, TensorError>
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
    pub fn try_astype<U>(&self) -> Result<_Tensor<U, Cpu, DEVICE, A>, TensorError>
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
    pub fn static_cast<Dst>(&self) -> std::result::Result<_Tensor<Dst, Cpu, DEVICE, A>, TensorError>
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
                        _backend: self._backend.clone(),
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
                    _backend: self._backend.clone(),
                    phantom: PhantomData,
                }),
            }
        } else {
            panic!("Cannot cast tensor to different type")
        }
    }

    /// check if two tensors are close to each other
    pub fn allclose<U: CommonBounds>(&self, other: &_Tensor<U, Cpu, DEVICE, A>) -> bool
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
                let abs_diff: f64 = (a_val - b_val).abs();
                let torlerance: f64 = 1.0e-8 + 1.0e-5 * b_val.abs();
                acc && abs_diff <= torlerance
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
    pub fn allclose<U: CommonBounds>(&self, other: &Tensor<U, Cpu, DEVICE, A>) -> bool
    where
        T: Cast<f64>,
        U: Cast<f64>,
    {
        self.inner.allclose(&other.inner)
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

impl<const N: usize, T: CommonBounds + ToBytes<Bytes = [u8; N]>, const DEVICE: usize, A> Save
    for Tensor<T, Cpu, DEVICE, A>
where
    A: Allocator,
    Tensor<T, Cpu, DEVICE, A>: hpt_traits::TensorCreator<T, Output = Tensor<T, Cpu, DEVICE, A>>,
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
        let data_loader: DataLoader<T> = data.clone().into();
        let meta = Meta {
            name: "".to_string(),
            compression_algo,
            endian,
            data_saver: Box::new(data_loader),
            compression_level: level,
        };
        let info = save(file, meta, len_so_far, *global_cnt)?;
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
            _backend: self._backend.clone(),
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

impl<T: CommonBounds, const DEVICE: usize, Al> TensorAlloc for Tensor<T, Cpu, DEVICE, Al>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Meta = T;
    fn _empty<S: Into<Shape>>(shape: S) -> std::result::Result<Self, TensorError>
    where
        Self: Sized,
    {
        <Self as TensorCreator<T>>::empty(shape)
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
