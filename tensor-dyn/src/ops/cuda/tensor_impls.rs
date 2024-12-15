use std::sync::Arc;

use crate::{tensor_base::_Tensor, Cuda, Tensor};
use cudarc::driver::{CudaDevice, DeviceRepr};
use tensor_common::{layout::Layout, pointer::Pointer, shape::Shape};
use tensor_traits::TensorCreator;
use tensor_traits::{CommonBounds, TensorAlloc, TensorInfo, TensorLike};
use tensor_types::convertion::Convertor;

impl<T> TensorLike<T> for _Tensor<T, Cuda>
where
    T: CommonBounds,
{
    fn as_raw(&self) -> &[T] {
        unimplemented!()
    }

    fn as_raw_mut(&mut self) -> &mut [T] {
        unimplemented!()
    }

    fn contiguous(&self) -> anyhow::Result<Self> {
        unimplemented!()
    }
}

impl<T, const DEVICE_ID: usize> TensorInfo<T> for _Tensor<T, Cuda, DEVICE_ID>
where
    T: CommonBounds,
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
    fn strides(&self) -> &tensor_common::strides::Strides {
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

impl<T, const DEVICE_ID: usize> TensorInfo<T> for &_Tensor<T, Cuda, DEVICE_ID>
where
    T: CommonBounds,
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

    fn strides(&self) -> &tensor_common::strides::Strides {
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

impl<T: CommonBounds + DeviceRepr, const DEVICE_ID: usize> TensorAlloc
    for _Tensor<T, Cuda, DEVICE_ID>
{
    type Meta = T;
    fn _empty<S: Into<Shape>>(shape: S) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        Self::empty(shape)
    }
}

// impl<T: CommonBounds> TensorIterator<'_, T> for _Tensor<T, Cuda> {}

impl<T: CommonBounds + DeviceRepr, const DEVICE_ID: usize> _Tensor<T, Cuda, DEVICE_ID> {
    // /// copy the data from the other tensor to this tensor
    // pub fn assign(&mut self, other: &_Tensor<T>) {
    //     self.par_iter_mut_simd()
    //         .zip(other.par_iter_simd())
    //         .for_each(|(a, b)| {
    //             *a = b;
    //         });
    // }

    // /// cast the tensor to the new type
    // pub fn astype<U>(&self) -> anyhow::Result<_Tensor<U, Cuda>>
    // where
    //     U: CommonBounds,
    //     T: IntoScalar<U>,
    // {
    //     // Create an empty tensor of the new type with the same shape.
    //     let mut ret: _Tensor<U, Cuda> = _Tensor::<U, Cuda>::empty(self.layout.shape().clone())?;

    //     // Parallel iteration to convert and copy each element to the new tensor.
    //     ret.as_raw_mut()
    //         .par_iter_mut()
    //         .zip(self.as_raw().par_iter())
    //         .for_each(|(a, &b)| {
    //             *a = b.into_scalar();
    //         });
    //     Ok(ret)
    // }

    // /// try to cast the tensor to the new type, if the type is the same, return the tensor itself, otherwise return the new tensor
    // pub fn try_astype<U>(&self) -> anyhow::Result<_Tensor<U, Cuda>>
    // where
    //     U: CommonBounds,
    //     T: IntoScalar<U>,
    // {
    //     if U::ID == T::ID {
    //         Ok(self.static_cast()?)
    //     } else {
    //         Ok(self.astype::<U>()?)
    //     }
    // }

    /// bitcast the tensor to the new type, the user must ensure the size of the new type is the same as the old type
    pub fn static_cast<Dst>(&self) -> anyhow::Result<_Tensor<Dst, Cuda, DEVICE_ID>>
    where
        Dst: CommonBounds,
    {
        if T::ID == Dst::ID {
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
                }),
            }
        } else {
            panic!("Cannot cast tensor to different type")
        }
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

    pub fn to_cpu(&self) -> anyhow::Result<Tensor<T>> {
        let mut data = _Tensor::<T>::empty(self.layout.shape().clone()).unwrap();
        let device = self.device();
        let ptr = unsafe {
            device.upgrade_device_ptr(
                self.data.ptr as u64,
                self.size(),
            )
        };
        self.device()
            .dtoh_sync_copy_into(&ptr, data.as_raw_mut())
            .unwrap();
        ptr.leak();
        Ok(data.into())
    }
    pub(crate) fn device(&self) -> Arc<CudaDevice> {
        self._backend._backend.device.clone()
    }
}

impl<T: CommonBounds + DeviceRepr, const DEVICE_ID: usize> Tensor<T, Cuda, DEVICE_ID> {
    /// copy the data from the cuda tensor to the cpu tensor
    pub fn to_cpu(&self) -> anyhow::Result<Tensor<T>> {
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

    // /// cast the tensor to the new type
    // pub fn astype<U>(&self) -> anyhow::Result<Tensor<U, Cuda>>
    // where
    //     U: CommonBounds,
    //     T: IntoScalar<U>,
    // {
    //     Ok(self.inner.astype()?.into())
    // }

    // /// try to cast the tensor to the new type, if the type is the same, return the tensor itself, otherwise return the new tensor
    // pub fn try_astype<U>(&self) -> anyhow::Result<Tensor<U, Cuda>>
    // where
    //     U: CommonBounds,
    //     T: IntoScalar<U>,
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

impl<T, const DEVICE_ID: usize> std::fmt::Display for _Tensor<T, Cuda, DEVICE_ID>
where
    T: CommonBounds + Convertor + DeviceRepr,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut data = _Tensor::<T>::empty(self.layout.shape().clone()).unwrap();
        let device = self.device();
        let ptr = unsafe { device.upgrade_device_ptr(self.ptr().ptr as u64, self.size()) };
        self.device()
            .dtoh_sync_copy_into(&ptr, data.as_raw_mut())
            .unwrap();
        ptr.leak();
        write!(f, "{}", data)
    }
}

impl<T, const DEVICE_ID: usize> std::fmt::Display for Tensor<T, Cuda, DEVICE_ID>
where
    T: CommonBounds + Convertor + DeviceRepr,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner.as_ref())
    }
}

// impl<T> std::fmt::Debug for _Tensor<T, Cuda>
// where
//     T: CommonBounds + Convertor,
// {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         let precision = DISPLAY_PRECISION.load(Ordering::Relaxed);
//         let lr_element_size = DISPLAY_LR_ELEMENTS.load(Ordering::Relaxed);
//         display(self, f, lr_element_size, precision, false)
//     }
// }

impl<T, const DEVICE_ID: usize> Into<Tensor<T, Cuda, DEVICE_ID>> for _Tensor<T, Cuda, DEVICE_ID> {
    fn into(self) -> Tensor<T, Cuda, DEVICE_ID> {
        Tensor { inner: self.into() }
    }
}

// impl<T> Into<_Tensor<T, Cuda>> for &_Tensor<T, Cuda>
// where
//     T: CommonBounds,
// {
//     fn into(self) -> _Tensor<T, Cuda> {
//         _Tensor {
//             data: self.data.clone(),
//             parent: self.parent.clone(),
//             layout: self.layout.clone(),
//             mem_layout: self.mem_layout.clone(),
//             _backend: self._backend.clone(),
//         }
//     }
// }

impl<T, const DEVICE_ID: usize> Into<Tensor<T, Cuda, DEVICE_ID>> for &Tensor<T, Cuda, DEVICE_ID> {
    fn into(self) -> Tensor<T, Cuda, DEVICE_ID> {
        Tensor {
            inner: self.inner.clone(),
        }
    }
}

// impl<'a, T> Into<_Tensor<T, Cuda>> for &'a [T] {
//     fn into(self) -> _Tensor<T, Cuda> {
//         let shape = vec![self.len() as i64];
//         let strides = vec![1];
//         let layout = Layout::new(shape, strides);
//         let mem_layout =
//             std::alloc::Layout::from_size_align(self.len() * size_of::<T>(), ALIGN).unwrap();
//         let ptr = CACHE.allocate(mem_layout.clone()).unwrap();
//         unsafe {
//             std::ptr::copy_nonoverlapping(self.as_ptr(), ptr as *mut T, self.len());
//         }
//         _Tensor {
//             #[cfg(feature = "bound_check")]
//             data: Pointer::new(ptr as *mut T, self.len() as i64),
//             #[cfg(not(feature = "bound_check"))]
//             data: Pointer::new(ptr as *mut T),
//             parent: None,
//             layout,
//             mem_layout: Arc::new(mem_layout),
//             _backend: Backend::new(ptr as u64),
//         }
//     }
// }
