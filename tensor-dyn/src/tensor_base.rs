use anyhow::Result;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use std::{
    fmt::{Debug, Display},
    sync::{atomic::Ordering, Arc},
};
use tensor_allocator::CACHE;
use tensor_common::{layout::Layout, pointer::Pointer, shape::Shape};
use tensor_display::display;
use tensor_iterator::{iterator_traits::ParStridedIteratorZip, TensorIterator};
use tensor_traits::tensor::{CommonBounds, TensorAlloc, TensorCreator, TensorInfo, TensorLike};
use tensor_types::{convertion::Convertor, into_scalar::IntoScalar};

use crate::{
    backend::{Backend, BackendTy, Buffer, Cpu},
    tensor::Tensor,
    ALIGN, DISPLAY_LR_ELEMENTS, DISPLAY_PRECISION,
};

/// This struct is the heart of the `DiffTensors` and `BasicTensors`. Both of them are just `wrappers` around this struct.
///
/// All the operations are happen on this struct.
///
/// # Properties
/// - `data`: The pointer to the data.
/// - `layout`: The layout of the tensor. We can get strides, shape, ndim, size from it.
/// - `parent`: The parent tensor of the tensor. parent is always the root tensor (`not a view`).
/// - `mem_layout`: std::alloc::layout, use for deallocate the memory and find cache in the allocator.
#[derive(Clone)]
pub struct _Tensor<T, B = Cpu>
where
    B: BackendTy + Buffer,
{
    pub(crate) data: Pointer<T>,
    pub(crate) parent: Option<Pointer<T>>,
    pub(crate) layout: Layout,
    pub(crate) mem_layout: Arc<std::alloc::Layout>,
    pub(crate) _backend: Backend<B>,
}

impl<T, B> Drop for _Tensor<T, B>
where
    B: BackendTy + Buffer,
{
    fn drop(&mut self) {
        match B::ID {
            0 => CACHE.deallocate(
                self._backend._backend.get_ptr() as *mut u8,
                &self.mem_layout,
            ),
            _ => {
                panic!("Invalid Backend ID")
            }
        }
    }
}

impl<T> TensorLike<T> for _Tensor<T>
where
    T: CommonBounds,
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

    fn contiguous(&self) -> anyhow::Result<Self> {
        use tensor_types::traits::VecTrait;
        let res = self
            .par_iter_simd()
            .strided_map_simd(
                |(res, x)| {
                    *res = x;
                },
                |(res, x)| {
                    // possibily a rust bug when we use sse vector,
                    // so we have to use ptr directly or hope rust is able to inline the `write_unaligned`

                    // let ptr = res.as_mut_ptr() as *mut T::Vec;
                    // unsafe {
                    //     ptr.write_unaligned(x);
                    // }
                    res.write_unaligned(x);
                },
            )
            .collect();
        Ok(res)
    }
}

impl<T> TensorInfo<T> for _Tensor<T>
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

impl<T> TensorInfo<T> for &_Tensor<T>
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

impl<T: CommonBounds> TensorAlloc for _Tensor<T> {
    type Meta = T;
    fn _empty<S: Into<Shape>>(shape: S) -> Result<Self>
    where
        Self: Sized,
    {
        Self::empty(shape)
    }
}

impl<T: CommonBounds> TensorIterator<'_, T> for _Tensor<T> {}

impl<T: CommonBounds> _Tensor<T> {
    /// copy the data from the other tensor to this tensor
    pub fn assign(&mut self, other: &_Tensor<T>) {
        self.par_iter_mut_simd()
            .zip(other.par_iter_simd())
            .for_each(|(a, b)| {
                *a = b;
            });
    }

    /// cast the tensor to the new type
    pub fn astype<U>(&self) -> Result<_Tensor<U>>
    where
        U: CommonBounds,
        T: IntoScalar<U>,
    {
        // Create an empty tensor of the new type with the same shape.
        let mut ret: _Tensor<U> = _Tensor::<U>::empty(self.layout.shape().clone())?;

        // Parallel iteration to convert and copy each element to the new tensor.
        ret.as_raw_mut()
            .par_iter_mut()
            .zip(self.as_raw().par_iter())
            .for_each(|(a, &b)| {
                *a = b.into_scalar();
            });
        Ok(ret)
    }

    /// try to cast the tensor to the new type, if the type is the same, return the tensor itself, otherwise return the new tensor
    pub fn try_astype<U>(&self) -> Result<_Tensor<U>>
    where
        U: CommonBounds,
        T: IntoScalar<U>,
    {
        if U::ID == T::ID {
            Ok(self.static_cast()?)
        } else {
            Ok(self.astype::<U>()?)
        }
    }

    /// bitcast the tensor to the new type, the user must ensure the size of the new type is the same as the old type
    pub fn static_cast<Dst>(&self) -> Result<_Tensor<Dst>>
    where
        Dst: CommonBounds,
    {
        if T::ID == Dst::ID {
            match self.parent.clone() {
                Some(parent) => {
                    #[cfg(feature = "bound_check")]
                    let new_parent = Pointer::new(parent.ptr as *mut Dst, parent.layout.clone());
                    #[cfg(not(feature = "bound_check"))]
                    let new_parent = Pointer::new(parent.ptr as *mut Dst);
                    Ok(_Tensor {
                        #[cfg(feature = "bound_check")]
                        data: Pointer::new(self.data.ptr as *mut Dst, self.layout.clone()),
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
                    data: Pointer::new(self.data.ptr as *mut Dst, self.layout.clone()),
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

    /// check if two tensors are close to each other
    pub fn allclose<U: CommonBounds>(&self, other: &_Tensor<U>) -> bool
    where
        T: Convertor,
        U: Convertor,
    {
        if self.shape() != other.shape() {
            return false;
        }
        let folder = self.par_iter().zip(other.par_iter()).fold(
            || true,
            |acc, (a, b)| {
                let a_val: f64 = a.to_f64();
                let b_val: f64 = b.to_f64();
                let abs_diff: f64 = (a_val - b_val).abs();
                let torlerance: f64 = 1.0e-8 + 1.0e-5 * b_val.abs();
                acc && abs_diff <= torlerance
            },
        );
        folder.reduce(|| true, |a, b| a && b)
    }
}

impl<T> Display for _Tensor<T>
where
    T: CommonBounds + Convertor,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let precision = DISPLAY_PRECISION.load(Ordering::Relaxed);
        let lr_element_size = DISPLAY_LR_ELEMENTS.load(Ordering::Relaxed);
        display(self, f, lr_element_size, precision, false)
    }
}

impl<T> Debug for _Tensor<T>
where
    T: CommonBounds + Convertor,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let precision = DISPLAY_PRECISION.load(Ordering::Relaxed);
        let lr_element_size = DISPLAY_LR_ELEMENTS.load(Ordering::Relaxed);
        display(self, f, lr_element_size, precision, false)
    }
}

impl<T> Into<Tensor<T>> for _Tensor<T> {
    fn into(self) -> Tensor<T> {
        Tensor { inner: self.into() }
    }
}

impl<T> Into<_Tensor<T>> for &_Tensor<T>
where
    T: CommonBounds,
{
    fn into(self) -> _Tensor<T> {
        _Tensor {
            data: self.data.clone(),
            parent: self.parent.clone(),
            layout: self.layout.clone(),
            mem_layout: self.mem_layout.clone(),
            _backend: self._backend.clone(),
        }
    }
}

impl<T> Into<Tensor<T>> for &Tensor<T> {
    fn into(self) -> Tensor<T> {
        Tensor {
            inner: self.inner.clone(),
        }
    }
}

impl<'a, T> Into<_Tensor<T>> for &'a [T] {
    fn into(self) -> _Tensor<T> {
        let shape = vec![self.len() as i64];
        let strides = vec![1];
        let layout = Layout::new(shape, strides);
        let mem_layout =
            std::alloc::Layout::from_size_align(self.len() * size_of::<T>(), ALIGN).unwrap();
        let ptr = CACHE.allocate(mem_layout.clone());
        unsafe {
            std::ptr::copy_nonoverlapping(self.as_ptr(), ptr as *mut T, self.len());
        }
        _Tensor {
            #[cfg(feature = "bound_check")]
            data: Pointer::new(ptr as *mut T, layout.clone()),
            #[cfg(not(feature = "bound_check"))]
            data: Pointer::new(ptr as *mut T),
            parent: None,
            layout,
            mem_layout: Arc::new(mem_layout),
            _backend: Backend::new(ptr as u64),
        }
    }
}
