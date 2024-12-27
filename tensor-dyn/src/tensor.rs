use std::{
    borrow::{Borrow, BorrowMut},
    fmt::{Debug, Display},
    sync::{atomic::Ordering, Arc},
};

use crate::{
    backend::{BackendTy, Buffer, Cpu},
    tensor_base::_Tensor,
    DISPLAY_LR_ELEMENTS, DISPLAY_PRECISION,
};
use tensor_common::{err_handler::ErrHandler, layout::Layout, pointer::Pointer, shape::Shape};
use tensor_dataloader::DataLoader;
use tensor_display::display;
use tensor_iterator::TensorIterator;
use tensor_traits::tensor::{CommonBounds, TensorAlloc, TensorCreator, TensorInfo, TensorLike};
use tensor_types::convertion::Convertor;

/// `Tensor` is alias of N-dimensional array.
///
/// # Properties
/// - `data`: The pointer to the data.
/// - `layout`: The layout of the tensor. We can get strides, shape, ndim, size from it.
/// - `parent`: The parent tensor of the tensor. parent is always the root tensor (`not a view`).
/// - `mem_layout`: std::alloc::layout, use for deallocate the memory and find cache in the allocator.
#[derive(Clone)]
pub struct Tensor<T, B = Cpu, const DEVICE_ID: usize = 0>
where
    B: BackendTy + Buffer,
{
    pub(crate) inner: Arc<_Tensor<T, B, DEVICE_ID>>,
}

impl<T> TensorLike<T> for Tensor<T>
where
    T: CommonBounds,
{
    fn as_raw(&self) -> &[T] {
        self.inner.as_raw()
    }

    fn as_raw_mut(&mut self) -> &mut [T] {
        let slice =
            unsafe { std::slice::from_raw_parts_mut(self.inner.ptr().ptr as *mut T, self.size()) };
        slice
    }

    fn contiguous(&self) -> std::result::Result<Self, ErrHandler> {
        Ok(_Tensor::contiguous(self.inner.as_ref())?.into())
    }
}

impl<T: CommonBounds> TensorIterator<'_, T> for Tensor<T> {}

impl<T> TensorInfo<T> for Tensor<T>
where
    T: CommonBounds,
{
    fn ptr(&self) -> Pointer<T> {
        self.inner.ptr().clone()
    }

    fn size(&self) -> usize {
        self.inner.layout().size() as usize
    }

    fn shape(&self) -> &Shape {
        self.inner.layout().shape()
    }

    fn strides(&self) -> &tensor_common::strides::Strides {
        self.inner.layout().strides()
    }

    fn layout(&self) -> &Layout {
        self.inner.layout()
    }

    fn parent(&self) -> Option<Pointer<T>> {
        self.inner.parent().clone()
    }

    fn ndim(&self) -> usize {
        self.inner.layout().ndim()
    }

    fn is_contiguous(&self) -> bool {
        self.inner.layout().is_contiguous()
    }
}

impl<T> TensorInfo<T> for &Tensor<T>
where
    T: CommonBounds,
{
    fn ptr(&self) -> Pointer<T> {
        self.inner.ptr().clone()
    }

    fn size(&self) -> usize {
        self.inner.layout().size() as usize
    }

    fn shape(&self) -> &Shape {
        self.inner.layout().shape()
    }

    fn strides(&self) -> &tensor_common::strides::Strides {
        self.inner.layout().strides()
    }

    fn layout(&self) -> &Layout {
        self.inner.layout()
    }

    fn parent(&self) -> Option<Pointer<T>> {
        self.inner.parent().clone()
    }

    fn ndim(&self) -> usize {
        self.inner.layout().ndim()
    }

    fn is_contiguous(&self) -> bool {
        self.inner.layout().is_contiguous()
    }
}

impl<T: CommonBounds> TensorAlloc for Tensor<T> {
    type Meta = T;
    fn _empty<S: Into<Shape>>(shape: S) -> std::result::Result<Self, ErrHandler>
    where
        Self: Sized,
    {
        Self::empty(shape)
    }
}

impl<T> Display for Tensor<T>
where
    T: CommonBounds + Convertor,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let precision = DISPLAY_PRECISION.load(Ordering::Relaxed);
        let lr_element_size = DISPLAY_LR_ELEMENTS.load(Ordering::Relaxed);
        display(self, f, lr_element_size, precision, false)
    }
}

impl<T> Debug for Tensor<T>
where
    T: CommonBounds + Convertor,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let precision = DISPLAY_PRECISION.load(Ordering::Relaxed);
        let lr_element_size = DISPLAY_LR_ELEMENTS.load(Ordering::Relaxed);
        display(self, f, lr_element_size, precision, false)
    }
}

impl<T, B: BackendTy + Buffer, const DEVICE_ID: usize> Borrow<_Tensor<T, B, DEVICE_ID>>
    for Tensor<T, B, DEVICE_ID>
where
    T: CommonBounds,
{
    fn borrow(&self) -> &_Tensor<T, B, DEVICE_ID> {
        &self.inner
    }
}

impl<T, B: BackendTy + Buffer, const DEVICE_ID: usize> Borrow<_Tensor<T, B, DEVICE_ID>>
    for &Tensor<T, B, DEVICE_ID>
where
    T: CommonBounds,
{
    fn borrow(&self) -> &_Tensor<T, B, DEVICE_ID> {
        &self.inner
    }
}

impl<T, B: BackendTy + Buffer, const DEVICE_ID: usize> Borrow<_Tensor<T, B, DEVICE_ID>>
    for &mut Tensor<T, B, DEVICE_ID>
where
    T: CommonBounds,
{
    fn borrow(&self) -> &_Tensor<T, B, DEVICE_ID> {
        &self.inner
    }
}

impl<T, B: BackendTy + Buffer + Clone, const DEVICE_ID: usize> BorrowMut<_Tensor<T, B, DEVICE_ID>>
    for &mut Tensor<T, B, DEVICE_ID>
where
    T: CommonBounds,
{
    fn borrow_mut(&mut self) -> &mut _Tensor<T, B, DEVICE_ID> {
        Arc::make_mut(&mut self.inner)
    }
}

impl<T, B: BackendTy + Buffer + Clone, const DEVICE_ID: usize> BorrowMut<_Tensor<T, B, DEVICE_ID>>
    for Tensor<T, B, DEVICE_ID>
where
    T: CommonBounds,
{
    fn borrow_mut(&mut self) -> &mut _Tensor<T, B, DEVICE_ID> {
        Arc::make_mut(&mut self.inner)
    }
}

impl<T, const DEVICE_ID: usize> From<Tensor<T, Cpu, DEVICE_ID>> for DataLoader<T> {
    fn from(value: Tensor<T, Cpu, DEVICE_ID>) -> Self {
        DataLoader::new(
            value.inner.layout.shape().clone(),
            value.inner.layout.strides().clone(),
            value.inner.data.ptr,
        )
    }
}
