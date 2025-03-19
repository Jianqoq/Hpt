use std::{
    borrow::{Borrow, BorrowMut},
    rc::Rc,
    sync::Arc,
};

use crate::tensor_base::_Tensor;
use hpt_allocator::{
    traits::{Allocator, AllocatorOutputRetrive},
    BackendTy, Buffer, Cpu,
};
use hpt_common::{
    error::base::TensorError, layout::layout::Layout, shape::shape::Shape, utils::pointer::Pointer,
};
use hpt_dataloader::CPUTensorCreator;
use hpt_iterator::TensorIterator;
use hpt_traits::{
    ops::creation::TensorCreator,
    tensor::{CommonBounds, TensorInfo, TensorLike},
};
use hpt_types::into_scalar::Cast;

/// `Tensor` is alias of N-dimensional array.
///
/// # Properties
/// - `data`: The pointer to the data.
/// - `layout`: The layout of the tensor. We can get strides, shape, ndim, size from it.
/// - `parent`: The parent tensor of the tensor. parent is always the root tensor (`not a view`).
/// - `mem_layout`: std::alloc::layout, use for deallocate the memory and find cache in the allocator.
#[derive(Clone)]
pub struct Tensor<T, B = Cpu, const DEVICE_ID: usize = 0, A = hpt_allocator::HptAllocator<B>>
where
    B: BackendTy + Buffer,
    A: Allocator,
{
    pub(crate) inner: Arc<_Tensor<T, B, DEVICE_ID, A>>,
}
use std::cell::RefCell;
/// `DiffTensor` is a tensor that has a gradient.
#[derive(Clone)]
pub struct DiffTensor<T, B = Cpu, const DEVICE_ID: usize = 0, A = hpt_allocator::HptAllocator<B>>
where
    B: BackendTy + Buffer,
    A: Allocator,
{
    pub(crate) inner: Tensor<T, B, DEVICE_ID, A>,
    pub(crate) grad: Rc<RefCell<Option<Tensor<T, B, DEVICE_ID, A>>>>,
    pub(crate) out_degree: Rc<RefCell<usize>>,
    pub(crate) backward:
        Rc<RefCell<dyn FnMut(Tensor<T, B, DEVICE_ID, A>) -> Result<bool, TensorError>>>,
}

impl<T, const DEVICE: usize, A> TensorLike<T> for Tensor<T, Cpu, DEVICE, A>
where
    T: CommonBounds,
    A: Allocator,
{
    fn as_raw(&self) -> &[T] {
        self.inner.as_raw()
    }

    fn as_raw_mut(&mut self) -> &mut [T] {
        let slice =
            unsafe { std::slice::from_raw_parts_mut(self.inner.ptr().ptr as *mut T, self.size()) };
        slice
    }
}

impl<'a, T: CommonBounds, const DEVICE: usize, Al> TensorIterator<'a, T>
    for Tensor<T, Cpu, DEVICE, Al>
where
    Al: Allocator + 'a,
    Al::Output: AllocatorOutputRetrive,
{
}

macro_rules! impl_tensor_info {
    ($tensor:ty) => {
        impl<T, B, const DEVICE: usize, A> TensorInfo<T> for $tensor
        where
            T: CommonBounds,
            B: BackendTy + Buffer,
            A: hpt_allocator::traits::Allocator,
        {
            fn ptr(&self) -> Pointer<T> {
                self.inner.as_ref().data.clone()
            }

            fn size(&self) -> usize {
                self.inner.as_ref().layout.size() as usize
            }

            fn shape(&self) -> &Shape {
                self.inner.as_ref().layout.shape()
            }

            fn strides(&self) -> &hpt_common::strides::strides::Strides {
                self.inner.as_ref().layout.strides()
            }

            fn layout(&self) -> &Layout {
                &self.inner.as_ref().layout
            }

            fn parent(&self) -> Option<Pointer<T>> {
                self.inner.as_ref().parent.clone()
            }

            fn ndim(&self) -> usize {
                self.inner.as_ref().layout.ndim()
            }

            fn is_contiguous(&self) -> bool {
                self.inner.as_ref().layout.is_contiguous()
            }
        }
    };
}

impl_tensor_info!(Tensor<T, B, DEVICE, A>);
impl_tensor_info!(&Tensor<T, B, DEVICE, A>);
impl_tensor_info!(&mut Tensor<T, B, DEVICE, A>);

impl<T, B: BackendTy + Buffer, const DEVICE_ID: usize, A> Borrow<_Tensor<T, B, DEVICE_ID, A>>
    for Tensor<T, B, DEVICE_ID, A>
where
    T: CommonBounds,
    A: Allocator,
{
    fn borrow(&self) -> &_Tensor<T, B, DEVICE_ID, A> {
        &self.inner
    }
}

impl<T, B: BackendTy + Buffer, const DEVICE_ID: usize, A> Borrow<_Tensor<T, B, DEVICE_ID, A>>
    for &Tensor<T, B, DEVICE_ID, A>
where
    T: CommonBounds,
    A: Allocator,
{
    fn borrow(&self) -> &_Tensor<T, B, DEVICE_ID, A> {
        &self.inner
    }
}

impl<T, B: BackendTy + Buffer, const DEVICE_ID: usize, A> Borrow<_Tensor<T, B, DEVICE_ID, A>>
    for &mut Tensor<T, B, DEVICE_ID, A>
where
    T: CommonBounds,
    A: Allocator,
{
    fn borrow(&self) -> &_Tensor<T, B, DEVICE_ID, A> {
        &self.inner
    }
}

impl<T, B: BackendTy + Buffer + Clone, const DEVICE_ID: usize, A>
    BorrowMut<_Tensor<T, B, DEVICE_ID, A>> for &mut Tensor<T, B, DEVICE_ID, A>
where
    T: CommonBounds,
    A: Allocator,
{
    fn borrow_mut(&mut self) -> &mut _Tensor<T, B, DEVICE_ID, A> {
        Arc::make_mut(&mut self.inner)
    }
}

impl<T, B: BackendTy + Buffer + Clone, const DEVICE_ID: usize, A>
    BorrowMut<_Tensor<T, B, DEVICE_ID, A>> for Tensor<T, B, DEVICE_ID, A>
where
    T: CommonBounds,
    A: Allocator,
{
    fn borrow_mut(&mut self) -> &mut _Tensor<T, B, DEVICE_ID, A> {
        Arc::make_mut(&mut self.inner)
    }
}

impl<T, B, const DEVICE: usize, A> CPUTensorCreator for Tensor<T, B, DEVICE, A>
where
    T: CommonBounds + bytemuck::AnyBitPattern,
    B: BackendTy + Buffer,
    A: Allocator,
    Tensor<T, Cpu, DEVICE, A::CpuAllocator>:
        TensorCreator<Output = Tensor<T, Cpu, DEVICE, A::CpuAllocator>>,
{
    type Output = Tensor<T, Cpu, DEVICE, A::CpuAllocator>;
    type Meta = T;
    fn empty<S: Into<Shape>>(shape: S) -> Result<Self::Output, TensorError> {
        <Tensor<T, Cpu, DEVICE, A::CpuAllocator> as TensorCreator>::empty(shape)
    }
}

impl<T, B: BackendTy + Buffer, const DEVICE: usize, Al> std::fmt::Debug for Tensor<T, B, DEVICE, Al>
where
    T: CommonBounds + Cast<f64>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("data", &self.inner.data)
            .field("shape", &self.shape())
            .field("strides", &self.strides())
            .field("parent", &self.parent())
            .field("align", &self.inner.mem_layout.align())
            .field("backend", &self.inner.backend)
            .finish()
    }
}
