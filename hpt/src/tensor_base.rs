use crate::backend::{BackendTy, Buffer, Cpu};
use hpt_allocator::{traits::Allocator, HptAllocator};
use hpt_common::{layout::layout::Layout, utils::pointer::Pointer};
use std::marker::PhantomData;
use std::sync::Arc;

/// This struct is the base of All Tensors.
///
/// All the operations are happen on this struct.
///
/// # Properties
/// - `data`: The pointer to the data.
/// - `layout`: The layout of the tensor. We can get strides, shape, ndim, size from it.
/// - `parent`: The parent tensor of the tensor. parent is always the root tensor (`not a view`).
/// - `mem_layout`: std::alloc::layout, use for deallocate the memory and find cache in the allocator.
#[derive(Clone)]
pub(crate) struct _Tensor<T, B = Cpu, const DEVICE_ID: usize = 0, A = HptAllocator<B>>
where
    B: BackendTy + Buffer,
    A: Allocator,
{
    pub(crate) data: Pointer<T>,
    pub(crate) parent: Option<Pointer<T>>,
    pub(crate) layout: Layout,
    pub(crate) mem_layout: Arc<std::alloc::Layout>,
    pub(crate) _backend: hpt_allocator::Backend<B>,
    pub(crate) phantom: PhantomData<A>,
}

impl<T, B, const DEVICE_ID: usize, A> Drop for _Tensor<T, B, DEVICE_ID, A>
where
    B: BackendTy + Buffer,
    A: Allocator,
{
    fn drop(&mut self) {
        let mut allocator = A::new();
        allocator.deallocate(
            self._backend._backend.get_ptr() as *mut u8,
            &self.mem_layout,
            DEVICE_ID,
        );
    }
}
