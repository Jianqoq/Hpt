use anyhow::Result;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use std::{
    fmt::{Debug, Display},
    sync::{atomic::Ordering, Arc},
};
use tensor_allocator::{CACHE, CUDA_CACHE};
use tensor_common::{layout::Layout, pointer::Pointer, shape::Shape};
use tensor_display::display;
use tensor_iterator::{
    iterator_traits::{ParStridedIteratorSimdZip, ParStridedIteratorZip},
    TensorIterator,
};
use tensor_traits::tensor::{CommonBounds, TensorAlloc, TensorCreator, TensorInfo, TensorLike};
use tensor_types::{convertion::Convertor, into_scalar::IntoScalar};

use crate::{
    backend::{Backend, BackendTy, Buffer, Cpu},
    tensor::Tensor,
    Cuda, ALIGN, DISPLAY_LR_ELEMENTS, DISPLAY_PRECISION,
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
pub(crate) struct _Tensor<T, B = Cpu, const DEVICE_ID: usize = 0>
where
    B: BackendTy + Buffer,
{
    pub(crate) data: Pointer<T>,
    pub(crate) parent: Option<Pointer<T>>,
    pub(crate) layout: Layout,
    pub(crate) mem_layout: Arc<std::alloc::Layout>,
    pub(crate) _backend: Backend<B>,
}

impl<T, B, const DEVICE_ID: usize> Drop for _Tensor<T, B, DEVICE_ID>
where
    B: BackendTy + Buffer,
{
    fn drop(&mut self) {
        match B::ID {
            0 => CACHE.deallocate(
                self._backend._backend.get_ptr() as *mut u8,
                &self.mem_layout,
            ),
            1 => {
                if let Ok(mut cuda_cache) = CUDA_CACHE.lock() {
                    cuda_cache.deallocate(
                        self._backend._backend.get_ptr() as *mut u8,
                        &self.mem_layout,
                        DEVICE_ID,
                    );
                } else {
                    panic!("CUDA_CACHE is poisoned");
                }
            }
            _ => {
                panic!("Invalid Backend ID")
            }
        }
    }
}
