use crate::{Backend, BackendTy, Buffer, Cpu};
use hpt_allocator::traits::Allocator;
use hpt_allocator::CACHE;
#[cfg(feature = "cuda")]
use hpt_allocator::CUDA_CACHE;
use hpt_common::{layout::layout::Layout, utils::pointer::Pointer};
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
            0 => {
                if let Ok(mut cpu_cache) = CACHE.lock() {
                    cpu_cache.deallocate(
                        self._backend._backend.get_ptr() as *mut u8,
                        &self.mem_layout,
                        DEVICE_ID,
                    );
                } else {
                    panic!("CUDA_CACHE is poisoned");
                }
            }
            #[cfg(feature = "cuda")]
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
