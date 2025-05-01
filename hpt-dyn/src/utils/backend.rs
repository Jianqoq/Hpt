#[cfg(feature = "cuda")]
use std::sync::Arc;

use hpt_allocator::{clone_storage, traits::Allocator};
use hpt_common::Pointer;

pub(crate) enum Backend {
    Cpu {
        should_drop: bool,
        ptr: Pointer<u8>,
        device_id: usize,
    },
    #[cfg(feature = "cuda")]
    Cuda {
        should_drop: bool,
        ptr: Pointer<u8>,
        device: Arc<cudarc::driver::CudaDevice>,
        cap: usize,
    },
}

impl Backend {
    pub fn new_cpu(ptr: Pointer<u8>, device_id: usize, should_drop: bool) -> Self {
        Self::Cpu {
            should_drop,
            ptr,
            device_id,
        }
    }
    #[cfg(feature = "cuda")]
    pub fn new_cuda(
        ptr: Pointer<u8>,
        device: Arc<cudarc::driver::CudaDevice>,
        should_drop: bool,
    ) -> Self {
        let cap_major = device.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        ).expect("failed to get compute capability major when creating cuda backend");
        let cap_minor = device.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        ).expect("failed to get compute capability minor when creating cuda backend");
        Self::Cuda {
            should_drop,
            ptr,
            device,
            cap: (cap_major * 10 + cap_minor) as usize,
        }
    }

    pub fn dealloc(&self, layout: std::alloc::Layout) {
        match self {
            Self::Cpu {
                should_drop,
                ptr,
                device_id,
            } => {
                use hpt_allocator::Cpu;
                let mut allocator = hpt_allocator::HptAllocator::<Cpu>::new();
                allocator.deallocate(ptr.ptr, &layout, *should_drop, *device_id);
            }
            #[cfg(feature = "cuda")]
            Self::Cuda {
                should_drop,
                ptr,
                device,
                ..
            } => {
                if *should_drop {
                    use hpt_allocator::Cuda;
                    let mut allocator = hpt_allocator::HptAllocator::<Cuda>::new();
                    allocator.deallocate(ptr.ptr, &layout, *should_drop, device.ordinal());
                }
            }
        }
    }
}

impl Clone for Backend {
    fn clone(&self) -> Self {
        match self {
            Self::Cpu {
                should_drop,
                ptr,
                device_id,
            } => {
                if let Ok(mut storage) = hpt_allocator::CPU_STORAGE.lock() {
                    clone_storage(ptr.ptr, *device_id, &mut storage);
                } else {
                    panic!("failed to lock CPU_STORAGE");
                }
                Self::Cpu {
                    should_drop: should_drop.clone(),
                    ptr: ptr.clone(),
                    device_id: device_id.clone(),
                }
            }
            #[cfg(feature = "cuda")]
            Self::Cuda {
                should_drop,
                ptr,
                device,
                cap,
            } => {
                if let Ok(mut storage) = hpt_allocator::CUDA_STORAGE.lock() {
                    clone_storage(ptr.ptr, device.ordinal(), &mut storage);
                } else {
                    panic!("failed to lock CUDA_STORAGE");
                }
                Self::Cuda {
                    should_drop: should_drop.clone(),
                    ptr: ptr.clone(),
                    device: device.clone(),
                    cap: cap.clone(),
                }
            }
        }
    }
}

impl std::fmt::Debug for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backend::Cpu {
                should_drop,
                ptr,
                device_id,
            } => f
                .debug_struct("Backend(Cpu)")
                .field("should_drop", should_drop)
                .field("ptr", ptr)
                .field("device_id", device_id)
                .finish(),
            #[cfg(feature = "cuda")]
            Backend::Cuda {
                should_drop,
                ptr,
                device,
                cap,
            } => f
                .debug_struct("Backend(Cuda)")
                .field("should_drop", should_drop)
                .field("ptr", ptr)
                .field("device_id", device.ordinal())
                .field("cap", cap)
                .finish(),
        }
    }
}
