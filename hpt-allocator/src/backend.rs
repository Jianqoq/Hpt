//! a module to define the backend of the tensor

#![allow(unused)]

use std::sync::Arc;

use crate::clone_storage;

/// Cpu backend
///
/// this backend stores the pointer of the data memory
pub struct Cpu {
    pub(crate) ptr: u64,
    pub(crate) device_id: usize,
}

#[cfg(feature = "cuda")]
/// Cuda backend
pub struct Cuda {
    pub(crate) ptr: u64,
    /// device
    pub device: Arc<cudarc::driver::CudaDevice>,
    /// compute capability
    pub cap: usize,
}

/// backend of tensor
///
/// this backend stores the pointer of the data memory
///
/// this backend is used when we `free` or `clone` the tensor
#[derive(Clone)]
pub struct Backend<B> {
    /// the backend of the tensor
    pub inner: B,
    /// should drop the data, the data comes from the user
    pub should_drop: bool,
}

impl<B: BackendTy> Backend<B> {
    /// get the should drop flag
    pub fn should_drop(&self) -> bool {
        self.should_drop
    }

    /// forget the backend
    pub fn forget(&mut self) {
        self.should_drop = false;
    }
}

impl<B: BackendTy> std::fmt::Debug for Backend<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match B::ID {
            0 => f.debug_struct("cpu").finish(),
            1 => f.debug_struct("cuda").finish(),
            _ => f.debug_struct("unknown").finish(),
        }
    }
}

impl Clone for Cpu {
    fn clone(&self) -> Self {
        if let Ok(mut storage) = crate::CPU_STORAGE.lock() {
            clone_storage(self.ptr as *mut u8, self.device_id, &mut storage);
        } else {
            panic!("failed to lock CPU_STORAGE");
        }
        Cpu {
            ptr: self.ptr,
            device_id: self.device_id,
        }
    }
}

impl Backend<Cpu> {
    /// create a new Cpu backend
    pub fn new(address: u64, device_id: usize, should_drop: bool) -> Self {
        Backend {
            inner: Cpu {
                ptr: address,
                device_id,
            },
            should_drop,
        }
    }
}

#[cfg(feature = "cuda")]
impl Clone for Cuda {
    fn clone(&self) -> Self {
        if let Ok(mut storage) = crate::CUDA_STORAGE.lock() {
            clone_storage(self.ptr as *mut u8, self.device.ordinal(), &mut storage);
        } else {
            panic!("failed to lock CUDA_STORAGE");
        }
        Cuda {
            ptr: self.ptr,
            device: self.device.clone(),
            cap: self.cap,
        }
    }
}

#[cfg(feature = "cuda")]
impl Backend<Cuda> {
    /// create a new Cuda backend
    pub fn new(address: u64, device: Arc<cudarc::driver::CudaDevice>, should_drop: bool) -> Self {
        let cap_major = device.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        ).expect("failed to get compute capability major when creating cuda backend");
        let cap_minor = device.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        ).expect("failed to get compute capability minor when creating cuda backend");
        Backend {
            inner: Cuda {
                ptr: address,
                device,
                cap: (cap_major * 10 + cap_minor) as usize,
            },
            should_drop,
        }
    }
}

/// trait for buffer
///
/// this trait is used to get the pointer of the data memory
pub trait Buffer {
    /// get the pointer of the data memory
    fn get_ptr(&self) -> u64;
}

impl Buffer for Cpu {
    fn get_ptr(&self) -> u64 {
        self.ptr
    }
}

#[cfg(feature = "cuda")]
impl Buffer for Cuda {
    fn get_ptr(&self) -> u64 {
        self.ptr
    }
}

/// backend id trait
///
/// this trait is used to get the id of the backend
///
/// 0: Cpu
///
/// 1: Cuda
///
/// 2: Wgpu
pub trait BackendTy {
    /// beackend id
    const ID: u8;
}

impl BackendTy for Cpu {
    const ID: u8 = 0;
}

#[cfg(feature = "cuda")]
impl BackendTy for Cuda {
    const ID: u8 = 1;
}
