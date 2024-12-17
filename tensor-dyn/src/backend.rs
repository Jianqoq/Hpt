//! a module to define the backend of the tensor

#![allow(unused)]

use std::sync::Arc;

use tensor_allocator::{clone_storage, cuda_clone_storage, CPU_STORAGE};

/// Cpu backend
///
/// this backend stores the pointer of the data memory
pub struct Cpu {
    pub(crate) ptr: u64,
}

/// Cuda backend
pub struct Cuda {
    pub(crate) ptr: u64,
    pub(crate) device: Arc<cudarc::driver::CudaDevice>,
    pub(crate) cap: usize,
}

/// Wgpu backend
pub struct Wgpu;

/// backend of tensor
///
/// this backend stores the pointer of the data memory
///
/// this backend is used when we `free` or `clone` the tensor
#[derive(Clone)]
pub struct Backend<B> {
    pub(crate) _backend: B,
}

impl Clone for Cpu {
    fn clone(&self) -> Self {
        // increment the reference count
        clone_storage(self.ptr as *mut u8);
        Cpu { ptr: self.ptr }
    }
}

impl Backend<Cpu> {
    /// create a new Cpu backend
    pub fn new(address: u64) -> Self {
        Backend {
            _backend: Cpu { ptr: address },
        }
    }
}

impl Clone for Cuda {
    fn clone(&self) -> Self {
        // increment the reference count
        cuda_clone_storage(self.ptr as *mut u8, self.device.ordinal());
        Cuda {
            ptr: self.ptr,
            device: self.device.clone(),
            cap: self.cap,
        }
    }
}

impl Backend<Cuda> {
    /// create a new Cuda backend
    pub fn new(address: u64, device: Arc<cudarc::driver::CudaDevice>) -> Self {
        let cap_major = device.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        ).expect("failed to get compute capability major when creating cuda backend");
        let cap_minor = device.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        ).expect("failed to get compute capability minor when creating cuda backend");
        Backend {
            _backend: Cuda {
                ptr: address,
                device,
                cap: (cap_major * 10 + cap_minor) as usize,
            },
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

// reserved for future use
impl BackendTy for Cuda {
    const ID: u8 = 1;
}

// reserved for future use
impl BackendTy for Wgpu {
    const ID: u8 = 2;
}
