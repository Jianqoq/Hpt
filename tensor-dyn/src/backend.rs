//! a module to define the backend of the tensor

#![allow(unused)]

use std::sync::Arc;

use tensor_allocator::{ clone_storage, CPU_STORAGE };

/// Cpu backend
/// 
/// this backend stores the pointer of the data memory
pub struct Cpu {
    pub(crate) ptr: u64,
}

/// Cuda backend
#[derive(Clone)]
pub struct Cuda;

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
        Cpu {
            ptr: self.ptr,
        }
    }
}

impl Backend<Cpu> {

    /// create a new Cpu backend
    pub fn new(address: u64) -> Self {
        Backend {
            _backend: Cpu {
                ptr: address,
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
