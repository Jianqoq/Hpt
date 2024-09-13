#![allow(unused)]

use std::sync::Arc;

use tensor_allocator::{ clone_storage, CPU_STORAGE };

pub struct Cpu {
    pub(crate) ptr: u64,
}

#[derive(Clone)]
pub struct Cuda;

pub struct Wgpu;

#[derive(Clone)]
pub struct Backend<B> {
    pub(crate) _backend: B,
}

impl Clone for Cpu {
    fn clone(&self) -> Self {
        clone_storage(self.ptr as *mut u8);
        Cpu {
            ptr: self.ptr,
        }
    }
}

impl Backend<Cpu> {
    pub fn new(address: u64) -> Self {
        Backend {
            _backend: Cpu {
                ptr: address,
            },
        }
    }
}

pub trait Buffer {
    fn get_ptr(&self) -> u64;
}

impl Buffer for Cpu {
    fn get_ptr(&self) -> u64 {
        self.ptr
    }
}

pub trait BackendTy {
    const ID: u8;
}

impl BackendTy for Cpu {
    const ID: u8 = 0;
}

impl BackendTy for Cuda {
    const ID: u8 = 1;
}

impl BackendTy for Wgpu {
    const ID: u8 = 2;
}
