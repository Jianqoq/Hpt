#![allow(unused)]

#[derive(Clone, Copy)]
pub struct Cpu {
    pub(crate) ptr: u64,
}

#[derive(Clone, Copy)]
pub struct Cuda;

#[derive(Clone, Copy)]
pub struct Wgpu {
    pub(crate) id: u64,
}

#[derive(Clone, Copy)]
pub struct Backend<B> {
    _backend: B,
}

pub trait TensorBackend {
    fn new(id: u64) -> Self;
}

impl TensorBackend for Backend<Cpu> {
    fn new(address: u64) -> Self {
        Backend {
            _backend: Cpu {
                ptr: address,
            },
        }
    }
}

impl TensorBackend for Backend<Cuda> {
    fn new(_id: u64) -> Self {
        Backend {
            _backend: Cuda,
        }
    }
}

impl TensorBackend for Backend<Wgpu> {
    fn new(id: u64) -> Self {
        Backend {
            _backend: Wgpu {
                id,
            },
        }
    }
}
