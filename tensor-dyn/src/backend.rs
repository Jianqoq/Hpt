#![allow(unused)]

use std::sync::Arc;

#[derive(Clone)]
pub struct Cpu {
    pub(crate) ptr: u64,
}

#[derive(Clone)]
pub struct Cuda;

#[derive(Clone)]
pub struct Wgpu {
    pub(crate) id: u64,
    pub(crate) device: Arc<wgpu::Device>,
}

#[derive(Clone)]
pub struct Backend<B> {
    _backend: B,
}

pub trait TensorBackend {
    fn new(id: u64) -> Self;
    fn wgpu_new(id: u64, device: Arc<wgpu::Device>) -> Self;
}

impl TensorBackend for Backend<Cpu> {
    fn new(address: u64) -> Self {
        Backend {
            _backend: Cpu {
                ptr: address,
            },
        }
    }

    fn wgpu_new(id: u64, device: Arc<wgpu::Device>) -> Self {
        todo!()
    }
}

impl TensorBackend for Backend<Cuda> {
    fn new(_id: u64) -> Self {
        Backend {
            _backend: Cuda,
        }
    }

    fn wgpu_new(id: u64, device: Arc<wgpu::Device>) -> Self {
        todo!()
    }
}

impl TensorBackend for Backend<Wgpu> {
    fn new(id: u64) -> Self {
        todo!()
    }

    fn wgpu_new(id: u64, device: Arc<wgpu::Device>) -> Self {
        Backend {
            _backend: Wgpu {
                id,
                device,
            },
        }
    }
}
