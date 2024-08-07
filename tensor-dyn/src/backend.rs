#![allow(unused)]

use std::sync::Arc;

use tensor_allocator::BufferWrapper;
use wgpu::Buffer;

use crate::ops::wgpu::buffer_helper::WgpuDevice;

#[derive(Clone)]
pub struct Cpu {
    pub(crate) ptr: u64,
}

#[derive(Clone)]
pub struct Cuda;

#[derive(Clone)]
pub struct Wgpu {
    pub(crate) buffer: BufferWrapper,
    pub(crate) device: WgpuDevice,
}

impl Wgpu {
    pub fn device(&self) -> &wgpu::Device {
        &self.device.device.device
    }
    pub fn queue(&self) -> &wgpu::Queue {
        &self.device.queue
    }
}

#[derive(Clone)]
pub struct Backend<B> {
    pub(crate) _backend: B,
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

impl Backend<Cuda> {
    pub fn wgpu_new(id: u64, device: Arc<wgpu::Device>) -> Self {
        todo!()
    }
}

impl Backend<Wgpu> {
    pub fn wgpu_new(id: u64, device: &WgpuDevice, buffer: BufferWrapper) -> Self {
        Backend {
            _backend: Wgpu {
                buffer,
                device: device.clone(),
            },
        }
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
