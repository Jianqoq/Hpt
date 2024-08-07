#![allow(unused)]

use std::sync::Arc;

use tensor_allocator::{ BufferWrapper, CPU_STORAGE, WGPU_STORAGE };
use wgpu::Buffer;

use crate::ops::wgpu::buffer_helper::WgpuDevice;

pub struct Cpu {
    pub(crate) ptr: u64,
}

#[derive(Clone)]
pub struct Cuda;

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

impl Clone for Cpu {
    fn clone(&self) -> Self {
        unsafe {
            CPU_STORAGE.lock()
                .unwrap()
                .entry(self.ptr as *mut u8)
                .and_modify(|v| {
                    *v += 1;
                });
        }
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

impl Backend<Cuda> {
    pub fn wgpu_new(id: u64, device: Arc<wgpu::Device>) -> Self {
        todo!()
    }
}

impl Clone for Wgpu {
    fn clone(&self) -> Self {
        unsafe {
            WGPU_STORAGE.lock()
                .unwrap()
                .entry(self.buffer.buffer.global_id())
                .and_modify(|v| {
                    *v += 1;
                });
        }
        Wgpu {
            buffer: self.buffer.clone(),
            device: self.device.clone(),
        }
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

pub trait BackendDevice {
    fn wgpu_device(&self) -> &WgpuDevice;
    fn buffer(&self) -> &BufferWrapper;
}

impl BackendDevice for Cpu {
    fn wgpu_device(&self) -> &WgpuDevice {
        panic!("Cpu backend does not have a device")
    }
    fn buffer(&self) -> &BufferWrapper {
        panic!("Cpu backend does not have a buffer")
    }
}

impl BackendDevice for Cuda {
    fn wgpu_device(&self) -> &WgpuDevice {
        panic!("Cuda backend does not have a device")
    }
    fn buffer(&self) -> &BufferWrapper {
        panic!("Cuda backend does not have a buffer")
    }
}

impl BackendDevice for Wgpu {
    fn wgpu_device(&self) -> &WgpuDevice {
        &self.device
    }
    fn buffer(&self) -> &BufferWrapper {
        &self.buffer
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
