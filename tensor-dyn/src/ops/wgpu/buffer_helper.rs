use std::{ num::NonZeroU32, ops::Deref, sync::Arc };

use bytemuck::Pod;
use tensor_allocator::DeviceWrapper;
use tensor_common::shape_utils::mt_intervals;
use wgpu::{
    util::DeviceExt,
    BufferUsages,
    Dx12Compiler,
    Gles3MinorVersion,
    InstanceFlags,
    Limits,
    RequestAdapterOptions,
};

#[derive(Clone)]
pub struct WgpuDevice {
    pub(crate) device: Arc<DeviceWrapper>,
    pub(crate) queue: Arc<wgpu::Queue>,
}

impl WgpuDevice {
    pub fn new(backends: wgpu::Backends, features: wgpu::Features) -> Self {
        let (device, queue) = pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends,
                flags: InstanceFlags::VALIDATION,
                dx12_shader_compiler: Dx12Compiler::Dxc { dxil_path: None, dxc_path: None },
                gles_minor_version: Gles3MinorVersion::default(),
            });

            // `request_adapter` instantiates the general connection to the GPU
            let adapter = instance
                .request_adapter(
                    &(RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        compatible_surface: None,
                        force_fallback_adapter: false,
                    })
                ).await
                .unwrap();

            let adapter_limits = adapter.limits();

            // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
            //  `features` being the available features.
            let limits = wgpu::Limits {
                max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
                max_buffer_size: adapter_limits.max_buffer_size,
                max_storage_buffers_per_shader_stage: 12,
                ..wgpu::Limits::default()
            };

            adapter
                .request_device(
                    &(wgpu::DeviceDescriptor {
                        label: None,
                        required_features: features,
                        required_limits: limits,
                        memory_hints: wgpu::MemoryHints::Performance,
                    }),
                    None
                ).await
                .unwrap()
        });

        Self {
            device: Arc::new(DeviceWrapper {
                device: Arc::new(device),
            }),
            queue: Arc::new(queue),
        }
    }
    pub fn device(&self) -> &DeviceWrapper {
        &self.device
    }
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}

impl Deref for WgpuDevice {
    type Target = DeviceWrapper;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

pub fn create_buffer<T: Pod>(
    device: &wgpu::Device,
    data: &[T],
    usages: BufferUsages,
    label: Option<&str>
) -> wgpu::Buffer {
    device.create_buffer_init(
        &(wgpu::util::BufferInitDescriptor {
            label,
            contents: bytemuck::cast_slice(data),
            usage: usages,
        })
    )
}

pub fn chunk_buffer<T: Pod>(
    device: &wgpu::Device,
    arr: &[T],
    inner_loop_size: usize,
    max_memory_per_pipeline: usize
) -> Vec<wgpu::Buffer> {
    let outer_loop_size = arr.len() / inner_loop_size;
    let num_pipeline = (outer_loop_size + max_memory_per_pipeline - 1) / max_memory_per_pipeline;
    let intervals = mt_intervals(outer_loop_size, num_pipeline);

    let mut buffers = Vec::with_capacity(num_pipeline);
    for (start, end) in intervals {
        let buffer = device.create_buffer_init(
            &(wgpu::util::BufferInitDescriptor {
                label: Some(format!("chunk_buffer_{}_{}", start, end).as_str()),
                contents: bytemuck::cast_slice(
                    arr[start * inner_loop_size..end * inner_loop_size].as_ref()
                ),
                usage: BufferUsages::STORAGE,
            })
        );
        buffers.push(buffer);
    }
    buffers
}

pub fn create_bg_layout(binding: u32, read_only: bool, size: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: NonZeroU32::new(size),
    }
}
