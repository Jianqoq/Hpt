use std::num::NonZeroU32;

use bytemuck::Pod;
use tensor_common::shape_utils::mt_intervals;
use wgpu::{ util::DeviceExt, BufferUsages };

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
