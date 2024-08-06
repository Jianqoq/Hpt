use bytemuck::Pod;
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
