use tensor_dyn::backend::Wgpu;
use tensor_dyn::ops::wgpu::buffer_helper::WgpuDevice;
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::FloatUaryOps;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let device = WgpuDevice::new(
        wgpu::Backends::VULKAN | wgpu::Backends::DX12,
        wgpu::Features::SHADER_INT64 |
            wgpu::Features::TIMESTAMP_QUERY |
            wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS |
            wgpu::Features::SHADER_F64
    );
    let a = _Tensor::<u32, Wgpu>::arange(0, 1024 * 1024, &device).await?;
    let res1 = a.sin()?;
    println!("{}", res1);
    Ok(())
}
