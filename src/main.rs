use tensor_dyn::set_global_display_precision;
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::Random;
use tensor_dyn::NormalReduce;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // let device = WgpuDevice::new(
    //     wgpu::Backends::VULKAN | wgpu::Backends::DX12,
    //     wgpu::Features::SHADER_INT64 |
    //         wgpu::Features::TIMESTAMP_QUERY |
    //         wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS |
    //         wgpu::Features::SHADER_F64
    // );
    // let a = _Tensor::<f32, Wgpu>::arange(0, 8, &device).await?;
    // let res1 = a.sinh().await?;
    // println!("{}", res1);
    set_global_display_precision(7);
    let a = _Tensor::<f32>::randn(&[8, 2048, 4, 2048])?;
    let mut i = 0;
    let now = std::time::Instant::now();
    for _ in 0..100 {
        let _ = a.sum([0, 1], false);
        i += 1;
    }
    println!("{:?}", now.elapsed() / 100);
    println!("{:?}", i);
    Ok(())
}
