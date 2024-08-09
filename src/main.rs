use tensor_dyn::set_global_display_lr_elements;
use tensor_dyn::set_global_display_precision;
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::Random;
use tensor_dyn::NormalReduce;
use tensor_dyn::StridedIterator;

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
    set_global_display_lr_elements(6);
    let a = _Tensor::<f32>::randn(&[8, 2048, 2048])?;
    let now = std::time::Instant::now();
    a.iter_mut().zip(a.iter()).for_each(|(x, y)| {
        *x = y;
    });
    println!("{:?}", now.elapsed() / 100);
    println!("{:?}", a);
    Ok(())
}
