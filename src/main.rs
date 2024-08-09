use tensor_dyn::set_global_display_precision;
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::Random;
use tensor_dyn::NormalReduce;
use tensor_dyn::Matmul;
use tensor_dyn::TensorCreator;
use tensor_dyn::ShapeManipulate;

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
    let a = _Tensor::<f32>::arange(0, 4 * 4 * 4 * 4)?.reshape(&[4, 4, 4, 4])?;
    let now = std::time::Instant::now();
    let res = a.sum([0, 2], false)?;
    println!("{:?}", now.elapsed() / 100);
    println!("{}", res);
    Ok(())
}
