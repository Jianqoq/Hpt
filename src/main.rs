use tensor_dyn::{ set_global_display_precision, TensorCreator };
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

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
    let a = _Tensor::<f32>::new([
        [1f32, 2.0],
        [3.0, 4.0],
    ]);
    let indices = _Tensor::<i64>::new([
        [0, 0],
        [1, 0],
    ]);
    let mut i = 0;
    let now = std::time::Instant::now();
    for _ in 0..1 {
        let d = a.gather_elements(&indices, 1)?;
        println!("{}", d);
        i += 1;
    }
    println!("{:?}", now.elapsed() / 100);
    println!("{}", i);
    Ok(())
}
