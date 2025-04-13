use candle_core::Tensor as CandleTensor;
use hpt::ops::*;
use hpt::types::{bf16, f16};
use hpt::utils::{set_display_elements, set_seed};
use hpt::{error::TensorError, Tensor};
fn main() -> Result<(), TensorError> {
    let test_times = 1000;
    const N: i64 = 32;
    const CHANNEL: i64 = 512;
    let batch = 1;
    let in_channel = CHANNEL;
    let out_channel = CHANNEL;
    let height = N;
    let width = N;
    let kernel_height = 3;
    let kernel_width = 3;
    // 196 * 4608 * 512
    // let mut test_a = Tensor::<f32>::arange(0, batch * in_channel * height * width)?;
    // let mut test_b = test_a.reshape([batch, height, width, in_channel])?;
    // let a = Tensor::<f32>::ones([batch, height, width, in_channel])?;
    // let kernel = Tensor::<f32>::ones([kernel_height, kernel_width, in_channel, out_channel])?;
    // let now = std::time::Instant::now();
    // for _ in 0..test_times {
    //     let b = a.conv2d(&kernel, None, [1, 1], [(0, 0), (0, 0)], [1, 1])?;
    //     test_a = b;
    // }
    // println!("conv2d time: {:?}", now.elapsed() / test_times);

    let a = Tensor::<f16>::ones([640, 640])?;
    let b = Tensor::<f16>::ones([640, 640])?;
    let now = std::time::Instant::now();
    // for _ in 0..test_times {
    let c = a.matmul(&b)?;
    println!("c: {}", c);
    // }
    println!("gemm time: {:?}", now.elapsed() / test_times);
    let a3 = CandleTensor::ones(
        [4, 4]
            .into_iter()
            .map(|x| x as usize)
            .collect::<Vec<usize>>(),
            candle_core::DType::F16,
        &candle_core::Device::Cpu,
    )
    .unwrap();
    let c3 = CandleTensor::ones(
        [4, 4]
            .into_iter()
            .map(|x| x as usize)
            .collect::<Vec<usize>>(),
            candle_core::DType::F16,
        &candle_core::Device::Cpu,
    )
    .unwrap();
    let now = std::time::Instant::now();
    // for _ in 0..test_times {
        let c = a3.matmul(&c3).unwrap();
    // }
    println!("c: {}", c);
    Ok(())
}
