use hpt::ops::*;
use hpt::types::{bf16, f16};
use hpt::utils::{set_display_elements, set_seed};
use hpt::{error::TensorError, Tensor};
fn main() -> Result<(), TensorError> {
    let test_times = 1000;
    const N: i64 = 16;
    const CHANNEL: i64 = 512;
    let batch = 1;
    let in_channel = CHANNEL;
    let out_channel = CHANNEL;
    let height = N;
    let width = N;
    let kernel_height = 3;
    let kernel_width = 3;
    // let a = Tensor::<f32>::arange(0, batch * in_channel * height * width)?
    //     .reshape([batch, height, width, in_channel])?;
    // let kernel = Tensor::<f32>::arange(0, out_channel * in_channel * kernel_height * kernel_width)?
    //     .reshape([kernel_height, kernel_width, in_channel, out_channel])?;
    let mut test_a = Tensor::<f32>::arange(0, batch * in_channel * height * width)?;
    let mut test_b = test_a.reshape([batch, height, width, in_channel])?;
    let a = Tensor::<f32>::ones([batch, height, width, in_channel])?;
    let kernel = Tensor::<f32>::ones([kernel_height, kernel_width, in_channel, out_channel])?;
    let now = std::time::Instant::now();
    for _ in 0..test_times {
        let b = a.conv2d(&kernel, None, [1, 1], [(0, 0), (0, 0)], [1, 1])?;
        test_a = b;
    }
    println!("conv2d time: {:?}", now.elapsed() / test_times);
    // let now = std::time::Instant::now();
    // for _ in 0..test_times {
    //     let b_group = a.conv2d_group(&kernel, None, [1, 1], [(1, 1), (1, 1)], [1, 1], 1, None)?;
    //     test_b = b_group;
    // }
    // println!("conv2d_group time: {:?}", now.elapsed() / test_times);
    // assert!(test_a.allclose(&test_b, 1e-3, 1e-3));
    // println!("b: {}", test_a);
    // println!("b_group: {}", test_b);
    Ok(())
}
