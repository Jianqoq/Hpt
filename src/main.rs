use hpt::ops::*;
use hpt::types::f16;
use hpt::utils::{set_display_elements, set_seed};
use hpt::{error::TensorError, Tensor};
fn main() -> Result<(), TensorError> {
    let batch = 1;
    let in_channel = 256;
    let out_channel = 256;
    let height = 16;
    let width = 16;
    let kernel_height = 3;
    let kernel_width = 3;
    // let a = Tensor::<f32>::arange(0, batch * in_channel * height * width)?
    //     .reshape([batch, height, width, in_channel])?;
    // let kernel = Tensor::<f32>::arange(0, out_channel * in_channel * kernel_height * kernel_width)?
    //     .reshape([kernel_height, kernel_width, in_channel, out_channel])?;
    let a = Tensor::<f32>::randn([batch, height, width, in_channel])?;
    let kernel = Tensor::<f32>::randn([kernel_height, kernel_width, in_channel, out_channel])?;
    let now = std::time::Instant::now();
    for _ in 0..100 {
        let b = a.conv2d(&kernel, None, [1, 1], [(0, 0), (0, 0)], [1, 1], None)?;
    }
    println!("conv2d time: {:?}", now.elapsed() / 100);
    let now = std::time::Instant::now();
    for _ in 0..100 {
        let b_group = a.conv2d_group(&kernel, None, [1, 1], [(0, 0), (0, 0)], [1, 1], 1, None)?;
    }
    println!("conv2d_group time: {:?}", now.elapsed() / 100);
    // assert!(b.allclose(&b_group, 1e-4, 1e-4));
    // println!("b: {}", b);
    // println!("b_group: {}", b_group);
    Ok(())
}
