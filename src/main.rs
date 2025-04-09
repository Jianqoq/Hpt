
use hpt::ops::*;
use hpt::types::f16;
use hpt::utils::{set_display_elements, set_seed};
use hpt::{error::TensorError, Tensor};
fn main() -> Result<(), TensorError> {
    let batch = 1;
    let in_channel = 3;
    let out_channel = 3;
    let height = 4;
    let width = 4;
    let kernel_height = 3;
    let kernel_width = 3;
    let a = Tensor::<f32>::arange(0, batch * in_channel * height * width)?
        .reshape([batch, height, width, in_channel])?;
    let kernel = Tensor::<f32>::arange(0, out_channel * in_channel * kernel_height * kernel_width)?
        .reshape([kernel_height, kernel_width, in_channel, out_channel])?;
    let b = a.conv2d(&kernel, None, [1, 1], [(0, 0), (0, 0)], [1, 1], None)?;
    let b_group = a.conv2d_group(&kernel, None, [1, 1], [(0, 0), (0, 0)], [1, 1], 1, None)?;
    println!("{}", b);
    println!("{}", b_group);
    Ok(())
}
