// use std::ffi::CStr;
// use std::hint::black_box;

use backend::Cpu;
use convertion::{Convertor, FromScalar};
use into_scalar::IntoScalar;
use ops::cpu::conv_config::{Conv2dConfig, KernelParamAlgo};
// use half::bf16;
// use tch::{ Device, Kind, Tensor };
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;
use type_promote::NormalOut;

#[track_caller]
fn assert_eq(
    a: &_Tensor<f64>,
    kernel: [i64; 2],
    b: &tch::Tensor,
    block_size: [i64; 2],
    config: &mut Conv2dConfig<f64>
) -> anyhow::Result<()> {
    config.set_ci_block_size(block_size[0]);
    config.set_co_block_size(block_size[1]);
    let res = a
        .avg_pool2d(
            kernel,
            [1, 1],
            [
                (0, 0),
                (0, 0),
            ],
            [1, 1],
            Some(&config)
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let res2 = b.avg_pool2d(&kernel, &[1, 1], &[0, 0], false, true, None);
    // let res_slice = res.as_raw();
    // let res2 = unsafe { std::slice::from_raw_parts(res2.data_ptr() as *const f64, res.size()) };
    println!("{}", res);
    println!("{}", res2);
    // res_slice
    //     .iter()
    //     .zip(res2.iter())
    //     .for_each(|(a, b)| {
    //         if a != b {
    //             assert_eq!(a, b);
    //         }
    //     });
    Ok(())
}

fn common_input<T>([batch, in_channel, kernel_height, kernel_width, height, width]: [i64; 6])
    -> anyhow::Result<(_Tensor<T>, tch::Tensor)>
    where
        T: Convertor + FromScalar<i64> + NormalOut<T, Output = T> + CommonBounds,
        usize: IntoScalar<T>,
        i64: IntoScalar<T>
{
    let a = _Tensor::<T>
        ::arange(0, batch * in_channel * height * width)?
        .reshape([batch, in_channel, height, width])?
        .permute([0, 2, 3, 1])?
        .contiguous()?;

    let tch_a = tch::Tensor
        ::arange(batch * in_channel * height * width, (tch::Kind::Double, tch::Device::Cpu))
        .reshape(&[batch, in_channel, height, width]);
    Ok((a, tch_a))
}
fn main() -> anyhow::Result<()> {
    let kernel = [3, 3];
    let (a, tch_a) = common_input([1, 130, 3, 3, 130, 130])?;
    let mut config = Conv2dConfig::<f64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
    assert_eq(&a, kernel, &tch_a, [3, 120], &mut config)?;
    Ok(())
}
