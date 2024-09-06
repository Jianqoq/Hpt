#![allow(unused_imports)]
use tensor_dyn::{set_num_threads, TensorInfo};
use tensor_dyn::ops::cpu::convolutions::conv_config::{ Conv2dConfig, KernelParamAlgo };
use tensor_dyn::{ tensor_base::_Tensor, TensorCreator };
use tensor_dyn::ShapeManipulate;
use tch;

#[test]
fn test_case0() -> anyhow::Result<()> {
    let kernel = _Tensor::<f64>
        ::arange(0, 3 * 128 * 3 * 3)?
        .reshape([128, 3, 3, 3])?
        .permute([2, 3, 1, 0])?
        .contiguous()?;
    let a = _Tensor::<f64>
        ::arange(0, 1 * 3 * 128 * 128)?
        .reshape([1, 3, 128, 128])?
        .permute([0, 2, 3, 1])?
        .contiguous()?;
    let mut config = Conv2dConfig::<f64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
    config.set_ci_block_size(3);
    config.set_co_block_size(128);
    let res = a
        .conv2d(
            &kernel,
            [1, 1],
            [
                (0, 0),
                (0, 0),
            ],
            [1, 1],
            &Some(config)
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let kernel = tch::Tensor
        ::arange(3 * 128 * 3 * 3, (tch::Kind::Double, tch::Device::Cpu))
        .reshape(&[128, 3, 3, 3]);
    let a = tch::Tensor
        ::arange(1 * 3 * 128 * 128, (tch::Kind::Double, tch::Device::Cpu))
        .reshape(&[1, 3, 128, 128]);

    let res2 = a.conv2d(&kernel, None::<tch::Tensor>, &[1, 1], &[0, 0], &[1, 1], 1);
    let res_slice = res.as_raw();
    let res2 = unsafe { std::slice::from_raw_parts(res2.data_ptr() as *const f64, res.size()) };
    res_slice
        .iter()
        .zip(res2.iter())
        .for_each(|(a, b)| {
            assert!((a - b).abs() < 1e-6);
        });
    Ok(())
}

#[test]
fn test_case1() -> anyhow::Result<()> {
    set_num_threads(1);
    let kernel = _Tensor::<f64>
        ::arange(0, 3 * 128 * 3 * 3)?
        .reshape([128, 3, 3, 3])?
        .permute([2, 3, 1, 0])?
        .contiguous()?;
    let a = _Tensor::<f64>
        ::arange(0, 1 * 3 * 128 * 128)?
        .reshape([1, 3, 128, 128])?
        .permute([0, 2, 3, 1])?
        .contiguous()?;
    let mut config = Conv2dConfig::<f64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
    config.set_ci_block_size(3);
    config.set_co_block_size(120);
    let res = a
        .conv2d(
            &kernel,
            [1, 1],
            [
                (0, 0),
                (0, 0),
            ],
            [1, 1],
            &Some(config)
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let kernel = tch::Tensor
        ::arange(3 * 128 * 3 * 3, (tch::Kind::Double, tch::Device::Cpu))
        .reshape(&[128, 3, 3, 3]);
    let a = tch::Tensor
        ::arange(1 * 3 * 128 * 128, (tch::Kind::Double, tch::Device::Cpu))
        .reshape(&[1, 3, 128, 128]);

    let res2 = a.conv2d(&kernel, None::<tch::Tensor>, &[1, 1], &[0, 0], &[1, 1], 1);
    let res_slice = res.as_raw();
    let res2 = unsafe { std::slice::from_raw_parts(res2.data_ptr() as *const f64, res.size()) };
    res_slice
        .iter()
        .zip(res2.iter())
        .for_each(|(a, b)| {
            assert!((a - b).abs() < 1e-6);
        });
    Ok(())
}
