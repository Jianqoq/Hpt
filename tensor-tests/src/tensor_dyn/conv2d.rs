#![allow(unused)]
use tensor_dyn::{ set_global_display_lr_elements, set_num_threads, CommonBounds, TensorInfo };
use tensor_dyn::ops::cpu::convolutions::conv_config::{ Conv2dConfig, KernelParamAlgo };
use tensor_dyn::{ tensor_base::_Tensor, TensorCreator };
use tensor_dyn::ShapeManipulate;
use tensor_types::convertion::{ Convertor, FromScalar };
use tch;
use tensor_types::into_scalar::IntoScalar;
use tensor_types::type_promote::NormalOut;

fn common_input<T>([batch, out_channel, in_channel, kernel_height, kernel_width, height, width]: [
    i64;
    7
])
    -> anyhow::Result<(_Tensor<T>, _Tensor<T>, tch::Tensor, tch::Tensor)>
    where
        T: Convertor + FromScalar<i64> + NormalOut<T, Output = T> + CommonBounds,
        usize: IntoScalar<T>,
        i64: IntoScalar<T>
{
    let kernel = _Tensor::<T>
        ::arange(0, in_channel * out_channel * kernel_height * kernel_width)?
        .reshape([out_channel, in_channel, kernel_height, kernel_width])?
        .permute([2, 3, 1, 0])?
        .contiguous()?;
    let a = _Tensor::<T>
        ::arange(0, batch * in_channel * height * width)?
        .reshape([batch, in_channel, height, width])?
        .permute([0, 2, 3, 1])?
        .contiguous()?;

    let tch_kernel = tch::Tensor
        ::arange(in_channel * out_channel * kernel_height * kernel_width, (
            tch::Kind::Int64,
            tch::Device::Cpu,
        ))
        .reshape(&[out_channel, in_channel, kernel_height, kernel_width]);
    let tch_a = tch::Tensor
        ::arange(batch * in_channel * height * width, (tch::Kind::Int64, tch::Device::Cpu))
        .reshape(&[batch, in_channel, height, width]);
    Ok((kernel, a, tch_kernel, tch_a))
}

#[test]
fn test_case0() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 128, 3, 3, 3, 128, 128])?;
    let mut config = Conv2dConfig::<i64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
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
            Some(&config)
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let res2 = tch_a.conv2d(&tch_kernel, None::<tch::Tensor>, &[1, 1], &[0, 0], &[1, 1], 1);
    let res_slice = res.as_raw();
    let res2 = unsafe { std::slice::from_raw_parts(res2.data_ptr() as *const i64, res.size()) };
    res_slice
        .iter()
        .zip(res2.iter())
        .for_each(|(a, b)| {
            assert!(a == b);
        });

    config.set_ci_block_size(1);
    config.set_co_block_size(16);
    let res = a
        .conv2d(
            &kernel,
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
    let res_slice = res.as_raw();
    res_slice
        .iter()
        .zip(res2.iter())
        .for_each(|(a, b)| {
            if a != b {
                println!("{} != {}", a, b);
            }
        });
    Ok(())
}

#[test]
fn test_case1() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 128, 3, 3, 3, 130, 130])?;
    let mut config = Conv2dConfig::<i64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
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
            Some(&config)
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let res2 = tch_a.conv2d(&tch_kernel, None::<tch::Tensor>, &[1, 1], &[0, 0], &[1, 1], 1);
    let res_slice = res.as_raw();
    let res2 = unsafe { std::slice::from_raw_parts(res2.data_ptr() as *const i64, res.size()) };
    res_slice
        .iter()
        .zip(res2.iter())
        .for_each(|(a, b)| {
            assert!(a == b);
        });
    Ok(())
}

#[test]
fn test_case2() -> anyhow::Result<()> {
    set_num_threads(1);
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 128, 3, 3, 3, 128, 128])?;
    let mut config = Conv2dConfig::<i64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
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
            Some(&config)
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let res2 = tch_a.conv2d(&tch_kernel, None::<tch::Tensor>, &[1, 1], &[0, 0], &[1, 1], 1);
    let res_slice = res.as_raw();
    let res2 = unsafe { std::slice::from_raw_parts(res2.data_ptr() as *const i64, res.size()) };
    res_slice
        .iter()
        .zip(res2.iter())
        .for_each(|(a, b)| {
            assert!(a == b);
        });
    Ok(())
}

#[test]
fn test_case3() -> anyhow::Result<()> {
    set_num_threads(1);
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 130, 3, 3, 3, 130, 130])?;
    let mut config = Conv2dConfig::<i64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
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
            Some(&config)
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let res2 = tch_a.conv2d(&tch_kernel, None::<tch::Tensor>, &[1, 1], &[0, 0], &[1, 1], 1);
    let res_slice = res.as_raw();
    let res2 = unsafe { std::slice::from_raw_parts(res2.data_ptr() as *const i64, res.size()) };
    res_slice
        .iter()
        .zip(res2.iter())
        .for_each(|(a, b)| {
            assert!(a == b);
        });
    Ok(())
}

#[test]
fn test_case4() -> anyhow::Result<()> {
    set_num_threads(10);
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 128, 3, 3, 3, 128, 128])?;
    let mut config = Conv2dConfig::<i64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
    config.set_ci_block_size(2);
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
            Some(&config)
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let res2 = tch_a.conv2d(&tch_kernel, None::<tch::Tensor>, &[1, 1], &[0, 0], &[1, 1], 1);
    let res_slice = res.as_raw();
    let res2 = unsafe { std::slice::from_raw_parts(res2.data_ptr() as *const i64, res.size()) };
    res_slice
        .iter()
        .zip(res2.iter())
        .for_each(|(a, b)| {
            assert!(a == b);
        });
    Ok(())
}

#[test]
fn test_case5() -> anyhow::Result<()> {
    set_num_threads(1);
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 128, 3, 3, 3, 127, 127])?;
    let mut config = Conv2dConfig::<i64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
    config.set_ci_block_size(2);
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
            Some(&config)
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let res2 = tch_a.conv2d(&tch_kernel, None::<tch::Tensor>, &[1, 1], &[0, 0], &[1, 1], 1);
    let res_slice = res.as_raw();
    let res2 = unsafe { std::slice::from_raw_parts(res2.data_ptr() as *const i64, res.size()) };
    res_slice
        .iter()
        .zip(res2.iter())
        .for_each(|(a, b)| {
            assert!(a == b);
        });
    Ok(())
}

#[test]
fn test_case6() -> anyhow::Result<()> {
    set_num_threads(1);
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 130, 3, 3, 3, 130, 130])?;
    let mut config = Conv2dConfig::<i64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
    config.set_ci_block_size(2);
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
            Some(&config)
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let res2 = tch_a.conv2d(&tch_kernel, None::<tch::Tensor>, &[1, 1], &[0, 0], &[1, 1], 1);
    let res_slice = res.as_raw();
    let res2 = unsafe { std::slice::from_raw_parts(res2.data_ptr() as *const i64, res.size()) };
    res_slice
        .iter()
        .zip(res2.iter())
        .for_each(|(a, b)| {
            assert!(a == b);
        });
    Ok(())
}
