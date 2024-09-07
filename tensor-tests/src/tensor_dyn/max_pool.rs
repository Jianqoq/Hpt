#![allow(unused)]
use tensor_dyn::{ set_global_display_lr_elements, set_num_threads, CommonBounds, TensorInfo };
use tensor_dyn::ops::cpu::convolutions::conv_config::{ Conv2dConfig, KernelParamAlgo };
use tensor_dyn::{ tensor_base::_Tensor, TensorCreator };
use tensor_dyn::ShapeManipulate;
use tensor_types::convertion::{ Convertor, FromScalar };
use tch;
use tensor_types::into_scalar::IntoScalar;
use tensor_types::type_promote::NormalOut;

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
        ::arange(batch * in_channel * height * width, (tch::Kind::Int64, tch::Device::Cpu))
        .reshape(&[batch, in_channel, height, width]);
    Ok((a, tch_a))
}

fn assert_eq(
    a: &_Tensor<i64>,
    b: &tch::Tensor,
    block_size: i64,
    kernel_size: [i64; 2],
    config: &mut Conv2dConfig<i64>
) -> anyhow::Result<()> {
    config.set_co_block_size(block_size);
    let res = a
        .max_pool2d(
            kernel_size,
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
    let res2 = b.max_pool2d(&kernel_size, &[1, 1], &[0, 0], &[1, 1], false);
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
fn test_case0() -> anyhow::Result<()> {
    let (a, tch_a) = common_input([1, 128, 3, 3, 128, 128])?;
    let mut config = Conv2dConfig::<i64>::new(3, 3, [3, 3], KernelParamAlgo::Greedy);
    assert_eq(&a, &tch_a, 4, [3, 3], &mut config)?;
    assert_eq(&a, &tch_a, 16, [3, 3], &mut config)?;
    assert_eq(&a, &tch_a, 1, [3, 3], &mut config)?;

    let (a, tch_a) = common_input([1, 128, 3, 3, 5, 5])?;
    let mut config = Conv2dConfig::<i64>::new(3, 3, [3, 3], KernelParamAlgo::Greedy);
    assert_eq(&a, &tch_a, 4, [3, 3], &mut config)?;
    assert_eq(&a, &tch_a, 16, [3, 3], &mut config)?;
    assert_eq(&a, &tch_a, 1, [3, 3], &mut config)?;

    let (a, tch_a) = common_input([1, 2, 3, 3, 5, 5])?;
    let mut config = Conv2dConfig::<i64>::new(3, 2, [3, 3], KernelParamAlgo::Greedy);
    assert_eq(&a, &tch_a, 1, [3, 3], &mut config)?;
    Ok(())
}

#[test]
fn test_case1() -> anyhow::Result<()> {
    let (a, tch_a) = common_input([1, 128, 3, 3, 130, 130])?;
    let mut config = Conv2dConfig::<i64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
    assert_eq(&a, &tch_a, 4, [3, 3], &mut config)?;
    assert_eq(&a, &tch_a, 16, [3, 3], &mut config)?;
    assert_eq(&a, &tch_a, 1, [3, 3], &mut config)?;
    Ok(())
}

#[test]
fn test_case2() -> anyhow::Result<()> {
    let (a, tch_a) = common_input([1, 128, 3, 3, 128, 128])?;
    let mut config = Conv2dConfig::<i64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
    assert_eq(&a, &tch_a, 4, [3, 3], &mut config)?;
    assert_eq(&a, &tch_a, 16, [3, 3], &mut config)?;
    assert_eq(&a, &tch_a, 1, [3, 3], &mut config)?;
    Ok(())
}

#[test]
fn test_case3() -> anyhow::Result<()> {
    let (a, tch_a) = common_input([1, 130, 3, 3, 130, 130])?;
    let mut config = Conv2dConfig::<i64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
    assert_eq(&a, &tch_a, 4, [3, 3], &mut config)?;
    assert_eq(&a, &tch_a, 16, [3, 3], &mut config)?;
    assert_eq(&a, &tch_a, 1, [3, 3], &mut config)?;
    Ok(())
}

#[test]
fn test_case4() -> anyhow::Result<()> {
    let (a, tch_a) = common_input([1, 128, 3, 3, 128, 128])?;
    let mut config = Conv2dConfig::<i64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
    assert_eq(&a, &tch_a, 4, [3, 3], &mut config)?;
    assert_eq(&a, &tch_a, 16, [3, 3], &mut config)?;
    assert_eq(&a, &tch_a, 1, [3, 3], &mut config)?;
    Ok(())
}

#[test]
fn test_case5() -> anyhow::Result<()> {
    let (a, tch_a) = common_input([1, 128, 3, 3, 127, 127])?;
    let mut config = Conv2dConfig::<i64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
    assert_eq(&a, &tch_a, 4, [3, 3], &mut config)?;
    assert_eq(&a, &tch_a, 16, [3, 3], &mut config)?;
    assert_eq(&a, &tch_a, 1, [3, 3], &mut config)?;
    Ok(())
}

#[test]
fn test_case6() -> anyhow::Result<()> {
    let (a, tch_a) = common_input([1, 130, 3, 3, 130, 130])?;
    let mut config = Conv2dConfig::<i64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
    assert_eq(&a, &tch_a, 4, [3, 3], &mut config)?;
    assert_eq(&a, &tch_a, 16, [3, 3], &mut config)?;
    assert_eq(&a, &tch_a, 1, [3, 3], &mut config)?;
    Ok(())
}

#[test]
fn test_case7() -> anyhow::Result<()> {
    let (a, tch_a) = common_input([1, 130, 3, 3, 128, 128])?;
    let mut config = Conv2dConfig::<i64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
    assert_eq(&a, &tch_a, 4, [3, 3], &mut config)?;
    assert_eq(&a, &tch_a, 16, [3, 3], &mut config)?;
    assert_eq(&a, &tch_a, 1, [3, 3], &mut config)?;
    Ok(())
}
