#![allow(unused)]
use rayon::iter::{ IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator };
use tensor_dyn::{ set_global_display_lr_elements, set_num_threads, CommonBounds, TensorInfo };
use tensor_dyn::ops::cpu::conv_config::{ Conv2dConfig, KernelParamAlgo };
use tensor_dyn::{ tensor_base::_Tensor, TensorCreator };
use tensor_dyn::ShapeManipulate;
use tensor_types::convertion::{ Convertor, FromScalar };
use tch;
use tensor_types::into_scalar::IntoScalar;
use tensor_types::type_promote::NormalOut;
use tensor_dyn::TensorLike;

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

fn assert_eq(
    a: &_Tensor<i64>,
    a_kernel: &_Tensor<i64>,
    b: &tch::Tensor,
    b_kernel: &tch::Tensor,
    block_size: [i64; 2],
    config: &mut Conv2dConfig<i64>
) -> anyhow::Result<()> {
    config.set_ci_block_size(block_size[0]);
    config.set_co_block_size(block_size[1]);
    let res = a
        .conv2d(
            &a_kernel,
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
    let res2 = b.conv2d(&b_kernel, None::<tch::Tensor>, &[1, 1], &[0, 0], &[1, 1], 1);
    let res_slice = res.as_raw();
    let res2 = unsafe { std::slice::from_raw_parts(res2.data_ptr() as *const i64, res.size()) };
    res_slice
        .par_iter()
        .zip(res2.par_iter())
        .for_each(|(a, b)| {
            if a != b {
                assert_eq!(a, b);
            }
        });
    Ok(())
}

#[test]
fn test_case0() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 128, 3, 3, 3, 64, 64])?;
    let mut config = Conv2dConfig::<i64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [3, 128], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1], &mut config)?;

    // test when outwidth is less than regnum
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 128, 3, 3, 3, 5, 5])?;
    let mut config = Conv2dConfig::<i64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [3, 128], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1], &mut config)?;

    let (kernel, a, tch_kernel, tch_a) = common_input([1, 2, 3, 3, 3, 5, 5])?;
    let mut config = Conv2dConfig::<i64>::new(128, 2, [3, 3], KernelParamAlgo::Greedy);
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [3, 1], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1], &mut config)?;
    Ok(())
}

#[test]
fn test_case1() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 128, 3, 3, 3, 130, 130])?;
    let mut config = Conv2dConfig::<i64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [3, 128], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1], &mut config)?;

    let (kernel, a, tch_kernel, tch_a) = common_input([1, 128, 3, 3, 3, 5, 5])?;
    let mut config = Conv2dConfig::<i64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [3, 128], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1], &mut config)?;
    Ok(())
}

#[test]
fn test_case2() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 128, 3, 3, 3, 128, 128])?;
    let mut config = Conv2dConfig::<i64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [3, 120], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1], &mut config)?;

    let (kernel, a, tch_kernel, tch_a) = common_input([1, 128, 3, 3, 3, 5, 5])?;
    let mut config = Conv2dConfig::<i64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [3, 128], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1], &mut config)?;
    Ok(())
}

#[test]
fn test_case3() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 130, 3, 3, 3, 130, 130])?;
    let mut config = Conv2dConfig::<i64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [3, 120], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1], &mut config)?;

    let (kernel, a, tch_kernel, tch_a) = common_input([1, 128, 3, 3, 3, 5, 5])?;
    let mut config = Conv2dConfig::<i64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [3, 128], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1], &mut config)?;
    Ok(())
}

#[test]
fn test_case4() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 128, 3, 3, 3, 128, 128])?;
    let mut config = Conv2dConfig::<i64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [2, 128], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1], &mut config)?;

    let (kernel, a, tch_kernel, tch_a) = common_input([1, 128, 3, 3, 3, 5, 5])?;
    let mut config = Conv2dConfig::<i64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [3, 128], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1], &mut config)?;
    Ok(())
}

#[test]
fn test_case5() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 128, 3, 3, 3, 127, 127])?;
    let mut config = Conv2dConfig::<i64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [2, 128], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1], &mut config)?;

    let (kernel, a, tch_kernel, tch_a) = common_input([1, 128, 3, 3, 3, 5, 5])?;
    let mut config = Conv2dConfig::<i64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [3, 128], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1], &mut config)?;
    Ok(())
}

#[test]
fn test_case6() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 130, 3, 3, 3, 130, 130])?;
    let mut config = Conv2dConfig::<i64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [2, 120], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1], &mut config)?;

    let (kernel, a, tch_kernel, tch_a) = common_input([1, 128, 3, 3, 3, 5, 5])?;
    let mut config = Conv2dConfig::<i64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [3, 128], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1], &mut config)?;
    Ok(())
}

#[test]
fn test_case7() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 130, 3, 3, 3, 128, 128])?;
    let mut config = Conv2dConfig::<i64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [2, 120], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1], &mut config)?;

    let (kernel, a, tch_kernel, tch_a) = common_input([1, 128, 3, 3, 3, 5, 5])?;
    let mut config = Conv2dConfig::<i64>::new(128, 3, [3, 3], KernelParamAlgo::Greedy);
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [3, 128], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16], &mut config)?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1], &mut config)?;
    Ok(())
}
