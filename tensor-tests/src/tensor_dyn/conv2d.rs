#![allow(unused)]
use rayon::iter::{ IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator };
use tch;
use tensor_dyn::ops::cpu::conv_config::{ Conv2dConfig, KernelParamAlgo };
use tensor_dyn::ShapeManipulate;
use tensor_dyn::TensorLike;
use tensor_dyn::{ set_global_display_lr_elements, set_num_threads, CommonBounds, TensorInfo };
use tensor_dyn::{ tensor_base::_Tensor, TensorCreator };
use tensor_types::convertion::{ Convertor, FromScalar };
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

#[track_caller]
fn assert_eq(
    a: &_Tensor<i64>,
    a_kernel: &_Tensor<i64>,
    b: &tch::Tensor,
    b_kernel: &tch::Tensor,
    block_size: [i64; 2]
) -> anyhow::Result<()> {
    let res = a
        .conv2d(
            &a_kernel,
            None,
            [1, 1],
            [
                (0, 0),
                (0, 0),
            ],
            [1, 1],
            |x| x
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
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 32, 3, 3, 3, 16, 16])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [3, 32])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1])?;

    // test when outwidth is less than regnum
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 32, 3, 3, 3, 5, 5])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [3, 32])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1])?;

    let (kernel, a, tch_kernel, tch_a) = common_input([1, 2, 3, 3, 3, 5, 5])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [3, 1])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1])?;
    Ok(())
}

#[test]
fn test_case1() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 32, 3, 3, 3, 20, 20])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [3, 32])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1])?;

    let (kernel, a, tch_kernel, tch_a) = common_input([1, 32, 3, 3, 3, 5, 5])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [3, 32])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1])?;
    Ok(())
}

#[test]
fn test_case2() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 32, 3, 3, 3, 20, 20])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [3, 20])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1])?;

    let (kernel, a, tch_kernel, tch_a) = common_input([1, 32, 3, 3, 3, 5, 5])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [3, 20])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1])?;
    Ok(())
}

#[test]
fn test_case3() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 32, 3, 3, 3, 20, 20])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [3, 28])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1])?;

    let (kernel, a, tch_kernel, tch_a) = common_input([1, 32, 3, 3, 3, 5, 5])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [3, 32])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1])?;
    Ok(())
}

#[test]
fn test_case4() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 32, 3, 3, 3, 20, 20])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [2, 32])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1])?;

    let (kernel, a, tch_kernel, tch_a) = common_input([1, 32, 3, 3, 3, 5, 5])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [3, 32])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1])?;
    Ok(())
}

#[test]
fn test_case5() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 32, 3, 3, 3, 19, 19])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [2, 32])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1])?;

    let (kernel, a, tch_kernel, tch_a) = common_input([1, 32, 3, 3, 3, 5, 5])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [3, 32])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1])?;
    Ok(())
}

#[test]
fn test_case6() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 30, 3, 3, 3, 20, 20])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [2, 28])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1])?;

    let (kernel, a, tch_kernel, tch_a) = common_input([1, 30, 3, 3, 3, 5, 5])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [3, 30])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1])?;
    Ok(())
}

#[test]
fn test_case7() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 30, 3, 3, 3, 20, 20])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [2, 28])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1])?;

    let (kernel, a, tch_kernel, tch_a) = common_input([1, 30, 3, 3, 3, 5, 5])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [3, 30])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 16])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, [1, 1])?;
    Ok(())
}
