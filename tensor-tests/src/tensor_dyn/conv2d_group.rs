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

fn common_input<T>([batch, out_channel, in_channel, kernel_height, kernel_width, height, width, groups]: [
    i64;
    8
])
    -> anyhow::Result<(_Tensor<T>, _Tensor<T>, tch::Tensor, tch::Tensor)>
    where
        T: Convertor + FromScalar<i64> + NormalOut<T, Output = T> + CommonBounds,
        usize: IntoScalar<T>,
        i64: IntoScalar<T>
{
    let kernel = _Tensor::<T>
        ::arange(0, in_channel / groups * out_channel * kernel_height * kernel_width)?
        .reshape([out_channel, in_channel / groups, kernel_height, kernel_width])?
        .permute([2, 3, 1, 0])?
        .contiguous()?;
    let a = _Tensor::<T>
        ::arange(0, batch * in_channel * height * width)?
        .reshape([batch, in_channel, height, width])?
        .permute([0, 2, 3, 1])?
        .contiguous()?;

    let tch_kernel = tch::Tensor
        ::arange(in_channel / groups * out_channel * kernel_height * kernel_width, (
            tch::Kind::Int64,
            tch::Device::Cpu,
        ))
        .reshape(&[out_channel, in_channel / groups, kernel_height, kernel_width]);
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
    groups: i64
) -> anyhow::Result<()> {
    let res = a
        .conv2d_group(
            &a_kernel,
            None,
            [1, 1],
            [
                (0, 0),
                (0, 0),
            ],
            [1, 1],
            groups,
            None
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let res2 = b.conv2d(&b_kernel, None::<tch::Tensor>, &[1, 1], &[0, 0], &[1, 1], groups);
    // println!("{}", res);
    // println!("{}", res2);
    let res_slice = res.as_raw();
    let res2 = unsafe { std::slice::from_raw_parts(res2.data_ptr() as *const i64, res.size()) };
    res_slice
        .iter()
        .zip(res2.iter())
        .for_each(|(a, b)| {
            if a != b {
                assert_eq!(a, b);
            }
        });
    Ok(())
}

#[track_caller]
fn assert_eq_pad(
    a: &_Tensor<i64>,
    a_kernel: &_Tensor<i64>,
    b: &tch::Tensor,
    b_kernel: &tch::Tensor,
    groups: i64
) -> anyhow::Result<()> {
    let res = a
        .conv2d_group(
            &a_kernel,
            None,
            [1, 1],
            [
                (2, 2),
                (2, 2),
            ],
            [1, 1],
            groups,
            None
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let res2 = b.conv2d(&b_kernel, None::<tch::Tensor>, &[1, 1], &[2, 2], &[1, 1], groups);
    let res_slice = res.as_raw();
    let res2 = unsafe { std::slice::from_raw_parts(res2.data_ptr() as *const i64, res.size()) };
    res_slice
        .iter()
        .zip(res2.iter())
        .for_each(|(a, b)| {
            if a != b {
                assert_eq!(a, b);
            }
        });
    Ok(())
}

#[track_caller]
fn assert_eq_bias(
    a: &_Tensor<i64>,
    a_kernel: &_Tensor<i64>,
    b: &tch::Tensor,
    b_kernel: &tch::Tensor,
    groups: i64
) -> anyhow::Result<()> {
    let bias = _Tensor::<i64>::arange(0i64, *a_kernel.shape().last().unwrap())?;
    let res = a
        .conv2d_group(
            &a_kernel,
            Some(&bias),
            [1, 1],
            [
                (0, 0),
                (0, 0),
            ],
            [1, 1],
            groups,
            None
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let tch_bias = tch::Tensor::arange(b_kernel.size()[0], (tch::Kind::Int64, tch::Device::Cpu));
    let res2 = b.conv2d(&b_kernel, Some(tch_bias), &[1, 1], &[0, 0], &[1, 1], groups);
    let res_slice = res.as_raw();
    let res2 = unsafe { std::slice::from_raw_parts(res2.data_ptr() as *const i64, res.size()) };
    res_slice
        .iter()
        .zip(res2.iter())
        .for_each(|(a, b)| {
            if a != b {
                assert_eq!(a, b);
            }
        });
    Ok(())
}

#[track_caller]
fn assert_eq_bias_pad(
    a: &_Tensor<i64>,
    a_kernel: &_Tensor<i64>,
    b: &tch::Tensor,
    b_kernel: &tch::Tensor,
    groups: i64
) -> anyhow::Result<()> {
    let bias = _Tensor::<i64>::arange(0i64, *a_kernel.shape().last().unwrap())?;
    let res = a
        .conv2d_group(
            &a_kernel,
            Some(&bias),
            [1, 1],
            [
                (2, 2),
                (2, 2),
            ],
            [1, 1],
            groups,
            None
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let tch_bias = tch::Tensor::arange(b_kernel.size()[0], (tch::Kind::Int64, tch::Device::Cpu));
    let res2 = b.conv2d(&b_kernel, Some(tch_bias), &[1, 1], &[2, 2], &[1, 1], groups);
    let res_slice = res.as_raw();
    let res2 = unsafe { std::slice::from_raw_parts(res2.data_ptr() as *const i64, res.size()) };
    res_slice
        .iter()
        .zip(res2.iter())
        .for_each(|(a, b)| {
            if a != b {
                assert_eq!(a, b);
            }
        });
    Ok(())
}

#[track_caller]
fn assert_eq_bias_pad_relu(
    a: &_Tensor<i64>,
    a_kernel: &_Tensor<i64>,
    b: &tch::Tensor,
    b_kernel: &tch::Tensor,
    groups: i64
) -> anyhow::Result<()> {
    let bias = _Tensor::<i64>::arange(0i64, *a_kernel.shape().last().unwrap())?;
    let res = a
        .conv2d_group(
            &a_kernel,
            Some(&bias),
            [1, 1],
            [
                (2, 2),
                (2, 2),
            ],
            [1, 1],
            groups,
            None
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let tch_bias = tch::Tensor::arange(b_kernel.size()[0], (tch::Kind::Int64, tch::Device::Cpu));
    let res2 = b.conv2d(&b_kernel, Some(tch_bias), &[1, 1], &[2, 2], &[1, 1], groups);
    let res_slice = res.as_raw();
    let res2 = unsafe { std::slice::from_raw_parts(res2.data_ptr() as *const i64, res.size()) };
    res_slice
        .iter()
        .zip(res2.iter())
        .for_each(|(a, b)| {
            if a != b {
                assert_eq!(a, b);
            }
        });
    Ok(())
}

#[test]
fn test_case0() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 6, 3, 3, 3, 4, 4, 3])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel, 3)?;

    // test when outwidth is less than regnum
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 33, 3, 3, 3, 5, 5, 3])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    // assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel, 1)?;
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 15, 3, 3, 3, 5, 5, 3])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel, 3)?;

    Ok(())
}

#[test]
fn test_case1() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 33, 3, 3, 3, 20, 20, 3])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel, 3)?;

    let (kernel, a, tch_kernel, tch_a) = common_input([1, 33, 3, 3, 3, 5, 5, 3])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    Ok(())
}

#[test]
fn test_case2() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 80, 8, 3, 3, 20, 20, 8])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, 8)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel, 8)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel, 8)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel, 8)?;
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 80, 8, 3, 3, 5, 5, 8])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, 8)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel, 8)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel, 8)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel, 8)?;
    Ok(())
}

#[test]
fn test_case3() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 80, 8, 3, 3, 20, 20, 8])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, 8)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel, 8)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel, 8)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel, 8)?;

    let (kernel, a, tch_kernel, tch_a) = common_input([1, 80, 8, 3, 3, 5, 5, 8])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, 8)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel, 8)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel, 8)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel, 8)?;
    Ok(())
}

#[test]
fn test_case4() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 33, 3, 3, 3, 20, 20, 3])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel, 3)?;

    let (kernel, a, tch_kernel, tch_a) = common_input([1, 33, 3, 3, 3, 5, 5, 3])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    Ok(())
}

#[test]
fn test_case5() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 33, 3, 3, 3, 19, 19, 3])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel, 3)?;

    let (kernel, a, tch_kernel, tch_a) = common_input([1, 33, 3, 3, 3, 5, 5, 3])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    Ok(())
}

#[test]
fn test_case6() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 30, 3, 3, 3, 20, 20, 3])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel, 3)?;

    let (kernel, a, tch_kernel, tch_a) = common_input([1, 30, 3, 3, 3, 5, 5, 3])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    Ok(())
}

#[test]
fn test_case7() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 30, 3, 3, 3, 20, 20, 3])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel, 3)?;

    let (kernel, a, tch_kernel, tch_a) = common_input([1, 30, 3, 3, 3, 5, 5, 3])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    Ok(())
}
