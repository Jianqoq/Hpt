#![allow(unused)]
use hpt::common::cpu::TensorLike;
use hpt::common::TensorInfo;
use hpt::ops::Contiguous;
use hpt::ops::CudaConv;
use hpt::ops::ShapeManipulate;
use hpt::ops::TensorCreator;
use hpt::Tensor;
use hpt_types::into_scalar::Cast;
use hpt_types::type_promote::NormalOut;
use hpt_types::type_promote::NormalOutUnary;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use tch;

type Type = f64;
const TCH_TYPE: tch::Kind = tch::Kind::Double;

fn common_input(
    [batch, out_channel, in_channel, kernel_height, kernel_width, height, width, groups]: [i64; 8],
) -> anyhow::Result<(Tensor<Type>, Tensor<Type>, tch::Tensor, tch::Tensor)> {
    let kernel = Tensor::<Type>::arange(
        0,
        (in_channel / groups) * out_channel * kernel_height * kernel_width,
    )?
    .reshape([
        out_channel,
        in_channel / groups,
        kernel_height,
        kernel_width,
    ])?
    .permute([0, 2, 3, 1])?
    .contiguous()?;
    let a = Tensor::<Type>::arange(0, batch * in_channel * height * width)?
        .reshape([batch, in_channel, height, width])?
        .permute([0, 2, 3, 1])?
        .contiguous()?;

    let tch_kernel = tch::Tensor::arange(
        (in_channel / groups) * out_channel * kernel_height * kernel_width,
        (TCH_TYPE, tch::Device::Cpu),
    )
    .reshape(&[
        out_channel,
        in_channel / groups,
        kernel_height,
        kernel_width,
    ]);
    let tch_a = tch::Tensor::arange(
        batch * in_channel * height * width,
        (TCH_TYPE, tch::Device::Cpu),
    )
    .reshape(&[batch, in_channel, height, width]);
    Ok((kernel, a, tch_kernel, tch_a))
}

#[track_caller]
fn assert_eq(
    a: &Tensor<Type>,
    a_kernel: &Tensor<Type>,
    b: &tch::Tensor,
    b_kernel: &tch::Tensor,
    groups: i64,
) -> anyhow::Result<()> {
    let a = a.to_cuda::<0>()?;
    let a_kernel = a_kernel.to_cuda::<0>()?;
    let res = a
        .dwconv2d(&a_kernel, None, [1, 1], [0, 0], [1, 1], None)?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let res2 = b.conv2d(
        &b_kernel,
        None::<tch::Tensor>,
        &[1, 1],
        &[0, 0],
        &[1, 1],
        groups,
    );
    let res_cpu = res.to_cpu::<0>()?;
    let res_slice = res_cpu.as_raw();
    let res2 = unsafe { std::slice::from_raw_parts(res2.data_ptr() as *const Type, res.size()) };
    res_slice.iter().zip(res2.iter()).for_each(|(a, b)| {
        if a != b {
            assert_eq!(a, b);
        }
    });
    Ok(())
}

#[track_caller]
fn assert_eq_pad(
    a: &Tensor<Type>,
    a_kernel: &Tensor<Type>,
    b: &tch::Tensor,
    b_kernel: &tch::Tensor,
    groups: i64,
) -> anyhow::Result<()> {
    let a = a.to_cuda::<0>()?;
    let a_kernel = a_kernel.to_cuda::<0>()?;
    let res = a
        .dwconv2d(&a_kernel, None, [1, 1], [2, 2], [1, 1], None)?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let res2 = b.conv2d(
        &b_kernel,
        None::<tch::Tensor>,
        &[1, 1],
        &[2, 2],
        &[1, 1],
        groups,
    );
    let res_cpu = res.to_cpu::<0>()?;
    let res_slice = res_cpu.as_raw();
    let res2 = unsafe { std::slice::from_raw_parts(res2.data_ptr() as *const Type, res.size()) };
    res_slice.iter().zip(res2.iter()).for_each(|(a, b)| {
        if a != b {
            assert_eq!(a, b);
        }
    });
    Ok(())
}

#[track_caller]
fn assert_eq_bias(
    a: &Tensor<Type>,
    a_kernel: &Tensor<Type>,
    b: &tch::Tensor,
    b_kernel: &tch::Tensor,
    groups: i64,
) -> anyhow::Result<()> {
    let bias = Tensor::<Type>::arange(0i64, *a_kernel.shape().first().unwrap())?;
    let a = a.to_cuda::<0>()?;
    let a_kernel = a_kernel.to_cuda::<0>()?;
    let res = a
        .dwconv2d(
            &a_kernel,
            Some(&bias.to_cuda::<0>()?),
            [1, 1],
            [0, 0],
            [1, 1],
            None,
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let tch_bias = tch::Tensor::arange(
        *a_kernel.shape().first().unwrap(),
        (TCH_TYPE, tch::Device::Cpu),
    );
    let res2 = b.conv2d(&b_kernel, Some(tch_bias), &[1, 1], &[0, 0], &[1, 1], groups);
    let res_cpu = res.to_cpu::<0>()?;
    let res_slice = res_cpu.as_raw();
    let res2 = unsafe { std::slice::from_raw_parts(res2.data_ptr() as *const Type, res.size()) };
    res_slice.iter().zip(res2.iter()).for_each(|(a, b)| {
        if a != b {
            assert_eq!(a, b);
        }
    });
    Ok(())
}

#[track_caller]
fn assert_eq_bias_pad(
    a: &Tensor<Type>,
    a_kernel: &Tensor<Type>,
    b: &tch::Tensor,
    b_kernel: &tch::Tensor,
    groups: i64,
) -> anyhow::Result<()> {
    let bias = Tensor::<Type>::arange(0i64, *a_kernel.shape().first().unwrap())?;
    let a = a.to_cuda::<0>()?;
    let a_kernel = a_kernel.to_cuda::<0>()?;
    let res = a
        .dwconv2d(
            &a_kernel,
            Some(&bias.to_cuda::<0>()?),
            [1, 1],
            [2, 2],
            [1, 1],
            None,
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let tch_bias = tch::Tensor::arange(b_kernel.size()[0], (TCH_TYPE, tch::Device::Cpu));
    let res2 = b.conv2d(&b_kernel, Some(tch_bias), &[1, 1], &[2, 2], &[1, 1], groups);
    let res_cpu = res.to_cpu::<0>()?;
    let res_slice = res_cpu.as_raw();
    let res2 = unsafe { std::slice::from_raw_parts(res2.data_ptr() as *const Type, res.size()) };
    res_slice.iter().zip(res2.iter()).for_each(|(a, b)| {
        if a != b {
            assert_eq!(a, b);
        }
    });
    Ok(())
}

#[test]
fn test_case0() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 3, 3, 3, 3, 4, 4, 3])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel, 3)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel, 3)?;

    // test when outwidth is less than regnum
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 8, 8, 3, 3, 5, 5, 8])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, 8)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel, 8)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel, 8)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel, 8)?;
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 16, 16, 3, 3, 5, 5, 16])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, 16)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel, 16)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel, 16)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel, 16)?;

    Ok(())
}

#[test]
fn test_case1() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 33, 33, 3, 3, 3, 3, 33])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, 33)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel, 33)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel, 33)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel, 33)?;
    Ok(())
}

#[test]
fn test_case2() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 80, 80, 3, 3, 20, 20, 80])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, 80)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel, 80)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel, 80)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel, 80)?;
    Ok(())
}

#[test]
fn test_case5() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 19, 19, 3, 3, 19, 19, 19])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, 19)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel, 19)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel, 19)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel, 19)?;
    Ok(())
}

#[test]
fn test_case6() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 30, 30, 3, 3, 20, 20, 30])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel, 30)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel, 30)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel, 30)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel, 30)?;
    Ok(())
}
