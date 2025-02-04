#![allow(unused)]
use rayon::iter::{ IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator };
use tch;
use tensor_dyn::NormalOutUnary;
use tensor_dyn::ShapeManipulate;
use tensor_dyn::TensorLike;
use tensor_dyn::{ set_global_display_lr_elements, set_num_threads, CommonBounds, TensorInfo };
use tensor_dyn::{ Tensor, TensorCreator };
use tensor_types::cast::Cast;
use tensor_types::type_promote::NormalOut;

fn common_input([
    batch,
    out_channel,
    in_channel,
    kernel_height,
    kernel_width,
    height,
    width,
    groups,
]: [i64; 8])
    -> anyhow::Result<(Tensor<i64>, Tensor<i64>, tch::Tensor, tch::Tensor)>
{
    let kernel = Tensor::<i64>
        ::arange(0, (in_channel / groups) * out_channel * kernel_height * kernel_width)?
        .reshape([out_channel, in_channel / groups, kernel_height, kernel_width])?
        .permute([2, 3, 1, 0])?
        .contiguous()?;
    let a = Tensor::<i64>
        ::arange(0, batch * in_channel * height * width)?
        .reshape([batch, in_channel, height, width])?
        .permute([0, 2, 3, 1])?
        .contiguous()?;

    let tch_kernel = tch::Tensor
        ::arange((in_channel / groups) * out_channel * kernel_height * kernel_width, (
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
    a: &Tensor<i64>,
    a_kernel: &Tensor<i64>,
    b: &tch::Tensor,
    b_kernel: &tch::Tensor,
    groups: i64
) -> anyhow::Result<()> {
    let res = a
        .dwconv2d(
            &a_kernel,
            None,
            [1, 1],
            [
                (0, 0),
                (0, 0),
            ],
            [1, 1],
            None
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let res2 = b.conv2d(&b_kernel, None::<tch::Tensor>, &[1, 1], &[0, 0], &[1, 1], groups);
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
    a: &Tensor<i64>,
    a_kernel: &Tensor<i64>,
    b: &tch::Tensor,
    b_kernel: &tch::Tensor,
    groups: i64
) -> anyhow::Result<()> {
    let res = a
        .dwconv2d(
            &a_kernel,
            None,
            [1, 1],
            [
                (2, 2),
                (2, 2),
            ],
            [1, 1],
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
    a: &Tensor<i64>,
    a_kernel: &Tensor<i64>,
    b: &tch::Tensor,
    b_kernel: &tch::Tensor,
    groups: i64
) -> anyhow::Result<()> {
    let bias = Tensor::<i64>::arange(0i64, *a_kernel.shape().last().unwrap())?;
    let res = a
        .dwconv2d(
            &a_kernel,
            Some(&bias),
            [1, 1],
            [
                (0, 0),
                (0, 0),
            ],
            [1, 1],
            None
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let tch_bias = tch::Tensor::arange(*a_kernel.shape().last().unwrap(), (tch::Kind::Int64, tch::Device::Cpu));
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
    a: &Tensor<i64>,
    a_kernel: &Tensor<i64>,
    b: &tch::Tensor,
    b_kernel: &tch::Tensor,
    groups: i64
) -> anyhow::Result<()> {
    let bias = Tensor::<i64>::arange(0i64, *a_kernel.shape().last().unwrap())?;
    let res = a
        .dwconv2d(
            &a_kernel,
            Some(&bias),
            [1, 1],
            [
                (2, 2),
                (2, 2),
            ],
            [1, 1],
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
    a: &Tensor<i64>,
    a_kernel: &Tensor<i64>,
    b: &tch::Tensor,
    b_kernel: &tch::Tensor,
    groups: i64
) -> anyhow::Result<()> {
    let bias = Tensor::<i64>::arange(0i64, *a_kernel.shape().last().unwrap())?;
    let res = a
        .dwconv2d(
            &a_kernel,
            Some(&bias),
            [1, 1],
            [
                (2, 2),
                (2, 2),
            ],
            [1, 1],
            Some(|x| x._relu())
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let tch_bias = tch::Tensor::arange(b_kernel.size()[0], (tch::Kind::Int64, tch::Device::Cpu));
    let res2 = b.conv2d(&b_kernel, Some(tch_bias), &[1, 1], &[2, 2], &[1, 1], groups).relu();
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