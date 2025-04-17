#![allow(unused)]
use hpt::common::cpu::TensorLike;
use hpt::common::TensorInfo;
use hpt::ops::Contiguous;
use hpt::ops::Conv;
use hpt::ops::ShapeManipulate;
use hpt::ops::TensorCreator;
use hpt::Tensor;
use hpt_types::into_scalar::Cast;
use hpt_types::type_promote::NormalOut;
use hpt_types::type_promote::NormalOutUnary;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use tch;

pub(crate) type TestTypes = f32;
static TCH_TEST_TYPES: tch::Kind = tch::Kind::Float;
static TEST_RTOL: TestTypes = 1e-3;
static TEST_ATOL: TestTypes = 1e-3;
static EPSILON: TestTypes = 1e-5;

fn common_input(
    [batch, out_channel, in_channel, kernel_height, kernel_width, height, width, groups]: [i64; 8],
) -> anyhow::Result<(
    Tensor<TestTypes>,
    Tensor<TestTypes>,
    tch::Tensor,
    tch::Tensor,
)> {
    let kernel = Tensor::<TestTypes>::arange(
        0,
        (in_channel / groups) * out_channel * kernel_height * kernel_width,
    )?
    .reshape([
        out_channel,
        in_channel / groups,
        kernel_height,
        kernel_width,
    ])?
    .permute([2, 3, 1, 0])?
    .contiguous()?;
    let a = Tensor::<TestTypes>::arange(0, batch * in_channel * height * width)?
        .reshape([batch, in_channel, height, width])?
        .permute([0, 2, 3, 1])?
        .contiguous()?;

    let tch_kernel = tch::Tensor::arange(
        (in_channel / groups) * out_channel * kernel_height * kernel_width,
        (TCH_TEST_TYPES, tch::Device::Cpu),
    )
    .reshape(&[
        out_channel,
        in_channel / groups,
        kernel_height,
        kernel_width,
    ]);
    let tch_a = tch::Tensor::arange(
        batch * in_channel * height * width,
        (TCH_TEST_TYPES, tch::Device::Cpu),
    )
    .reshape(&[batch, in_channel, height, width]);
    Ok((kernel, a, tch_kernel, tch_a))
}

#[track_caller]
fn assert_eq(
    a: &Tensor<TestTypes>,
    a_kernel: &Tensor<TestTypes>,
    b: &tch::Tensor,
    b_kernel: &tch::Tensor,
    groups: i64,
) -> anyhow::Result<()> {
    let res = a
        .dwconv2d(
            &a_kernel,
            None,
            [1, 1],
            [(0, 0), (0, 0)],
            [1, 1],
            None,
            None,
        )?
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
    let tch_res = unsafe {
        Tensor::<TestTypes>::from_raw(res2.data_ptr() as *mut TestTypes, &res.shape().to_vec())
    }?;
    assert!(res.allclose(&tch_res, TEST_RTOL, TEST_ATOL));
    Ok(())
}

#[track_caller]
fn assert_eq_pad(
    a: &Tensor<TestTypes>,
    a_kernel: &Tensor<TestTypes>,
    b: &tch::Tensor,
    b_kernel: &tch::Tensor,
    groups: i64,
) -> anyhow::Result<()> {
    let res = a
        .dwconv2d(
            &a_kernel,
            None,
            [1, 1],
            [(2, 2), (2, 2)],
            [1, 1],
            None,
            None,
        )?
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
    let tch_res = unsafe {
        Tensor::<TestTypes>::from_raw(res2.data_ptr() as *mut TestTypes, &res.shape().to_vec())
    }?;
    assert!(res.allclose(&tch_res, TEST_RTOL, TEST_ATOL));
    Ok(())
}

#[track_caller]
fn assert_eq_bias(
    a: &Tensor<TestTypes>,
    a_kernel: &Tensor<TestTypes>,
    b: &tch::Tensor,
    b_kernel: &tch::Tensor,
    groups: i64,
) -> anyhow::Result<()> {
    let bias = Tensor::<TestTypes>::arange(0i64, *a_kernel.shape().last().unwrap())?;
    let res = a
        .dwconv2d(
            &a_kernel,
            Some(&bias),
            [1, 1],
            [(0, 0), (0, 0)],
            [1, 1],
            None,
            None,
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let tch_bias = tch::Tensor::arange(
        *a_kernel.shape().last().unwrap(),
        (TCH_TEST_TYPES, tch::Device::Cpu),
    );
    let res2 = b.conv2d(&b_kernel, Some(tch_bias), &[1, 1], &[0, 0], &[1, 1], groups);
    let tch_res = unsafe {
        Tensor::<TestTypes>::from_raw(res2.data_ptr() as *mut TestTypes, &res.shape().to_vec())
    }?;
    assert!(res.allclose(&tch_res, TEST_RTOL, TEST_ATOL));
    Ok(())
}

#[track_caller]
fn assert_eq_bias_pad(
    a: &Tensor<TestTypes>,
    a_kernel: &Tensor<TestTypes>,
    b: &tch::Tensor,
    b_kernel: &tch::Tensor,
    groups: i64,
) -> anyhow::Result<()> {
    let bias = Tensor::<TestTypes>::arange(0i64, *a_kernel.shape().last().unwrap())?;
    let res = a
        .dwconv2d(
            &a_kernel,
            Some(&bias),
            [1, 1],
            [(2, 2), (2, 2)],
            [1, 1],
            None,
            None,
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let tch_bias = tch::Tensor::arange(b_kernel.size()[0], (TCH_TEST_TYPES, tch::Device::Cpu));
    let res2 = b.conv2d(&b_kernel, Some(tch_bias), &[1, 1], &[2, 2], &[1, 1], groups);
    let tch_res = unsafe {
        Tensor::<TestTypes>::from_raw(res2.data_ptr() as *mut TestTypes, &res.shape().to_vec())
    }?;
    assert!(res.allclose(&tch_res, TEST_RTOL, TEST_ATOL));
    Ok(())
}

#[track_caller]
fn assert_eq_bias_pad_relu(
    a: &Tensor<TestTypes>,
    a_kernel: &Tensor<TestTypes>,
    b: &tch::Tensor,
    b_kernel: &tch::Tensor,
    groups: i64,
) -> anyhow::Result<()> {
    let bias = Tensor::<TestTypes>::arange(0i64, *a_kernel.shape().last().unwrap())?;
    let res = a
        .dwconv2d(
            &a_kernel,
            Some(&bias),
            [1, 1],
            [(2, 2), (2, 2)],
            [1, 1],
            Some(|x| x._relu()),
            Some(|x| x._relu()),
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let tch_bias = tch::Tensor::arange(b_kernel.size()[0], (TCH_TEST_TYPES, tch::Device::Cpu));
    let res2 = b
        .conv2d(&b_kernel, Some(tch_bias), &[1, 1], &[2, 2], &[1, 1], groups)
        .relu();
    let tch_res = unsafe {
        Tensor::<TestTypes>::from_raw(res2.data_ptr() as *mut TestTypes, &res.shape().to_vec())
    }?;
    assert!(res.allclose(&tch_res, TEST_RTOL, TEST_ATOL));
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
