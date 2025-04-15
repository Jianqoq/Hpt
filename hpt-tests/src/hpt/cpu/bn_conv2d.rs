#![allow(unused)]
use super::assert_utils::assert_f32;
use super::assert_utils::assert_f64;
use hpt::common::cpu::TensorLike;
use hpt::common::TensorInfo;
use hpt::ops::Contiguous;
use hpt::ops::ConvBatchNorm;
use hpt::ops::ShapeManipulate;
use hpt::ops::TensorCreator;
use hpt::Tensor;
use hpt_types::into_scalar::Cast;
use hpt_types::type_promote::NormalOut;
use hpt_types::type_promote::NormalOutUnary;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use tch;
use hpt::ops::Conv;
use crate::TestTypes;
use crate::TCH_TEST_TYPES;
use crate::TEST_ATOL;
use crate::TEST_RTOL;


fn common_input(
    [batch, out_channel, in_channel, kernel_height, kernel_width, height, width]: [i64; 7],
) -> anyhow::Result<(Tensor<TestTypes>, Tensor<TestTypes>, tch::Tensor, tch::Tensor)> {
    let kernel = Tensor::<TestTypes>::arange(0, in_channel * out_channel * kernel_height * kernel_width)?
        .reshape([out_channel, in_channel, kernel_height, kernel_width])?
        .permute([2, 3, 1, 0])?
        .contiguous()?;
    let a = Tensor::<TestTypes>::arange(0, batch * in_channel * height * width)?
        .reshape([batch, in_channel, height, width])?
        .permute([0, 2, 3, 1])?
        .contiguous()?;

    let tch_kernel = tch::Tensor::arange(
        in_channel * out_channel * kernel_height * kernel_width,
        (TCH_TEST_TYPES, tch::Device::Cpu),
    )
    .reshape(&[out_channel, in_channel, kernel_height, kernel_width]);
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
) -> anyhow::Result<()> {
    let oc = *a_kernel.shape().last().unwrap();
    let mean =
        tch::Tensor::arange(oc, (TCH_TEST_TYPES, tch::Device::Cpu)).reshape(&[1, oc, 1, 1]);
    let var = tch::Tensor::arange(oc, (TCH_TEST_TYPES, tch::Device::Cpu)).reshape(&[1, oc, 1, 1]);
    let gamma =
        tch::Tensor::arange(oc, (TCH_TEST_TYPES, tch::Device::Cpu)).reshape(&[1, oc, 1, 1]);
    let beta =
        tch::Tensor::arange(oc, (TCH_TEST_TYPES, tch::Device::Cpu)).reshape(&[1, oc, 1, 1]);
    let hpt_mean = Tensor::<TestTypes>::arange(0, oc)?;
    let hpt_var = Tensor::<TestTypes>::arange(0, oc)?;
    let hpt_gamma = Tensor::<TestTypes>::arange(0, oc)?;
    let hpt_beta = Tensor::<TestTypes>::arange(0, oc)?;
    let res = a
        .batchnorm_conv2d(
            &a_kernel,
            &hpt_mean,
            &hpt_var,
            &hpt_gamma,
            &hpt_beta,
            None,
            half::f16::from_f32_const(1e-5),
            [1, 1],
            [(0, 0), (0, 0)],
            [1, 1],
            None,
            None,
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let res2 = b.conv2d(&b_kernel, None::<tch::Tensor>, &[1, 1], &[0, 0], &[1, 1], 1);
    let tch_res = gamma * ((res2 - mean) / (var + 1e-5).sqrt()) + beta;
    let tch_res = unsafe {
        Tensor::<TestTypes>::from_raw(tch_res.data_ptr() as *mut TestTypes, &res.shape().to_vec())
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
) -> anyhow::Result<()> {
    let oc = *a_kernel.shape().last().unwrap();
    let mean =
        tch::Tensor::arange(oc, (TCH_TEST_TYPES, tch::Device::Cpu)).reshape(&[1, oc, 1, 1]);
    let var = tch::Tensor::arange(oc, (TCH_TEST_TYPES, tch::Device::Cpu)).reshape(&[1, oc, 1, 1]);
    let gamma =
        tch::Tensor::arange(oc, (TCH_TEST_TYPES, tch::Device::Cpu)).reshape(&[1, oc, 1, 1]);
    let beta =
        tch::Tensor::arange(oc, (TCH_TEST_TYPES, tch::Device::Cpu)).reshape(&[1, oc, 1, 1]);
    let hpt_mean = Tensor::<TestTypes>::arange(0, oc)?;
    let hpt_var = Tensor::<TestTypes>::arange(0, oc)?;
    let hpt_gamma = Tensor::<TestTypes>::arange(0, oc)?;
    let hpt_beta = Tensor::<TestTypes>::arange(0, oc)?;
    let res = a
        .batchnorm_conv2d(
            &a_kernel,
            &hpt_mean,
            &hpt_var,
            &hpt_gamma,
            &hpt_beta,
            None,
            half::f16::from_f32_const(1e-5),
            [1, 1],
            [(2, 2), (2, 2)],
            [1, 1],
            None,
            None,
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let res2 = b.conv2d(&b_kernel, None::<tch::Tensor>, &[1, 1], &[2, 2], &[1, 1], 1);
    let tch_res = gamma * ((res2 - mean) / (var + 1e-5).sqrt()) + beta;
    let tch_res = unsafe {
        Tensor::<TestTypes>::from_raw(tch_res.data_ptr() as *mut TestTypes, &res.shape().to_vec())
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
) -> anyhow::Result<()> {
    let oc = *a_kernel.shape().last().unwrap();
    let mean =
        tch::Tensor::arange(oc, (TCH_TEST_TYPES, tch::Device::Cpu)).reshape(&[1, oc, 1, 1]);
    let var = tch::Tensor::arange(oc, (TCH_TEST_TYPES, tch::Device::Cpu)).reshape(&[1, oc, 1, 1]);
    let gamma =
        tch::Tensor::arange(oc, (TCH_TEST_TYPES, tch::Device::Cpu)).reshape(&[1, oc, 1, 1]);
    let beta =
        tch::Tensor::arange(oc, (TCH_TEST_TYPES, tch::Device::Cpu)).reshape(&[1, oc, 1, 1]);
    let hpt_mean = Tensor::<TestTypes>::arange(0, oc)?;
    let hpt_var = Tensor::<TestTypes>::arange(0, oc)?;
    let hpt_gamma = Tensor::<TestTypes>::arange(0, oc)?;
    let hpt_beta = Tensor::<TestTypes>::arange(0, oc)?;
    let bias = Tensor::<TestTypes>::arange(0i64, oc)?;
    let res = a
        .batchnorm_conv2d(
            &a_kernel,
            &hpt_mean,
            &hpt_var,
            &hpt_gamma,
            &hpt_beta,
            Some(&bias),
            half::f16::from_f32_const(1e-5),
            [1, 1],
            [(0, 0), (0, 0)],
            [1, 1],
            None,
            None,
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let tch_bias = tch::Tensor::arange(b_kernel.size()[0], (TCH_TEST_TYPES, tch::Device::Cpu));
    let res2 = b.conv2d(&b_kernel, Some(tch_bias), &[1, 1], &[0, 0], &[1, 1], 1);
    let tch_res = gamma * ((res2 - mean) / (var + 1e-5).sqrt()) + beta;
    let tch_res = unsafe {
        Tensor::<TestTypes>::from_raw(tch_res.data_ptr() as *mut TestTypes, &res.shape().to_vec())
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
) -> anyhow::Result<()> {
    let oc = *a_kernel.shape().last().unwrap();
    let mean =
        tch::Tensor::arange(oc, (TCH_TEST_TYPES, tch::Device::Cpu)).reshape(&[1, oc, 1, 1]);
    let var = tch::Tensor::arange(oc, (TCH_TEST_TYPES, tch::Device::Cpu)).reshape(&[1, oc, 1, 1]);
    let gamma =
        tch::Tensor::arange(oc, (TCH_TEST_TYPES, tch::Device::Cpu)).reshape(&[1, oc, 1, 1]);
    let beta =
        tch::Tensor::arange(oc, (TCH_TEST_TYPES, tch::Device::Cpu)).reshape(&[1, oc, 1, 1]);
    let hpt_mean = Tensor::<TestTypes>::arange(0, oc)?;
    let hpt_var = Tensor::<TestTypes>::arange(0, oc)?;
    let hpt_gamma = Tensor::<TestTypes>::arange(0, oc)?;
    let hpt_beta = Tensor::<TestTypes>::arange(0, oc)?;
    let bias = Tensor::<TestTypes>::arange(0i64, oc)?;
    let res = a
        .batchnorm_conv2d(
            &a_kernel,
            &hpt_mean,
            &hpt_var,
            &hpt_gamma,
            &hpt_beta,
            Some(&bias),
            half::f16::from_f32_const(1e-5),
            [1, 1],
            [(2, 2), (2, 2)],
            [1, 1],
            None,
            None,
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let tch_bias = tch::Tensor::arange(b_kernel.size()[0], (TCH_TEST_TYPES, tch::Device::Cpu));
    let res2 = b.conv2d(&b_kernel, Some(tch_bias), &[1, 1], &[2, 2], &[1, 1], 1);
    let tch_res = gamma * ((res2 - mean) / (var + 1e-5).sqrt()) + beta;
    let tch_res = unsafe {
        Tensor::<TestTypes>::from_raw(tch_res.data_ptr() as *mut TestTypes, &res.shape().to_vec())
    }?;
    assert!(res.allclose(&tch_res, TEST_RTOL, TEST_ATOL));
    Ok(())
}

#[track_caller]
fn assert_eq_bias_pad_relu6(
    a: &Tensor<TestTypes>,
    a_kernel: &Tensor<TestTypes>,
    b: &tch::Tensor,
    b_kernel: &tch::Tensor,
) -> anyhow::Result<()> {
    let oc = *a_kernel.shape().last().unwrap();
    let mean =
        tch::Tensor::arange(oc, (TCH_TEST_TYPES, tch::Device::Cpu)).reshape(&[1, oc, 1, 1]);
    let var = tch::Tensor::arange(oc, (TCH_TEST_TYPES, tch::Device::Cpu)).reshape(&[1, oc, 1, 1]);
    let gamma =
        tch::Tensor::arange(oc, (TCH_TEST_TYPES, tch::Device::Cpu)).reshape(&[1, oc, 1, 1]);
    let beta =
        tch::Tensor::arange(oc, (TCH_TEST_TYPES, tch::Device::Cpu)).reshape(&[1, oc, 1, 1]);
    let hpt_mean = Tensor::<TestTypes>::arange(0, oc)?;
    let hpt_var = Tensor::<TestTypes>::arange(0, oc)?;
    let hpt_gamma = Tensor::<TestTypes>::arange(0, oc)?;
    let hpt_beta = Tensor::<TestTypes>::arange(0, oc)?;
    let bias = Tensor::<TestTypes>::arange(0i64, *a_kernel.shape().last().unwrap())?;
    let res = a
        .batchnorm_conv2d(
            &a_kernel,
            &hpt_mean,
            &hpt_var,
            &hpt_gamma,
            &hpt_beta,
            Some(&bias),
            half::f16::from_f32_const(1e-5),
            [1, 1],
            [(2, 2), (2, 2)],
            [1, 1],
            Some(|x| x._relu6()),
            Some(|x| x._relu6()),
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let tch_bias = tch::Tensor::arange(b_kernel.size()[0], (TCH_TEST_TYPES, tch::Device::Cpu));
    let res2 = b.conv2d(&b_kernel, Some(tch_bias), &[1, 1], &[2, 2], &[1, 1], 1);
    let tch_res = (gamma * ((res2 - mean) / (var + 1e-5).sqrt()) + beta).relu6();
    let tch_res = unsafe {
        Tensor::<TestTypes>::from_raw(tch_res.data_ptr() as *mut TestTypes, &res.shape().to_vec())
    }?;
    assert!(res.allclose(&tch_res, TEST_RTOL, TEST_ATOL));
    Ok(())
}

#[test]
fn test_case0() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 32, 6, 3, 3, 16, 16])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad_relu6(&a, &kernel, &tch_a, &tch_kernel)?;
    // test when outwidth is less than regnum
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 1, 2, 3, 3, 5, 5])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad_relu6(&a, &kernel, &tch_a, &tch_kernel)?;
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 2, 6, 3, 3, 5, 5])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad_relu6(&a, &kernel, &tch_a, &tch_kernel)?;

    Ok(())
}

#[test]
fn test_case1() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 32, 6, 3, 3, 20, 20])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad_relu6(&a, &kernel, &tch_a, &tch_kernel)?;
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 32, 6, 3, 3, 5, 5])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad_relu6(&a, &kernel, &tch_a, &tch_kernel)?;
    Ok(())
}

#[test]
fn test_case5() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 32, 6, 3, 3, 19, 19])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad_relu6(&a, &kernel, &tch_a, &tch_kernel)?;
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 32, 6, 3, 3, 5, 5])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad_relu6(&a, &kernel, &tch_a, &tch_kernel)?;
    Ok(())
}

#[test]
fn test_case6() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 30, 6, 3, 3, 20, 20])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad_relu6(&a, &kernel, &tch_a, &tch_kernel)?;
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 30, 6, 3, 3, 5, 5])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad_relu6(&a, &kernel, &tch_a, &tch_kernel)?;
    Ok(())
}

#[test]
fn test_case8() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 32, 3, 1, 1, 16, 16])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad_relu6(&a, &kernel, &tch_a, &tch_kernel)?;
    // test when outwidth is less than regnum
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 1, 1, 1, 1, 5, 5])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad_relu6(&a, &kernel, &tch_a, &tch_kernel)?;
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 2, 3, 1, 1, 5, 5])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad_relu6(&a, &kernel, &tch_a, &tch_kernel)?;

    Ok(())
}

#[test]
fn test_case9() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 32, 3, 1, 1, 20, 20])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad_relu6(&a, &kernel, &tch_a, &tch_kernel)?;
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 32, 3, 1, 1, 5, 5])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad_relu6(&a, &kernel, &tch_a, &tch_kernel)?;
    Ok(())
}

#[test]
fn test_case13() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 32, 3, 1, 1, 19, 19])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad_relu6(&a, &kernel, &tch_a, &tch_kernel)?;
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 32, 3, 1, 1, 5, 5])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad_relu6(&a, &kernel, &tch_a, &tch_kernel)?;
    Ok(())
}

#[test]
fn test_case14() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 30, 3, 1, 1, 20, 20])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad_relu6(&a, &kernel, &tch_a, &tch_kernel)?;
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 30, 3, 1, 1, 5, 5])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_bias_pad_relu6(&a, &kernel, &tch_a, &tch_kernel)?;
    Ok(())
}
