#![allow(unused)]
use hpt::common::cpu::TensorLike;
use hpt::common::TensorInfo;
use hpt::ops::Contiguous;
use hpt::ops::Conv;
use hpt::ops::CudaConv;
use hpt::ops::ShapeManipulate;
use hpt::ops::TensorCreator;
use hpt::Tensor;
use hpt_types::into_scalar::Cast;
use hpt_types::type_promote::NormalOut;
use hpt_types::type_promote::NormalOutUnary;
use rand::Rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use tch;

use super::assert_utils::assert_f64;

type Type = f64;
const TCH_TYPE: tch::Kind = tch::Kind::Double;

fn common_input(
    [in_channel, out_channel, kernel_height, kernel_width, height, width]: [i64; 6],
) -> anyhow::Result<(Tensor<Type>, Tensor<Type>, tch::Tensor, tch::Tensor)> {
    let batch = 1;
    let tch_kernel = tch::Tensor::randn(
        [out_channel, in_channel, kernel_height, kernel_width],
        (TCH_TYPE, tch::Device::Cpu),
    );
    let tch_a = tch::Tensor::randn(
        [batch, in_channel, height, width],
        (TCH_TYPE, tch::Device::Cpu),
    );
    let mut kernel = Tensor::<Type>::empty([out_channel, in_channel, kernel_height, kernel_width])?;
    let size = kernel.size();
    kernel.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_kernel.data_ptr() as *const Type, size)
    });
    let mut a = Tensor::<Type>::empty([batch, in_channel, height, width])?;
    let size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const Type, size)
    });
    Ok((
        kernel.permute([0, 2, 3, 1])?.contiguous()?,
        a.permute([0, 2, 3, 1])?.contiguous()?,
        tch_kernel,
        tch_a,
    ))
}

#[track_caller]
fn assert_eq(
    a: &Tensor<Type>,
    a_kernel: &Tensor<Type>,
    b: &tch::Tensor,
    b_kernel: &tch::Tensor,
) -> anyhow::Result<()> {
    let a = a.to_cuda::<0>()?;
    let a_kernel = a_kernel.to_cuda::<0>()?;
    let res = a
        .conv2d(&a_kernel, None, [1, 1], [0, 0], [1, 1], None)?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let tch_res = b.conv2d(&b_kernel, None::<tch::Tensor>, &[1, 1], &[0, 0], &[1, 1], 1);
    let res_cpu = res.to_cpu::<0>()?;
    let res_slice = res_cpu.as_raw();
    let res2 = unsafe { std::slice::from_raw_parts(tch_res.data_ptr() as *const Type, res.size()) };
    res_slice.iter().zip(res2.iter()).for_each(|(a, b)| {
        assert_f64(*a, *b, 0.05, &res, &tch_res).expect("assert_f64 failed");
    });
    Ok(())
}

#[track_caller]
fn assert_eq_pad(
    a: &Tensor<Type>,
    a_kernel: &Tensor<Type>,
    b: &tch::Tensor,
    b_kernel: &tch::Tensor,
) -> anyhow::Result<()> {
    let a = a.to_cuda::<0>()?;
    let a_kernel = a_kernel.to_cuda::<0>()?;
    let res = a
        .conv2d(&a_kernel, None, [1, 1], [2, 2], [1, 1], None)?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let tch_res = b.conv2d(&b_kernel, None::<tch::Tensor>, &[1, 1], &[2, 2], &[1, 1], 1);
    let res_cpu = res.to_cpu::<0>()?;
    let res_slice = res_cpu.as_raw();
    let res2 = unsafe { std::slice::from_raw_parts(tch_res.data_ptr() as *const Type, res.size()) };
    res_slice.iter().zip(res2.iter()).for_each(|(a, b)| {
        assert_f64(*a, *b, 0.05, &res, &tch_res).expect("assert_f64 failed");
    });
    Ok(())
}

#[track_caller]
fn assert_eq_bias(
    a: &Tensor<Type>,
    a_kernel: &Tensor<Type>,
    b: &tch::Tensor,
    b_kernel: &tch::Tensor,
) -> anyhow::Result<()> {
    let a = a.to_cuda::<0>()?;
    let a_kernel = a_kernel.to_cuda::<0>()?;
    let tch_bias = tch::Tensor::randn(b_kernel.size()[0], (tch::Kind::Double, tch::Device::Cpu));
    let mut bias = Tensor::<Type>::empty([*a_kernel.shape().first().unwrap() as i64])?;
    let size = bias.size();
    bias.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_bias.data_ptr() as *const Type, size)
    });
    let res = a
        .conv2d(
            &a_kernel,
            Some(&bias.to_cuda::<0>()?),
            [1, 1],
            [0, 0],
            [1, 1],
            None,
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let tch_res = b.conv2d(&b_kernel, Some(&tch_bias), &[1, 1], &[0, 0], &[1, 1], 1);
    let res_cpu = res.to_cpu::<0>()?;
    let res_slice = res_cpu.as_raw();
    let res2 = unsafe { std::slice::from_raw_parts(tch_res.data_ptr() as *const Type, res.size()) };
    res_slice.iter().zip(res2.iter()).for_each(|(a, b)| {
        assert_f64(*a, *b, 0.05, &res, &tch_res).expect("assert_f64 failed");
    });
    Ok(())
}

#[track_caller]
fn assert_eq_bias_pad(
    a: &Tensor<Type>,
    a_kernel: &Tensor<Type>,
    b: &tch::Tensor,
    b_kernel: &tch::Tensor,
) -> anyhow::Result<()> {
    let a = a.to_cuda::<0>()?;
    let a_kernel = a_kernel.to_cuda::<0>()?;
    let tch_bias = tch::Tensor::randn(b_kernel.size()[0], (tch::Kind::Double, tch::Device::Cpu));
    let mut bias = Tensor::<Type>::empty([*a_kernel.shape().first().unwrap() as i64])?;
    let size = bias.size();
    bias.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_bias.data_ptr() as *const Type, size)
    });
    let res = a
        .conv2d(
            &a_kernel,
            Some(&bias.to_cuda::<0>()?),
            [1, 1],
            [2, 2],
            [1, 1],
            None,
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let tch_res = b.conv2d(&b_kernel, Some(&tch_bias), &[1, 1], &[2, 2], &[1, 1], 1);
    let res_cpu = res.to_cpu::<0>()?;
    let res_slice = res_cpu.as_raw();
    let res2 = unsafe { std::slice::from_raw_parts(tch_res.data_ptr() as *const Type, res.size()) };
    res_slice.iter().zip(res2.iter()).for_each(|(a, b)| {
        assert_f64(*a, *b, 0.05, &res, &tch_res).expect("assert_f64 failed");
    });
    Ok(())
}

#[test]
fn test() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    for i in 0..100 {
        let in_channel = rng.random_range(1..=32);
        let out_channel = rng.random_range(1..=32);
        let kernel_height = rng.random_range(1..=5);
        let kernel_width = rng.random_range(1..=5);
        let height = rng.random_range(10..=32);
        let width = rng.random_range(10..=32);
        let (kernel, a, tch_kernel, tch_a) = common_input([
            in_channel,
            out_channel,
            kernel_height,
            kernel_width,
            height,
            width,
        ])?;
        assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
        assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
        assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel)?;
        assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    }
    Ok(())
}
