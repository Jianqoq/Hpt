#![allow(unused)]
use hpt::ShapeManipulate;
use hpt::TensorLike;
use hpt::TypeCommon;
use hpt::{set_display_elements, set_num_threads, CommonBounds, TensorInfo};
use hpt::{Conv, Tensor, TensorCreator};
use hpt_types::into_scalar::Cast;
use hpt_types::type_promote::NormalOut;
use rand::Rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use tch;

use super::assert_utils::assert_f64;
type Type = f64;

fn common_input(
    [batch, out_channel, in_channel, kernel_height, kernel_width, height, width, groups]: [i64; 8],
) -> anyhow::Result<(Tensor<Type>, Tensor<Type>, tch::Tensor, tch::Tensor)> {
    let tch_kernel = tch::Tensor::randn(
        [
            out_channel,
            in_channel / groups,
            kernel_height,
            kernel_width,
        ],
        (tch::Kind::Double, tch::Device::Cpu),
    );
    let mut kernel = Tensor::<Type>::empty([
        out_channel,
        in_channel / groups,
        kernel_height,
        kernel_width,
    ])?;
    let size = kernel.size();
    kernel.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_kernel.data_ptr() as *const Type, size)
    });

    let tch_a = tch::Tensor::randn(
        [batch, in_channel, height, width],
        (tch::Kind::Double, tch::Device::Cpu),
    );
    let mut a = Tensor::<Type>::empty([batch, in_channel, height, width])?;
    let size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const Type, size)
    });

    Ok((
        kernel.permute([2, 3, 1, 0])?.contiguous()?,
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
    groups: i64,
) -> anyhow::Result<()> {
    let res = a
        .conv2d_group(
            &a_kernel,
            None,
            [1, 1],
            [(0, 0), (0, 0)],
            [1, 1],
            groups,
            None,
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let tch_res = b.conv2d(
        &b_kernel,
        None::<tch::Tensor>,
        &[1, 1],
        &[0, 0],
        &[1, 1],
        groups,
    );
    let res_slice = res.as_raw();
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
    groups: i64,
) -> anyhow::Result<()> {
    let res = a
        .conv2d_group(
            &a_kernel,
            None,
            [1, 1],
            [(2, 2), (2, 2)],
            [1, 1],
            groups,
            None,
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let tch_res = b.conv2d(
        &b_kernel,
        None::<tch::Tensor>,
        &[1, 1],
        &[2, 2],
        &[1, 1],
        groups,
    );
    let res_slice = res.as_raw();
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
    groups: i64,
) -> anyhow::Result<()> {
    let tch_bias = tch::Tensor::randn(b_kernel.size()[0], (tch::Kind::Double, tch::Device::Cpu));
    let mut bias = Tensor::<Type>::empty([*a_kernel.shape().last().unwrap()])?;
    let size = bias.size();
    bias.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_bias.data_ptr() as *const Type, size)
    });
    let res = a
        .conv2d_group(
            &a_kernel,
            Some(&bias),
            [1, 1],
            [(0, 0), (0, 0)],
            [1, 1],
            groups,
            None,
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let tch_res = b.conv2d(&b_kernel, Some(tch_bias), &[1, 1], &[0, 0], &[1, 1], groups);
    let res_slice = res.as_raw();
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
    groups: i64,
) -> anyhow::Result<()> {
    let tch_bias = tch::Tensor::randn(b_kernel.size()[0], (tch::Kind::Double, tch::Device::Cpu));
    let mut bias = Tensor::<Type>::empty([*a_kernel.shape().last().unwrap()])?;
    let size = bias.size();
    bias.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_bias.data_ptr() as *const Type, size)
    });
    let res = a
        .conv2d_group(
            &a_kernel,
            Some(&bias),
            [1, 1],
            [(2, 2), (2, 2)],
            [1, 1],
            groups,
            None,
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let tch_res = b.conv2d(&b_kernel, Some(tch_bias), &[1, 1], &[2, 2], &[1, 1], groups);
    let res_slice = res.as_raw();
    let res2 = unsafe { std::slice::from_raw_parts(tch_res.data_ptr() as *const Type, res.size()) };
    res_slice.iter().zip(res2.iter()).for_each(|(a, b)| {
        assert_f64(*a, *b, 0.05, &res, &tch_res).expect("assert_f64 failed");
    });
    Ok(())
}

#[track_caller]
fn assert_eq_bias_pad_relu(
    a: &Tensor<Type>,
    a_kernel: &Tensor<Type>,
    b: &tch::Tensor,
    b_kernel: &tch::Tensor,
    groups: i64,
) -> anyhow::Result<()> {
    let tch_bias = tch::Tensor::randn(b_kernel.size()[0], (tch::Kind::Double, tch::Device::Cpu));
    let mut bias = Tensor::<Type>::empty([*a_kernel.shape().last().unwrap()])?;
    let size = bias.size();
    bias.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_bias.data_ptr() as *const Type, size)
    });
    let res = a
        .conv2d_group(
            &a_kernel,
            Some(&bias),
            [1, 1],
            [(2, 2), (2, 2)],
            [1, 1],
            groups,
            None,
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let tch_res = b.conv2d(&b_kernel, Some(tch_bias), &[1, 1], &[2, 2], &[1, 1], groups);
    let res_slice = res.as_raw();
    let res2 = unsafe { std::slice::from_raw_parts(tch_res.data_ptr() as *const Type, res.size()) };
    res_slice.iter().zip(res2.iter()).for_each(|(a, b)| {
        assert_f64(*a, *b, 0.05, &res, &tch_res).expect("assert_f64 failed");
    });
    Ok(())
}

#[test]
fn test() -> anyhow::Result<()> {
    let mut rng = rand::thread_rng();
    for i in 0..100 {
        let groups = rng.gen_range(1..=4);

        let in_channel = rng.gen_range(1..=16) * groups;
        let out_channel = rng.gen_range(1..=16) * groups;

        let kernel_height = rng.gen_range(1..=5);
        let kernel_width = rng.gen_range(1..=5);
        let height = rng.gen_range(10..=32);
        let width = rng.gen_range(10..=32);
        let batch = rng.gen_range(1..=4);

        let (kernel, a, tch_kernel, tch_a) = common_input([
            batch,
            out_channel,
            in_channel,
            kernel_height,
            kernel_width,
            height,
            width,
            groups,
        ])?;
        assert_eq(&a, &kernel, &tch_a, &tch_kernel, groups)?;
        assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel, groups)?;
        assert_eq_bias(&a, &kernel, &tch_a, &tch_kernel, groups)?;
        assert_eq_bias_pad(&a, &kernel, &tch_a, &tch_kernel, groups)?;
    }
    Ok(())
}
