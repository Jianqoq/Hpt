#![allow(unused)]
use hpt::common::cpu::TensorLike;
use hpt::common::TensorInfo;
use hpt::ops::Contiguous;
use hpt::ops::NormalPooling;
use hpt::ops::ShapeManipulate;
use hpt::ops::TensorCreator;
use hpt::Tensor;
use hpt_types::into_scalar::Cast;
use hpt_types::type_promote::NormalOut;
use rand::Rng;
use tch;

use super::assert_utils::assert_f64;

fn common_input(
    [batch, in_channel, kernel_height, kernel_width, height, width]: [i64; 6],
) -> anyhow::Result<(Tensor<i64>, Tensor<i64>, tch::Tensor, tch::Tensor)> {
    let kernel = Tensor::<i64>::arange(0, kernel_height * kernel_width)?
        .reshape([kernel_height, kernel_width])?;
    let a = Tensor::<i64>::arange(0, batch * in_channel * height * width)?
        .reshape([batch, in_channel, height, width])?
        .permute([0, 2, 3, 1])?
        .contiguous()?;

    let tch_kernel = tch::Tensor::arange(
        kernel_height * kernel_width,
        (tch::Kind::Int64, tch::Device::Cpu),
    )
    .reshape(&[kernel_height, kernel_width]);
    let tch_a = tch::Tensor::arange(
        batch * in_channel * height * width,
        (tch::Kind::Int64, tch::Device::Cpu),
    )
    .reshape(&[batch, in_channel, height, width]);
    Ok((kernel, a, tch_kernel, tch_a))
}

#[track_caller]
fn assert_eq(
    a: &Tensor<i64>,
    a_kernel: &Tensor<i64>,
    b: &tch::Tensor,
    b_kernel: &tch::Tensor,
) -> anyhow::Result<()> {
    let res = a
        .maxpool2d(a_kernel.shape(), [1, 1], [(0, 0), (0, 0)], [1, 1])?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let tch_res = b.max_pool2d(&b_kernel.size(), &[1, 1], &[0, 0], [1, 1], false);
    let res_slice = res.as_raw();
    let res2 = unsafe { std::slice::from_raw_parts(tch_res.data_ptr() as *const i64, res.size()) };
    res_slice.iter().zip(res2.iter()).for_each(|(a, b)| {
        assert_eq!(*a, *b);
    });
    Ok(())
}

#[track_caller]
fn assert_eq_pad(
    a: &Tensor<i64>,
    a_kernel: &Tensor<i64>,
    b: &tch::Tensor,
    b_kernel: &tch::Tensor,
) -> anyhow::Result<()> {
    let res = a
        .maxpool2d(a_kernel.shape(), [1, 1], [(2, 2), (2, 2)], [1, 1])?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let tch_res = b.max_pool2d(&b_kernel.size(), &[1, 1], &[2, 2], [1, 1], false);
    let res_slice = res.as_raw();
    let res2 = unsafe { std::slice::from_raw_parts(tch_res.data_ptr() as *const i64, res.size()) };
    res_slice.iter().zip(res2.iter()).for_each(|(a, b)| {
        assert_eq!(*a, *b);
    });
    Ok(())
}

#[test]
fn test() -> anyhow::Result<()> {
    let mut rng = rand::thread_rng();

    for _ in 0..1000 {
        let batch = rng.gen_range(1..=4);
        let channel = rng.gen_range(1..=16);

        let kernel_height = rng.gen_range(1..=5);
        let kernel_width = rng.gen_range(1..=5);

        let height = rng.gen_range(8..=32);
        let width = rng.gen_range(8..=32);
        let (kernel, a, tch_kernel, tch_a) =
            common_input([batch, channel, kernel_height, kernel_width, height, width])?;
        assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    }

    Ok(())
}

#[test]
fn test_pad() -> anyhow::Result<()> {
    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        let batch = rng.gen_range(1..=4);
        let channel = rng.gen_range(1..=16);

        let kernel_height = rng.gen_range(4..=7);
        let kernel_width = rng.gen_range(4..=7);

        let height = rng.gen_range(8..=32);
        let width = rng.gen_range(8..=32);
        let (kernel, a, tch_kernel, tch_a) =
            common_input([batch, channel, kernel_height, kernel_width, height, width])?;
        assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    }

    Ok(())
}
