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

use crate::TestTypes;
use crate::TCH_TEST_TYPES;
use crate::TEST_ATOL;
use crate::TEST_RTOL;

fn common_input(
    [batch, in_channel, kernel_height, kernel_width, height, width]: [i64; 6],
) -> anyhow::Result<(
    Tensor<TestTypes>,
    Tensor<TestTypes>,
    tch::Tensor,
    tch::Tensor,
)> {
    let kernel = Tensor::<TestTypes>::arange(0, kernel_height * kernel_width)?
        .reshape([kernel_height, kernel_width])?;
    let a = Tensor::<TestTypes>::arange(0, batch * in_channel * height * width)?
        .reshape([batch, in_channel, height, width])?
        .permute([0, 2, 3, 1])?
        .contiguous()?;

    let tch_kernel = tch::Tensor::arange(
        kernel_height * kernel_width,
        (TCH_TEST_TYPES, tch::Device::Cpu),
    )
    .reshape(&[kernel_height, kernel_width]);
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
    let res = a
        .maxpool2d(a_kernel.shape(), [1, 1], [(0, 0), (0, 0)], [1, 1])?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let tch_res = b.max_pool2d(&b_kernel.size(), &[1, 1], &[0, 0], [1, 1], false);
    let tch_res = unsafe {
        Tensor::<TestTypes>::from_raw(tch_res.data_ptr() as *mut TestTypes, &res.shape().to_vec())
    }?;
    assert!(res.allclose(&tch_res, TEST_ATOL, TEST_RTOL));
    Ok(())
}

#[track_caller]
fn assert_eq_pad(
    a: &Tensor<TestTypes>,
    a_kernel: &Tensor<TestTypes>,
    b: &tch::Tensor,
    b_kernel: &tch::Tensor,
) -> anyhow::Result<()> {
    let res = a
        .maxpool2d(a_kernel.shape(), [1, 1], [(2, 2), (2, 2)], [1, 1])?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let tch_res = b.max_pool2d(&b_kernel.size(), &[1, 1], &[2, 2], [1, 1], false);
    let tch_res = unsafe {
        Tensor::<TestTypes>::from_raw(tch_res.data_ptr() as *mut TestTypes, &res.shape().to_vec())
    }?;
    assert!(res.allclose(&tch_res, TEST_ATOL, TEST_RTOL));
    Ok(())
}

#[test]
fn test() -> anyhow::Result<()> {
    let mut rng = rand::rng();

    for _ in 0..1000 {
        let batch = rng.random_range(1..=4);
        let channel = rng.random_range(1..=16);

        let kernel_height = rng.random_range(1..=5);
        let kernel_width = rng.random_range(1..=5);

        let height = rng.random_range(8..=32);
        let width = rng.random_range(8..=32);
        let (kernel, a, tch_kernel, tch_a) =
            common_input([batch, channel, kernel_height, kernel_width, height, width])?;
        assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    }

    Ok(())
}

#[test]
fn test_pad() -> anyhow::Result<()> {
    let mut rng = rand::rng();

    for _ in 0..100 {
        let batch = rng.random_range(1..=4);
        let channel = rng.random_range(1..=16);

        let kernel_height = rng.random_range(4..=7);
        let kernel_width = rng.random_range(4..=7);

        let height = rng.random_range(8..=32);
        let width = rng.random_range(8..=32);
        let (kernel, a, tch_kernel, tch_a) =
            common_input([batch, channel, kernel_height, kernel_width, height, width])?;
        assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    }

    Ok(())
}
