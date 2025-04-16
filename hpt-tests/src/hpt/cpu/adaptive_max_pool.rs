#![allow(unused)]
use hpt::common::cpu::TensorLike;
use hpt::common::TensorInfo;
use hpt::{
    ops::{Contiguous, NormalPooling, ShapeManipulate, TensorCreator},
    Tensor,
};
use hpt_types::into_scalar::Cast;
use hpt_types::type_promote::NormalOut;
use rand::Rng;
use tch;

use crate::TCH_TEST_TYPES;
use crate::{TestTypes, TEST_ATOL, TEST_RTOL};
fn common_input(
    [batch, in_channel, height, width]: [i64; 4],
) -> anyhow::Result<(Tensor<TestTypes>, tch::Tensor)> {
    let a = Tensor::<TestTypes>::arange(0, batch * in_channel * height * width)?
        .reshape([batch, in_channel, height, width])?
        .permute([0, 2, 3, 1])?
        .contiguous()?;
    let tch_a = tch::Tensor::arange(
        batch * in_channel * height * width,
        (TCH_TEST_TYPES, tch::Device::Cpu),
    )
    .reshape(&[batch, in_channel, height, width]);
    Ok((a, tch_a))
}

#[track_caller]
fn assert_eq(a: &Tensor<TestTypes>, b: &tch::Tensor) -> anyhow::Result<()> {
    let res = a
        .adaptive_maxpool2d([2, 2])?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let (tch_res, _) = b.adaptive_max_pool2d([2, 2]);
    let tch_res = unsafe {
        Tensor::<TestTypes>::from_raw(tch_res.data_ptr() as *mut TestTypes, &res.shape().to_vec())
    }?;
    assert!(res.allclose(&tch_res, TEST_RTOL, TEST_ATOL));
    Ok(())
}

#[test]
fn test_case0() -> anyhow::Result<()> {
    let mut rng = rand::rng();

    for _ in 0..100 {
        let batch = rng.random_range(1..=4);
        let channel = rng.random_range(1..=16);
        let height = rng.random_range(8..=32);
        let width = rng.random_range(8..=32);
        let (a, tch_a) = common_input([batch, channel, height, width])?;
        assert_eq(&a, &tch_a)?;
    }

    Ok(())
}
