#![allow(unused)]
use super::assert_utils::assert_f64;
use hpt::common::cpu::TensorLike;
use hpt::common::TensorInfo;
use hpt::ops::Contiguous;
use hpt::ops::ShapeManipulate;
use hpt::ops::{FloatOutPooling, TensorCreator};
use hpt::Tensor;
use hpt_types::into_scalar::Cast;
use hpt_types::type_promote::NormalOut;
use rand::Rng;
use tch;
fn common_input(
    [batch, in_channel, height, width]: [i64; 4],
) -> anyhow::Result<(Tensor<f64>, tch::Tensor)> {
    let a = Tensor::<f64>::arange(0, batch * in_channel * height * width)?
        .reshape([batch, in_channel, height, width])?
        .permute([0, 2, 3, 1])?
        .contiguous()?;
    let tch_a = tch::Tensor::arange(
        batch * in_channel * height * width,
        (tch::Kind::Double, tch::Device::Cpu),
    )
    .reshape(&[batch, in_channel, height, width]);
    Ok((a, tch_a))
}

#[track_caller]
fn assert_eq(a: &Tensor<f64>, b: &tch::Tensor) -> anyhow::Result<()> {
    let res = a
        .adaptive_avgpool2d([2, 2])?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let tch_res = b.adaptive_avg_pool2d([2, 2]);
    let res_slice = res.as_raw();
    let res2 = unsafe { std::slice::from_raw_parts(tch_res.data_ptr() as *const f64, res.size()) };
    res_slice.iter().zip(res2.iter()).for_each(|(a, b)| {
        assert_f64(*a, *b, 0.05, &res, &tch_res);
    });
    Ok(())
}

#[test]
fn test() -> anyhow::Result<()> {
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
