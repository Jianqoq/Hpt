#![allow(unused)]
use hpt_core::set_num_threads;
use hpt_core::ShapeManipulate;
use hpt_core::TensorLike;
use hpt_core::{CommonBounds, TensorInfo};
use hpt_core::{NormalPooling, Tensor, TensorCreator};
use hpt_types::into_scalar::Cast;
use hpt_types::type_promote::NormalOut;
use rand::Rng;
use tch;

use super::assert_utils::assert_f64;

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
        .adaptive_maxpool2d([2, 2])?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let (tch_res, _) = b.adaptive_max_pool2d([2, 2]);
    let res_slice = res.as_raw();
    let res2 = unsafe { std::slice::from_raw_parts(tch_res.data_ptr() as *const f64, res.size()) };
    res_slice.iter().zip(res2.iter()).for_each(|(a, b)| {
        assert_f64(*a, *b, 0.05, &res, &tch_res);
    });
    Ok(())
}

#[test]
fn test_case0() -> anyhow::Result<()> {
    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        let batch = rng.gen_range(1..=4);
        let channel = rng.gen_range(1..=16);
        let height = rng.gen_range(8..=32);
        let width = rng.gen_range(8..=32);
        let (a, tch_a) = common_input([batch, channel, height, width])?;
        assert_eq(&a, &tch_a)?;
    }

    Ok(())
}
