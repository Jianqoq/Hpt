#![allow(unused)]
use duplicate::duplicate_item;
use rand::Rng;
use tch::Tensor;
use tensor_common::slice;
use tensor_dyn::{ backend::Cpu, TensorCreator };
use tensor_dyn::{set_num_threads, ShapeManipulate};
use tensor_dyn::TensorInfo;
use tensor_dyn::TensorLike;
use tensor_macros::match_selection;
use tensor_common::slice::Slice;

use super::assert_utils::assert_f64;

#[allow(unused)]
#[track_caller]
fn assert_eq_f64(hpt_res: &tensor_dyn::tensor::Tensor<f64>, tch_res: &Tensor) {
    let a_raw = if hpt_res.strides().contains(&0) {
        let size = hpt_res
            .shape()
            .iter()
            .zip(hpt_res.strides().iter())
            .filter(|(sp, s)| **s != 0)
            .fold(1, |acc, (sp, _)| acc * sp);
        unsafe { std::slice::from_raw_parts(tch_res.data_ptr() as *const f64, size as usize) }
    } else {
        unsafe { std::slice::from_raw_parts(tch_res.data_ptr() as *const f64, hpt_res.size()) }
    };
    let b_raw = hpt_res.as_raw();
    a_raw
        .iter()
        .zip(b_raw.iter())
        .for_each(|(a, b)| {
            assert_f64(*a, *b, 0.05, hpt_res, tch_res);
        });
}

fn common_input(
    end: i64,
    shape: &[i64]
) -> anyhow::Result<(tensor_dyn::tensor::Tensor<f64>, Tensor)> {
    let a = tensor_dyn::tensor::Tensor::<f64>::arange(0, end)?.reshape(shape)?;
    let tch_a = Tensor::arange(end, (tch::Kind::Int64, tch::Device::Cpu)).reshape(shape);
    Ok((a, tch_a))
}

#[duplicate_item(
    func                    hpt_method      tch_method;
    [test_softmax]          [softmax]       [softmax];
    [test_logsoftmax]       [log_softmax]   [log_softmax];
)]
#[test]
fn func() -> anyhow::Result<()> {
    let mut rng = rand::thread_rng();
    for _ in 0..100 {
        let len = rng.gen_range(1..3);
        let mut shape = Vec::with_capacity(len);
        for _ in 0..len {
            shape.push(rng.gen_range(1..10));
        }
        let (a, tch_a) = common_input(shape.iter().product::<i64>(), &shape)?;

        let dim = rng.gen_range(0..len) as i64;
        let res = a.hpt_method(dim)?;
        let tch_res = tch_a.tch_method(dim, tch::Kind::Double);
        assert_eq_f64(&res, &tch_res);
    }
    Ok(())
}