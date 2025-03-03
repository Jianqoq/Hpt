#![allow(unused)]
use duplicate::duplicate_item;
use hpt::set_display_elements;
use hpt::TensorInfo;
use hpt::TensorLike;
use hpt::{set_num_threads, ShapeManipulate};
use hpt::{Cpu, TensorCreator};
use hpt_common::slice;
use hpt_macros::select;
use rand::Rng;
use tch::Tensor;

use super::assert_utils::assert_f64;

#[allow(unused)]
#[track_caller]
fn assert_eq_f64(hpt_res: &hpt::tensor::Tensor<f64>, tch_res: &Tensor) {
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
    a_raw.iter().zip(b_raw.iter()).for_each(|(a, b)| {
        assert_f64(*a, *b, 0.05, hpt_res, tch_res);
    });
}

fn common_input(end: i64, shape: &[i64]) -> anyhow::Result<(hpt::tensor::Tensor<f64>, Tensor)> {
    let a = hpt::tensor::Tensor::<f64>::arange(0, end)?.reshape(shape)?;
    let tch_a = Tensor::arange(end, (tch::Kind::Double, tch::Device::Cpu)).reshape(shape);
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

#[test]
fn test_layernorm() -> anyhow::Result<()> {
    let mut rng = rand::thread_rng();
    for _ in 0..1000 {
        let len = rng.gen_range(1..3);
        let mut shape = Vec::with_capacity(len);
        for _ in 0..len {
            shape.push(rng.gen_range(1..10));
        }
        let (a, tch_a) = common_input(shape.iter().product::<i64>(), &shape)?;

        let to_normalize_dims = rng.gen_range(1..=len);
        let mut shape = a.shape().iter().rev();
        let mut normalized_shape = vec![0; to_normalize_dims];
        for i in (0..to_normalize_dims).rev() {
            normalized_shape[i] = *shape.next().unwrap();
        }
        let (gamma, tch_gamma) =
            common_input(normalized_shape.iter().product::<i64>(), &normalized_shape)?;
        let (beta, tch_beta) =
            common_input(normalized_shape.iter().product::<i64>(), &normalized_shape)?;
        let res = a.layernorm(&normalized_shape, Some(&gamma), Some(&beta), 1e-5)?;
        let tch_res = tch_a.layer_norm(
            &normalized_shape,
            Some(tch_gamma),
            Some(tch_beta),
            1e-5,
            false,
        );
        assert_eq_f64(&res, &tch_res);
    }
    Ok(())
}
