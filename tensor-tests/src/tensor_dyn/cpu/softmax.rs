#![allow(unused)]
use duplicate::duplicate_item;
use rand::Rng;
use tch::Tensor;
use tensor_common::slice;
use tensor_dyn::{ backend::Cpu, TensorCreator };
use tensor_dyn::ShapeManipulate;
use tensor_dyn::TensorInfo;
use tensor_dyn::TensorLike;
use tensor_macros::match_selection;
use tensor_common::slice::Slice;

#[allow(unused)]
#[track_caller]
fn assert_eq_f64(b: &tensor_dyn::tensor::Tensor<f64>, a: &Tensor) {
    let a_raw = if b.strides().contains(&0) {
        let size = b
            .shape()
            .iter()
            .zip(b.strides().iter())
            .filter(|(sp, s)| **s != 0)
            .fold(1, |acc, (sp, _)| acc * sp);
        unsafe { std::slice::from_raw_parts(a.data_ptr() as *const f64, size as usize) }
    } else {
        unsafe { std::slice::from_raw_parts(a.data_ptr() as *const f64, b.size()) }
    };
    let b_raw = b.as_raw();
    let tolerance = 2.5e-16;
    let caller = core::panic::Location::caller();
    a_raw
        .iter()
        .zip(b_raw.iter())
        .for_each(|(a, b)| {
            let abs_diff = (a - b).abs();
            let relative_diff = abs_diff / b.abs().max(f64::EPSILON);

            if abs_diff > tolerance && relative_diff > tolerance {
                panic!(
                    "{} != {} (abs_diff: {}, relative_diff: {}), at {}",
                    a,
                    b,
                    abs_diff,
                    relative_diff,
                    caller
                );
            }
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
    for _ in 0..1000 {
        let len = rng.gen_range(1..5);
        let mut shape = Vec::with_capacity(len);
        for _ in 0..len {
            shape.push(rng.gen_range(1..10));
        }
        let (a, tch_a) = common_input(shape.iter().product::<i64>(), &shape)?;

        let dim = rng.gen_range(0..len) as i64;
        println!("dim: {}", dim);
        println!("shape: {:?}", shape);
        let res = a.hpt_method(dim)?;
        let tch_res = tch_a.tch_method(dim, tch::Kind::Double);
        assert_eq_f64(&res, &tch_res);
    }
    Ok(())
}

#[test]
fn func() -> anyhow::Result<()> {
    let mut shape = vec![8, 2, 3, 1];
    let (a, tch_a) = common_input(shape.iter().product::<i64>(), &shape)?;
    let res = a.softmax(1)?;
    let tch_res = tch_a.softmax(1, tch::Kind::Double);
    assert_eq_f64(&res, &tch_res);
    Ok(())
}
