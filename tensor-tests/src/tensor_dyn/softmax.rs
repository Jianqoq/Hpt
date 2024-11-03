#![allow(unused)]
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

fn common_input<const N: usize>(
    end: i64,
    shape: [i64; N]
) -> anyhow::Result<(tensor_dyn::tensor::Tensor<f64, Cpu>, Tensor)> {
    let a = tensor_dyn::tensor::Tensor::<f64, Cpu>::arange(0, end)?.reshape(&shape)?;
    let tch_a = Tensor::arange(end, (tch::Kind::Int64, tch::Device::Cpu)).reshape(&shape);
    Ok((a, tch_a))
}

#[test]
fn test_softmax_axis_0() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let res = a.softmax(0)?;
    let tch_res = tch_a.softmax(0, tch::Kind::Double);
    assert_eq_f64(&res, &tch_res);
    Ok(())
}

#[test]
fn test_softmax_axis_1() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let res = a.softmax(1)?;
    let tch_res = tch_a.softmax(1, tch::Kind::Double);
    assert_eq_f64(&res, &tch_res);
    Ok(())
}

#[test]
fn test_softmax_axis_2() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 2 * 5, [2, 2, 5])?;
    let res = a.softmax(2)?;
    let tch_res = tch_a.softmax(2, tch::Kind::Double);
    assert_eq_f64(&res, &tch_res);
    Ok(())
}

#[test]
fn test_softmax_axis_0_step() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = slice!(a[:, 1:5:2, 2:9:2])?;
    let tch_a = tch_a.slice(1, 1, 5, 2).slice(2, 2, 9, 2);
    let res = a.softmax(0)?;
    let tch_res = tch_a.softmax(0, tch::Kind::Double);
    assert_eq_f64(&res, &tch_res);
    Ok(())
}

#[test]
fn test_softmax_axis_1_step() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = slice!(a[:, 1:5:2, 2:9:2])?;
    let tch_a = tch_a.slice(1, 1, 5, 2).slice(2, 2, 9, 2);
    let res = a.softmax(1)?;
    let tch_res = tch_a.softmax(1, tch::Kind::Double);
    assert_eq_f64(&res, &tch_res);
    Ok(())
}

#[test]
fn test_softmax_axis_2_step() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = slice!(a[:, 1:5:2, 2:9:2])?;
    let tch_a = tch_a.slice(1, 1, 5, 2).slice(2, 2, 9, 2);
    let res = a.softmax(2)?;
    let tch_res = tch_a.softmax(2, tch::Kind::Double);
    assert_eq_f64(&res, &tch_res);
    Ok(())
}
