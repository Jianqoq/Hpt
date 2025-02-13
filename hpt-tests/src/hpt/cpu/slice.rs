#![allow(unused)]
use hpt_common::slice;
use hpt_common::slice::Slice;
use hpt::ShapeManipulate;
use hpt::Tensor;
use hpt::TensorLike;
use hpt::{TensorCreator, TensorInfo};
use hpt_macros::match_selection;

#[allow(unused)]
fn assert_eq(b: &Tensor<i32>, a: &tch::Tensor) {
    let a_raw = if b.strides().contains(&0) {
        let size = b
            .shape()
            .iter()
            .zip(b.strides().iter())
            .filter(|(sp, s)| **s != 0)
            .fold(1, |acc, (sp, _)| acc * sp);
        unsafe { std::slice::from_raw_parts(a.data_ptr() as *const i32, size as usize) }
    } else {
        unsafe { std::slice::from_raw_parts(a.data_ptr() as *const i32, b.size()) }
    };
    let b_raw = b.as_raw();

    a_raw.iter().zip(b_raw.iter()).for_each(|(a, b)| {
        if a != b {
            panic!("{} != {}", a, b);
        }
    });
}

#[test]
fn test_slice() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::arange(100, (tch::Kind::Int, tch::Device::Cpu)).reshape(&[10, 10]);
    let a = Tensor::<i32>::arange(0, 100)?.reshape(&[10, 10])?;
    let a = slice!(a[1:9, 1:9])?.contiguous()?;
    let tch_a = tch_a.slice(1, 1, 9, 1).slice(0, 1, 9, 1).contiguous();
    assert_eq(&a, &tch_a);
    let a = slice!(a[2:9, 2:9])?.contiguous()?;
    let tch_a = tch_a.slice(1, 2, 9, 1).slice(0, 2, 9, 1).contiguous();
    assert_eq(&a, &tch_a);

    Ok(())
}

#[test]
fn test_uncontinuous_slice() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::arange(100, (tch::Kind::Int, tch::Device::Cpu)).reshape(&[10, 10]);
    let a = Tensor::<i32>::arange(0, 100)?.reshape(&[10, 10])?;
    let tch_a = tch_a.permute(&[1, 0]);
    let a = a.permute(&[1, 0])?;
    let a = slice!(a[1:9:2, 1:9:2])?.contiguous()?;
    let tch_a = tch_a.slice(1, 1, 9, 2).slice(0, 1, 9, 2).contiguous();
    assert_eq(&a, &tch_a);

    let a = slice!(a[2:9:2, 2:9:2])?.contiguous()?;
    let tch_a = tch_a.slice(1, 2, 9, 2).slice(0, 2, 9, 2).contiguous();
    assert_eq(&a, &tch_a);
    Ok(())
}
