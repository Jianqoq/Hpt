#![allow(unused_imports)]

use tch::Tensor;
use tensor_common::slice;
use tensor_dyn::{ tensor_base::_Tensor, TensorCreator };
use tensor_dyn::ShapeManipulate;
use tensor_dyn::TensorLike;
use tensor_dyn::TensorInfo;
use tensor_macros::match_selection;
use tensor_common::slice::Slice;

#[allow(unused)]
fn assert_eq(b: &_Tensor<i64>, a: &Tensor) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const i64, b.size()) };
    let b_raw = b.as_raw();

    for i in 0..b.size() {
        if a_raw[i] != b_raw[i] {
            panic!(
                "{} != {}, bytes: {:?}, {:?}",
                a_raw[i],
                b_raw[i],
                a_raw[i].to_ne_bytes(),
                b_raw[i].to_ne_bytes()
            );
        }
    }
}

#[test]
fn test_2dim() -> anyhow::Result<()> {
    let a = _Tensor::<i64>::arange(0, 1000000)?.reshape(&[1000, 1000])?;
    let tch_a = tch::Tensor
        ::arange(1000000, (tch::Kind::Int64, tch::Device::Cpu))
        .reshape(&[1000, 1000]);
    let (_, b_values) = a.topk(5, 0, true, true)?;
    let (tch_b_values, _) = tch_a.topk(5, 0, true, true);
    assert_eq(&b_values, &tch_b_values);

    let (_, b_values) = a.topk(5, 0, false, true)?;
    let (tch_b_values, _) = tch_a.topk(5, 0, false, true);
    assert_eq(&b_values, &tch_b_values);

    let (_, b_values) = a.topk(5, 0, false, false)?;
    let (tch_b_values, _) = tch_a.topk(5, 0, false, false);
    assert_eq(&b_values, &tch_b_values);

    let (_, b_values) = a.topk(5, 1, true, true)?;
    let (tch_b_values, _) = tch_a.topk(5, 1, true, true);
    assert_eq(&b_values, &tch_b_values);

    let (_, b_values) = a.topk(5, 1, false, true)?;
    let (tch_b_values, _) = tch_a.topk(5, 1, false, true);
    assert_eq(&b_values, &tch_b_values);

    let (_, b_values) = a.topk(5, 1, false, false)?;
    let (tch_b_values, _) = tch_a.topk(5, 1, false, false);
    assert_eq(&b_values, &tch_b_values);
    Ok(())
}

#[test]
fn test_2dim_uncontiguous() -> anyhow::Result<()> {
    let a = _Tensor::<i64>::arange(0, 1000000)?.reshape(&[1000, 1000])?.permute(&[1, 0])?;
    let tch_a = tch::Tensor
        ::arange(1000000, (tch::Kind::Int64, tch::Device::Cpu))
        .reshape(&[1000, 1000])
        .permute(&[1, 0]);
    let (_, b_values) = a.topk(5, 0, true, true)?;
    let (tch_b_values, _) = tch_a.topk(5, 0, true, true);
    assert_eq(&b_values, &tch_b_values);

    let (_, b_values) = a.topk(5, 0, false, true)?;
    let (tch_b_values, _) = tch_a.topk(5, 0, false, true);
    assert_eq(&b_values, &tch_b_values);

    let (_, b_values) = a.topk(5, 0, false, false)?;
    let (tch_b_values, _) = tch_a.topk(5, 0, false, false);
    assert_eq(&b_values, &tch_b_values);

    let (_, b_values) = a.topk(5, 1, true, true)?;
    let (tch_b_values, _) = tch_a.topk(5, 1, true, true);
    assert_eq(&b_values, &tch_b_values);

    let (_, b_values) = a.topk(5, 1, false, true)?;
    let (tch_b_values, _) = tch_a.topk(5, 1, false, true);
    assert_eq(&b_values, &tch_b_values);

    let (_, b_values) = a.topk(5, 1, false, false)?;
    let (tch_b_values, _) = tch_a.topk(5, 1, false, false);
    assert_eq(&b_values, &tch_b_values);
    Ok(())
}

#[test]
fn test_2dim_uncontiguous_sub_tensor() -> anyhow::Result<()> {
    let a = _Tensor::<i64>::arange(0, 1000000)?.reshape(&[1000, 1000])?.permute(&[1, 0])?;
    let a = slice!(a[20:70:3, 20:70:3])?;
    let tch_a = tch::Tensor
        ::arange(1000000, (tch::Kind::Int64, tch::Device::Cpu))
        .reshape(&[1000, 1000])
        .permute(&[1, 0]);
    let tch_a = tch_a.slice(0, 20, 70, 3).slice(1, 20, 70, 3);
    let (_, b_values) = a.topk(5, 0, true, true)?;
    let (tch_b_values, _) = tch_a.topk(5, 0, true, true);
    assert_eq(&b_values, &tch_b_values);

    let (_, b_values) = a.topk(5, 0, false, true)?;
    let (tch_b_values, _) = tch_a.topk(5, 0, false, true);
    assert_eq(&b_values, &tch_b_values);

    let (_, b_values) = a.topk(5, 0, false, false)?;
    let (tch_b_values, _) = tch_a.topk(5, 0, false, false);
    assert_eq(&b_values, &tch_b_values);

    let (_, b_values) = a.topk(5, 1, true, true)?;
    let (tch_b_values, _) = tch_a.topk(5, 1, true, true);
    assert_eq(&b_values, &tch_b_values);

    let (_, b_values) = a.topk(5, 1, false, true)?;
    let (tch_b_values, _) = tch_a.topk(5, 1, false, true);
    assert_eq(&b_values, &tch_b_values);

    let (_, b_values) = a.topk(5, 1, false, false)?;
    let (tch_b_values, _) = tch_a.topk(5, 1, false, false);
    assert_eq(&b_values, &tch_b_values);
    Ok(())
}

#[test]
fn test_3dim() -> anyhow::Result<()> {
    let a = _Tensor::<i64>::arange(0, 1000000)?.reshape(&[100, 100, 100])?;
    let tch_a = tch::Tensor
        ::arange(1000000, (tch::Kind::Int64, tch::Device::Cpu))
        .reshape(&[100, 100, 100]);
    let (_, b_values) = a.topk(50, 0, true, true)?;
    let (tch_b_values, _) = tch_a.topk(50, 0, true, true);
    assert_eq(&b_values, &tch_b_values);

    let (_, b_values) = a.topk(50, 0, false, true)?;
    let (tch_b_values, _) = tch_a.topk(50, 0, false, true);
    assert_eq(&b_values, &tch_b_values);

    let (_, b_values) = a.topk(50, 1, true, true)?;
    let (tch_b_values, _) = tch_a.topk(50, 1, true, true);
    assert_eq(&b_values, &tch_b_values);

    let (_, b_values) = a.topk(50, 1, false, true)?;
    let (tch_b_values, _) = tch_a.topk(50, 1, false, true);
    assert_eq(&b_values, &tch_b_values);

    let (_, b_values) = a.topk(50, 2, true, true)?;
    let (tch_b_values, _) = tch_a.topk(50, 2, true, true);
    assert_eq(&b_values, &tch_b_values);

    let (_, b_values) = a.topk(50, 2, false, true)?;
    let (tch_b_values, _) = tch_a.topk(50, 2, false, true);
    assert_eq(&b_values, &tch_b_values);
    Ok(())
}

#[test]
fn test_3dim_uncontiguous() -> anyhow::Result<()> {
    let a = _Tensor::<i64>::arange(0, 1000000)?.reshape(&[100, 100, 100])?.permute(&[1, 0, 2])?;
    let tch_a = tch::Tensor
        ::arange(1000000, (tch::Kind::Int64, tch::Device::Cpu))
        .reshape(&[100, 100, 100])
        .permute(&[1, 0, 2]);
    let (_, b_values) = a.topk(50, 0, true, true)?;
    let (tch_b_values, _) = tch_a.topk(50, 0, true, true);
    assert_eq(&b_values, &tch_b_values);

    let (_, b_values) = a.topk(50, 0, false, true)?;
    let (tch_b_values, _) = tch_a.topk(50, 0, false, true);
    assert_eq(&b_values, &tch_b_values);

    let (_, b_values) = a.topk(50, 1, true, true)?;
    let (tch_b_values, _) = tch_a.topk(50, 1, true, true);
    assert_eq(&b_values, &tch_b_values);

    let (_, b_values) = a.topk(50, 1, false, true)?;
    let (tch_b_values, _) = tch_a.topk(50, 1, false, true);
    assert_eq(&b_values, &tch_b_values);

    let (_, b_values) = a.topk(50, 2, true, true)?;
    let (tch_b_values, _) = tch_a.topk(50, 2, true, true);
    assert_eq(&b_values, &tch_b_values);

    let (_, b_values) = a.topk(50, 2, false, true)?;
    let (tch_b_values, _) = tch_a.topk(50, 2, false, true);
    assert_eq(&b_values, &tch_b_values);
    Ok(())
}

#[test]
fn test_3dim_uncontiguous_sub_tensor() -> anyhow::Result<()> {
    let a = _Tensor::<i64>::arange(0, 1000000)?.reshape(&[100, 100, 100])?.permute(&[1, 0, 2])?;
    let tch_a = tch::Tensor
        ::arange(1000000, (tch::Kind::Int64, tch::Device::Cpu))
        .reshape(&[100, 100, 100])
        .permute(&[1, 0, 2]);
    let a = slice!(a[20:80:3, 20:80:3, 20:80:3])?;
    let tch_a = tch_a.slice(0, 20, 80, 3).slice(1, 20, 80, 3).slice(2, 20, 80, 3);
    let (_, b_values) = a.topk(15, 0, true, true)?;
    let (tch_b_values, _) = tch_a.topk(15, 0, true, true);
    assert_eq(&b_values, &tch_b_values);

    let (_, b_values) = a.topk(15, 0, false, true)?;
    let (tch_b_values, _) = tch_a.topk(15, 0, false, true);
    assert_eq(&b_values, &tch_b_values);

    let (_, b_values) = a.topk(15, 1, true, true)?;
    let (tch_b_values, _) = tch_a.topk(15, 1, true, true);
    assert_eq(&b_values, &tch_b_values);

    let (_, b_values) = a.topk(15, 1, false, true)?;
    let (tch_b_values, _) = tch_a.topk(15, 1, false, true);
    assert_eq(&b_values, &tch_b_values);

    let (_, b_values) = a.topk(15, 2, true, true)?;
    let (tch_b_values, _) = tch_a.topk(15, 2, true, true);
    assert_eq(&b_values, &tch_b_values);

    let (_, b_values) = a.topk(15, 2, false, true)?;
    let (tch_b_values, _) = tch_a.topk(15, 2, false, true);
    assert_eq(&b_values, &tch_b_values);
    Ok(())
}
