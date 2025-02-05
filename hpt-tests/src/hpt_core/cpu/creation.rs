#![allow(unused_imports)]
use hpt_common::slice;
use hpt_common::slice::Slice;
use hpt_core::ShapeManipulate;
use hpt_core::TensorInfo;
use hpt_core::TensorLike;
use hpt_core::WindowOps;
use hpt_core::{Tensor, TensorCreator};
use hpt_macros::match_selection;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use super::assert_utils::assert_f64;

#[allow(unused)]
fn assert_eq(hpt_res: &Tensor<f64>, tch_res: &tch::Tensor) {
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
        .for_each(|(a, b)| assert_f64(*a, *b, 0.05, hpt_res, tch_res).unwrap());
}

#[test]
fn test_new() -> anyhow::Result<()> {
    let a = Tensor::<f64>::new(&[10.0, 10.0]);
    assert_eq!(a.as_raw(), &[10.0, 10.0]);
    Ok(())
}

#[test]
#[should_panic]
fn test_allocate_too_large() {
    let _a = Tensor::<f64>::empty(&[i64::MAX]).unwrap();
}

#[test]
fn test_arange() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::arange(100, (tch::Kind::Double, tch::Device::Cpu));
    let a = Tensor::<f64>::arange(0, 100)?;
    assert_eq(&a, &tch_a);
    Ok(())
}

#[test]
fn test_hamming() -> anyhow::Result<()> {
    let tch_a =
        tch::Tensor::hamming_window_periodic(1000, true, (tch::Kind::Double, tch::Device::Cpu));
    let a = Tensor::<f64>::hamming_window(1000, true)?;
    assert_eq(&a, &tch_a);
    Ok(())
}

#[test]
fn test_hann() -> anyhow::Result<()> {
    let tch_a =
        tch::Tensor::hann_window_periodic(1000, true, (tch::Kind::Double, tch::Device::Cpu));
    let a = Tensor::<f64>::hann_window(1000, true)?;
    assert_eq(&a, &tch_a);
    Ok(())
}

#[test]
#[allow(unused)]
fn test_blackman_window() -> anyhow::Result<()> {
    let tch_a =
        tch::Tensor::blackman_window_periodic(1000, true, (tch::Kind::Double, tch::Device::Cpu));
    let a = Tensor::<f64>::blackman_window(1000, true)?;
    Ok(())
}

#[test]
fn test_zeros() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::zeros(&[1000], (tch::Kind::Double, tch::Device::Cpu));
    let a = Tensor::<f64>::zeros(&[1000])?;
    assert_eq(&a, &tch_a);
    Ok(())
}

#[test]
fn test_full() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::full(&[1000], 1.0, (tch::Kind::Double, tch::Device::Cpu));
    let a = Tensor::<f64>::full(1.0, &[1000])?;
    assert_eq(&a, &tch_a);
    Ok(())
}

#[test]
fn test_eye() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::eye(10, (tch::Kind::Double, tch::Device::Cpu));
    let a = Tensor::<f64>::eye(10, 10, 0)?;
    assert_eq(&a, &tch_a);
    Ok(())
}

#[test]
fn test_tril() -> anyhow::Result<()> {
    fn assert(diagnal: i64) -> anyhow::Result<()> {
        let tch_a =
            tch::Tensor::randn(&[10, 10], (tch::Kind::Double, tch::Device::Cpu)).tril(diagnal);
        let mut a = Tensor::<f64>::empty(&[10, 10])?;
        let a_size = a.size();
        a.as_raw_mut().copy_from_slice(unsafe {
            std::slice::from_raw_parts(tch_a.data_ptr() as *const f64, a_size)
        });
        let b = a.tril(diagnal)?;
        assert_eq(&b, &tch_a);
        Ok(())
    }
    assert(0)?;
    assert(1)?;
    assert(-1)?;
    assert(-2)?;
    assert(2)?;
    Ok(())
}

#[test]
fn test_identity() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::eye(10, (tch::Kind::Double, tch::Device::Cpu));
    let a = Tensor::<f64>::identity(10)?;
    assert_eq(&a, &tch_a);
    Ok(())
}
