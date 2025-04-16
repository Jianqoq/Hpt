#![allow(unused_imports)]
use hpt::common::cpu::TensorLike;
use hpt::common::TensorInfo;
use hpt::ops::Contiguous;
use hpt::ops::ShapeManipulate;
use hpt::ops::TensorCreator;
use hpt::ops::WindowOps;
use hpt::types::TypeCommon;
use hpt::Tensor;
use hpt_common::slice;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::TestTypes;
use crate::TCH_TEST_TYPES;
use crate::TEST_ATOL;
use crate::TEST_RTOL;

#[allow(unused)]
fn assert_eq(hpt_res: &Tensor<TestTypes>, tch_res: &tch::Tensor) -> anyhow::Result<()> {
    let tch_res = unsafe {
        Tensor::<TestTypes>::from_raw(
            tch_res.data_ptr() as *mut TestTypes,
            &hpt_res.shape().to_vec(),
        )
    }?;
    assert!(hpt_res.allclose(&tch_res, TEST_RTOL, TEST_ATOL));
    Ok(())
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
    let tch_a = tch::Tensor::arange(100, (TCH_TEST_TYPES, tch::Device::Cpu));
    let a = Tensor::<TestTypes>::arange(0, 100)?;
    assert_eq(&a, &tch_a)?;
    Ok(())
}

#[test]
fn test_hamming() -> anyhow::Result<()> {
    let tch_a =
        tch::Tensor::hamming_window_periodic(1000, true, (TCH_TEST_TYPES, tch::Device::Cpu));
    let a = Tensor::<TestTypes>::hamming_window(1000, true)?;
    assert_eq(&a, &tch_a)?;
    Ok(())
}

#[test]
fn test_hann() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::hann_window_periodic(1000, true, (TCH_TEST_TYPES, tch::Device::Cpu));
    let a = Tensor::<TestTypes>::hann_window(1000, true)?;
    assert_eq(&a, &tch_a)?;
    Ok(())
}

#[test]
#[allow(unused)]
fn test_blackman_window() -> anyhow::Result<()> {
    let tch_a =
        tch::Tensor::blackman_window_periodic(1000, true, (TCH_TEST_TYPES, tch::Device::Cpu));
    let a = Tensor::<TestTypes>::blackman_window(1000, true)?;
    assert_eq(&a, &tch_a)?;
    Ok(())
}

#[test]
fn test_zeros() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::zeros(&[1000], (TCH_TEST_TYPES, tch::Device::Cpu));
    let a = Tensor::<TestTypes>::zeros(&[1000])?;
    assert_eq(&a, &tch_a)?;
    Ok(())
}

#[test]
fn test_full() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::full(&[1000], 1.0, (TCH_TEST_TYPES, tch::Device::Cpu));
    let a = Tensor::<TestTypes>::full(TestTypes::ONE, &[1000])?;
    assert_eq(&a, &tch_a)?;
    Ok(())
}

#[test]
fn test_eye() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::eye(10, (TCH_TEST_TYPES, tch::Device::Cpu));
    let a = Tensor::<TestTypes>::eye(10, 10, 0)?;
    assert_eq(&a, &tch_a)?;
    Ok(())
}

#[test]
fn test_tril() -> anyhow::Result<()> {
    fn assert(diagnal: i64) -> anyhow::Result<()> {
        let tch_a = tch::Tensor::randn(&[10, 10], (TCH_TEST_TYPES, tch::Device::Cpu)).tril(diagnal);
        let mut a = Tensor::<TestTypes>::empty(&[10, 10])?;
        let a_size = a.size();
        a.as_raw_mut().copy_from_slice(unsafe {
            std::slice::from_raw_parts(tch_a.data_ptr() as *const TestTypes, a_size)
        });
        let b = a.tril(diagnal)?;
        assert_eq(&b, &tch_a)?;
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
    let tch_a = tch::Tensor::eye(10, (TCH_TEST_TYPES, tch::Device::Cpu));
    let a = Tensor::<TestTypes>::identity(10)?;
    assert_eq(&a, &tch_a)?;
    Ok(())
}
