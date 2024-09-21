#![allow(unused_imports)]
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serial_test::serial;
use tch::Tensor;
use tensor_common::slice;
use tensor_common::slice::Slice;
use tensor_dyn::ShapeManipulate;
use tensor_dyn::TensorInfo;
use tensor_dyn::TensorLike;
use tensor_dyn::{tensor_base::_Tensor, TensorCreator};
use tensor_macros::match_selection;

#[allow(unused)]
fn assert_eq(b: &_Tensor<f64>, a: &Tensor) {
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

    a_raw.iter().zip(b_raw.iter()).for_each(|(a, b)| {
        let abs_diff = (a - b).abs();
        let relative_diff = abs_diff / b.abs().max(f64::EPSILON);

        if abs_diff > tolerance && relative_diff > tolerance {
            println!(
                "{} != {} (abs_diff: {}, relative_diff: {})",
                a, b, abs_diff, relative_diff
            );
        }
    });
}

#[test]
#[serial]
fn test_arange() -> anyhow::Result<()> {
    let tch_a = Tensor::arange(100, (tch::Kind::Double, tch::Device::Cpu));
    let a = _Tensor::<f64>::arange(0, 100)?;
    assert_eq(&a, &tch_a);
    Ok(())
}

#[test]
#[serial]
fn test_hamming() -> anyhow::Result<()> {
    let tch_a = Tensor::hamming_window_periodic(1000, true, (tch::Kind::Double, tch::Device::Cpu));
    let a = tensor_dyn::tensor::Tensor::<f64>::hamming_window(1000, true)?;
    assert_eq(&a, &tch_a);
    Ok(())
}

#[test]
#[serial]
fn test_hann() -> anyhow::Result<()> {
    let tch_a = Tensor::hann_window_periodic(1000, true, (tch::Kind::Double, tch::Device::Cpu));
    let a = tensor_dyn::tensor::Tensor::<f64>::hann_window(1000, true)?;
    assert_eq(&a, &tch_a);
    Ok(())
}

#[test]
#[serial]
#[allow(unused)]
fn test_blackman_window() -> anyhow::Result<()> {
    let tch_a = Tensor::blackman_window_periodic(1000, true, (tch::Kind::Double, tch::Device::Cpu));
    let a = _Tensor::<f64>::blackman_window(1000, true)?;
    Ok(())
}

#[test]
#[serial]
fn test_zeros() -> anyhow::Result<()> {
    let tch_a = Tensor::zeros(&[1000], (tch::Kind::Double, tch::Device::Cpu));
    let a = _Tensor::<f64>::zeros(&[1000])?;
    assert_eq(&a, &tch_a);
    Ok(())
}

#[test]
#[serial]
fn test_full() -> anyhow::Result<()> {
    let tch_a = Tensor::full(&[1000], 1.0, (tch::Kind::Double, tch::Device::Cpu));
    let a = _Tensor::<f64>::full(1.0, &[1000])?;
    assert_eq(&a, &tch_a);
    Ok(())
}

#[test]
#[serial]
fn test_eye() -> anyhow::Result<()> {
    let tch_a = Tensor::eye(10, (tch::Kind::Double, tch::Device::Cpu));
    let a = _Tensor::<f64>::eye(10, 10, 0)?;
    assert_eq(&a, &tch_a);
    Ok(())
}

#[test]
#[serial]
fn test_tril() -> anyhow::Result<()> {
    fn assert(diagnal: i64) -> anyhow::Result<()> {
        let tch_a = Tensor::randn(&[10, 10], (tch::Kind::Double, tch::Device::Cpu)).tril(diagnal);
        let mut a = _Tensor::<f64>::empty(&[10, 10])?;
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
#[serial]
fn test_identity() -> anyhow::Result<()> {
    let tch_a = Tensor::eye(10, (tch::Kind::Double, tch::Device::Cpu));
    let a = _Tensor::<f64>::identity(10)?;
    assert_eq(&a, &tch_a);
    Ok(())
}
