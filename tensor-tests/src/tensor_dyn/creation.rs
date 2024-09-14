#![allow(unused_imports)]
use rayon::iter::{ IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator };
use tch::Tensor;
use tensor_common::slice;
use tensor_dyn::{ tensor_base::_Tensor, TensorCreator };
use tensor_dyn::TensorInfo;
use tensor_dyn::ShapeManipulate;
use tensor_macros::match_selection;
use tensor_common::slice::Slice;
use tensor_dyn::slice::SliceOps;

#[allow(unused)]
fn assert_eq(b: &_Tensor<f64>, a: &Tensor) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const f64, b.size()) };
    let b_raw = b.as_raw();
    let tolerance = 2.5e-16;

    for i in 0..b.size() {
        let abs_diff = (a_raw[i] - b_raw[i]).abs();
        let relative_diff = abs_diff / b_raw[i].abs().max(f64::EPSILON);

        if abs_diff > tolerance && relative_diff > tolerance {
            panic!(
                "{} != {} (abs_diff: {}, relative_diff: {})",
                a_raw[i],
                b_raw[i],
                abs_diff,
                relative_diff
            );
        }
    }
}

#[test]
fn test_arange() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::arange(100, (tch::Kind::Double, tch::Device::Cpu));
    let a = _Tensor::<f64>::arange(0, 100)?;
    assert_eq(&a, &tch_a);
    Ok(())
}

#[test]
fn test_hamming() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::hamming_window_periodic(1000, true, (
        tch::Kind::Double,
        tch::Device::Cpu,
    ));
    let a = _Tensor::<f64>::hamming_window(1000, true)?;
    assert_eq(&a, &tch_a);
    Ok(())
}

#[test]
fn test_hann() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::hann_window_periodic(1000, true, (
        tch::Kind::Double,
        tch::Device::Cpu,
    ));
    let a = _Tensor::<f64>::hann_window(1000, true)?;
    assert_eq(&a, &tch_a);
    Ok(())
}

#[test]
fn test_zeros() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::zeros(&[1000], (tch::Kind::Double, tch::Device::Cpu));
    let a = _Tensor::<f64>::zeros(&[1000])?;
    assert_eq(&a, &tch_a);
    Ok(())
}

#[test]
fn test_full() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::full(&[1000], 1.0, (tch::Kind::Double, tch::Device::Cpu));
    let a = _Tensor::<f64>::full(1.0, &[1000])?;
    assert_eq(&a, &tch_a);
    Ok(())
}

#[test]
fn test_eye() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::eye(10, (tch::Kind::Double, tch::Device::Cpu));
    let a = _Tensor::<f64>::eye(10, 10, 0)?;
    assert_eq(&a, &tch_a);
    Ok(())
}

#[test]
fn test_tril() -> anyhow::Result<()> {
    fn assert(diagnal: i64) -> anyhow::Result<()> {
        let tch_a = tch::Tensor
            ::randn(&[10, 10], (tch::Kind::Double, tch::Device::Cpu))
            .tril(diagnal);
        let a = _Tensor::<f64>::empty(&[10, 10])?;
        a.as_raw_mut().copy_from_slice(unsafe {
            std::slice::from_raw_parts(tch_a.data_ptr() as *const f64, a.size())
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
fn test_transpose() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::randn(&[10, 10], (tch::Kind::Double, tch::Device::Cpu));
    let a = _Tensor::<f64>::empty(&[10, 10])?;
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const f64, a.size())
    });
    let b = a.transpose(0, 1)?;
    let tch_b = tch_a.transpose(0, 1);
    assert_eq(&b, &tch_b);
    assert_eq!(&tch_b.size(), b.shape().inner());
    Ok(())
}

#[test]
fn test_unsqueeze() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::randn(&[10], (tch::Kind::Double, tch::Device::Cpu));
    let a = _Tensor::<f64>::empty(&[10])?;
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const f64, a.size())
    });
    let b = a.unsqueeze(0)?;
    let tch_b = tch_a.unsqueeze(0);
    let b = b.unsqueeze(1)?;
    let tch_b = tch_b.unsqueeze(1);
    assert_eq(&b, &tch_b);
    assert_eq!(&tch_b.size(), b.shape().inner());
    Ok(())
}

#[test]
fn test_squeeze() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::randn(&[1, 10, 1], (tch::Kind::Double, tch::Device::Cpu));
    let a = _Tensor::<f64>::empty(&[1, 10, 1])?;
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const f64, a.size())
    });
    let b = a.squeeze(0)?;
    let tch_b = tch_a.squeeze_dim(0);
    let b = b.squeeze(1)?;
    let tch_b = tch_b.squeeze_dim(1);
    assert_eq(&b, &tch_b);
    assert_eq!(&tch_b.size(), b.shape().inner());
    Ok(())
}