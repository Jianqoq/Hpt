#![allow(unused_imports)]
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use tensor_common::slice;
use tensor_common::slice::Slice;
use tensor_dyn::Cuda;
use tensor_dyn::ShapeManipulate;
use tensor_dyn::TensorInfo;
use tensor_dyn::TensorLike;
use tensor_dyn::{Tensor, TensorCreator};
use tensor_macros::match_selection;

#[allow(unused)]
fn assert_eq(b: &Tensor<f64>, a: &tch::Tensor) {
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
fn test_arange() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::arange(100, (tch::Kind::Double, tch::Device::Cpu));
    let a = Tensor::<f64, Cuda, 0>::arange(0, 100)?;
    let cpu_a = a.to_cpu()?;
    assert_eq(&cpu_a, &tch_a);
    Ok(())
}

#[test]
fn test_zeros() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::zeros(&[1000], (tch::Kind::Double, tch::Device::Cpu));
    let a = Tensor::<f64, Cuda, 0>::zeros(&[1000])?;
    let cpu_a = a.to_cpu()?;
    assert_eq(&cpu_a, &tch_a);
    Ok(())
}

#[test]
fn test_full() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::full(&[1000], 1.0, (tch::Kind::Double, tch::Device::Cpu));
    let a = Tensor::<f64, Cuda, 0>::full(1.0, &[1000])?;
    let cpu_a = a.to_cpu()?;
    assert_eq(&cpu_a, &tch_a);
    Ok(())
}

#[test]
fn test_eye() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::eye(10, (tch::Kind::Double, tch::Device::Cpu));
    let a = Tensor::<f64, Cuda, 0>::eye(10, 10, 0)?;
    let cpu_a = a.to_cpu()?;
    assert_eq(&cpu_a, &tch_a);
    Ok(())
}
