#![allow(unused_imports)]
use hpt::common::cpu::TensorLike;
use hpt::common::TensorInfo;
use hpt::ops::*;
use hpt::Tensor;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

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
            panic!(
                "{} != {} (abs_diff: {}, relative_diff: {})",
                a, b, abs_diff, relative_diff
            );
        }
    });
}
#[test]
fn test_transpose() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::randn(&[10, 10], (tch::Kind::Double, tch::Device::Cpu));
    let mut a = Tensor::<f64>::empty(&[10, 10])?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const f64, a_size)
    });
    let b = a.transpose(0, 1)?;
    let tch_b = tch_a.transpose(0, 1);
    assert_eq(&b, &tch_b);
    assert_eq!(&tch_b.size(), b.shape().inner());
    Ok(())
}

#[test]
fn test_mt() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::randn(&[10, 10, 10], (tch::Kind::Double, tch::Device::Cpu));
    let mut a = Tensor::<f64>::empty(&[10, 10, 10])?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const f64, a_size)
    });
    let b = a.mt()?;
    let tch_b = tch_a.permute(&[2, 1, 0][..]);
    assert_eq(&b, &tch_b);
    assert_eq!(&tch_b.size(), b.shape().inner());
    Ok(())
}

#[should_panic(expected = "transpose failed")]
#[test]
fn test_transpose_panic() {
    let a = Tensor::<f64>::empty(&[10]).unwrap();
    a.t().expect("transpose failed");
}

#[test]
fn test_swap_axes() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::randn(&[10, 10, 10], (tch::Kind::Double, tch::Device::Cpu));
    let mut a = Tensor::<f64>::empty(&[10, 10, 10])?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const f64, a_size)
    });
    let b = a.swap_axes(0, 1)?;
    let tch_b = tch_a.permute(&[1, 0, 2][..]);
    assert_eq(&b, &tch_b);
    assert_eq!(&tch_b.size(), b.shape().inner());
    Ok(())
}

#[test]
fn test_unsqueeze() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::randn(&[10], (tch::Kind::Double, tch::Device::Cpu));
    let mut a = Tensor::<f64>::empty(&[10])?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const f64, a_size)
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
    let mut a = Tensor::<f64>::empty(&[1, 10, 1])?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const f64, a_size)
    });
    let b = a.squeeze(0)?;
    let tch_b = tch_a.squeeze_dim(0);
    let b = b.squeeze(1)?;
    let tch_b = tch_b.squeeze_dim(1);
    assert_eq(&b, &tch_b);
    assert_eq!(&tch_b.size(), b.shape().inner());
    Ok(())
}

#[should_panic(expected = "squeeze failed")]
#[test]
fn test_squeeze_panic() {
    let a = Tensor::<f64>::empty(&[1, 10, 1]).unwrap();
    a.squeeze(1).expect("squeeze failed");
}

#[test]
fn test_expand() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::randn(&[1, 10, 1], (tch::Kind::Double, tch::Device::Cpu));
    let mut a = Tensor::<f64>::empty(&[1, 10, 1])?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const f64, a_size)
    });
    let b = a.expand(&[10, 10, 10])?;
    let tch_b = tch_a.expand(&[10, 10, 10], true);
    assert_eq(&b, &tch_b);
    assert_eq!(&tch_b.size(), b.shape().inner());
    Ok(())
}

#[test]
fn test_flatten() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::randn(&[10, 10, 10], (tch::Kind::Double, tch::Device::Cpu));
    let mut a = Tensor::<f64>::empty(&[10, 10, 10])?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const f64, a_size)
    });
    let b = a.flatten(1, 2)?;
    let tch_b = tch_a.flatten(1, 2);
    assert_eq(&b, &tch_b);
    assert_eq!(&tch_b.size(), b.shape().inner());

    let b = a.flatten(0, 2)?;
    let tch_b = tch_a.flatten(0, 2);
    assert_eq(&b, &tch_b);
    assert_eq!(&tch_b.size(), b.shape().inner());

    let b = a.flatten(2, 2)?;
    let tch_b = tch_a.flatten(2, 2);
    assert_eq(&b, &tch_b);
    assert_eq!(&tch_b.size(), b.shape().inner());

    let b = a.flatten(0, 1)?;
    let tch_b = tch_a.flatten(0, 1);
    assert_eq(&b, &tch_b);
    assert_eq!(&tch_b.size(), b.shape().inner());

    Ok(())
}

#[test]
fn test_split() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::randn(&[10, 10, 10], (tch::Kind::Double, tch::Device::Cpu));
    let mut a = Tensor::<f64>::empty(&[10, 10, 10])?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const f64, a_size)
    });
    let b = a.split(&[2, 5], 1)?;
    let tch_b = tch_a.split_with_sizes(&[2, 3, 5], 1);
    b.iter()
        .zip(tch_b.iter())
        .for_each(|(b, tch_b)| assert_eq(b, tch_b));
    Ok(())
}

#[test]
fn test_tile() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::randn(&[10, 10, 10], (tch::Kind::Double, tch::Device::Cpu));
    let mut a = Tensor::<f64>::empty(&[10, 10, 10])?;
    let size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const f64, size)
    });
    let b = a.tile(&[2, 3, 4])?;
    let tch_b = tch_a.tile(&[2, 3, 4]);
    assert_eq(&b, &tch_b);
    assert_eq!(&tch_b.size(), b.shape().inner());
    Ok(())
}

#[test]
fn test_reshape() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::randn(&[10, 10, 10], (tch::Kind::Double, tch::Device::Cpu));
    let mut a = Tensor::<f64>::empty(&[10, 10, 10])?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const f64, a_size)
    });
    let b = a.reshape(&[10, 100])?;
    let tch_b = tch_a.reshape(&[10, 100]);
    assert_eq(&b, &tch_b);
    assert_eq!(&tch_b.size(), b.shape().inner());
    Ok(())
}

#[should_panic(expected = "reshape failed")]
#[test]
fn test_reshape_panic() {
    Tensor::<f64>::empty(&[10, 10, 10])
        .unwrap()
        .reshape(&[10, 100, 10])
        .expect("reshape failed");
}

#[test]
fn test_reshape_uncontiguous() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::randn(&[10, 10, 10], (tch::Kind::Double, tch::Device::Cpu));
    let mut a = Tensor::<f64>::empty(&[10, 10, 10])?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const f64, a_size)
    });
    let b = a.permute([1, 0, 2])?.reshape(&[10, 100])?;
    let tch_b = tch_a.permute(&[1, 0, 2][..]).reshape(&[10, 100]);
    assert_eq(&b, &tch_b);
    assert_eq!(&tch_b.size(), b.shape().inner());
    Ok(())
}

#[test]
fn test_concat() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::randn(&[10, 10, 10], (tch::Kind::Double, tch::Device::Cpu));
    let tch_b = tch::Tensor::randn(&[10, 10, 10], (tch::Kind::Double, tch::Device::Cpu));
    let mut a = Tensor::<f64>::empty(&[10, 10, 10])?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const f64, a_size)
    });
    let mut b = Tensor::<f64>::empty(&[10, 10, 10])?;
    let b_size = b.size();
    b.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_b.data_ptr() as *const f64, b_size)
    });
    let c = Tensor::<f64>::concat(vec![a, b], 1, false)?;
    let tch_c = tch::Tensor::cat(&[&tch_a, &tch_b], 1);
    assert_eq(&c, &tch_c);
    assert_eq!(&tch_c.size(), c.shape().inner());
    Ok(())
}

#[test]
fn test_uncontiguous_concat() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::randn(&[10, 10, 10], (tch::Kind::Double, tch::Device::Cpu));
    let tch_b = tch::Tensor::randn(&[10, 10, 10], (tch::Kind::Double, tch::Device::Cpu));
    let mut a = Tensor::<f64>::empty(&[10, 10, 10])?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const f64, a_size)
    });
    let mut b = Tensor::<f64>::empty(&[10, 10, 10])?;
    let b_size = b.size();
    b.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_b.data_ptr() as *const f64, b_size)
    });
    let a = a.permute([1, 0, 2])?;
    let tch_a = tch_a.permute(&[1, 0, 2][..]);
    let b = b.permute([1, 0, 2])?;
    let tch_b = tch_b.permute(&[1, 0, 2][..]);
    let c = Tensor::<f64>::concat(vec![a, b], 1, false)?;
    let tch_c = tch::Tensor::cat(&[&tch_a, &tch_b], 1);
    assert_eq(&c, &tch_c);
    assert_eq!(&tch_c.size(), c.shape().inner());
    Ok(())
}
