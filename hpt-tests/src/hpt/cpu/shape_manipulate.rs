#![allow(unused_imports)]
use hpt::common::cpu::TensorLike;
use hpt::common::TensorInfo;
use hpt::ops::*;
use hpt::Tensor;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::TestTypes;
use crate::TCH_TEST_TYPES;
use crate::TEST_ATOL;
use crate::TEST_RTOL;

#[test]
fn test_transpose() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::randn(&[10, 10], (TCH_TEST_TYPES, tch::Device::Cpu));
    let mut a = Tensor::<TestTypes>::empty(&[10, 10])?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const TestTypes, a_size)
    });
    let b = a.transpose(0, 1)?;
    let tch_b = tch_a.transpose(0, 1);
    assert_eq!(&tch_b.size(), b.shape().inner());
    let tch_b = unsafe {
        Tensor::<TestTypes>::from_raw(tch_b.data_ptr() as *mut TestTypes, &b.shape().to_vec())
    }?;
    assert!(b.allclose(&tch_b, TEST_RTOL, TEST_ATOL));
    Ok(())
}

#[test]
fn test_mt() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::randn(&[10, 10, 10], (TCH_TEST_TYPES, tch::Device::Cpu));
    let mut a = Tensor::<TestTypes>::empty(&[10, 10, 10])?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const TestTypes, a_size)
    });
    let b = a.mt()?;
    let tch_b = tch_a.permute(&[2, 1, 0][..]).contiguous();
    assert_eq!(&tch_b.size(), b.shape().inner());
    let tch_b = unsafe {
        Tensor::<TestTypes>::from_raw(tch_b.data_ptr() as *mut TestTypes, &b.shape().to_vec())
    }?;
    assert!(b.allclose(&tch_b, TEST_RTOL, TEST_ATOL));
    Ok(())
}

#[should_panic(expected = "transpose failed")]
#[test]
fn test_transpose_panic() {
    let a = Tensor::<TestTypes>::empty(&[10]).unwrap();
    a.t().expect("transpose failed");
}

#[test]
fn test_swap_axes() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::randn(&[10, 10, 10], (TCH_TEST_TYPES, tch::Device::Cpu));
    let mut a = Tensor::<TestTypes>::empty(&[10, 10, 10])?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const TestTypes, a_size)
    });
    let b = a.swap_axes(0, 1)?;
    let tch_b = tch_a.permute(&[1, 0, 2][..]).contiguous();
    assert_eq!(&tch_b.size(), b.shape().inner());
    let tch_b = unsafe {
        Tensor::<TestTypes>::from_raw(tch_b.data_ptr() as *mut TestTypes, &b.shape().to_vec())
    }?;
    assert!(b.allclose(&tch_b, TEST_RTOL, TEST_ATOL));
    Ok(())
}

#[test]
fn test_unsqueeze() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::randn(&[10], (TCH_TEST_TYPES, tch::Device::Cpu));
    let mut a = Tensor::<TestTypes>::empty(&[10])?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const TestTypes, a_size)
    });
    let b = a.unsqueeze(0)?;
    let tch_b = tch_a.unsqueeze(0);
    let b = b.unsqueeze(1)?;
    let tch_b = tch_b.unsqueeze(1);
    assert_eq!(&tch_b.size(), b.shape().inner());
    let tch_b = unsafe {
        Tensor::<TestTypes>::from_raw(tch_b.data_ptr() as *mut TestTypes, &b.shape().to_vec())
    }?;
    assert!(b.allclose(&tch_b, TEST_RTOL, TEST_ATOL));
    Ok(())
}

#[test]
fn test_squeeze() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::randn(&[1, 10, 1], (TCH_TEST_TYPES, tch::Device::Cpu));
    let mut a = Tensor::<TestTypes>::empty(&[1, 10, 1])?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const TestTypes, a_size)
    });
    let b = a.squeeze(0)?;
    let tch_b = tch_a.squeeze_dim(0);
    let b = b.squeeze(1)?;
    let tch_b = tch_b.squeeze_dim(1);
    assert_eq!(&tch_b.size(), b.shape().inner());
    let tch_b = unsafe {
        Tensor::<TestTypes>::from_raw(tch_b.data_ptr() as *mut TestTypes, &b.shape().to_vec())
    }?;
    assert!(b.allclose(&tch_b, TEST_RTOL, TEST_ATOL));
    Ok(())
}

#[should_panic(expected = "squeeze failed")]
#[test]
fn test_squeeze_panic() {
    let a = Tensor::<TestTypes>::empty(&[1, 10, 1]).unwrap();
    a.squeeze(1).expect("squeeze failed");
}

#[test]
fn test_expand() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::randn(&[1, 10, 1], (TCH_TEST_TYPES, tch::Device::Cpu));
    let mut a = Tensor::<TestTypes>::empty(&[1, 10, 1])?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const TestTypes, a_size)
    });
    let b = a.expand(&[10, 10, 10])?.contiguous()?;
    let tch_b = tch_a.expand(&[10, 10, 10], true).contiguous();
    assert_eq!(&tch_b.size(), b.shape().inner());
    let tch_b = unsafe {
        Tensor::<TestTypes>::from_raw(tch_b.data_ptr() as *mut TestTypes, &b.shape().to_vec())
    }?;
    assert!(b.allclose(&tch_b, TEST_RTOL, TEST_ATOL));
    Ok(())
}

#[test]
fn test_flatten() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::randn(&[10, 10, 10], (TCH_TEST_TYPES, tch::Device::Cpu));
    let mut a = Tensor::<TestTypes>::empty(&[10, 10, 10])?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const TestTypes, a_size)
    });
    let b = a.flatten(1, 2)?;
    let tch_b = tch_a.flatten(1, 2);
    assert_eq!(&tch_b.size(), b.shape().inner());
    let tch_b = unsafe {
        Tensor::<TestTypes>::from_raw(tch_b.data_ptr() as *mut TestTypes, &b.shape().to_vec())
    }?;
    assert!(b.allclose(&tch_b, TEST_RTOL, TEST_ATOL));

    let b = a.flatten(0, 2)?;
    let tch_b = tch_a.flatten(0, 2);
    assert_eq!(&tch_b.size(), b.shape().inner());
    let tch_b = unsafe {
        Tensor::<TestTypes>::from_raw(tch_b.data_ptr() as *mut TestTypes, &b.shape().to_vec())
    }?;
    assert!(b.allclose(&tch_b, TEST_RTOL, TEST_ATOL));

    let b = a.flatten(2, 2)?;
    let tch_b = tch_a.flatten(2, 2);
    assert_eq!(&tch_b.size(), b.shape().inner());
    let tch_b = unsafe {
        Tensor::<TestTypes>::from_raw(tch_b.data_ptr() as *mut TestTypes, &b.shape().to_vec())
    }?;
    assert!(b.allclose(&tch_b, TEST_RTOL, TEST_ATOL));

    let b = a.flatten(0, 1)?;
    let tch_b = tch_a.flatten(0, 1);
    assert_eq!(&tch_b.size(), b.shape().inner());
    let tch_b = unsafe {
        Tensor::<TestTypes>::from_raw(tch_b.data_ptr() as *mut TestTypes, &b.shape().to_vec())
    }?;
    assert!(b.allclose(&tch_b, TEST_RTOL, TEST_ATOL));

    Ok(())
}

#[test]
fn test_split() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::randn(&[10, 10, 10], (TCH_TEST_TYPES, tch::Device::Cpu));
    let mut a = Tensor::<TestTypes>::empty(&[10, 10, 10])?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const TestTypes, a_size)
    });
    let b = a.split(&[2, 5], 1)?;
    let tch_b = tch_a.split_with_sizes(&[2, 3, 5], 1);
    b.iter()
        .zip(tch_b.iter())
        .for_each(|(b, tch_b)| {
            let tch_b = tch_b.contiguous();
            let tch_b = unsafe {
                Tensor::<TestTypes>::from_raw(tch_b.data_ptr() as *mut TestTypes, &b.shape().to_vec())
            }.expect("failed to create tensor from raw pointer");
            assert!(b.allclose(&tch_b, TEST_RTOL, TEST_ATOL));
        });
    Ok(())
}

#[test]
fn test_tile() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::randn(&[10, 10, 10], (TCH_TEST_TYPES, tch::Device::Cpu));
    let mut a = Tensor::<TestTypes>::empty(&[10, 10, 10])?;
    let size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const TestTypes, size)
    });
    let b = a.tile(&[2, 3, 4])?;
    let tch_b = tch_a.tile(&[2, 3, 4]);
    assert_eq!(&tch_b.size(), b.shape().inner());
    let tch_b = unsafe {
        Tensor::<TestTypes>::from_raw(tch_b.data_ptr() as *mut TestTypes, &b.shape().to_vec())
    }?;
    assert!(b.allclose(&tch_b, TEST_RTOL, TEST_ATOL));
    Ok(())
}

#[test]
fn test_reshape() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::randn(&[10, 10, 10], (TCH_TEST_TYPES, tch::Device::Cpu));
    let mut a = Tensor::<TestTypes>::empty(&[10, 10, 10])?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const TestTypes, a_size)
    });
    let b = a.reshape(&[10, 100])?;
    let tch_b = tch_a.reshape(&[10, 100]);
    assert_eq!(&tch_b.size(), b.shape().inner());
    let tch_b = unsafe {
        Tensor::<TestTypes>::from_raw(tch_b.data_ptr() as *mut TestTypes, &b.shape().to_vec())
    }?;
    assert!(b.allclose(&tch_b, TEST_RTOL, TEST_ATOL));
    Ok(())
}

#[should_panic(expected = "reshape failed")]
#[test]
fn test_reshape_panic() {
    Tensor::<TestTypes>::empty(&[10, 10, 10])
        .unwrap()
        .reshape(&[10, 100, 10])
        .expect("reshape failed");
}

#[test]
fn test_reshape_uncontiguous() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::randn(&[10, 10, 10], (TCH_TEST_TYPES, tch::Device::Cpu));
    let mut a = Tensor::<TestTypes>::empty(&[10, 10, 10])?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const TestTypes, a_size)
    });
    let b = a.permute([1, 0, 2])?.reshape(&[10, 100])?;
    let tch_b = tch_a.permute(&[1, 0, 2][..]).reshape(&[10, 100]);
    assert_eq!(&tch_b.size(), b.shape().inner());
    let tch_b = unsafe {
        Tensor::<TestTypes>::from_raw(tch_b.data_ptr() as *mut TestTypes, &b.shape().to_vec())
    }?;
    assert!(b.allclose(&tch_b, TEST_RTOL, TEST_ATOL));
    Ok(())
}

#[test]
fn test_concat() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::randn(&[10, 10, 10], (TCH_TEST_TYPES, tch::Device::Cpu));
    let tch_b = tch::Tensor::randn(&[10, 10, 10], (TCH_TEST_TYPES, tch::Device::Cpu));
    let mut a = Tensor::<TestTypes>::empty(&[10, 10, 10])?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const TestTypes, a_size)
    });
    let mut b = Tensor::<TestTypes>::empty(&[10, 10, 10])?;
    let b_size = b.size();
    b.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_b.data_ptr() as *const TestTypes, b_size)
    });
    let c = Tensor::<TestTypes>::concat(vec![a, b], 1, false)?;
    let tch_c = tch::Tensor::cat(&[&tch_a, &tch_b], 1);
    assert_eq!(&tch_c.size(), c.shape().inner());
    let tch_c = unsafe {
        Tensor::<TestTypes>::from_raw(tch_c.data_ptr() as *mut TestTypes, &c.shape().to_vec())
    }?;
    assert!(c.allclose(&tch_c, TEST_RTOL, TEST_ATOL));
    Ok(())
}

#[test]
fn test_uncontiguous_concat() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::randn(&[10, 10, 10], (TCH_TEST_TYPES, tch::Device::Cpu));
    let tch_b = tch::Tensor::randn(&[10, 10, 10], (TCH_TEST_TYPES, tch::Device::Cpu));
    let mut a = Tensor::<TestTypes>::empty(&[10, 10, 10])?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const TestTypes, a_size)
    });
    let mut b = Tensor::<TestTypes>::empty(&[10, 10, 10])?;
    let b_size = b.size();
    b.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_b.data_ptr() as *const TestTypes, b_size)
    });
    let a = a.permute([1, 0, 2])?;
    let tch_a = tch_a.permute(&[1, 0, 2][..]);
    let b = b.permute([1, 0, 2])?;
    let tch_b = tch_b.permute(&[1, 0, 2][..]);
    let c = Tensor::<TestTypes>::concat(vec![a, b], 1, false)?;
    let tch_c = tch::Tensor::cat(&[&tch_a, &tch_b], 1);
    assert_eq!(&tch_c.size(), c.shape().inner());
    let tch_c = unsafe {
        Tensor::<TestTypes>::from_raw(tch_c.data_ptr() as *mut TestTypes, &c.shape().to_vec())
    }?;
    assert!(c.allclose(&tch_c, TEST_RTOL, TEST_ATOL));
    Ok(())
}
