#![allow(unused_imports)]
use hpt::common::cpu::TensorLike;
use hpt::common::TensorInfo;
use hpt::ops::*;
use hpt::slice;
use hpt::Tensor;
use rand::Rng;

use crate::TestTypes;
use crate::TCH_TEST_TYPES;
use crate::TEST_ATOL;
use crate::TEST_RTOL;

#[test]
fn test() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    for _ in 0..100 {
        let ndim = rng.random_range(1..=3);
        let shape = (0..ndim)
            .map(|_| rng.random_range(1..=10))
            .collect::<Vec<_>>();
        let tch_a = tch::Tensor::randn(&shape, (TCH_TEST_TYPES, tch::Device::Cpu));
        let mut a = Tensor::<TestTypes>::empty(&shape)?;
        a.as_raw_mut().copy_from_slice(unsafe {
            std::slice::from_raw_parts(tch_a.data_ptr() as *mut TestTypes, tch_a.numel())
        });

        for i in 0..ndim {
            let k = rng.random_range(1..=shape[i]);
            let (_, b_values) = a.topk(k as i64, i as i64, true, true)?;
            let (tch_b_values, _) = tch_a.topk(k as i64, i as i64, true, true);
            let tch_res = unsafe {
                Tensor::<TestTypes>::from_raw(
                    tch_b_values.data_ptr() as *mut TestTypes,
                    &b_values.shape().to_vec(),
                )
            }?;
            assert!(b_values.allclose(&tch_res, TEST_RTOL, TEST_ATOL));
        }
    }
    Ok(())
}

#[test]
fn test_uncontiguous() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    for _ in 0..100 {
        let ndim = rng.random_range(1..=5);
        let shape = (0..ndim)
            .map(|_| rng.random_range(1..=10))
            .collect::<Vec<_>>();
        let tch_a = tch::Tensor::randn(&shape, (TCH_TEST_TYPES, tch::Device::Cpu));
        let mut a = Tensor::<TestTypes>::empty(&shape)?;
        a.as_raw_mut().copy_from_slice(unsafe {
            std::slice::from_raw_parts(tch_a.data_ptr() as *mut TestTypes, tch_a.numel())
        });
        let permute_shape = (0..ndim).rev().collect::<Vec<_>>();
        let a = a.permute(&permute_shape)?;
        let tch_a = tch_a.permute(&permute_shape);
        for i in 0..ndim {
            let k = rng.random_range(1..=a.shape()[i as usize]);
            let (_, b_values) = a.topk(k as i64, i as i64, true, true)?;
            let (tch_b_values, _) = tch_a.topk(k as i64, i as i64, true, true);
            let tch_res = unsafe {
                Tensor::<TestTypes>::from_raw(
                    tch_b_values.data_ptr() as *mut TestTypes,
                    &b_values.shape().to_vec(),
                )
            }?;
            assert!(b_values.allclose(&tch_res, TEST_RTOL, TEST_ATOL));
        }
    }
    Ok(())
}

#[test]
fn test_2dim_uncontiguous_sub_tensor() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    let ndim = 5;
    for _ in 0..100 {
        let shape = (0..ndim)
            .map(|_| rng.random_range(1..=10))
            .collect::<Vec<_>>();
        let tch_a = tch::Tensor::randn(&shape, (TCH_TEST_TYPES, tch::Device::Cpu));
        let mut a = Tensor::<TestTypes>::empty(&shape)?;
        a.as_raw_mut().copy_from_slice(unsafe {
            std::slice::from_raw_parts(tch_a.data_ptr() as *mut TestTypes, tch_a.numel())
        });
        let permute_shape = (0..ndim).rev().collect::<Vec<_>>();
        let a = a.permute(&permute_shape)?;
        let tch_a = tch_a.permute(&permute_shape);

        let dim0_min = rng.random_range(0..a.shape()[0]);
        let dim0_max = rng.random_range((dim0_min + 1)..=a.shape()[0]);
        let dim0_step = if dim0_max > dim0_min {
            rng.random_range(1..=(dim0_max - dim0_min).min(3))
        } else {
            1
        };
        let dim1_min = rng.random_range(0..a.shape()[1]);
        let dim1_max = rng.random_range((dim1_min + 1)..=a.shape()[1]);
        let dim1_step = if dim1_max > dim1_min {
            rng.random_range(1..=(dim1_max - dim1_min).min(3))
        } else {
            1
        };
        let dim2_min = rng.random_range(0..a.shape()[2]);
        let dim2_max = rng.random_range((dim2_min + 1)..=a.shape()[2]);
        let dim2_step = if dim2_max > dim2_min {
            rng.random_range(1..=(dim2_max - dim2_min).min(3))
        } else {
            1
        };
        let dim3_min = rng.random_range(0..a.shape()[3]);
        let dim3_max = rng.random_range((dim3_min + 1)..=a.shape()[3]);
        let dim3_step = if dim3_max > dim3_min {
            rng.random_range(1..=(dim3_max - dim3_min).min(3))
        } else {
            1
        };
        let dim4_min = rng.random_range(0..a.shape()[4]);
        let dim4_max = rng.random_range((dim4_min + 1)..=a.shape()[4]);
        let dim4_step = if dim4_max > dim4_min {
            rng.random_range(1..=(dim4_max - dim4_min).min(3))
        } else {
            1
        };

        let a = slice!(a[dim0_min:dim0_max:dim0_step, dim1_min:dim1_max:dim1_step, dim2_min:dim2_max:dim2_step, dim3_min:dim3_max:dim3_step, dim4_min:dim4_max:dim4_step])?;
        let tch_a = tch_a
            .slice(0, dim0_min, dim0_max, dim0_step)
            .slice(1, dim1_min, dim1_max, dim1_step)
            .slice(2, dim2_min, dim2_max, dim2_step)
            .slice(3, dim3_min, dim3_max, dim3_step)
            .slice(4, dim4_min, dim4_max, dim4_step);

        for i in 0..ndim {
            let k = rng.random_range(1..=a.shape()[i as usize]);
            let (_, b_values) = a.topk(k, i, true, true)?;
            let (tch_b_values, _) = tch_a.topk(k, i, true, true);
            let tch_res = unsafe {
                Tensor::<TestTypes>::from_raw(
                    tch_b_values.data_ptr() as *mut TestTypes,
                    &b_values.shape().to_vec(),
                )
            }?;
            assert!(b_values.allclose(&tch_res, TEST_RTOL, TEST_ATOL));
        }
    }
    Ok(())
}
