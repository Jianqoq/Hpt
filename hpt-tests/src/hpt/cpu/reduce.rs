#![allow(unused)]
use core::f64;

use duplicate::duplicate_item;
use hpt::common::TensorInfo;
use hpt::ops::*;
use hpt::slice;
use hpt::{backend::Cpu, common::cpu::TensorLike};
use rand::seq::SliceRandom;
use rand::Rng;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use tch::Tensor;

use crate::utils::random_utils::generate_all_combinations;
use crate::TestTypes;
use crate::TCH_TEST_TYPES;
use crate::TEST_ATOL;
use crate::TEST_RTOL;

#[track_caller]
fn assert_eq(a: &hpt::Tensor<TestTypes>, b: &Tensor) {
    let raw = a.as_raw();
    if a.size() != b.size().into_iter().product::<i64>() as usize {
        println!("a size {:?}", a.shape());
        println!("b size {:?}", b.size());
    }
    let tch_res = unsafe {
        hpt::Tensor::<TestTypes>::from_raw(b.data_ptr() as *mut TestTypes, &a.shape().to_vec())
    }.expect("Failed to convert tch tensor to hpt tensor");
    assert!(a.allclose(&tch_res, TEST_ATOL, TEST_RTOL));
}

#[track_caller]
fn assert_eq_i64(a: &hpt::Tensor<i64>, b: &Tensor) {
    let raw = a.as_raw();
    let tch_res = unsafe {
        hpt::Tensor::<i64>::from_raw(b.data_ptr() as *mut i64, &a.shape().to_vec())
    }.expect("Failed to convert tch tensor to hpt tensor");
    assert!(a.allclose(&tch_res, 0, 0));
}

#[track_caller]
fn assert_eq_bool(a: &hpt::Tensor<bool>, b: &Tensor) {
    let raw = a.as_raw();
    let tch_raw = unsafe { core::slice::from_raw_parts(b.data_ptr() as *const bool, a.size()) };
    let caller = core::panic::Location::caller();
    raw.par_iter().zip(tch_raw.par_iter()).for_each(|(a, b)| {
        if a != b {
            panic!("{} != {}, at {}", a, b, caller)
        }
    });
}

fn common_input(end: i64, shape: &[i64]) -> anyhow::Result<(hpt::Tensor<TestTypes, Cpu>, Tensor)> {
    let a = hpt::Tensor::<TestTypes, Cpu>::arange(0, end)?.reshape(shape)?;
    let tch_a = Tensor::arange(end, (TCH_TEST_TYPES, tch::Device::Cpu)).reshape(shape);
    Ok((a, tch_a))
}

#[duplicate_item(
    func                hpt_method      hpt_inplace      tch_method;
    [test_sum]          [sum]           [sum_]           [sum_dim_intlist];
    [test_nan_sum]      [nansum]        [nansum_]        [nansum];
)]
#[test]
fn func() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    for _ in 0..100 {
        let ndim = rng.random_range(1..=3);
        let shape = (0..ndim)
            .map(|_| rng.random_range(1..=32))
            .collect::<Vec<i64>>();
        let (a, tch_a) = common_input(shape.iter().product(), shape.as_slice())?;
        let combinations = generate_all_combinations(&(0..ndim).collect::<Vec<_>>());
        for axes in combinations {
            let sum = a.hpt_method(&axes, true)?;
            let tch_sum = tch_a.tch_method(axes.as_slice(), true, TCH_TEST_TYPES);
            assert_eq(&sum, &tch_sum);
            let sum = a.hpt_inplace(&axes, true, true, sum)?;
            assert_eq(&sum, &tch_sum);
        }
    }
    Ok(())
}

#[duplicate_item(
    func                                hpt_method      tch_method;
    [test_uncontiguous_sum]             [sum]           [sum_dim_intlist];
    [test_uncontiguous_nan_sum]         [nansum]        [nansum];
)]
#[test]
fn func() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    for idx in 0..100 {
        let ndim = rng.random_range(1..=3usize);
        let shape = (0..ndim)
            .map(|_| rng.random_range(1..=32))
            .collect::<Vec<i64>>();
        let (a, tch_a) = common_input(shape.iter().product(), shape.as_slice())?;
        let mut axes = (0..ndim).map(|i| i as i64).collect::<Vec<_>>();
        axes.shuffle(&mut rng);
        let a = a.permute(&axes)?;
        let tch_a = tch_a.permute(&axes);
        let combinations = generate_all_combinations(&(0..ndim).collect::<Vec<_>>());
        for axes in combinations {
            let sum = a.hpt_method(&axes, true)?;
            let tch_sum = tch_a.tch_method(axes.as_slice(), true, TCH_TEST_TYPES);
            assert_eq(&sum, &tch_sum);
        }
    }
    Ok(())
}

#[duplicate_item(
    func                             hpt_method      tch_method;
    [test_sub_tensor_sum]            [sum]           [sum_dim_intlist];
    [test_sub_tensor_nansum]         [nansum]        [nansum];
)]
#[test]
fn func() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    for _ in 0..100 {
        let shape = [
            rng.random_range(1..32),
            rng.random_range(1..32),
            rng.random_range(1..32),
        ];
        let (a, tch_a) = common_input(shape.iter().product(), &shape)?;
        let dim0_max = if shape[0] > 1 {
            rng.random_range(1..shape[0])
        } else {
            1
        };
        let dim0_min = if dim0_max > 0 {
            rng.random_range(0..dim0_max)
        } else {
            0
        };

        let dim1_max = if shape[1] > 1 {
            rng.random_range(1..shape[1])
        } else {
            1
        };
        let dim1_min = if dim1_max > 0 {
            rng.random_range(0..dim1_max)
        } else {
            0
        };

        let dim2_max = if shape[2] > 1 {
            rng.random_range(1..shape[2])
        } else {
            1
        };
        let dim2_min = if dim2_max > 0 {
            rng.random_range(0..dim2_max)
        } else {
            0
        };

        let a = slice!(a[dim0_min:dim0_max, dim1_min:dim1_max, dim2_min:dim2_max])?;
        let tch_a = tch_a
            .slice(0, dim0_min, dim0_max, 1)
            .slice(1, dim1_min, dim1_max, 1)
            .slice(2, dim2_min, dim2_max, 1);
        let sum = a.hpt_method(0, true)?;
        let tch_sum = tch_a.tch_method(0, true, TCH_TEST_TYPES);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_method(1, false)?;
        let tch_sum = tch_a.tch_method(1, false, TCH_TEST_TYPES);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_method(2, false)?;
        let tch_sum = tch_a.tch_method(2, false, TCH_TEST_TYPES);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_method([0, 1], false)?;
        let tch_sum = tch_a.tch_method(&[0, 1][..], false, TCH_TEST_TYPES);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_method([0, 2], false)?;
        let tch_sum = tch_a.tch_method(&[0, 2][..], false, TCH_TEST_TYPES);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_method([1, 2], false)?;
        let tch_sum = tch_a.tch_method(&[1, 2][..], false, TCH_TEST_TYPES);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_method([0, 1, 2], false)?;
        let tch_sum = tch_a.tch_method(&[0, 1, 2][..], false, TCH_TEST_TYPES);
        assert_eq(&sum, &tch_sum);
    }
    Ok(())
}

#[duplicate_item(
    func                                  hpt_method      tch_method;
    [test_sub_tensor_sum_step]            [sum]           [sum_dim_intlist];
    [test_sub_tensor_nansum_step]         [nansum]        [nansum];
)]
#[test]
fn func() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    for _ in 0..100 {
        let shape = [
            rng.random_range(1..32),
            rng.random_range(1..32),
            rng.random_range(1..32),
        ];
        let (a, tch_a) = common_input(shape.iter().product(), &shape)?;
        let dim0_max = if shape[0] > 1 {
            rng.random_range(1..shape[0])
        } else {
            1
        };
        let dim0_min = if dim0_max > 0 {
            rng.random_range(0..dim0_max)
        } else {
            0
        };
        let dim0_step = if dim0_max > dim0_min {
            rng.random_range(1..=(dim0_max - dim0_min).min(2))
        } else {
            1
        };

        let dim1_max = if shape[1] > 1 {
            rng.random_range(1..shape[1])
        } else {
            1
        };
        let dim1_min = if dim1_max > 0 {
            rng.random_range(0..dim1_max)
        } else {
            0
        };

        let dim1_step = if dim1_max > dim1_min {
            rng.random_range(1..=(dim1_max - dim1_min).min(2))
        } else {
            1
        };

        let dim2_max = if shape[2] > 1 {
            rng.random_range(1..shape[2])
        } else {
            1
        };
        let dim2_min = if dim2_max > 0 {
            rng.random_range(0..dim2_max)
        } else {
            0
        };

        let dim2_step = if dim2_max > dim2_min {
            rng.random_range(1..=(dim2_max - dim2_min).min(2))
        } else {
            1
        };

        let a = slice!(
            a[dim0_min:dim0_max:dim0_step,
             dim1_min:dim1_max:dim1_step, 
             dim2_min:dim2_max:dim2_step])?;
        let tch_a = tch_a
            .slice(0, dim0_min, dim0_max, dim0_step)
            .slice(1, dim1_min, dim1_max, dim1_step)
            .slice(2, dim2_min, dim2_max, dim2_step);
        let sum = a.hpt_method(0, true)?;
        let tch_sum = tch_a.tch_method(0, true, TCH_TEST_TYPES);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_method(1, false)?;
        let tch_sum = tch_a.tch_method(1, false, TCH_TEST_TYPES);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_method(2, false)?;
        let tch_sum = tch_a.tch_method(2, false, TCH_TEST_TYPES);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_method([0, 1], false)?;
        let tch_sum = tch_a.tch_method(&[0, 1][..], false, TCH_TEST_TYPES);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_method([0, 2], false)?;
        let tch_sum = tch_a.tch_method(&[0, 2][..], false, TCH_TEST_TYPES);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_method([1, 2], false)?;
        let tch_sum = tch_a.tch_method(&[1, 2][..], false, TCH_TEST_TYPES);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_method([0, 1, 2], false)?;
        let tch_sum = tch_a.tch_method(&[0, 1, 2][..], false, TCH_TEST_TYPES);
        assert_eq(&sum, &tch_sum);
    }
    Ok(())
}

#[test]
fn test_prod() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, &[2, 5, 10])?;
    let prod = a.prod(0, false)?;
    let tch_prod = tch_a.prod_dim_int(0, false, TCH_TEST_TYPES);
    assert_eq(&prod, &tch_prod);

    let prod = a.prod(1, false)?;
    let tch_prod = tch_a.prod_dim_int(1, false, TCH_TEST_TYPES);
    assert_eq(&prod, &tch_prod);

    let prod = a.prod(2, false)?;
    let tch_prod = tch_a.prod_dim_int(2, false, TCH_TEST_TYPES);
    assert_eq(&prod, &tch_prod);

    Ok(())
}

#[test]
fn test_nanprod() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, &[2, 5, 10])?;
    let prod = a.nanprod(0, false)?;
    let tch_prod = tch_a.prod_dim_int(0, false, TCH_TEST_TYPES);
    assert_eq(&prod, &tch_prod);

    let prod = a.nanprod(1, false)?;
    let tch_prod = tch_a.prod_dim_int(1, false, TCH_TEST_TYPES);
    assert_eq(&prod, &tch_prod);

    let prod = a.nanprod(2, false)?;
    let tch_prod = tch_a.prod_dim_int(2, false, TCH_TEST_TYPES);
    assert_eq(&prod, &tch_prod);

    Ok(())
}

#[test]
fn test_mean() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, &[2, 5, 10])?;
    let mean = a.mean(0, false)?;
    let tch_mean = tch_a.mean_dim(0, false, TCH_TEST_TYPES);
    assert_eq(&mean, &tch_mean);
    let mean = a.mean(1, false)?;
    let tch_mean = tch_a.mean_dim(1, false, TCH_TEST_TYPES);
    assert_eq(&mean, &tch_mean);
    let mean = a.mean(2, false)?;
    let tch_mean = tch_a.mean_dim(2, false, TCH_TEST_TYPES);
    assert_eq(&mean, &tch_mean);
    let mean = a.mean([0, 1], false)?;
    let tch_mean = tch_a.mean_dim(&[0, 1][..], false, TCH_TEST_TYPES);
    assert_eq(&mean, &tch_mean);
    let mean = a.mean([0, 2], false)?;
    let tch_mean = tch_a.mean_dim(&[0, 2][..], false, TCH_TEST_TYPES);
    assert_eq(&mean, &tch_mean);
    let mean = a.mean([1, 2], false)?;
    let tch_mean = tch_a.mean_dim(&[1, 2][..], false, TCH_TEST_TYPES);
    assert_eq(&mean, &tch_mean);
    let mean = a.mean([0, 1, 2], false)?;
    let tch_mean = tch_a.mean_dim(&[0, 1, 2][..], false, TCH_TEST_TYPES);
    assert_eq(&mean, &tch_mean);
    Ok(())
}

#[test]
fn test_logsumexp() -> anyhow::Result<()> {
    // TODO: make logsumexp stable
    // let (a, tch_a) = common_input(2 * 5 * 10, &[2, 5, 10])?;
    // let mean = a.logsumexp(0, false)?;
    // let tch_mean = tch_a
    //     .logsumexp(0, false)
    //     .to_dtype(TCH_TEST_TYPES, false, true);
    // assert_eq(&mean, &tch_mean);
    // let mean = a.logsumexp(1, false)?;
    // let tch_mean = tch_a
    //     .logsumexp(1, false)
    //     .to_dtype(TCH_TEST_TYPES, false, true);
    // assert_eq(&mean, &tch_mean);
    // let mean = a.logsumexp(2, false)?;
    // let tch_mean = tch_a
    //     .logsumexp(2, false)
    //     .to_dtype(TCH_TEST_TYPES, false, true);
    // assert_eq(&mean, &tch_mean);
    // let mean = a.logsumexp([0, 1], false)?;
    // let tch_mean = tch_a
    //     .logsumexp(&[0, 1][..], false)
    //     .to_dtype(TCH_TEST_TYPES, false, true);
    // assert_eq(&mean, &tch_mean);
    // let mean = a.logsumexp([0, 2], false)?;
    // let tch_mean = tch_a
    //     .logsumexp(&[0, 2][..], false)
    //     .to_dtype(TCH_TEST_TYPES, false, true);
    // assert_eq(&mean, &tch_mean);
    // let mean = a.logsumexp([1, 2], false)?;
    // let tch_mean = tch_a
    //     .logsumexp(&[1, 2][..], false)
    //     .to_dtype(TCH_TEST_TYPES, false, true);
    // assert_eq(&mean, &tch_mean);
    // let mean = a.logsumexp([0, 1, 2], false)?;
    // let tch_mean = tch_a
    //     .logsumexp(&[0, 1, 2][..], false)
    //     .to_dtype(TCH_TEST_TYPES, false, true);
    // assert_eq(&mean, &tch_mean);
    Ok(())
}

#[test]
fn test_max() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, &[2, 5, 10])?;
    let max = a.max(0, false)?;
    let (tch_max, _) = tch_a.max_dim(0, false);
    assert_eq(&max, &tch_max);

    let max = a.max(1, false)?;
    let (tch_max, _) = tch_a.max_dim(1, false);
    assert_eq(&max, &tch_max);

    let max = a.max(2, false)?;
    let (tch_max, _) = tch_a.max_dim(2, false);
    assert_eq(&max, &tch_max);

    Ok(())
}

#[test]
fn test_min() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, &[2, 5, 10])?;
    let min = a.min(0, false)?;
    let (tch_min, _) = tch_a.min_dim(0, false);
    assert_eq(&min, &tch_min);

    let min = a.min(1, false)?;
    let (tch_min, _) = tch_a.min_dim(1, false);
    assert_eq(&min, &tch_min);

    let min = a.min(2, false)?;
    let (tch_min, _) = tch_a.min_dim(2, false);
    assert_eq(&min, &tch_min);
    Ok(())
}

#[test]
fn test_sum_square() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, &[2, 5, 10])?;
    let sum = a.sum_square(0, false)?;
    let tch_sum = tch_a
        .pow_tensor_scalar(2)
        .sum_dim_intlist(0, false, TCH_TEST_TYPES);
    assert_eq(&sum, &tch_sum);

    let sum = a.sum_square(1, false)?;
    let tch_sum = tch_a
        .pow_tensor_scalar(2)
        .sum_dim_intlist(1, false, TCH_TEST_TYPES);
    assert_eq(&sum, &tch_sum);

    let sum = a.sum_square(2, false)?;
    let tch_sum = tch_a
        .pow_tensor_scalar(2)
        .sum_dim_intlist(2, false, TCH_TEST_TYPES);
    assert_eq(&sum, &tch_sum);

    let sum = a.sum_square([0, 1], false)?;
    let tch_sum = tch_a
        .pow_tensor_scalar(2)
        .sum_dim_intlist(&[0, 1][..], false, TCH_TEST_TYPES);
    assert_eq(&sum, &tch_sum);

    let sum = a.sum_square([0, 2], false)?;
    let tch_sum = tch_a
        .pow_tensor_scalar(2)
        .sum_dim_intlist(&[0, 2][..], false, TCH_TEST_TYPES);
    assert_eq(&sum, &tch_sum);

    let sum = a.sum_square([1, 2], false)?;
    let tch_sum = tch_a
        .pow_tensor_scalar(2)
        .sum_dim_intlist(&[1, 2][..], false, TCH_TEST_TYPES);
    assert_eq(&sum, &tch_sum);

    let sum = a.sum_square([0, 1, 2], false)?;
    let tch_sum =
        tch_a
            .pow_tensor_scalar(2)
            .sum_dim_intlist(&[0, 1, 2][..], false, TCH_TEST_TYPES);
    assert_eq(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_reducel1() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, &[2, 5, 10])?;
    let sum = a.reducel1(0, false)?;
    let res = Tensor::empty(sum.shape().inner(), (TCH_TEST_TYPES, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 1, 0, false)?;
    assert_eq(&sum, &tch_sum);
    let sum = a.reducel1(1, false)?;
    let res = Tensor::empty(sum.shape().inner(), (TCH_TEST_TYPES, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 1, 1, false)?;
    assert_eq(&sum, &tch_sum);
    let sum = a.reducel1(2, false)?;
    let res = Tensor::empty(sum.shape().inner(), (TCH_TEST_TYPES, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 1, 2, false)?;
    assert_eq(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_reducel2() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(1 * 1 * 10, &[1, 1, 10])?;
    let sum = a.reducel2(0, false)?;
    let res = Tensor::empty(sum.shape().inner(), (TCH_TEST_TYPES, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 2, 0, false)?;
    assert_eq(&sum, &tch_sum);
    let sum = a.reducel2(1, false)?;
    let res = Tensor::empty(sum.shape().inner(), (TCH_TEST_TYPES, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 2, 1, false)?;
    assert_eq(&sum, &tch_sum);
    let sum = a.reducel2(2, false)?;
    let res = Tensor::empty(sum.shape().inner(), (TCH_TEST_TYPES, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 2, 2, false)?;
    assert_eq(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_reducel3() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, &[2, 5, 10])?;
    let sum = a.reducel3(0, false)?;
    let res = Tensor::empty(sum.shape().inner(), (TCH_TEST_TYPES, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 3, 0, false)?;
    assert_eq(&sum, &tch_sum);
    let sum = a.reducel3(1, false)?;
    let res = Tensor::empty(sum.shape().inner(), (TCH_TEST_TYPES, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 3, 1, false)?;
    assert_eq(&sum, &tch_sum);
    let sum = a.reducel3(2, false)?;
    let res = Tensor::empty(sum.shape().inner(), (TCH_TEST_TYPES, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 3, 2, false)?;
    assert_eq(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_argmin() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, &[2, 5, 10])?;
    let sum = a.argmin(0, false)?;
    let tch_sum = tch_a.argmin(0, false);
    assert_eq_i64(&sum, &tch_sum);
    let sum = a.argmin(1, false)?;
    let tch_sum = tch_a.argmin(1, false);
    assert_eq_i64(&sum, &tch_sum);
    let sum = a.argmin(2, false)?;
    let tch_sum = tch_a.argmin(2, false);
    assert_eq_i64(&sum, &tch_sum);
    let a = a.permute([1, 0, 2])?;
    let tch_a = tch_a.permute(&[1, 0, 2][..]);
    let sum = a.argmin(0, false)?;
    let tch_sum = tch_a.argmin(0, false);
    assert_eq_i64(&sum, &tch_sum);
    let sum = a.argmin(1, false)?;
    let tch_sum = tch_a.argmin(1, false);
    assert_eq_i64(&sum, &tch_sum);
    let sum = a.argmin(2, false)?;
    let tch_sum = tch_a.argmin(2, false);
    assert_eq_i64(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_argmax() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, &[2, 5, 10])?;
    let sum = a.argmax(0, false)?;
    let tch_sum = tch_a.argmax(0, false);
    assert_eq_i64(&sum, &tch_sum);
    let sum = a.argmax(1, false)?;
    let tch_sum = tch_a.argmax(1, false);
    assert_eq_i64(&sum, &tch_sum);
    let sum = a.argmax(2, false)?;
    let tch_sum = tch_a.argmax(2, false);
    assert_eq_i64(&sum, &tch_sum);
    let a = a.permute([1, 0, 2])?;
    let tch_a = tch_a.permute(&[1, 0, 2][..]);
    let sum = a.argmax(0, false)?;
    let tch_sum = tch_a.argmax(0, false);
    assert_eq_i64(&sum, &tch_sum);
    let sum = a.argmax(1, false)?;
    let tch_sum = tch_a.argmax(1, false);
    assert_eq_i64(&sum, &tch_sum);
    let sum = a.argmax(2, false)?;
    let tch_sum = tch_a.argmax(2, false);
    assert_eq_i64(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_all() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, &[2, 5, 10])?;
    let sum = a.all(0, false)?;
    let tch_sum = tch_a.all_dims(0, false);
    assert_eq_bool(&sum, &tch_sum);

    let sum = a.all(1, false)?;
    let tch_sum = tch_a.all_dims(1, false);
    assert_eq_bool(&sum, &tch_sum);

    let sum = a.all(2, false)?;
    let tch_sum = tch_a.all_dims(2, false);
    assert_eq_bool(&sum, &tch_sum);

    let sum = a.all([0, 1], false)?;
    let tch_sum = tch_a.all_dims(&[0, 1][..], false);
    assert_eq_bool(&sum, &tch_sum);

    let sum = a.all([0, 2], false)?;
    let tch_sum = tch_a.all_dims(&[0, 2][..], false);
    assert_eq_bool(&sum, &tch_sum);

    let sum = a.all([1, 2], false)?;
    let tch_sum = tch_a.all_dims(&[1, 2][..], false);
    assert_eq_bool(&sum, &tch_sum);

    let sum = a.all([0, 1, 2], false)?;
    let tch_sum = tch_a.all_dims(&[0, 1, 2][..], false);
    assert_eq_bool(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_any() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, &[2, 5, 10])?;
    let sum = a.any(0, false)?;
    let tch_sum = tch_a.any_dims(0, false);
    assert_eq_bool(&sum, &tch_sum);

    let sum = a.any(1, false)?;
    let tch_sum = tch_a.any_dims(1, false);
    assert_eq_bool(&sum, &tch_sum);

    let sum = a.any(2, false)?;
    let tch_sum = tch_a.any_dims(2, false);
    assert_eq_bool(&sum, &tch_sum);

    let sum = a.any([0, 1], false)?;
    let tch_sum = tch_a.any_dims(&[0, 1][..], false);
    assert_eq_bool(&sum, &tch_sum);

    let sum = a.any([0, 2], false)?;
    let tch_sum = tch_a.any_dims(&[0, 2][..], false);
    assert_eq_bool(&sum, &tch_sum);

    let sum = a.any([1, 2], false)?;
    let tch_sum = tch_a.any_dims(&[1, 2][..], false);
    assert_eq_bool(&sum, &tch_sum);

    let sum = a.any([0, 1, 2], false)?;
    let tch_sum = tch_a.any_dims(&[0, 1, 2][..], false);
    assert_eq_bool(&sum, &tch_sum);
    Ok(())
}
