#![allow(unused)]
use core::f64;

use hpt::Cpu;
use duplicate::duplicate_item;
use hpt::*;
use hpt_common::slice;
use rand::Rng;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use tch::Tensor;

#[track_caller]
fn assert_eq(a: &hpt::tensor::Tensor<i64>, b: &Tensor) {
    let raw = a.as_raw();
    if a.size() != b.size().into_iter().product::<i64>() as usize {
        println!("a size {:?}", a.shape());
        println!("b size {:?}", b.size());
    }
    let tch_raw = unsafe {
        core::slice::from_raw_parts(
            b.data_ptr() as *const i64,
            b.size().into_iter().product::<i64>() as usize,
        )
    };
    let caller = core::panic::Location::caller();
    raw.par_iter().zip(tch_raw.par_iter()).for_each(|(a, b)| {
        if a != b {
            panic!("{} != {}, at {}", a, b, caller)
        }
    });
}

#[track_caller]
fn assert_eq_bool(a: &hpt::tensor::Tensor<bool>, b: &Tensor) {
    let raw = a.as_raw();
    let tch_raw = unsafe { core::slice::from_raw_parts(b.data_ptr() as *const bool, a.size()) };
    let caller = core::panic::Location::caller();
    raw.par_iter().zip(tch_raw.par_iter()).for_each(|(a, b)| {
        if a != b {
            panic!("{} != {}, at {}", a, b, caller)
        }
    });
}

#[allow(unused)]
#[track_caller]
fn assert_eq_f64(b: &hpt::tensor::Tensor<f64>, a: &Tensor) {
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
    let caller = core::panic::Location::caller();
    a_raw.iter().zip(b_raw.iter()).for_each(|(a, b)| {
        let abs_diff = (*a - *b).abs();
        let rel_diff = if *a == 0.0 && *b == 0.0 {
            0.0
        } else {
            abs_diff / (a.abs() + b.abs() + f64::EPSILON)
        };

        if rel_diff > 0.05 {
            panic!("{} != {} (relative_diff: {})", *a, *b, rel_diff);
        }
    });
}

#[allow(unused)]
#[track_caller]
fn assert_eq_f64_10(b: &hpt::tensor::Tensor<f64>, a: &Tensor) {
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
    let tolerance = 10e-10;
    let caller = core::panic::Location::caller();
    a_raw.iter().zip(b_raw.iter()).for_each(|(a, b)| {
        let abs_diff = (*a - *b).abs();
        let rel_diff = if *a == 0.0 && *b == 0.0 {
            0.0
        } else {
            abs_diff / (a.abs() + b.abs() + f64::EPSILON)
        };

        if rel_diff > 0.05 {
            panic!("{} != {} (relative_diff: {})", *a, *b, rel_diff);
        }
    });
}

fn common_input<const N: usize>(
    end: i64,
    shape: [i64; N],
) -> anyhow::Result<(hpt::tensor::Tensor<i64, Cpu>, Tensor)> {
    let a = hpt::tensor::Tensor::<i64, Cpu>::arange(0, end)?.reshape(&shape)?;
    let tch_a = Tensor::arange(end, (tch::Kind::Int64, tch::Device::Cpu)).reshape(&shape);
    Ok((a, tch_a))
}

fn common_input_f64<const N: usize>(
    end: i64,
    shape: [i64; N],
) -> anyhow::Result<(hpt::tensor::Tensor<f64, Cpu>, Tensor)> {
    let tch_a = Tensor::randn(&shape, (tch::Kind::Double, tch::Device::Cpu)).reshape(&shape);
    let mut a = hpt::tensor::Tensor::<f64, Cpu>::empty(&shape)?;
    let a_size = a.size();
    let raw_mut = a.as_raw_mut();
    let tch_raw = unsafe { core::slice::from_raw_parts_mut(tch_a.data_ptr() as *mut f64, a_size) };
    raw_mut
        .par_iter_mut()
        .zip(tch_raw.par_iter())
        .for_each(|(a, b)| {
            *a = *b;
        });
    Ok((a, tch_a))
}

#[duplicate_item(
    func                hpt_method      hpt_inplace      tch_method;
    [test_sum]          [sum]           [sum_]           [sum_dim_intlist];
    [test_nan_sum]      [nansum]        [nansum_]        [nansum];
)]
#[test]
fn func() -> anyhow::Result<()> {
    let mut rng = rand::thread_rng();
    for _ in 0..100 {
        let shape = [
            rng.gen_range(1..32),
            rng.gen_range(1..32),
            rng.gen_range(1..32),
        ];
        let (a, tch_a) = common_input(shape.iter().product(), shape)?;
        let sum = a.hpt_method(1, true)?;
        let tch_sum = tch_a.tch_method(1, true, tch::Kind::Int64);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_inplace(1, true, true, sum)?;
        assert_eq(&sum, &tch_sum);

        let sum = a.hpt_method(2, true)?;
        let tch_sum = tch_a.tch_method(2, true, tch::Kind::Int64);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_inplace(2, true, true, sum)?;
        assert_eq(&sum, &tch_sum);

        let sum = a.hpt_method([0, 1], true)?;
        let tch_sum = tch_a.tch_method(&[0, 1][..], true, tch::Kind::Int64);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_inplace([0, 1], true, true, sum)?;
        assert_eq(&sum, &tch_sum);

        let sum = a.hpt_method([0, 2], true)?;
        let tch_sum = tch_a.tch_method(&[0, 2][..], true, tch::Kind::Int64);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_inplace([0, 2], true, true, sum)?;
        assert_eq(&sum, &tch_sum);

        let sum = a.hpt_method([1, 2], true)?;
        let tch_sum = tch_a.tch_method(&[1, 2][..], true, tch::Kind::Int64);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_inplace([1, 2], true, true, sum)?;
        assert_eq(&sum, &tch_sum);

        let sum = a.hpt_method([0, 1, 2], true)?;
        let tch_sum = tch_a.tch_method(&[0, 1, 2][..], true, tch::Kind::Int64);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_inplace([0, 1, 2], true, true, sum)?;
        assert_eq(&sum, &tch_sum);
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
    let mut rng = rand::thread_rng();
    for _ in 0..100 {
        let shape = [
            rng.gen_range(1..32),
            rng.gen_range(1..32),
            rng.gen_range(1..32),
            rng.gen_range(1..32),
        ];
        let (a, tch_a) = common_input(shape.iter().product(), shape)?;
        let a = a.permute([1, 0, 3, 2])?;
        let tch_a = tch_a.permute(&[1, 0, 3, 2][..]);
        let sum = a.hpt_method(0, false)?;
        let tch_sum = tch_a.tch_method(0, false, tch::Kind::Int64);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_method(1, false)?;
        let tch_sum = tch_a.tch_method(1, false, tch::Kind::Int64);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_method(2, false)?;
        let tch_sum = tch_a.tch_method(2, false, tch::Kind::Int64);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_method([0, 1], false)?;
        let tch_sum = tch_a.tch_method(&[0, 1][..], false, tch::Kind::Int64);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_method([0, 2], false)?;
        let tch_sum = tch_a.tch_method(&[0, 2][..], false, tch::Kind::Int64);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_method([1, 2], false)?;
        let tch_sum = tch_a.tch_method(&[1, 2][..], false, tch::Kind::Int64);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_method([0, 1, 2], false)?;
        let tch_sum = tch_a.tch_method(&[0, 1, 2][..], false, tch::Kind::Int64);
        assert_eq(&sum, &tch_sum);
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
    let mut rng = rand::thread_rng();
    for _ in 0..100 {
        let shape = [
            rng.gen_range(1..32),
            rng.gen_range(1..32),
            rng.gen_range(1..32),
        ];
        let (a, tch_a) = common_input(shape.iter().product(), shape)?;
        let dim0_max = if shape[0] > 1 {
            rng.gen_range(1..shape[0])
        } else {
            1
        };
        let dim0_min = if dim0_max > 0 {
            rng.gen_range(0..dim0_max)
        } else {
            0
        };

        let dim1_max = if shape[1] > 1 {
            rng.gen_range(1..shape[1])
        } else {
            1
        };
        let dim1_min = if dim1_max > 0 {
            rng.gen_range(0..dim1_max)
        } else {
            0
        };

        let dim2_max = if shape[2] > 1 {
            rng.gen_range(1..shape[2])
        } else {
            1
        };
        let dim2_min = if dim2_max > 0 {
            rng.gen_range(0..dim2_max)
        } else {
            0
        };

        let a = slice!(a[dim0_min:dim0_max, dim1_min:dim1_max, dim2_min:dim2_max])?;
        let tch_a = tch_a
            .slice(0, dim0_min, dim0_max, 1)
            .slice(1, dim1_min, dim1_max, 1)
            .slice(2, dim2_min, dim2_max, 1);
        let sum = a.hpt_method(0, true)?;
        let tch_sum = tch_a.tch_method(0, true, tch::Kind::Int64);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_method(1, false)?;
        let tch_sum = tch_a.tch_method(1, false, tch::Kind::Int64);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_method(2, false)?;
        let tch_sum = tch_a.tch_method(2, false, tch::Kind::Int64);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_method([0, 1], false)?;
        let tch_sum = tch_a.tch_method(&[0, 1][..], false, tch::Kind::Int64);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_method([0, 2], false)?;
        let tch_sum = tch_a.tch_method(&[0, 2][..], false, tch::Kind::Int64);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_method([1, 2], false)?;
        let tch_sum = tch_a.tch_method(&[1, 2][..], false, tch::Kind::Int64);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_method([0, 1, 2], false)?;
        let tch_sum = tch_a.tch_method(&[0, 1, 2][..], false, tch::Kind::Int64);
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
    let mut rng = rand::thread_rng();
    for _ in 0..100 {
        let shape = [
            rng.gen_range(1..32),
            rng.gen_range(1..32),
            rng.gen_range(1..32),
        ];
        let (a, tch_a) = common_input(shape.iter().product(), shape)?;
        let dim0_max = if shape[0] > 1 {
            rng.gen_range(1..shape[0])
        } else {
            1
        };
        let dim0_min = if dim0_max > 0 {
            rng.gen_range(0..dim0_max)
        } else {
            0
        };
        let dim0_step = if dim0_max > dim0_min {
            rng.gen_range(1..=(dim0_max - dim0_min).min(2))
        } else {
            1
        };

        let dim1_max = if shape[1] > 1 {
            rng.gen_range(1..shape[1])
        } else {
            1
        };
        let dim1_min = if dim1_max > 0 {
            rng.gen_range(0..dim1_max)
        } else {
            0
        };

        let dim1_step = if dim1_max > dim1_min {
            rng.gen_range(1..=(dim1_max - dim1_min).min(2))
        } else {
            1
        };

        let dim2_max = if shape[2] > 1 {
            rng.gen_range(1..shape[2])
        } else {
            1
        };
        let dim2_min = if dim2_max > 0 {
            rng.gen_range(0..dim2_max)
        } else {
            0
        };

        let dim2_step = if dim2_max > dim2_min {
            rng.gen_range(1..=(dim2_max - dim2_min).min(2))
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
        let tch_sum = tch_a.tch_method(0, true, tch::Kind::Int64);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_method(1, false)?;
        let tch_sum = tch_a.tch_method(1, false, tch::Kind::Int64);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_method(2, false)?;
        let tch_sum = tch_a.tch_method(2, false, tch::Kind::Int64);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_method([0, 1], false)?;
        let tch_sum = tch_a.tch_method(&[0, 1][..], false, tch::Kind::Int64);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_method([0, 2], false)?;
        let tch_sum = tch_a.tch_method(&[0, 2][..], false, tch::Kind::Int64);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_method([1, 2], false)?;
        let tch_sum = tch_a.tch_method(&[1, 2][..], false, tch::Kind::Int64);
        assert_eq(&sum, &tch_sum);
        let sum = a.hpt_method([0, 1, 2], false)?;
        let tch_sum = tch_a.tch_method(&[0, 1, 2][..], false, tch::Kind::Int64);
        assert_eq(&sum, &tch_sum);
    }
    Ok(())
}

#[test]
fn test_prod() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let prod = a.prod(0, false)?;
    let tch_prod = tch_a.prod_dim_int(0, false, tch::Kind::Int64);
    assert_eq(&prod, &tch_prod);

    let prod = a.prod(1, false)?;
    let tch_prod = tch_a.prod_dim_int(1, false, tch::Kind::Int64);
    assert_eq(&prod, &tch_prod);

    let prod = a.prod(2, false)?;
    let tch_prod = tch_a.prod_dim_int(2, false, tch::Kind::Int64);
    assert_eq(&prod, &tch_prod);

    Ok(())
}

#[test]
fn test_nanprod() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let prod = a.nanprod(0, false)?;
    let tch_prod = tch_a.prod_dim_int(0, false, tch::Kind::Int64);
    assert_eq(&prod, &tch_prod);

    let prod = a.nanprod(1, false)?;
    let tch_prod = tch_a.prod_dim_int(1, false, tch::Kind::Int64);
    assert_eq(&prod, &tch_prod);

    let prod = a.nanprod(2, false)?;
    let tch_prod = tch_a.prod_dim_int(2, false, tch::Kind::Int64);
    assert_eq(&prod, &tch_prod);

    Ok(())
}

#[test]
fn test_uncontiguous_prod() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = a.permute([1, 0, 2])?;
    let tch_a = tch_a.permute(&[1, 0, 2][..]);
    let sum = a.prod(0, false)?;
    let tch_sum = tch_a.prod_dim_int(0, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.prod(1, false)?;
    let tch_sum = tch_a.prod_dim_int(1, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.prod(2, false)?;
    let tch_sum = tch_a.prod_dim_int(2, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_uncontiguous_prod2() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = a.permute([1, 2, 0])?;
    let tch_a = tch_a.permute(&[1, 2, 0][..]);
    let sum = a.prod(0, false)?;
    let tch_sum = tch_a.prod_dim_int(0, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.prod(1, false)?;
    let tch_sum = tch_a.prod_dim_int(1, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.prod(2, false)?;
    let tch_sum = tch_a.prod_dim_int(2, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_sub_tensor_prod() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = slice!(a[:, 1:3, 2:5])?;
    let tch_a = tch_a.slice(1, 1, 3, 1).slice(2, 2, 5, 1);
    let sum = a.prod(0, false)?;
    let tch_sum = tch_a.prod_dim_int(0, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.prod(1, false)?;
    let tch_sum = tch_a.prod_dim_int(1, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.prod(2, false)?;
    let tch_sum = tch_a.prod_dim_int(2, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_sub_tensor_prod_step() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = slice!(a[:, 1:5:2, 2:9:2])?;
    let tch_a = tch_a.slice(1, 1, 5, 2).slice(2, 2, 9, 2);
    let sum = a.prod(0, false)?;
    let tch_sum = tch_a.prod_dim_int(0, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.prod(1, false)?;
    let tch_sum = tch_a.prod_dim_int(1, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.prod(2, false)?;
    let tch_sum = tch_a.prod_dim_int(2, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_mean() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let mean = a.mean(0, false)?;
    let tch_mean = tch_a.mean_dim(0, false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.mean(1, false)?;
    let tch_mean = tch_a.mean_dim(1, false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.mean(2, false)?;
    let tch_mean = tch_a.mean_dim(2, false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.mean([0, 1], false)?;
    let tch_mean = tch_a.mean_dim(&[0, 1][..], false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.mean([0, 2], false)?;
    let tch_mean = tch_a.mean_dim(&[0, 2][..], false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.mean([1, 2], false)?;
    let tch_mean = tch_a.mean_dim(&[1, 2][..], false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.mean([0, 1, 2], false)?;
    let tch_mean = tch_a.mean_dim(&[0, 1, 2][..], false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    Ok(())
}

#[test]
fn test_uncontiguous_mean() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = a.permute([1, 0, 2])?;
    let tch_a = tch_a.permute(&[1, 0, 2][..]);
    let mean = a.mean(0, false)?;
    let tch_mean = tch_a.mean_dim(0, false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.mean(1, false)?;
    let tch_mean = tch_a.mean_dim(1, false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.mean(2, false)?;
    let tch_mean = tch_a.mean_dim(2, false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.mean([0, 1], false)?;
    let tch_mean = tch_a.mean_dim(&[0, 1][..], false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.mean([0, 2], false)?;
    let tch_mean = tch_a.mean_dim(&[0, 2][..], false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.mean([1, 2], false)?;
    let tch_mean = tch_a.mean_dim(&[1, 2][..], false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.mean([0, 1, 2], false)?;
    let tch_mean = tch_a.mean_dim(&[0, 1, 2][..], false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    Ok(())
}

#[test]
fn test_uncontiguous_mean2() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = a.permute([1, 2, 0])?;
    let tch_a = tch_a.permute(&[1, 2, 0][..]);
    let mean = a.mean(0, false)?;
    let tch_mean = tch_a.mean_dim(0, false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.mean(1, false)?;
    let tch_mean = tch_a.mean_dim(1, false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.mean(2, false)?;
    let tch_mean = tch_a.mean_dim(2, false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.mean([0, 1], false)?;
    let tch_mean = tch_a.mean_dim(&[0, 1][..], false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.mean([0, 2], false)?;
    let tch_mean = tch_a.mean_dim(&[0, 2][..], false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.mean([1, 2], false)?;
    let tch_mean = tch_a.mean_dim(&[1, 2][..], false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.mean([0, 1, 2], false)?;
    let tch_mean = tch_a.mean_dim(&[0, 1, 2][..], false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    Ok(())
}

#[test]
fn test_sub_tensor_mean() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = slice!(a[:, 1:3, 2:5])?;
    let tch_a = tch_a.slice(1, 1, 3, 1).slice(2, 2, 5, 1);
    let mean = a.mean(0, false)?;
    let tch_mean = tch_a.mean_dim(0, false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.mean(1, false)?;
    let tch_mean = tch_a.mean_dim(1, false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.mean(2, false)?;
    let tch_mean = tch_a.mean_dim(2, false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.mean([0, 1], false)?;
    let tch_mean = tch_a.mean_dim(&[0, 1][..], false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.mean([0, 2], false)?;
    let tch_mean = tch_a.mean_dim(&[0, 2][..], false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.mean([1, 2], false)?;
    let tch_mean = tch_a.mean_dim(&[1, 2][..], false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.mean([0, 1, 2], false)?;
    let tch_mean = tch_a.mean_dim(&[0, 1, 2][..], false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    Ok(())
}

#[test]
fn test_sub_tensor_mean_step() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = slice!(a[:, 1:5:2, 2:9:2])?;
    let tch_a = tch_a.slice(1, 1, 5, 2).slice(2, 2, 9, 2);
    let mean = a.mean(0, false)?;
    let tch_mean = tch_a.mean_dim(0, false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.mean(1, false)?;
    let tch_mean = tch_a.mean_dim(1, false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.mean(2, false)?;
    let tch_mean = tch_a.mean_dim(2, false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.mean([0, 1], false)?;
    let tch_mean = tch_a.mean_dim(&[0, 1][..], false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.mean([0, 2], false)?;
    let tch_mean = tch_a.mean_dim(&[0, 2][..], false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.mean([1, 2], false)?;
    let tch_mean = tch_a.mean_dim(&[1, 2][..], false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.mean([0, 1, 2], false)?;
    let tch_mean = tch_a.mean_dim(&[0, 1, 2][..], false, tch::Kind::Double);
    assert_eq_f64(&mean, &tch_mean);
    Ok(())
}

#[test]
fn test_logsumexp() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let mean = a.logsumexp(0, false)?;
    let tch_mean = tch_a
        .logsumexp(0, false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.logsumexp(1, false)?;
    let tch_mean = tch_a
        .logsumexp(1, false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.logsumexp(2, false)?;
    let tch_mean = tch_a
        .logsumexp(2, false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.logsumexp([0, 1], false)?;
    let tch_mean = tch_a
        .logsumexp(&[0, 1][..], false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.logsumexp([0, 2], false)?;
    let tch_mean = tch_a
        .logsumexp(&[0, 2][..], false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.logsumexp([1, 2], false)?;
    let tch_mean = tch_a
        .logsumexp(&[1, 2][..], false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.logsumexp([0, 1, 2], false)?;
    let tch_mean = tch_a
        .logsumexp(&[0, 1, 2][..], false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    Ok(())
}

#[test]
fn test_uncontiguous_logsumexp() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = a.permute([1, 0, 2])?;
    let tch_a = tch_a.permute(&[1, 0, 2][..]);
    let mean = a.logsumexp(0, false)?;
    let tch_mean = tch_a
        .logsumexp(0, false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.logsumexp(1, false)?;
    let tch_mean = tch_a
        .logsumexp(1, false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.logsumexp(2, false)?;
    let tch_mean = tch_a
        .logsumexp(2, false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.logsumexp([0, 1], false)?;
    let tch_mean = tch_a
        .logsumexp(&[0, 1][..], false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.logsumexp([0, 2], false)?;
    let tch_mean = tch_a
        .logsumexp(&[0, 2][..], false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.logsumexp([1, 2], false)?;
    let tch_mean = tch_a
        .logsumexp(&[1, 2][..], false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.logsumexp([0, 1, 2], false)?;
    let tch_mean = tch_a
        .logsumexp(&[0, 1, 2][..], false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    Ok(())
}

#[test]
fn test_uncontiguous_logsumexp2() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = a.permute([1, 2, 0])?;
    let tch_a = tch_a.permute(&[1, 2, 0][..]);
    let mean = a.logsumexp(0, false)?;
    let tch_mean = tch_a
        .logsumexp(0, false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.logsumexp(1, false)?;
    let tch_mean = tch_a
        .logsumexp(1, false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.logsumexp(2, false)?;
    let tch_mean = tch_a
        .logsumexp(2, false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.logsumexp([0, 1], false)?;
    let tch_mean = tch_a
        .logsumexp(&[0, 1][..], false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.logsumexp([0, 2], false)?;
    let tch_mean = tch_a
        .logsumexp(&[0, 2][..], false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.logsumexp([1, 2], false)?;
    let tch_mean = tch_a
        .logsumexp(&[1, 2][..], false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.logsumexp([0, 1, 2], false)?;
    let tch_mean = tch_a
        .logsumexp(&[0, 1, 2][..], false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    Ok(())
}

#[test]
fn test_sub_tensor_logsumexp() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = slice!(a[:, 1:3, 2:5])?;
    let tch_a = tch_a.slice(1, 1, 3, 1).slice(2, 2, 5, 1);
    let mean = a.logsumexp(0, false)?;
    let tch_mean = tch_a
        .logsumexp(0, false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.logsumexp(1, false)?;
    let tch_mean = tch_a
        .logsumexp(1, false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.logsumexp(2, false)?;
    let tch_mean = tch_a
        .logsumexp(2, false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.logsumexp([0, 1], false)?;
    let tch_mean = tch_a
        .logsumexp(&[0, 1][..], false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.logsumexp([0, 2], false)?;
    let tch_mean = tch_a
        .logsumexp(&[0, 2][..], false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.logsumexp([1, 2], false)?;
    let tch_mean = tch_a
        .logsumexp(&[1, 2][..], false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.logsumexp([0, 1, 2], false)?;
    let tch_mean = tch_a
        .logsumexp(&[0, 1, 2][..], false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    Ok(())
}

#[test]
fn test_sub_tensor_logsumexp_step() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = slice!(a[:, 1:5:2, 2:9:2])?;
    let tch_a = tch_a.slice(1, 1, 5, 2).slice(2, 2, 9, 2);
    let mean = a.logsumexp(0, false)?;
    let tch_mean = tch_a
        .logsumexp(0, false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.logsumexp(1, false)?;
    let tch_mean = tch_a
        .logsumexp(1, false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.logsumexp(2, false)?;
    let tch_mean = tch_a
        .logsumexp(2, false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.logsumexp([0, 1], false)?;
    let tch_mean = tch_a
        .logsumexp(&[0, 1][..], false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.logsumexp([0, 2], false)?;
    let tch_mean = tch_a
        .logsumexp(&[0, 2][..], false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.logsumexp([1, 2], false)?;
    let tch_mean = tch_a
        .logsumexp(&[1, 2][..], false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    let mean = a.logsumexp([0, 1, 2], false)?;
    let tch_mean = tch_a
        .logsumexp(&[0, 1, 2][..], false)
        .to_dtype(tch::Kind::Double, false, true);
    assert_eq_f64(&mean, &tch_mean);
    Ok(())
}

#[test]
fn test_max() -> anyhow::Result<()> {
    let (a, tch_a) = common_input_f64(2 * 5 * 10, [2, 5, 10])?;
    let max = a.max(0, false)?;
    let (tch_max, _) = tch_a.max_dim(0, false);
    assert_eq_f64(&max, &tch_max);

    let max = a.max(1, false)?;
    let (tch_max, _) = tch_a.max_dim(1, false);
    assert_eq_f64(&max, &tch_max);

    let max = a.max(2, false)?;
    let (tch_max, _) = tch_a.max_dim(2, false);
    assert_eq_f64(&max, &tch_max);

    Ok(())
}

#[test]
fn test_uncontiguous_max() -> anyhow::Result<()> {
    let (a, tch_a) = common_input_f64(2 * 5 * 10, [2, 5, 10])?;
    let a = a.permute([1, 0, 2])?;
    let tch_a = tch_a.permute(&[1, 0, 2][..]);
    let max = a.max(0, false)?;
    let (tch_max, _) = tch_a.max_dim(0, false);
    assert_eq_f64(&max, &tch_max);
    let max = a.max(1, false)?;
    let (tch_max, _) = tch_a.max_dim(1, false);
    assert_eq_f64(&max, &tch_max);
    let max = a.max(2, false)?;
    let (tch_max, _) = tch_a.max_dim(2, false);
    Ok(())
}

#[test]
fn test_uncontiguous_max2() -> anyhow::Result<()> {
    let (a, tch_a) = common_input_f64(2 * 5 * 10, [2, 5, 10])?;
    let a = a.permute([1, 2, 0])?;
    let tch_a = tch_a.permute(&[1, 2, 0][..]);
    let max = a.max(0, false)?;
    let (tch_max, _) = tch_a.max_dim(0, false);
    assert_eq_f64(&max, &tch_max);
    let max = a.max(1, false)?;
    let (tch_max, _) = tch_a.max_dim(1, false);
    assert_eq_f64(&max, &tch_max);
    let max = a.max(2, false)?;
    let (tch_max, _) = tch_a.max_dim(2, false);
    Ok(())
}

#[test]
fn test_sub_tensor_max() -> anyhow::Result<()> {
    let (a, tch_a) = common_input_f64(2 * 5 * 10, [2, 5, 10])?;
    let a = slice!(a[:, 1:3, 2:5])?;
    let tch_a = tch_a.slice(1, 1, 3, 1).slice(2, 2, 5, 1);
    let max = a.max(0, false)?;
    let (tch_max, _) = tch_a.max_dim(0, false);
    assert_eq_f64(&max, &tch_max);
    let max = a.max(1, false)?;
    let (tch_max, _) = tch_a.max_dim(1, false);
    assert_eq_f64(&max, &tch_max);
    let max = a.max(2, false)?;
    let (tch_max, _) = tch_a.max_dim(2, false);
    assert_eq_f64(&max, &tch_max);
    Ok(())
}

#[test]
fn test_sub_tensor_max_step() -> anyhow::Result<()> {
    let (a, tch_a) = common_input_f64(2 * 5 * 10, [2, 5, 10])?;
    let a = slice!(a[:, 1:5:2, 2:9:2])?;
    let tch_a = tch_a.slice(1, 1, 5, 2).slice(2, 2, 9, 2);
    let max = a.max(0, false)?;
    let (tch_max, _) = tch_a.max_dim(0, false);
    assert_eq_f64(&max, &tch_max);
    let max = a.max(1, false)?;
    let (tch_max, _) = tch_a.max_dim(1, false);
    assert_eq_f64(&max, &tch_max);
    let max = a.max(2, false)?;
    let (tch_max, _) = tch_a.max_dim(2, false);
    assert_eq_f64(&max, &tch_max);
    Ok(())
}

#[test]
fn test_min() -> anyhow::Result<()> {
    let (a, tch_a) = common_input_f64(2 * 5 * 10, [2, 5, 10])?;
    let min = a.min(0, false)?;
    let (tch_min, _) = tch_a.min_dim(0, false);
    assert_eq_f64(&min, &tch_min);

    let min = a.min(1, false)?;
    let (tch_min, _) = tch_a.min_dim(1, false);
    assert_eq_f64(&min, &tch_min);

    let min = a.min(2, false)?;
    let (tch_min, _) = tch_a.min_dim(2, false);
    assert_eq_f64(&min, &tch_min);
    Ok(())
}

#[test]
fn test_uncontiguous_min() -> anyhow::Result<()> {
    let (a, tch_a) = common_input_f64(2 * 5 * 10, [2, 5, 10])?;
    let a = a.permute([1, 0, 2])?;
    let tch_a = tch_a.permute(&[1, 0, 2][..]);
    let min = a.min(0, false)?;
    let (tch_min, _) = tch_a.min_dim(0, false);
    assert_eq_f64(&min, &tch_min);
    let min = a.min(1, false)?;
    let (tch_min, _) = tch_a.min_dim(1, false);
    assert_eq_f64(&min, &tch_min);
    let min = a.min(2, false)?;
    let (tch_min, _) = tch_a.min_dim(2, false);
    Ok(())
}

#[test]
fn test_uncontiguous_min2() -> anyhow::Result<()> {
    let (a, tch_a) = common_input_f64(2 * 5 * 10, [2, 5, 10])?;
    let a = a.permute([1, 2, 0])?;
    let tch_a = tch_a.permute(&[1, 2, 0][..]);
    let min = a.min(0, false)?;
    let (tch_min, _) = tch_a.min_dim(0, false);
    assert_eq_f64(&min, &tch_min);
    let min = a.min(1, false)?;
    let (tch_min, _) = tch_a.min_dim(1, false);
    assert_eq_f64(&min, &tch_min);
    let min = a.min(2, false)?;
    let (tch_min, _) = tch_a.min_dim(2, false);
    Ok(())
}

#[test]
fn test_sub_tensor_min() -> anyhow::Result<()> {
    let (a, tch_a) = common_input_f64(2 * 5 * 10, [2, 5, 10])?;
    let a = slice!(a[:, 1:3, 2:5])?;
    let tch_a = tch_a.slice(1, 1, 3, 1).slice(2, 2, 5, 1);
    let min = a.min(0, false)?;
    let (tch_min, _) = tch_a.min_dim(0, false);
    assert_eq_f64(&min, &tch_min);
    let min = a.min(1, false)?;
    let (tch_min, _) = tch_a.min_dim(1, false);
    assert_eq_f64(&min, &tch_min);
    let min = a.min(2, false)?;
    let (tch_min, _) = tch_a.min_dim(2, false);
    assert_eq_f64(&min, &tch_min);
    Ok(())
}

#[test]
fn test_sub_tensor_min_step() -> anyhow::Result<()> {
    let (a, tch_a) = common_input_f64(2 * 5 * 10, [2, 5, 10])?;
    let a = slice!(a[:, 1:5:2, 2:9:2])?;
    let tch_a = tch_a.slice(1, 1, 5, 2).slice(2, 2, 9, 2);
    let min = a.min(0, false)?;
    let (tch_min, _) = tch_a.min_dim(0, false);
    assert_eq_f64(&min, &tch_min);
    let min = a.min(1, false)?;
    let (tch_min, _) = tch_a.min_dim(1, false);
    assert_eq_f64(&min, &tch_min);
    let min = a.min(2, false)?;
    let (tch_min, _) = tch_a.min_dim(2, false);
    assert_eq_f64(&min, &tch_min);
    Ok(())
}

#[test]
fn test_sum_square() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let sum = a.sum_square(0, false)?;
    let tch_sum = tch_a
        .pow_tensor_scalar(2)
        .sum_dim_intlist(0, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);

    let sum = a.sum_square(1, false)?;
    let tch_sum = tch_a
        .pow_tensor_scalar(2)
        .sum_dim_intlist(1, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);

    let sum = a.sum_square(2, false)?;
    let tch_sum = tch_a
        .pow_tensor_scalar(2)
        .sum_dim_intlist(2, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);

    let sum = a.sum_square([0, 1], false)?;
    let tch_sum = tch_a
        .pow_tensor_scalar(2)
        .sum_dim_intlist(&[0, 1][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);

    let sum = a.sum_square([0, 2], false)?;
    let tch_sum = tch_a
        .pow_tensor_scalar(2)
        .sum_dim_intlist(&[0, 2][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);

    let sum = a.sum_square([1, 2], false)?;
    let tch_sum = tch_a
        .pow_tensor_scalar(2)
        .sum_dim_intlist(&[1, 2][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);

    let sum = a.sum_square([0, 1, 2], false)?;
    let tch_sum =
        tch_a
            .pow_tensor_scalar(2)
            .sum_dim_intlist(&[0, 1, 2][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_uncontiguous_sum_square() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = a.permute([1, 0, 2])?;
    let tch_a = tch_a.permute(&[1, 0, 2][..]);
    let sum = a.sum_square(0, false)?;
    let tch_sum = tch_a
        .pow_tensor_scalar(2)
        .sum_dim_intlist(0, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum_square(1, false)?;
    let tch_sum = tch_a
        .pow_tensor_scalar(2)
        .sum_dim_intlist(1, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum_square(2, false)?;
    let tch_sum = tch_a
        .pow_tensor_scalar(2)
        .sum_dim_intlist(2, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum_square([0, 1], false)?;
    let tch_sum = tch_a
        .pow_tensor_scalar(2)
        .sum_dim_intlist(&[0, 1][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum_square([0, 2], false)?;
    let tch_sum = tch_a
        .pow_tensor_scalar(2)
        .sum_dim_intlist(&[0, 2][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum_square([1, 2], false)?;
    let tch_sum = tch_a
        .pow_tensor_scalar(2)
        .sum_dim_intlist(&[1, 2][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum_square([0, 1, 2], false)?;
    let tch_sum =
        tch_a
            .pow_tensor_scalar(2)
            .sum_dim_intlist(&[0, 1, 2][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_sub_tensor_sum_square() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = slice!(a[:, 1:3, 2:5])?;
    let tch_a = tch_a.slice(1, 1, 3, 1).slice(2, 2, 5, 1);
    let sum = a.sum_square(0, false)?;
    let tch_sum = tch_a
        .pow_tensor_scalar(2)
        .sum_dim_intlist(0, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum_square(1, false)?;
    let tch_sum = tch_a
        .pow_tensor_scalar(2)
        .sum_dim_intlist(1, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum_square(2, false)?;
    let tch_sum = tch_a
        .pow_tensor_scalar(2)
        .sum_dim_intlist(2, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum_square([0, 1], false)?;
    let tch_sum = tch_a
        .pow_tensor_scalar(2)
        .sum_dim_intlist(&[0, 1][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum_square([0, 2], false)?;
    let tch_sum = tch_a
        .pow_tensor_scalar(2)
        .sum_dim_intlist(&[0, 2][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum_square([1, 2], false)?;
    let tch_sum = tch_a
        .pow_tensor_scalar(2)
        .sum_dim_intlist(&[1, 2][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum_square([0, 1, 2], false)?;
    let tch_sum =
        tch_a
            .pow_tensor_scalar(2)
            .sum_dim_intlist(&[0, 1, 2][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_reducel1() -> anyhow::Result<()> {
    let (a, tch_a) = common_input_f64(2 * 5 * 10, [2, 5, 10])?;
    let sum = a.reducel1(0, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 1, 0, false)?;
    assert_eq_f64(&sum, &tch_sum);
    let sum = a.reducel1(1, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 1, 1, false)?;
    assert_eq_f64(&sum, &tch_sum);
    let sum = a.reducel1(2, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 1, 2, false)?;
    assert_eq_f64(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_uncontiguous_reducel1() -> anyhow::Result<()> {
    let (a, tch_a) = common_input_f64(2 * 5 * 10, [2, 5, 10])?;
    let a = a.permute([1, 0, 2])?;
    let tch_a = tch_a.permute(&[1, 0, 2][..]);
    let sum = a.reducel1(0, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 1, 0, false)?;
    assert_eq_f64(&sum, &tch_sum);
    let sum = a.reducel1(1, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 1, 1, false)?;
    assert_eq_f64(&sum, &tch_sum);
    let sum = a.reducel1(2, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 1, 2, false)?;
    assert_eq_f64(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_sub_tensor_reducel1() -> anyhow::Result<()> {
    let (a, tch_a) = common_input_f64(2 * 5 * 10, [2, 5, 10])?;
    let a = slice!(a[:, 1:3, 2:5])?;
    let tch_a = tch_a.slice(1, 1, 3, 1).slice(2, 2, 5, 1);
    let sum = a.reducel1(0, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 1, 0, false)?;
    assert_eq_f64(&sum, &tch_sum);
    let sum = a.reducel1(1, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 1, 1, false)?;
    assert_eq_f64(&sum, &tch_sum);
    let sum = a.reducel1(2, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 1, 2, false)?;
    assert_eq_f64(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_sub_tensor_reducel1_step() -> anyhow::Result<()> {
    let (a, tch_a) = common_input_f64(2 * 5 * 10, [2, 5, 10])?;
    let a = slice!(a[:, 1:5:2, 2:9:2])?;
    let tch_a = tch_a.slice(1, 1, 5, 2).slice(2, 2, 9, 2);
    let sum = a.reducel1(0, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 1, 0, false)?;
    assert_eq_f64(&sum, &tch_sum);
    let sum = a.reducel1(1, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 1, 1, false)?;
    assert_eq_f64(&sum, &tch_sum);
    let sum = a.reducel1(2, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 1, 2, false)?;
    assert_eq_f64(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_reducel2() -> anyhow::Result<()> {
    let (a, tch_a) = common_input_f64(1 * 1 * 10, [1, 1, 10])?;
    let sum = a.reducel2(0, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 2, 0, false)?;
    assert_eq_f64(&sum, &tch_sum);
    let sum = a.reducel2(1, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 2, 1, false)?;
    assert_eq_f64(&sum, &tch_sum);
    let sum = a.reducel2(2, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 2, 2, false)?;
    assert_eq_f64(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_uncontiguous_reducel2() -> anyhow::Result<()> {
    let (a, tch_a) = common_input_f64(2 * 5 * 10, [2, 5, 10])?;
    let a = a.permute([1, 0, 2])?;
    let tch_a = tch_a.permute(&[1, 0, 2][..]);
    let sum = a.reducel2(0, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 2, 0, false)?;
    assert_eq_f64(&sum, &tch_sum);
    let sum = a.reducel2(1, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 2, 1, false)?;
    assert_eq_f64(&sum, &tch_sum);
    let sum = a.reducel2(2, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 2, 2, false)?;
    assert_eq_f64(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_sub_tensor_reducel2() -> anyhow::Result<()> {
    let (a, tch_a) = common_input_f64(2 * 5 * 10, [2, 5, 10])?;
    let a = slice!(a[:, 1:3, 2:5])?;
    let tch_a = tch_a.slice(1, 1, 3, 1).slice(2, 2, 5, 1);
    let sum = a.reducel2(0, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 2, 0, false)?;
    assert_eq_f64(&sum, &tch_sum);
    let sum = a.reducel2(1, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 2, 1, false)?;
    assert_eq_f64(&sum, &tch_sum);
    let sum = a.reducel2(2, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 2, 2, false)?;
    assert_eq_f64(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_sub_tensor_reducel2_step() -> anyhow::Result<()> {
    let (a, tch_a) = common_input_f64(2 * 5 * 10, [2, 5, 10])?;
    let a = slice!(a[:, 1:5:2, 2:9:2])?;
    let tch_a = tch_a.slice(1, 1, 5, 2).slice(2, 2, 9, 2);
    let sum = a.reducel2(0, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 2, 0, false)?;
    assert_eq_f64(&sum, &tch_sum);
    let sum = a.reducel2(1, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 2, 1, false)?;
    assert_eq_f64(&sum, &tch_sum);
    let sum = a.reducel2(2, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 2, 2, false)?;
    assert_eq_f64(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_reducel3() -> anyhow::Result<()> {
    let (a, tch_a) = common_input_f64(2 * 5 * 10, [2, 5, 10])?;
    let sum = a.reducel3(0, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 3, 0, false)?;
    assert_eq_f64(&sum, &tch_sum);
    let sum = a.reducel3(1, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 3, 1, false)?;
    assert_eq_f64(&sum, &tch_sum);
    let sum = a.reducel3(2, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 3, 2, false)?;
    assert_eq_f64(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_uncontiguous_reducel3() -> anyhow::Result<()> {
    let (a, tch_a) = common_input_f64(2 * 5 * 10, [2, 5, 10])?;
    let a = a.permute([1, 0, 2])?;
    let tch_a = tch_a.permute(&[1, 0, 2][..]);
    let sum = a.reducel3(0, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 3, 0, false)?;
    assert_eq_f64(&sum, &tch_sum);
    let sum = a.reducel3(1, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 3, 1, false)?;
    assert_eq_f64(&sum, &tch_sum);
    let sum = a.reducel3(2, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 3, 2, false)?;
    assert_eq_f64(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_sub_tensor_reducel3() -> anyhow::Result<()> {
    let (a, tch_a) = common_input_f64(2 * 5 * 10, [2, 5, 10])?;
    let a = slice!(a[:, 1:3, 2:5])?;
    let tch_a = tch_a.slice(1, 1, 3, 1).slice(2, 2, 5, 1);
    let sum = a.reducel3(0, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 3, 0, false)?;
    assert_eq_f64(&sum, &tch_sum);
    let sum = a.reducel3(1, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 3, 1, false)?;
    assert_eq_f64(&sum, &tch_sum);
    let sum = a.reducel3(2, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 3, 2, false)?;
    assert_eq_f64(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_sub_tensor_reducel3_step() -> anyhow::Result<()> {
    let (a, tch_a) = common_input_f64(2 * 5 * 10, [2, 5, 10])?;
    let a = slice!(a[:, 1:5:2, 2:9:2])?;
    let tch_a = tch_a.slice(1, 1, 5, 2).slice(2, 2, 9, 2);
    let sum = a.reducel3(0, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 3, 0, false)?;
    assert_eq_f64(&sum, &tch_sum);
    let sum = a.reducel3(1, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 3, 1, false)?;
    assert_eq_f64(&sum, &tch_sum);
    let sum = a.reducel3(2, false)?;
    let res = Tensor::empty(sum.shape().inner(), (tch::Kind::Double, tch::Device::Cpu));
    let tch_sum = tch_a.f_norm_out(&res, 3, 2, false)?;
    assert_eq_f64(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_argmin() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let sum = a.argmin(0, false)?;
    let tch_sum = tch_a.argmin(0, false);
    assert_eq(&sum, &tch_sum);
    let sum = a.argmin(1, false)?;
    let tch_sum = tch_a.argmin(1, false);
    assert_eq(&sum, &tch_sum);
    let sum = a.argmin(2, false)?;
    let tch_sum = tch_a.argmin(2, false);
    assert_eq(&sum, &tch_sum);
    let a = a.permute([1, 0, 2])?;
    let tch_a = tch_a.permute(&[1, 0, 2][..]);
    let sum = a.argmin(0, false)?;
    let tch_sum = tch_a.argmin(0, false);
    assert_eq(&sum, &tch_sum);
    let sum = a.argmin(1, false)?;
    let tch_sum = tch_a.argmin(1, false);
    assert_eq(&sum, &tch_sum);
    let sum = a.argmin(2, false)?;
    let tch_sum = tch_a.argmin(2, false);
    assert_eq(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_uncontiguous_argmin() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = a.permute([1, 0, 2])?;
    let tch_a = tch_a.permute(&[1, 0, 2][..]);
    let sum = a.argmin(0, false)?;
    let tch_sum = tch_a.argmin(0, false);
    assert_eq(&sum, &tch_sum);
    let sum = a.argmin(1, false)?;
    let tch_sum = tch_a.argmin(1, false);
    assert_eq(&sum, &tch_sum);
    let sum = a.argmin(2, false)?;
    let tch_sum = tch_a.argmin(2, false);
    assert_eq(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_uncontiguous_argmin2() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = a.permute([1, 2, 0])?;
    let tch_a = tch_a.permute(&[1, 2, 0][..]);
    let sum = a.argmin(0, false)?;
    let tch_sum = tch_a.argmin(0, false);
    assert_eq(&sum, &tch_sum);
    let sum = a.argmin(1, false)?;
    let tch_sum = tch_a.argmin(1, false);
    assert_eq(&sum, &tch_sum);
    let sum = a.argmin(2, false)?;
    let tch_sum = tch_a.argmin(2, false);
    assert_eq(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_sub_tensor_argmin() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = slice!(a[:, 1:3, 2:5])?;
    let tch_a = tch_a.slice(1, 1, 3, 1).slice(2, 2, 5, 1);
    let sum = a.argmin(0, false)?;
    let tch_sum = tch_a.argmin(0, false);
    assert_eq(&sum, &tch_sum);
    let sum = a.argmin(1, false)?;
    let tch_sum = tch_a.argmin(1, false);
    assert_eq(&sum, &tch_sum);
    let sum = a.argmin(2, false)?;
    let tch_sum = tch_a.argmin(2, false);
    assert_eq(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_sub_tensor_argmin_step() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = slice!(a[:, 1:5:2, 2:9:2])?;
    let tch_a = tch_a.slice(1, 1, 5, 2).slice(2, 2, 9, 2);
    let sum = a.argmin(0, false)?;
    let tch_sum = tch_a.argmin(0, false);
    assert_eq(&sum, &tch_sum);
    let sum = a.argmin(1, false)?;
    let tch_sum = tch_a.argmin(1, false);
    assert_eq(&sum, &tch_sum);
    let sum = a.argmin(2, false)?;
    let tch_sum = tch_a.argmin(2, false);
    assert_eq(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_argmax() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let sum = a.argmax(0, false)?;
    let tch_sum = tch_a.argmax(0, false);
    assert_eq(&sum, &tch_sum);
    let sum = a.argmax(1, false)?;
    let tch_sum = tch_a.argmax(1, false);
    assert_eq(&sum, &tch_sum);
    let sum = a.argmax(2, false)?;
    let tch_sum = tch_a.argmax(2, false);
    assert_eq(&sum, &tch_sum);
    let a = a.permute([1, 0, 2])?;
    let tch_a = tch_a.permute(&[1, 0, 2][..]);
    let sum = a.argmax(0, false)?;
    let tch_sum = tch_a.argmax(0, false);
    assert_eq(&sum, &tch_sum);
    let sum = a.argmax(1, false)?;
    let tch_sum = tch_a.argmax(1, false);
    assert_eq(&sum, &tch_sum);
    let sum = a.argmax(2, false)?;
    let tch_sum = tch_a.argmax(2, false);
    assert_eq(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_uncontiguous_argmax() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = a.permute([1, 0, 2])?;
    let tch_a = tch_a.permute(&[1, 0, 2][..]);
    let sum = a.argmax(0, false)?;
    let tch_sum = tch_a.argmax(0, false);
    assert_eq(&sum, &tch_sum);
    let sum = a.argmax(1, false)?;
    let tch_sum = tch_a.argmax(1, false);
    assert_eq(&sum, &tch_sum);
    let sum = a.argmax(2, false)?;
    let tch_sum = tch_a.argmax(2, false);
    assert_eq(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_uncontiguous_argmax2() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = a.permute([1, 2, 0])?;
    let tch_a = tch_a.permute(&[1, 2, 0][..]);
    let sum = a.argmax(0, false)?;
    let tch_sum = tch_a.argmax(0, false);
    assert_eq(&sum, &tch_sum);
    let sum = a.argmax(1, false)?;
    let tch_sum = tch_a.argmax(1, false);
    assert_eq(&sum, &tch_sum);
    let sum = a.argmax(2, false)?;
    let tch_sum = tch_a.argmax(2, false);
    assert_eq(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_sub_tensor_argmax() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = slice!(a[:, 1:3, 2:5])?;
    let tch_a = tch_a.slice(1, 1, 3, 1).slice(2, 2, 5, 1);
    let sum = a.argmax(0, false)?;
    let tch_sum = tch_a.argmax(0, false);
    assert_eq(&sum, &tch_sum);
    let sum = a.argmax(1, false)?;
    let tch_sum = tch_a.argmax(1, false);
    assert_eq(&sum, &tch_sum);
    let sum = a.argmax(2, false)?;
    let tch_sum = tch_a.argmax(2, false);
    assert_eq(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_sub_tensor_argmax_step() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = slice!(a[:, 1:5:2, 2:9:2])?;
    let tch_a = tch_a.slice(1, 1, 5, 2).slice(2, 2, 9, 2);
    let sum = a.argmax(0, false)?;
    let tch_sum = tch_a.argmax(0, false);
    assert_eq(&sum, &tch_sum);
    let sum = a.argmax(1, false)?;
    let tch_sum = tch_a.argmax(1, false);
    assert_eq(&sum, &tch_sum);
    let sum = a.argmax(2, false)?;
    let tch_sum = tch_a.argmax(2, false);
    assert_eq(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_all() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
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
fn test_uncontiguous_all() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = a.permute([1, 0, 2])?;
    let tch_a = tch_a.permute(&[1, 0, 2][..]);
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
fn test_sub_tensor_all() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = slice!(a[:, 1:3, 2:5])?;
    let tch_a = tch_a.slice(1, 1, 3, 1).slice(2, 2, 5, 1);
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
fn test_sub_tensor_all_step() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = slice!(a[:, 1:5:2, 2:9:2])?;
    let tch_a = tch_a.slice(1, 1, 5, 2).slice(2, 2, 9, 2);
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
fn test_uncontiguous_all2() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = a.permute([1, 2, 0])?;
    let tch_a = tch_a.permute(&[1, 2, 0][..]);
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
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
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

#[test]
fn test_uncontiguous_any() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = a.permute([1, 0, 2])?;
    let tch_a = tch_a.permute(&[1, 0, 2][..]);
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

#[test]
fn test_sub_tensor_any() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = slice!(a[:, 1:3, 2:5])?;
    let tch_a = tch_a.slice(1, 1, 3, 1).slice(2, 2, 5, 1);
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

#[test]
fn test_sub_tensor_any_step() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = slice!(a[:, 1:5:2, 2:9:2])?;
    let tch_a = tch_a.slice(1, 1, 5, 2).slice(2, 2, 9, 2);
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

#[test]
fn test_uncontiguous_any2() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = a.permute([1, 2, 0])?;
    let tch_a = tch_a.permute(&[1, 2, 0][..]);
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
