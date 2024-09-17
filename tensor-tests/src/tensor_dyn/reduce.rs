#![allow(unused)]
use backend::Cpu;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use tch::Tensor;
use tensor_base::_Tensor;
use tensor_common::slice;
use tensor_common::slice::Slice;
use tensor_dyn::slice::SliceOps;
use tensor_dyn::*;

#[track_caller]
fn assert_eq(a: &_Tensor<i64>, b: &Tensor) {
    let raw = a.as_raw();
    let tch_raw = unsafe { core::slice::from_raw_parts(b.data_ptr() as *const i64, a.size()) };
    let caller = core::panic::Location::caller();
    raw.par_iter()
        .zip(tch_raw.par_iter())
        .for_each(|(a, b)| {
            if a != b {
                panic!("{} != {}, at {}", a, b, caller)
            }
        });
}

#[allow(unused)]
#[track_caller]
fn assert_eq_f64(b: &_Tensor<f64>, a: &Tensor) {
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
        let abs_diff = (a - b).abs();
        let relative_diff = abs_diff / b.abs().max(f64::EPSILON);

        if abs_diff > tolerance && relative_diff > tolerance {
            println!(
                "{} != {} (abs_diff: {}, relative_diff: {}), at {}",
                a, b, abs_diff, relative_diff, caller
            );
        }
    });
}

fn common_input<const N: usize>(
    end: i64,
    shape: [i64; N],
) -> anyhow::Result<(_Tensor<i64, Cpu>, Tensor)> {
    let a = _Tensor::<i64, Cpu>::arange(0, end)?.reshape(&shape)?;
    let tch_a = Tensor::arange(end, (tch::Kind::Int64, tch::Device::Cpu)).reshape(&shape);
    Ok((a, tch_a))
}

fn common_input_f64<const N: usize>(
    end: i64,
    shape: [i64; N],
) -> anyhow::Result<(_Tensor<f64, Cpu>, Tensor)> {
    let tch_a = Tensor::randn(&shape, (tch::Kind::Double, tch::Device::Cpu)).reshape(&shape);
    let a = _Tensor::<f64, Cpu>::empty(&shape)?;
    let raw_mut = a.as_raw_mut();
    let tch_raw =
        unsafe { core::slice::from_raw_parts_mut(tch_a.data_ptr() as *mut f64, a.size()) };
    raw_mut
        .par_iter_mut()
        .zip(tch_raw.par_iter())
        .for_each(|(a, b)| *a = *b);
    Ok((a, tch_a))
}

#[test]
fn test_sum() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let sum = a.sum(0, false)?;
    let tch_sum = tch_a.sum_dim_intlist(0, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum(1, false)?;
    let tch_sum = tch_a.sum_dim_intlist(1, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum(2, false)?;
    let tch_sum = tch_a.sum_dim_intlist(2, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum([0, 1], false)?;
    let tch_sum = tch_a.sum_dim_intlist(&[0, 1][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum([0, 2], false)?;
    let tch_sum = tch_a.sum_dim_intlist(&[0, 2][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum([1, 2], false)?;
    let tch_sum = tch_a.sum_dim_intlist(&[1, 2][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum([0, 1, 2], false)?;
    let tch_sum = tch_a.sum_dim_intlist(&[0, 1, 2][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_uncontiguous_sum() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = a.permute([1, 0, 2])?;
    let tch_a = tch_a.permute(&[1, 0, 2][..]);
    let sum = a.sum(0, false)?;
    let tch_sum = tch_a.sum_dim_intlist(0, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum(1, false)?;
    let tch_sum = tch_a.sum_dim_intlist(1, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum(2, false)?;
    let tch_sum = tch_a.sum_dim_intlist(2, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum([0, 1], false)?;
    let tch_sum = tch_a.sum_dim_intlist(&[0, 1][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum([0, 2], false)?;
    let tch_sum = tch_a.sum_dim_intlist(&[0, 2][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum([1, 2], false)?;
    let tch_sum = tch_a.sum_dim_intlist(&[1, 2][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum([0, 1, 2], false)?;
    let tch_sum = tch_a.sum_dim_intlist(&[0, 1, 2][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_sub_tensor_sum() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = slice!(a[:, 1:3, 2:5])?;
    let tch_a = tch_a.slice(1, 1, 3, 1).slice(2, 2, 5, 1);
    let sum = a.sum(0, false)?;
    let tch_sum = tch_a.sum_dim_intlist(0, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum(1, false)?;
    let tch_sum = tch_a.sum_dim_intlist(1, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum(2, false)?;
    let tch_sum = tch_a.sum_dim_intlist(2, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum([0, 1], false)?;
    let tch_sum = tch_a.sum_dim_intlist(&[0, 1][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum([0, 2], false)?;
    let tch_sum = tch_a.sum_dim_intlist(&[0, 2][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum([1, 2], false)?;
    let tch_sum = tch_a.sum_dim_intlist(&[1, 2][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum([0, 1, 2], false)?;
    let tch_sum = tch_a.sum_dim_intlist(&[0, 1, 2][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_sub_tensor_sum_step() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = slice!(a[:, 1:5:2, 2:9:2])?;
    let tch_a = tch_a.slice(1, 1, 5, 2).slice(2, 2, 9, 2);
    let sum = a.sum(0, false)?;
    let tch_sum = tch_a.sum_dim_intlist(0, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum(1, false)?;
    let tch_sum = tch_a.sum_dim_intlist(1, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum(2, false)?;
    let tch_sum = tch_a.sum_dim_intlist(2, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum([0, 1], false)?;
    let tch_sum = tch_a.sum_dim_intlist(&[0, 1][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum([0, 2], false)?;
    let tch_sum = tch_a.sum_dim_intlist(&[0, 2][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum([1, 2], false)?;
    let tch_sum = tch_a.sum_dim_intlist(&[1, 2][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum([0, 1, 2], false)?;
    let tch_sum = tch_a.sum_dim_intlist(&[0, 1, 2][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_uncontiguous_sum2() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = a.permute([1, 2, 0])?;
    let tch_a = tch_a.permute(&[1, 2, 0][..]);
    let sum = a.sum(0, false)?;
    let tch_sum = tch_a.sum_dim_intlist(0, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum(1, false)?;
    let tch_sum = tch_a.sum_dim_intlist(1, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum(2, false)?;
    let tch_sum = tch_a.sum_dim_intlist(2, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum([0, 1], false)?;
    let tch_sum = tch_a.sum_dim_intlist(&[0, 1][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum([0, 2], false)?;
    let tch_sum = tch_a.sum_dim_intlist(&[0, 2][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum([1, 2], false)?;
    let tch_sum = tch_a.sum_dim_intlist(&[1, 2][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum([0, 1, 2], false)?;
    let tch_sum = tch_a.sum_dim_intlist(&[0, 1, 2][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    Ok(())
}

#[test]
fn test_prod() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(8 * 8096 * 2048, [8, 8096, 2048])?;
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
    println!("num threads: {}", rayon::current_num_threads());
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    println!("a: {:?}", a);
    let mean = a.mean(0, false)?;
    println!("mean: {:?}", mean);
    let tch_mean = tch_a.mean_dim(0, false, tch::Kind::Double);
    println!("tch_mean: {}", tch_mean);
    assert_eq_f64(&mean, &tch_mean);
    // let mean = a.mean(1, false)?;
    // let tch_mean = tch_a.mean_dim(1, false, tch::Kind::Double);
    // assert_eq_f64(&mean, &tch_mean);
    // let mean = a.mean(2, false)?;
    // let tch_mean = tch_a.mean_dim(2, false, tch::Kind::Double);
    // assert_eq_f64(&mean, &tch_mean);
    // let mean = a.mean([0, 1], false)?;
    // let tch_mean = tch_a.mean_dim(&[0, 1][..], false, tch::Kind::Double);
    // assert_eq_f64(&mean, &tch_mean);
    // let mean = a.mean([0, 2], false)?;
    // let tch_mean = tch_a.mean_dim(&[0, 2][..], false, tch::Kind::Double);
    // assert_eq_f64(&mean, &tch_mean);
    // let mean = a.mean([1, 2], false)?;
    // let tch_mean = tch_a.mean_dim(&[1, 2][..], false, tch::Kind::Double);
    // assert_eq_f64(&mean, &tch_mean);
    // let mean = a.mean([0, 1, 2], false)?;
    // let tch_mean = tch_a.mean_dim(&[0, 1, 2][..], false, tch::Kind::Double);
    // assert_eq_f64(&mean, &tch_mean);
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
fn test_uncontiguous_reducel12() -> anyhow::Result<()> {
    let (a, tch_a) = common_input_f64(2 * 5 * 10, [2, 5, 10])?;
    let a = a.permute([1, 2, 0])?;
    let tch_a = tch_a.permute(&[1, 2, 0][..]);
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
    let (a, tch_a) = common_input_f64(2 * 5 * 10, [2, 5, 10])?;
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