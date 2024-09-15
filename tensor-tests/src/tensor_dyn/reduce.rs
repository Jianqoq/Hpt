#![allow(unused)]

use backend::Cpu;
use rayon::iter::{ IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator };
use tensor_base::_Tensor;
use tensor_dyn::*;

fn assert_eq(a: &_Tensor<i64>, b: &tch::Tensor) {
    let raw = a.as_raw();
    let tch_raw = unsafe { core::slice::from_raw_parts(b.data_ptr() as *const i64, a.size()) };
    raw.par_iter()
        .zip(tch_raw.par_iter())
        .for_each(|(a, b)| assert_eq!(a, b));
}

fn assert_eq_f64(a: &_Tensor<f64>, b: &tch::Tensor) {
    let raw = a.as_raw();
    let tch_raw = unsafe { core::slice::from_raw_parts(b.data_ptr() as *const f64, a.size()) };
    raw.par_iter()
        .zip(tch_raw.par_iter())
        .for_each(|(a, b)| assert_eq!(a, b));
}

fn common_input<const N: usize>(
    end: i64,
    shape: [i64; N]
) -> anyhow::Result<(_Tensor<i64, Cpu>, tch::Tensor)> {
    let a = _Tensor::<i64, Cpu>::arange(0, end)?.reshape(&shape)?;
    let tch_a = tch::Tensor::arange(end, (tch::Kind::Int64, tch::Device::Cpu)).reshape(&shape);
    Ok((a, tch_a))
}

#[test]
fn test_sum() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2 * 5 * 10])?;
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
fn test_argmin() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(8 * 8096 * 2048, [8, 8096, 2048])?;
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
fn test_argmax() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(8 * 8096 * 2048, [8, 8096, 2048])?;
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