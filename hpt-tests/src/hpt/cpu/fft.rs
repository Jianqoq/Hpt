#![allow(unused)]

use hpt::common::cpu::TensorLike;
use hpt::common::TensorInfo;
use hpt::ops::{AdvancedOps, FFTOps};
use hpt::ops::{Random, ShapeManipulate};
use hpt::utils::set_num_threads;
use hpt::{backend::Cpu, ops::TensorCreator};
use num_complex::{Complex64, ComplexFloat};
use rand::Rng;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use tch::Tensor;

use crate::utils::random_utils::generate_all_combinations;

fn common_input(shape: &[i64]) -> anyhow::Result<(hpt::Tensor<Complex64, Cpu>, Tensor)> {
    let tch_a = Tensor::randn(shape, (tch::Kind::ComplexDouble, tch::Device::Cpu)).reshape(shape);
    let mut a = hpt::Tensor::<Complex64, Cpu>::empty(shape)?;
    let a_size = a.size();
    let raw_mut = a.as_raw_mut();
    let tch_raw =
        unsafe { core::slice::from_raw_parts_mut(tch_a.data_ptr() as *mut Complex64, a_size) };
    raw_mut
        .par_iter_mut()
        .zip(tch_raw.par_iter())
        .for_each(|(a, b)| {
            *a = *b;
        });
    Ok((a, tch_a))
}

#[allow(unused)]
#[track_caller]
fn assert_eq(b: &hpt::Tensor<Complex64>, a: &Tensor) {
    let a_raw = if b.strides().contains(&0) {
        let size = b
            .shape()
            .iter()
            .zip(b.strides().iter())
            .filter(|(sp, s)| **s != 0)
            .fold(1, |acc, (sp, _)| acc * sp);
        unsafe { std::slice::from_raw_parts(a.data_ptr() as *const Complex64, size as usize) }
    } else {
        unsafe { std::slice::from_raw_parts(a.data_ptr() as *const Complex64, b.size()) }
    };
    let b_raw = b.as_raw();
    let tolerance = 2.5e-16;
    let caller = core::panic::Location::caller();
    a_raw.iter().zip(b_raw.iter()).for_each(|(a, b)| {
        let abs_diff = (*a - *b).abs();
        let rel_diff = if *a == Complex64::new(0.0, 0.0) && *b == Complex64::new(0.0, 0.0) {
            0.0
        } else {
            abs_diff / (a.abs() + b.abs() + f64::EPSILON)
        };

        if rel_diff > 0.05 {
            panic!("{} != {} (relative_diff: {})", *a, *b, rel_diff);
        }
    });
}

#[test]
fn fftn_test() -> anyhow::Result<()> {
    let mut rng = rand::thread_rng();
    for _ in 0..100 {
        let ndim = rng.gen_range(1..=3);
        let shape = (0..ndim)
            .map(|_| rng.gen_range(1..=32))
            .collect::<Vec<i64>>();
        let (a, tch_a) = common_input(shape.as_slice())?;
        let combinations = generate_all_combinations(&(0..ndim).collect::<Vec<_>>());
        for axes in combinations {
            let s = axes
                .iter()
                .map(|i| rng.gen_range(1..=32))
                .collect::<Vec<i64>>();
            let res = a.fftn(&s, axes.as_slice(), Some("backward"))?;
            let tch_res = tch_a
                .fft_fftn(Some(s.as_slice()), axes.as_slice(), "backward")
                .contiguous();
            assert_eq(&res, &tch_res);
            let res = a.fftn(&s, axes.as_slice(), Some("forward"))?;
            let tch_res = tch_a
                .fft_fftn(Some(s.as_slice()), axes.as_slice(), "forward")
                .contiguous();
            assert_eq(&res, &tch_res);
            let res = a.fftn(&s, axes.as_slice(), Some("ortho"))?;
            let tch_res = tch_a
                .fft_fftn(Some(s.as_slice()), axes.as_slice(), "ortho")
                .contiguous();
            assert_eq(&res, &tch_res);
        }
    }
    Ok(())
}

#[test]
fn ifftn_test() -> anyhow::Result<()> {
    let mut rng = rand::thread_rng();
    for _ in 0..100 {
        let ndim = rng.gen_range(1..=3);
        let shape = (0..ndim)
            .map(|_| rng.gen_range(1..=32))
            .collect::<Vec<i64>>();
        let (a, tch_a) = common_input(shape.as_slice())?;
        let combinations = generate_all_combinations(&(0..ndim).collect::<Vec<_>>());
        for axes in combinations {
            let s = axes
                .iter()
                .map(|i| rng.gen_range(1..=32))
                .collect::<Vec<i64>>();
            let res = a.ifftn(&s, axes.as_slice(), Some("backward"))?;
            let tch_res = tch_a
                .fft_ifftn(Some(s.as_slice()), axes.as_slice(), "backward")
                .contiguous();
            assert_eq(&res, &tch_res);
            let res = a.ifftn(&s, axes.as_slice(), Some("forward"))?;
            let tch_res = tch_a
                .fft_ifftn(Some(s.as_slice()), axes.as_slice(), "forward")
                .contiguous();
            assert_eq(&res, &tch_res);
            let res = a.ifftn(&s, axes.as_slice(), Some("ortho"))?;
            let tch_res = tch_a
                .fft_ifftn(Some(s.as_slice()), axes.as_slice(), "ortho")
                .contiguous();
            assert_eq(&res, &tch_res);
        }
    }
    Ok(())
}
