#![allow(unused_imports)]
use crate::TestTypes;
use crate::TCH_TEST_TYPES;
use crate::TEST_ATOL;
use crate::TEST_RTOL;
use hpt::common::cpu::TensorLike;
use hpt::common::TensorInfo;
use hpt::ops::*;
use hpt::slice;
use hpt::Tensor;
use rand::Rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
#[allow(unused)]
fn assert_eq(b: &Tensor<TestTypes>, a: &tch::Tensor) {
    let tch_b = unsafe {
        Tensor::<TestTypes>::from_raw(a.data_ptr() as *mut TestTypes, &b.shape().to_vec())
    }
    .expect("from_raw failed");
    assert!(b.allclose(&tch_b, TEST_RTOL, TEST_ATOL));
}

#[allow(unused)]
fn no_assert(b: &Tensor<TestTypes>, a: &tch::Tensor) {}

#[allow(unused)]
fn assert_eq_bool(b: &Tensor<bool>, a: &tch::Tensor) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const bool, b.size()) };
    let b_raw = b.as_raw();

    for i in 0..b.size() {
        if a_raw[i] != b_raw[i] {
            panic!("{} != {}", a_raw[i], b_raw[i]);
        }
    }
}

macro_rules! test_unarys {
    (
        $name:ident,
        $assert_method:ident,
        $tch_method:ident($($tch_args:expr),*),
        $hpt_method:ident($($hpt_args:expr),*)
    ) => {
        paste::paste! {
            #[test]
            fn [<test _ $name>]() -> anyhow::Result<()> {
                let mut rng = rand::rng();
                for _ in 0..100 {
                    let shape = [
                        rng.random_range(1..1024),
                    ];
                    let tch_a = tch::Tensor::randn(shape, (TCH_TEST_TYPES, tch::Device::Cpu));
                    let mut a = Tensor::<TestTypes>::empty(shape)?;
                    let a_size = a.size();
                    a.as_raw_mut().copy_from_slice(unsafe {
                        std::slice::from_raw_parts(tch_a.data_ptr() as *const TestTypes, a_size)
                    });
                    let b = a.$hpt_method($($hpt_args),*)?;
                    let tch_b = tch_a.$tch_method($($tch_args),*);
                    $assert_method(&b, &tch_b);
                }
                Ok(())
            }
        }
    };
}

macro_rules! test_unarys_out {
    (
        $name:ident,
        $assert_method:ident,
        $tch_method:ident($($tch_args:expr),*),
        $hpt_method:ident($($hpt_args:expr),*)
    ) => {
        paste::paste! {
            #[test]
            fn [<test _ $name>]() -> anyhow::Result<()> {
                let mut rng = rand::rng();
                for _ in 0..100 {
                    let shape = [
                        rng.random_range(1..1024),
                    ];
                    let tch_a = tch::Tensor::randn(shape, (TCH_TEST_TYPES, tch::Device::Cpu));
                    let mut a = Tensor::<TestTypes>::empty(shape)?;
                    let a_size = a.size();
                    a.as_raw_mut().copy_from_slice(unsafe {
                        std::slice::from_raw_parts(tch_a.data_ptr() as *const TestTypes, a_size)
                    });
                    let b = a.$hpt_method($($hpt_args,)* &mut a.clone())?;
                    let tch_b = tch_a.$tch_method($($tch_args),*);
                    $assert_method(&b, &tch_b);
                }
                Ok(())
            }
        }
    };
}

test_unarys!(abs, assert_eq, abs(), abs());
test_unarys_out!(abs_, assert_eq, abs(), abs_());
test_unarys!(acos, assert_eq, acos(), acos());
test_unarys_out!(acos_, assert_eq, acos(), acos_());
test_unarys!(acosh, assert_eq, acosh(), acosh());
test_unarys_out!(acosh_, assert_eq, acosh(), acosh_());
test_unarys!(asin, assert_eq, asin(), asin());
test_unarys_out!(asin_, assert_eq, asin(), asin_());
test_unarys!(asinh, assert_eq, asinh(), asinh());
test_unarys_out!(asinh_, assert_eq, asinh(), asinh_());
test_unarys!(atan, assert_eq, atan(), atan());
test_unarys_out!(atan_, assert_eq, atan(), atan_());
test_unarys!(atanh, assert_eq, atanh(), atanh()); // not sure why precision is huge
test_unarys_out!(atanh_, assert_eq, atanh(), atanh_());
test_unarys!(ceil, assert_eq, ceil(), ceil());
test_unarys_out!(ceil_, assert_eq, ceil(), ceil_());
test_unarys!(cos, assert_eq, cos(), cos());
test_unarys_out!(cos_, assert_eq, cos(), cos_());
test_unarys!(cosh, assert_eq, cosh(), cosh());
test_unarys_out!(cosh_, assert_eq, cosh(), cosh_());
test_unarys!(erf, assert_eq, erf(), erf());
test_unarys!(exp, assert_eq, exp(), exp());
test_unarys_out!(exp_, assert_eq, exp(), exp_());
test_unarys!(floor, assert_eq, floor(), floor());
test_unarys!(log, assert_eq, log(), ln());
test_unarys_out!(log_, assert_eq, log(), ln_());
test_unarys!(log10, assert_eq, log10(), log10());
test_unarys_out!(log10_, assert_eq, log10(), log10_());
test_unarys!(log2, assert_eq, log2(), log2());
test_unarys_out!(log2_, assert_eq, log2(), log2_());
test_unarys!(recip, assert_eq, reciprocal(), recip());
test_unarys_out!(recip_, assert_eq, reciprocal(), recip_());
test_unarys!(neg, assert_eq, neg(), neg());
test_unarys!(sigmoid, assert_eq, sigmoid(), sigmoid());
test_unarys_out!(sigmoid_, assert_eq, sigmoid(), sigmoid_());
test_unarys!(sin, assert_eq, sin(), sin());
test_unarys_out!(sin_, assert_eq, sin(), sin_());
test_unarys!(sinh, assert_eq, sinh(), sinh());
test_unarys_out!(sinh_, assert_eq, sinh(), sinh_());
test_unarys!(sqrt, assert_eq, sqrt(), sqrt());
test_unarys_out!(sqrt_, assert_eq, sqrt(), sqrt_());
test_unarys!(tan, assert_eq, tan(), tan());
test_unarys_out!(tan_, assert_eq, tan(), tan_());
test_unarys!(tanh, assert_eq, tanh(), tanh());
test_unarys_out!(tanh_, assert_eq, tanh(), tanh_());
test_unarys!(celu, assert_eq, celu(), celu(1.0));
test_unarys_out!(celu_, assert_eq, celu(), celu_(1.0));
test_unarys!(exp2, assert_eq, exp2(), exp2());
test_unarys_out!(exp2_, assert_eq, exp2(), exp2_());
test_unarys!(gelu, assert_eq, gelu("none"), gelu());
test_unarys_out!(gelu_, assert_eq, gelu("none"), gelu_());
test_unarys!(elu, assert_eq, elu(), elu(1.0));
test_unarys_out!(elu_, assert_eq, elu(), elu_(1.0));
test_unarys!(leaky_relu, assert_eq, leaky_relu(), leaky_relu(0.01));
test_unarys!(mish, assert_eq, mish(), mish());
test_unarys_out!(mish_, assert_eq, mish(), mish_());
test_unarys!(relu, assert_eq, relu(), relu());
test_unarys!(selu, assert_eq, selu(), selu());
test_unarys_out!(selu_, assert_eq, selu(), selu_());
test_unarys!(softplus, assert_eq, softplus(), softplus());
test_unarys!(round, assert_eq, round(), round());
test_unarys!(clip, assert_eq, clamp(0.0, 1.0), clamp(0.0, 1.0));
test_unarys!(dropout, no_assert, dropout(0.5, false), dropout(0.5));
test_unarys!(hard_sigmoid, assert_eq, hardsigmoid(), hard_sigmoid());
test_unarys!(hard_swish, assert_eq, hardswish(), hard_swish());
test_unarys_out!(hard_swish_, assert_eq, hardswish(), hard_swish_());

#[test]
fn test_sub_tensor_sin() -> anyhow::Result<()> {
    let a = Tensor::<TestTypes>::arange(0, 100)?.reshape([10, 10])?;
    let slice = slice!(a[3:8, 3:8])?;
    let b = slice.sin()?;
    let tch_a = tch::Tensor::arange(100, (TCH_TEST_TYPES, tch::Device::Cpu)).reshape(&[10, 10][..]);
    let tch_slice = tch_a.slice(0, 3, 8, 1).slice(1, 3, 8, 1);
    let tch_b = tch_slice.sin();
    assert_eq(&b, &tch_b);
    Ok(())
}

#[test]
fn test_cast() -> anyhow::Result<()> {
    let a = Tensor::<TestTypes>::arange(0, 100)?.reshape([10, 10])?;
    let b = a.astype::<bool>()?;
    let tch_a = tch::Tensor::arange(100, (TCH_TEST_TYPES, tch::Device::Cpu)).reshape(&[10, 10][..]);
    let tch_b = tch_a.to_kind(tch::Kind::Bool);
    assert_eq_bool(&b, &tch_b);
    Ok(())
}
