#![allow(unused_imports)]
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use tensor_common::slice;
use tensor_common::slice::Slice;
use tensor_dyn::FloatUaryOps;
use tensor_dyn::NormalUaryOps;
use tensor_dyn::ShapeManipulate;
use tensor_dyn::TensorInfo;
use tensor_dyn::TensorLike;
use tensor_dyn::{Tensor, TensorCreator};
use tensor_macros::match_selection;
use tensor_dyn::AdvanceOps;
#[allow(unused)]
fn assert_eq(b: &Tensor<f64>, a: &tch::Tensor) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const f64, b.size()) };
    let b_raw = b.as_raw();
    let tolerance = 1e-10;

    for i in 0..b.size() {
        let abs_diff = (a_raw[i] - b_raw[i]).abs();
        let relative_diff = abs_diff / b_raw[i].abs().max(f64::EPSILON);

        if abs_diff > tolerance && relative_diff > tolerance {
            panic!(
                "{} != {} (abs_diff: {}, relative_diff: {})",
                a_raw[i], b_raw[i], abs_diff, relative_diff
            );
        }
    }
}

#[allow(unused)]
fn no_assert(b: &Tensor<f64>, a: &tch::Tensor) {}

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
        $shapes:expr,
        $assert_method:ident,
        $tch_method:ident($($tch_args:expr),*),
        $hpt_method:ident($($hpt_args:expr),*)
    ) => {
        paste::paste! {
            #[test]
            fn [<test _ $name>]() -> anyhow::Result<()> {
                let tch_a = tch::Tensor::randn($shapes, (tch::Kind::Double, tch::Device::Cpu));
                let mut a = Tensor::<f64>::empty($shapes)?;
                let a_size = a.size();
                a.as_raw_mut().copy_from_slice(unsafe {
                    std::slice::from_raw_parts(tch_a.data_ptr() as *const f64, a_size)
                });
                let b = a.$hpt_method($($hpt_args),*)?;
                let tch_b = tch_a.$tch_method($($tch_args),*);
                $assert_method(&b, &tch_b);
                Ok(())
            }
        }
    };
}

macro_rules! test_unarys_out {
    (
        $name:ident,
        $shapes:expr,
        $assert_method:ident,
        $tch_method:ident($($tch_args:expr),*),
        $hpt_method:ident($($hpt_args:expr),*)
    ) => {
        paste::paste! {
            #[test]
            fn [<test _ $name>]() -> anyhow::Result<()> {
                let tch_a = tch::Tensor::randn($shapes, (tch::Kind::Double, tch::Device::Cpu));
                let mut a = Tensor::<f64>::empty($shapes)?;
                let a_size = a.size();
                a.as_raw_mut().copy_from_slice(unsafe {
                    std::slice::from_raw_parts(tch_a.data_ptr() as *const f64, a_size)
                });
                let b = a.$hpt_method($($hpt_args,)* &a)?;
                let tch_b = tch_a.$tch_method($($tch_args),*);
                $assert_method(&b, &tch_b);
                Ok(())
            }
        }
    };
}

test_unarys!(abs, [1000], assert_eq, abs(), abs());
test_unarys_out!(abs_, [1000], assert_eq, abs(), abs_());
test_unarys!(acos, [1000], assert_eq, acos(), acos());
test_unarys_out!(acos_, [1000], assert_eq, acos(), acos_());
test_unarys!(acosh, [1000], assert_eq, acosh(), acosh());
test_unarys_out!(acosh_, [1000], assert_eq, acosh(), acosh_());
test_unarys!(asin, [1000], assert_eq, asin(), asin());
test_unarys_out!(asin_, [1000], assert_eq, asin(), asin_());
test_unarys!(asinh, [1000], assert_eq, asinh(), asinh());
test_unarys_out!(asinh_, [1000], assert_eq, asinh(), asinh_());
test_unarys!(atan, [1000], assert_eq, atan(), atan());
test_unarys_out!(atan_, [1000], assert_eq, atan(), atan_());
test_unarys!(atanh, [1000], assert_eq, atanh(), atanh()); // not sure why precision is huge
test_unarys_out!(atanh_, [1000], assert_eq, atanh(), atanh_());
test_unarys!(ceil, [1000], assert_eq, ceil(), ceil());
test_unarys_out!(ceil_, [1000], assert_eq, ceil(), ceil_());
test_unarys!(cos, [1000], assert_eq, cos(), cos());
test_unarys_out!(cos_, [1000], assert_eq, cos(), cos_());
test_unarys!(cosh, [1000], assert_eq, cosh(), cosh());
test_unarys_out!(cosh_, [1000], assert_eq, cosh(), cosh_());
test_unarys!(erf, [1000], assert_eq, erf(), erf());
test_unarys!(exp, [1000], assert_eq, exp(), exp());
test_unarys_out!(exp_, [1000], assert_eq, exp(), exp_());
test_unarys!(floor, [1000], assert_eq, floor(), floor());
// test_unarys!(is_inf, [1000], assert_eq_bool, isinf(), is_inf());
// test_unarys!(is_nan, [1000], assert_eq_bool, isnan(), is_nan());
test_unarys!(log, [1000], assert_eq, log(), ln());
test_unarys_out!(log_, [1000], assert_eq, log(), ln_());
test_unarys!(log10, [1000], assert_eq, log10(), log10());
test_unarys_out!(log10_, [1000], assert_eq, log10(), log10_());
test_unarys!(log2, [1000], assert_eq, log2(), log2());
test_unarys_out!(log2_, [1000], assert_eq, log2(), log2_());
test_unarys!(recip, [1000], assert_eq, reciprocal(), recip());
test_unarys_out!(recip_, [1000], assert_eq, reciprocal(), recip_());
test_unarys!(neg, [1000], assert_eq, neg(), neg());
test_unarys!(sigmoid, [1000], assert_eq, sigmoid(), sigmoid());
test_unarys_out!(sigmoid_, [1000], assert_eq, sigmoid(), sigmoid_());
test_unarys!(sin, [1000], assert_eq, sin(), sin());
test_unarys_out!(sin_, [1000], assert_eq, sin(), sin_());
test_unarys!(sinh, [1000], assert_eq, sinh(), sinh());
test_unarys_out!(sinh_, [1000], assert_eq, sinh(), sinh_());
test_unarys!(sqrt, [1000], assert_eq, sqrt(), sqrt());
test_unarys_out!(sqrt_, [1000], assert_eq, sqrt(), sqrt_());
test_unarys!(tan, [1000], assert_eq, tan(), tan());
test_unarys_out!(tan_, [1000], assert_eq, tan(), tan_());
test_unarys!(tanh, [1000], assert_eq, tanh(), tanh());
test_unarys_out!(tanh_, [1000], assert_eq, tanh(), tanh_());
test_unarys!(celu, [1000], assert_eq, celu(), celu(1.0));
test_unarys_out!(celu_, [1000], assert_eq, celu(), celu_(1.0));
test_unarys!(exp2, [1000], assert_eq, exp2(), exp2());
test_unarys_out!(exp2_, [1000], assert_eq, exp2(), exp2_());
test_unarys!(gelu, [1000], assert_eq, gelu("none"), gelu());
test_unarys_out!(gelu_, [1000], assert_eq, gelu("none"), gelu_());
test_unarys!(elu, [1000], assert_eq, elu(), elu(1.0));
test_unarys_out!(elu_, [1000], assert_eq, elu(), elu_(1.0));
test_unarys!(
    leaky_relu,
    [1000],
    assert_eq,
    leaky_relu(),
    leaky_relu(0.01)
);
test_unarys!(mish, [1000], assert_eq, mish(), mish());
test_unarys_out!(mish_, [1000], assert_eq, mish(), mish_());
test_unarys!(relu, [1000], assert_eq, relu(), relu());
test_unarys!(selu, [1000], assert_eq, selu(), selu(None, None));
test_unarys_out!(selu_, [1000], assert_eq, selu(), selu_(None, None));
test_unarys!(softplus, [1000], assert_eq, softplus(), softplus());
test_unarys!(round, [1000], assert_eq, round(), round());
test_unarys!(clip, [1000], assert_eq, clamp(0.0, 1.0), clamp(0.0, 1.0));
test_unarys!(
    dropout,
    [1000],
    no_assert,
    dropout(0.5, false),
    dropout(0.5)
);
test_unarys!(
    fast_hard_sigmoid,
    [1000],
    assert_eq,
    hardsigmoid(),
    fast_hard_sigmoid()
);
test_unarys!(hard_swish, [1000], assert_eq, hardswish(), hard_swish());
test_unarys_out!(
    hard_swish_,
    [1000],
    assert_eq,
    hardswish(),
    hard_swish_()
);


#[test]
fn test_sub_tensor_sin() -> anyhow::Result<()> {
    let a = Tensor::<f64>::arange(0, 100)?.reshape([10, 10])?;
    let slice = slice!(a[3:8, 3:8])?;
    let b = slice.sin()?;
    let tch_a = tch::Tensor::arange(100, (tch::Kind::Double, tch::Device::Cpu)).reshape(&[10, 10][..]);
    let tch_slice = tch_a.slice(0, 3, 8, 1).slice(1, 3, 8, 1);
    let tch_b = tch_slice.sin();
    assert_eq(&b, &tch_b);
    Ok(())
}

#[test]
fn test_cast() -> anyhow::Result<()> {
    let a = Tensor::<f64>::arange(0, 100)?.reshape([10, 10])?;
    let b = a.astype::<bool>()?;
    let tch_a = tch::Tensor::arange(100, (tch::Kind::Double, tch::Device::Cpu)).reshape(&[10, 10][..]);
    let tch_b = tch_a.to_kind(tch::Kind::Bool);
    assert_eq_bool(&b, &tch_b);
    Ok(())
}
