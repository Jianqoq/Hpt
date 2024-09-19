#![allow(unused_imports)]
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use tch::Tensor;
use tensor_common::slice;
use tensor_common::slice::Slice;
use tensor_dyn::slice::SliceOps;
use tensor_dyn::ShapeManipulate;
use tensor_dyn::TensorInfo;
use tensor_dyn::{tensor_base::_Tensor, TensorCreator};
use tensor_macros::match_selection;
use tensor_dyn::FloatUaryOps;
use tensor_dyn::NormalUaryOps;
use tensor_dyn::Neg;

#[allow(unused)]
fn assert_eq(b: &_Tensor<f64>, a: &Tensor) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const f64, b.size()) };
    let b_raw = b.as_raw();
    let tolerance = 10e-16;

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
fn no_assert(b: &_Tensor<f64>, a: &Tensor) {}

#[allow(unused)]
fn assert_eq_bool(b: &_Tensor<bool>, a: &Tensor) {
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
                let a = _Tensor::<f64>::empty($shapes)?;
                a.as_raw_mut().copy_from_slice(unsafe {
                    std::slice::from_raw_parts(tch_a.data_ptr() as *const f64, a.size())
                });
                let b = a.$hpt_method($($hpt_args),*)?;
                let tch_b = tch_a.$tch_method($($tch_args),*);
                $assert_method(&b, &tch_b);
                Ok(())
            }
        }
    };
}

test_unarys!(abs, [1000], assert_eq, abs(), abs());
test_unarys!(acos, [1000], assert_eq, acos(), acos());
test_unarys!(acosh, [1000], assert_eq, acosh(), acosh());
test_unarys!(asin, [1000], assert_eq, asin(), asin());
test_unarys!(asinh, [1000], assert_eq, asinh(), asinh());
test_unarys!(atan, [1000], assert_eq, atan(), atan());
test_unarys!(atanh, [1000], assert_eq, atanh(), atanh()); // not sure why precision is huge
test_unarys!(ceil, [1000], assert_eq, ceil(), ceil());
test_unarys!(cos, [1000], assert_eq, cos(), cos());
test_unarys!(cosh, [1000], assert_eq, cosh(), cosh());
test_unarys!(erf, [1000], assert_eq, erf(), erf());
test_unarys!(exp, [1000], assert_eq, exp(), exp());
test_unarys!(floor, [1000], assert_eq, floor(), floor());
test_unarys!(is_inf, [1000], assert_eq_bool, isinf(), is_inf());
test_unarys!(is_nan, [1000], assert_eq_bool, isnan(), is_nan());
test_unarys!(log, [1000], assert_eq, log(), ln());
test_unarys!(log10, [1000], assert_eq, log10(), log10());
test_unarys!(log2, [1000], assert_eq, log2(), log2());
test_unarys!(recip, [1000], assert_eq, reciprocal(), recip());
test_unarys!(neg, [1000], assert_eq, neg(), neg());
test_unarys!(sigmoid, [1000], assert_eq, sigmoid(), sigmoid());
test_unarys!(sin, [1000], assert_eq, sin(), sin());
test_unarys!(sinh, [1000], assert_eq, sinh(), sinh());
test_unarys!(sqrt, [1000], assert_eq, sqrt(), sqrt());
test_unarys!(tan, [1000], assert_eq, tan(), tan());
test_unarys!(tanh, [1000], assert_eq, tanh(), tanh());
test_unarys!(celu, [1000], assert_eq, celu(), celu(1.0));
test_unarys!(exp2, [1000], assert_eq, exp2(), exp2());
test_unarys!(gelu, [1000], assert_eq, gelu("none"), gelu());
test_unarys!(elu, [1000], assert_eq, elu(), elu(1.0));
test_unarys!(
    leaky_relu,
    [1000],
    assert_eq,
    leaky_relu(),
    leaky_relu(0.01)
);
test_unarys!(mish, [1000], assert_eq, mish(), mish());
test_unarys!(relu, [1000], assert_eq, relu(), relu());
test_unarys!(selu, [1000], assert_eq, selu(), selu(None, None));
test_unarys!(softplus, [1000], assert_eq, softplus(), softplus());
test_unarys!(round, [1000], assert_eq, round(), round());
test_unarys!(clip, [1000], assert_eq, clamp(0.0, 1.0), clip(0.0, 1.0));
test_unarys!(
    dropout,
    [1000],
    no_assert,
    dropout(0.5, false),
    dropout(0.5)
);
test_unarys!(
    hard_sigmoid,
    [1000],
    assert_eq,
    hardsigmoid(),
    fast_hard_sigmoid()
);
test_unarys!(hard_swish, [1000], assert_eq, hardswish(), hard_swish());

#[test]
fn test_sub_tensor_sin() -> anyhow::Result<()> {
    let a = _Tensor::<f64>::arange(0, 100)?.reshape([10, 10])?;
    let slice = slice!(a[3:8, 3:8])?;
    let b = slice.sin()?;
    let tch_a =
        Tensor::arange(100, (tch::Kind::Double, tch::Device::Cpu)).reshape(&[10, 10][..]);
    let tch_slice = tch_a.slice(0, 3, 8, 1).slice(1, 3, 8, 1);
    let tch_b = tch_slice.sin();
    assert_eq(&b, &tch_b);
    Ok(())
}

#[test]
fn test_cast() -> anyhow::Result<()> {
    let a = _Tensor::<f64>::arange(0, 100)?.reshape([10, 10])?;
    let b = a.astype::<bool>()?;
    let tch_a =
        Tensor::arange(100, (tch::Kind::Double, tch::Device::Cpu)).reshape(&[10, 10][..]);
    let tch_b = tch_a.to_kind(tch::Kind::Bool);
    assert_eq_bool(&b, &tch_b);
    Ok(())
}
