use std::time::Duration;
use tensor_dyn::TensorCreator;
use criterion::{ black_box, criterion_group, BenchmarkId, Criterion };
use tch::{ Tensor, Kind, Device };
use tensor_dyn::{ tensor_base::_Tensor, Random };
use tensor_dyn::TensorInfo;
use crate::benchmarks::unary::float_cmp::assert_eq;
use tensor_dyn::FloatUaryOps;
use tensor_dyn::NormalUaryOps;
use tensor_dyn::Neg;
use tensor_dyn::TensorLike;

macro_rules! unary_bench_mark {
    (
        $name:expr,
        $shapes:expr,
        $tch_method:ident($($tch_args:expr),*),
        $hpt_method:ident($($hpt_args:expr),*)
    ) => {
        paste::paste! {
            #[cfg(any(feature = $name, feature = "unary"))]
            fn [<$name _benchmark>](c: &mut Criterion) {
                tensor_dyn::set_num_threads(num_cpus::get_physical());
                tch::set_num_threads(num_cpus::get_physical() as i32);
                let shapes = $shapes;
            
                let mut group = c.benchmark_group(concat!($name, " Benchmarks"));
                group.warm_up_time(Duration::new(1, 0)).measurement_time(Duration::new(3, 0)).sample_size(10);
                for idx in 0..shapes.len() {
                    let shape = shapes[idx];
                    let a = black_box(Tensor::randn(shape, (Kind::Float, Device::Cpu)));
                    let a2 = black_box(_Tensor::<f32>::randn(shape).unwrap());
                    group.bench_with_input(
                        BenchmarkId::new("torch", format!("tch {}", idx)),
                        &shapes[idx],
                        |b, _| {
                            b.iter(|| { a.$tch_method($($tch_args),*) });
                        }
                    );
                    group.bench_with_input(
                        BenchmarkId::new("hpt", format!("hpt {}", idx)),
                        &shapes[idx],
                        |b, _| {
                            b.iter(|| { a2.$hpt_method($($hpt_args),*).unwrap() });
                        }
                    );
                    let a = black_box(Tensor::randn(shape, (Kind::Double, Device::Cpu)));
                    let mut a2 = black_box(_Tensor::<f64>::empty(shape)).unwrap();
                    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const f64, a2.size()) };
                    let a2_raw_mut = a2.as_raw_mut();
                    a2_raw_mut.copy_from_slice(a_raw);
                    let a_sin = a.$tch_method($($tch_args),*);
                    let a2_sin = a2.$hpt_method($($hpt_args),*).unwrap();
                    assert_eq(&a_sin, &a2_sin);
                }
            
                group.finish();
            }
            #[cfg(any(feature = $name, feature = "unary"))]
            criterion_group!([<$name _benches>], [<$name _benchmark>]);
        }
    };
}

unary_bench_mark!(
    "sin",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    sin(),
    sin()
);
unary_bench_mark!(
    "cos",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    cos(),
    cos()
);
unary_bench_mark!(
    "tan",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    tan(),
    tan()
);
unary_bench_mark!(
    "asin",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    asin(),
    asin()
);
unary_bench_mark!(
    "acos",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    acos(),
    acos()
);
unary_bench_mark!(
    "atan",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    atan(),
    atan()
);
unary_bench_mark!(
    "sinh",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    sinh(),
    sinh()
);
unary_bench_mark!(
    "cosh",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    cosh(),
    cosh()
);
unary_bench_mark!(
    "tanh",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    tanh(),
    tanh()
);
unary_bench_mark!(
    "asinh",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    asinh(),
    asinh()
);
unary_bench_mark!(
    "acosh",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    acosh(),
    acosh()
);
unary_bench_mark!(
    "atanh",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    atanh(),
    atanh()
);
unary_bench_mark!(
    "selu",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    selu(),
    selu(None, None)
);
unary_bench_mark!(
    "sigmoid",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    sigmoid(),
    sigmoid()
);
unary_bench_mark!(
    "exp",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    exp(),
    exp()
);
unary_bench_mark!(
    "relu",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    relu(),
    relu()
);
unary_bench_mark!(
    "mish",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    mish(),
    mish()
);
unary_bench_mark!(
    "softplus",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    softplus(),
    softplus()
);
unary_bench_mark!(
    "relu6",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    relu6(),
    relu6()
);
unary_bench_mark!(
    "hard_sigmoid",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    hardsigmoid(),
    hard_sigmoid()
);
unary_bench_mark!(
    "gelu",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    gelu("none"),
    gelu()
);
unary_bench_mark!(
    "leaky_relu",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    leaky_relu(),
    leaky_relu(0.01)
);
unary_bench_mark!(
    "elu",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    elu(),
    elu(1.0)
);
unary_bench_mark!(
    "celu",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    celu(),
    celu(1.0)
);
unary_bench_mark!(
    "log10",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    log10(),
    log10()
);
unary_bench_mark!(
    "log2",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    log2(),
    log2()
);
unary_bench_mark!(
    "recip",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    reciprocal(),
    recip()
);
unary_bench_mark!(
    "sqrt",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    sqrt(),
    sqrt()
);
unary_bench_mark!(
    "exp2",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    exp2(),
    exp2()
);
unary_bench_mark!(
    "neg",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    neg(),
    neg()
);
unary_bench_mark!(
    "square",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    square(),
    square()
);
unary_bench_mark!(
    "abs",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    abs(),
    abs()
);
unary_bench_mark!(
    "ceil",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    ceil(),
    ceil()
);
unary_bench_mark!(
    "sign",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    sign(),
    sign()
);
unary_bench_mark!(
    "clip",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    clip(-1.0, 1.0),
    clip(-1.0, 1.0)
);