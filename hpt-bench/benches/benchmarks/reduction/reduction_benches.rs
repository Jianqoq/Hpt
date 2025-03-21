use criterion::{black_box, criterion_group, BenchmarkId, Criterion};
use hpt::common::cpu::TensorLike;
use hpt::common::TensorInfo;
use hpt::ops::*;
use hpt::Tensor;
use std::time::Duration;
use tch::{Device, Kind, Tensor as TchTensor};

#[allow(unused)]
fn assert_eq_i64(a: &TchTensor, b: &Tensor<i64>) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const i64, b.size()) };
    let b_raw = b.as_raw();
    assert_eq!(a_raw, b_raw);
}

macro_rules! reduction_bench_mark {
    (
        $name:expr,
        $shapes:expr,
        $axes:expr,
        $assert_method: ident,
        $tch_method:ident($($tch_args:expr),*),
        $hpt_method:ident($($hpt_args:expr),*)
    ) => {
        paste::paste! {
            #[cfg(any(feature = $name, feature = "reduction"))]
            fn [<$name _benchmark>](c: &mut Criterion) {
                hpt::utils::set_num_threads(num_cpus::get_physical());
                tch::set_num_threads(num_cpus::get_physical() as i32);
                let shapes = $shapes;
                let axes = $axes;
                let mut group = c.benchmark_group(concat!($name, " Benchmarks"));
                group.warm_up_time(Duration::new(1, 0)).measurement_time(Duration::new(3, 0)).sample_size(10);
                for idx in 0..shapes.len() {
                    let shape = shapes[idx];
                    let a = black_box(TchTensor::randn(shape, (Kind::Float, Device::Cpu)));
                    let a2 = black_box(Tensor::<f32, hpt::backend::Cpu>::randn(shape).unwrap());
                    for (i, axis) in axes.iter().enumerate() {
                        group.bench_with_input(
                            BenchmarkId::new("torch", format!("tch {}.{}", idx, i)),
                            &shapes[idx],
                            |b, _| {
                                b.iter(|| { a.$tch_method(axis.clone(), $($tch_args),*) });
                            }
                        );
                        group.bench_with_input(
                            BenchmarkId::new("hpt", format!("hpt {}.{}", idx, i)),
                            &shapes[idx],
                            |b, _| {
                                b.iter(|| { a2.$hpt_method(axis, $($hpt_args),*) });
                            }
                        );
                    }
                }

                group.finish();
            }
            #[cfg(any(feature = $name, feature = "reduction"))]
            criterion_group!([<$name _benches>], [<$name _benchmark>]);
        }
    };
}

reduction_bench_mark!(
    "sum",
    [[32, 32, 32], [64, 64, 64], [128, 128, 128],],
    [
        vec![0],
        // vec![1],
        // vec![2],
        // vec![0, 1],
        // vec![0, 2],
        // vec![1, 2],
        // vec![0, 1, 2]
    ],
    assert_eq_i64,
    sum_dim_intlist(false, Kind::Float),
    sum(false)
);

reduction_bench_mark!(
    "prod",
    [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ],
    [0, 1, 2],
    assert_eq_i64,
    prod_dim_int(false, Kind::Float),
    prod(false)
);
