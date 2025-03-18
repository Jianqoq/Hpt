use std::time::Duration;

use candle_core::Tensor as CandleTensor;
use criterion::{black_box, BenchmarkId, Criterion};
use hpt::ops::*;
use hpt::Tensor;
// use tch::{Device, Kind, Tensor as TchTensor};

fn matmul_f32_benchmark(c: &mut Criterion<crate::benchmarks::Timer>) {
    let num_threads = num_cpus::get_physical();
    hpt::utils::set_num_threads(num_threads);
    tch::set_num_threads(num_threads as i32);
    let mut ns = vec![];
    for i in 1..=80 {
        ns.push(64 * i);
    }

    let mut group = c.benchmark_group("matmul f16 Benchmarks");
    group
        .warm_up_time(Duration::new(1, 0))
        .measurement_time(Duration::new(3, 0))
        .sample_size(10);

    for n in ns {
        // let a = black_box(TchTensor::randn([n, n], (Kind::Float, Device::Cpu)));
        // let c = black_box(TchTensor::randn([n, n], (Kind::Float, Device::Cpu)));
        let a2 = black_box(Tensor::<half::f16>::randn([n, n]).unwrap());
        let c2 = black_box(Tensor::<half::f16>::randn([n, n]).unwrap());
        let a3 = black_box(
            CandleTensor::randn(
                half::f16::from_f32(0.0),
                half::f16::from_f32(1.0),
                [n, n]
                    .into_iter()
                    .map(|x| x as usize)
                    .collect::<Vec<usize>>(),
                &candle_core::Device::Cpu,
            )
            .unwrap(),
        );
        let c3 = black_box(
            CandleTensor::randn(
                half::f16::from_f32(0.0),
                half::f16::from_f32(1.0),
                [n, n]
                    .into_iter()
                    .map(|x| x as usize)
                    .collect::<Vec<usize>>(),
                &candle_core::Device::Cpu,
            )
            .unwrap(),
        );
        group.bench_with_input(BenchmarkId::new("hpt(builtin)", n), &n, |b, _| {
            b.iter(|| a2.matmul(&c2).unwrap());
        });
        // group.bench_with_input(BenchmarkId::new("torch", n), &n, |b, _| {
        //     b.iter(|| a.matmul(&c));
        // });
        group.bench_with_input(BenchmarkId::new("hpt(faer-gemm)", n), &n, |b, _| {
            b.iter(|| {
                a2.gemm(
                    &c2,
                    half::f16::from_f32(0.0),
                    half::f16::from_f32(1.0),
                    false,
                    false,
                    false,
                )
                .unwrap()
            });
        });
        group.bench_with_input(BenchmarkId::new("candle(mkl)", n), &n, |b, _| {
            b.iter(|| a3.matmul(&c3));
        });
    }

    group.finish();
}

pub fn matmul_f32_benches() {
    let criterion: criterion::Criterion<_> =
        (criterion::Criterion::default()).configure_from_args();
    let mut criterion = criterion.with_measurement(crate::benchmarks::Timer);
    matmul_f32_benchmark(&mut criterion);
}
