#![allow(unused_imports)]

use std::time::Duration;

use candle_core::Tensor as CandleTensor;
use criterion::{black_box, BenchmarkId, Criterion};
use half::f16;
use hpt::ops::*;
use hpt::Tensor;
use hpt_dyn::DType;
use hpt_dyn::Tensor as DynTensor;
use hpt_dyn::Device;

fn matmul_f32_benchmark(c: &mut Criterion<crate::benchmarks::Timer>) {
    let num_threads = num_cpus::get();
    hpt::utils::set_num_threads(num_threads);
    tch::set_num_threads(num_threads as i32);
    let mut ns = vec![];
    for i in 1..=80 {
        ns.push(64 * i);
    }

    let mut group = c.benchmark_group("matmul f32 Benchmarks");
    group
        .warm_up_time(Duration::new(1, 0))
        .measurement_time(Duration::new(3, 0))
        .sample_size(10);

    for n in ns {
        let a2 = black_box(Tensor::<f32>::randn([n, n]).unwrap());
        let c2 = black_box(Tensor::<f32>::randn([n, n]).unwrap());
        group.bench_with_input(BenchmarkId::new("hpt(builtin)", n), &n, |b, _| {
            b.iter(|| a2.matmul(&c2).unwrap());
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
