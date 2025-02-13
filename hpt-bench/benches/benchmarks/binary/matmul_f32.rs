use std::time::Duration;

use candle_core::Tensor as CandleTensor;
use criterion::{black_box, criterion_group, BenchmarkId, Criterion};
use hpt::Matmul;
use hpt::{Random, ShapeManipulate, Tensor};
use tch::{Device, Kind, Tensor as TchTensor};

fn matmul_f32_benchmark(c: &mut Criterion) {
    hpt::set_num_threads(num_cpus::get_physical());
    tch::set_num_threads(num_cpus::get_physical() as i32);

    let shapes = [
        [10, 4096, 2],
        [10, 4096, 4],
        [10, 4096, 8],
        [10, 4096, 16],
        [10, 4096, 32],
        [10, 4096, 64],
        [10, 4096, 128],
        [10, 4096, 256],
        [10, 4096, 512],
        [10, 4096, 1024],
        // [10, 2, 4096],
        // [10, 4, 4096],
        // [10, 8, 4096],
        // [10, 16, 4096],
        // [10, 32, 4096],
        // [10, 64, 4096],
        // [10, 128, 4096],
        // [10, 256, 4096],
        // [10, 512, 4096],
        // [10, 1024, 4096],
    ];

    let mut group = c.benchmark_group("matmul f16 Benchmarks");
    group
        .warm_up_time(Duration::new(1, 0))
        .measurement_time(Duration::new(3, 0))
        .sample_size(10);

    for idx in 0..shapes.len() {
        let a = black_box(TchTensor::randn(shapes[idx], (Kind::Float, Device::Cpu)));
        let c = black_box(TchTensor::randn(shapes[idx], (Kind::Float, Device::Cpu)));
        let a2 = black_box(Tensor::<f32>::randn(shapes[idx]).unwrap());
        let c2 = black_box(Tensor::<f32>::randn(shapes[idx]).unwrap());
        let a3 = black_box(
            CandleTensor::randn(
                0f32,
                1f32,
                shapes[idx]
                    .into_iter()
                    .map(|x| x as usize)
                    .collect::<Vec<usize>>(),
                &candle_core::Device::Cpu,
            )
            .unwrap(),
        );
        let c3 = black_box(
            CandleTensor::randn(
                0f32,
                1f32,
                shapes[idx]
                    .into_iter()
                    .map(|x| x as usize)
                    .collect::<Vec<usize>>(),
                &candle_core::Device::Cpu,
            )
            .unwrap(),
        );
        group.bench_with_input(BenchmarkId::new("hpt", idx), &idx, |b, _| {
            b.iter(|| a2.matmul(c2.t().unwrap()).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("torch", idx), &idx, |b, _| {
            b.iter(|| a.matmul(&c.transpose(1, 2)));
        });
        group.bench_with_input(BenchmarkId::new("candle", idx), &idx, |b, _| {
            b.iter(|| a3.matmul(&c3.t().unwrap()).unwrap());
        });
    }

    group.finish();
}

criterion_group!(matmul_f32_benches, matmul_f32_benchmark);
