use candle_core::Tensor as CandleTensor;
use criterion::{black_box, criterion_group, BenchmarkId, Criterion};
use hpt_core::{Random, Tensor};
use std::time::Duration;
use tch::{Device, Kind, Tensor as TchTensor};

pub(crate) fn add_f32_benchmarks(c: &mut Criterion) {
    hpt_core::set_num_threads(num_cpus::get_physical());
    tch::set_num_threads(num_cpus::get_physical() as i32);
    let shapes = [
        [96, 96, 96, 96],
        [98, 98, 98, 98],
        [100, 100, 100, 100],
        [102, 102, 102, 102],
        [104, 104, 104, 104],
    ];

    let mut group = c.benchmark_group("add f32 Benchmarks");
    group
        .warm_up_time(Duration::new(1, 0))
        .measurement_time(Duration::new(3, 0))
        .sample_size(10);
    for idx in 0..shapes.len() {
        let shape = shapes[idx];
        let a = black_box(TchTensor::randn(shape, (Kind::Float, Device::Cpu)));
        let c = black_box(TchTensor::randn(shape, (Kind::Float, Device::Cpu)));
        let a2 = black_box(Tensor::<f32>::randn(shape).unwrap());
        let c2 = black_box(Tensor::<f32>::randn(shape).unwrap());
        let a3 = black_box(
            CandleTensor::randn(
                0f32,
                1f32,
                shape
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
                shape
                    .into_iter()
                    .map(|x| x as usize)
                    .collect::<Vec<usize>>(),
                &candle_core::Device::Cpu,
            )
            .unwrap(),
        );
        group.bench_with_input(BenchmarkId::new("torch", idx), &shape, |b, _| {
            b.iter(|| &a + &c);
        });

        group.bench_with_input(BenchmarkId::new("hpt", idx), &shape, |b, _| {
            b.iter(|| &a2 + &c2);
        });

        group.bench_with_input(BenchmarkId::new("candle", idx), &shape, |b, _| {
            b.iter(|| &a3 + &c3);
        });
    }

    group.finish();
}

criterion_group!(add_f32_benches, add_f32_benchmarks);
