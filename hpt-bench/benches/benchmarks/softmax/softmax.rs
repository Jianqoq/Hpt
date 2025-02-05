use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use hpt_core::{Random, Tensor};
use std::time::Duration;
use tch::{Device, Kind, Tensor as TchTensor};

pub fn softmax_benchmark(c: &mut Criterion) {
    hpt_core::set_num_threads(num_cpus::get_physical());
    tch::set_num_threads(num_cpus::get_physical() as i32);
    let shapes = [[128, 128, 128], [256, 256, 256], [512, 512, 512]];
    let axes = [0, 1, 2];

    let mut group = c.benchmark_group("softmax Benchmarks");
    group
        .warm_up_time(Duration::new(1, 0))
        .measurement_time(Duration::new(3, 0))
        .sample_size(10);
    for idx in 0..shapes.len() {
        let shape = shapes[idx];
        let a = black_box(TchTensor::randn(shape, (Kind::Float, Device::Cpu)));
        let a2 = black_box(Tensor::<f32>::randn(shape).unwrap());
        for (i, axis) in axes.iter().enumerate() {
            group.bench_with_input(
                BenchmarkId::new("torch", format!("tch {}.{}", idx, i)),
                &shapes[idx],
                |b, _| {
                    b.iter(|| a.softmax(*axis, Kind::Float));
                },
            );
            group.bench_with_input(
                BenchmarkId::new("hpt", format!("hpt {}.{}", idx, i)),
                &shapes[idx],
                |b, _| {
                    b.iter(|| a2.softmax(*axis).unwrap());
                },
            );
        }
    }

    group.finish();
}

criterion_group!(softmax_benches, softmax_benchmark);
criterion_main!(softmax_benches);
