use criterion::{black_box, criterion_group, BenchmarkId, Criterion};
use hpt::ops::*;
use hpt::types::Complex32;
use hpt::Tensor;
use std::time::Duration;
use tch::{Device, Kind, Tensor as TchTensor};

fn fft_benchmark(c: &mut Criterion) {
    tch::set_num_threads(num_cpus::get_physical() as i32);
    hpt::utils::set_num_threads(num_cpus::get_physical());
    let shapes = [
        [16, 16, 16],
        [32, 32, 32],
        [64, 64, 64],
        [128, 128, 128],
        [256, 256, 256],
        [512, 512, 512],
    ];
    let mut group = c.benchmark_group(concat!("fft", " Benchmarks"));
    group
        .warm_up_time(Duration::new(1, 0))
        .measurement_time(Duration::new(3, 0))
        .sample_size(10);
    for idx in 0..shapes.len() {
        let inp_shape = shapes[idx];
        let a = black_box(TchTensor::empty(
            inp_shape,
            (Kind::ComplexFloat, Device::Cpu),
        ));
        let a2 = black_box(Tensor::<Complex32>::empty(inp_shape).unwrap());
        group.bench_with_input(
            BenchmarkId::new("torch", format!("tch {}", idx)),
            &shapes[idx],
            |b, _| {
                b.iter(|| a.fft_fftn(&inp_shape[..], &[0, 1, 2][..], "backward"));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("hpt", format!("hpt {}", idx)),
            &shapes[idx],
            |b, _| {
                b.iter(|| {
                    a2.fftn(&inp_shape[..], &[0, 1, 2][..], Some("backward"))
                        .unwrap()
                });
            },
        );
    }
    group.finish();
}

criterion_group!(fft_benches, fft_benchmark);
