use candle_core::Tensor as CandleTensor;
use criterion::{black_box, criterion_group, BenchmarkId, Criterion};
use hpt::{Random, Tensor};
use ndarray::{Array, Zip};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::time::Duration;
use tch::{Device, Kind, Tensor as TchTensor};

pub(crate) fn add_f32_benchmarks(c: &mut Criterion) {
    hpt::set_num_threads(num_cpus::get_physical());
    tch::set_num_threads(num_cpus::get_physical() as i32);
    let shapes = [[100, 100, 100, 100]];

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
        let a4 = black_box(Array::random(
            shape
                .into_iter()
                .map(|x| x as usize)
                .collect::<Vec<usize>>(),
            Uniform::new(0f32, 1f32),
        ));
        let c4 = black_box(Array::random(
            shape
                .into_iter()
                .map(|x| x as usize)
                .collect::<Vec<usize>>(),
            Uniform::new(0f32, 1f32),
        ));

        group.bench_with_input(BenchmarkId::new("torch", idx), &shape, |b, _| {
            b.iter(|| &a + &c);
        });

        group.bench_with_input(BenchmarkId::new("hpt", idx), &shape, |b, _| {
            b.iter(|| &a2 + &c2);
        });

        group.bench_with_input(BenchmarkId::new("candle", idx), &shape, |b, _| {
            b.iter(|| &a3 + &c3);
        });

        group.bench_with_input(BenchmarkId::new("ndarray", idx), &shape, |b, _| {
            b.iter(|| {
                let mut res = Array::<f32, _>::zeros(a4.dim());
                Zip::from(&mut res)
                    .and(&a4)
                    .and(&c4)
                    .par_for_each(|c, &a, &b| {
                        *c = a + b;
                    });
            });
        });
    }

    group.finish();
}

criterion_group!(add_f32_benches, add_f32_benchmarks);
