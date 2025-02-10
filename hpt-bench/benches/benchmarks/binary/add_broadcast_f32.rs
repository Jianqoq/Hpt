use std::time::Duration;

use candle_core::Tensor as CandleTensor;
use criterion::{black_box, criterion_group, BenchmarkId, Criterion};
use hpt_core::{Random, Tensor as HptTensor};
use ndarray::{Array, Zip};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use tch::{Device, Kind, Tensor as TchTensor};

fn add_f32_benchmark(c: &mut Criterion) {
    hpt_core::set_num_threads(num_cpus::get_physical());
    tch::set_num_threads(num_cpus::get_physical() as i32);
    let shapes = [[100, 100, 100, 100]];

    let mut group = c.benchmark_group("add f32 Benchmarks");
    group
        .warm_up_time(Duration::new(1, 0))
        .measurement_time(Duration::new(3, 0))
        .sample_size(10);
    for idx in 0..shapes.len() {
        let mut shape1 = shapes[idx];
        shape1[0] = 1;
        shape1[2] = 1;
        let mut shape2 = shapes[idx];
        shape2[1] = 1;
        shape2[3] = 1;
        let a = black_box(TchTensor::randn(shape1, (Kind::Float, Device::Cpu)));
        let c = black_box(TchTensor::randn(shape2, (Kind::Float, Device::Cpu)));
        let a2 = black_box(HptTensor::<f32>::randn(shape1).unwrap());
        let c2 = black_box(HptTensor::<f32>::randn(shape2).unwrap());
        let a3 = black_box(
            CandleTensor::randn(
                0f32,
                1f32,
                shape1
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
                shape2
                    .into_iter()
                    .map(|x| x as usize)
                    .collect::<Vec<usize>>(),
                &candle_core::Device::Cpu,
            )
            .unwrap(),
        );
        let a4 = black_box(Array::random(
            shape1
                .into_iter()
                .map(|x| x as usize)
                .collect::<Vec<usize>>(),
            Uniform::new(0f32, 1f32),
        ));
        let c4 = black_box(Array::random(
            shape2
                .into_iter()
                .map(|x| x as usize)
                .collect::<Vec<usize>>(),
            Uniform::new(0f32, 1f32),
        ));

        group.bench_with_input(BenchmarkId::new("torch", idx), &shapes[idx], |b, _| {
            b.iter(|| &a + &c);
        });

        group.bench_with_input(BenchmarkId::new("hpt", idx), &shapes[idx], |b, _| {
            b.iter(|| &a2 + &c2);
        });

        group.bench_with_input(BenchmarkId::new("candle", idx), &shapes[idx], |b, _| {
            b.iter(|| a3.broadcast_add(&c3).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("ndarray", idx), &shapes[idx], |b, _| {
            b.iter(|| {
                let mut res = Array::<f32, _>::zeros(
                    shapes[idx]
                        .into_iter()
                        .map(|x| x as usize)
                        .collect::<Vec<usize>>(),
                );
                let a4_broadcast = a4
                    .broadcast(
                        shapes[idx]
                            .into_iter()
                            .map(|x| x as usize)
                            .collect::<Vec<usize>>(),
                    )
                    .unwrap();
                let c4_broadcast = c4
                    .broadcast(
                        shapes[idx]
                            .into_iter()
                            .map(|x| x as usize)
                            .collect::<Vec<usize>>(),
                    )
                    .unwrap();
                Zip::from(&mut res)
                    .and(&a4_broadcast)
                    .and(&c4_broadcast)
                    .par_for_each(|c, &a, &b| {
                        *c = a + b;
                    });
            });
        });
    }

    group.finish();
}

criterion_group!(add_broadcast_f32_benches, add_f32_benchmark);
