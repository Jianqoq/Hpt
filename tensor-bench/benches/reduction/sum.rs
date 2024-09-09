use std::time::Duration;

use criterion::{ black_box, criterion_group, criterion_main, BenchmarkId, Criterion };
use tch::{ Tensor, Kind, Device };
use tensor_dyn::{ tensor_base::_Tensor, Random };

fn add_f16_benchmark(c: &mut Criterion) {
    tensor_dyn::set_num_threads(num_cpus::get_physical());
    tch::set_num_threads(num_cpus::get_physical() as i32);
    let shapes = [
        [64, 64, 64, 64],
        [128, 128, 128, 128],
        [256, 256, 256, 256],
        [384, 384, 384, 384],
        [512, 512, 512, 512],
    ];
    let axes = [
        vec![0],
        vec![1],
        vec![2],
        vec![3],
        vec![0, 1],
        vec![0, 2],
        vec![0, 3],
        vec![1, 2],
        vec![1, 3],
        vec![2, 3],
        vec![0, 1, 2],
        vec![0, 1, 3],
        vec![0, 2, 3],
        vec![1, 2, 3],
        vec![0, 1, 2, 3],
    ];

    let mut group = c.benchmark_group("sum Benchmarks");
    group.warm_up_time(Duration::new(1, 0)).measurement_time(Duration::new(3, 0)).sample_size(10);
    for idx in 0..shapes.len() {
        let shape = shapes[idx];
        let a = black_box(Tensor::randn(shape, (Kind::Float, Device::Cpu)));
        let a2 = black_box(_Tensor::<f32>::randn(shape).unwrap());
        for (i, axis) in axes.iter().enumerate() {
            group.bench_with_input(BenchmarkId::new("torch", idx + i), &shapes[idx], |b, _| {
                b.iter(|| { a.sum_dim_intlist(axis, false, Kind::Float) });
            });
            group.bench_with_input(BenchmarkId::new("hpt", idx + i + 1), &shapes[idx], |b, _| {
                b.iter(|| { a2.sum(axis, false) });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, add_f16_benchmark);
criterion_main!(benches);
