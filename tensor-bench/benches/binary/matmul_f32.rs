use std::time::Duration;

use criterion::{ black_box, criterion_group, criterion_main, BenchmarkId, Criterion };
use tch::{ Tensor, Kind, Device };
use tensor_dyn::{ tensor_base::_Tensor, Random };
use half::f16;
use tensor_dyn::Matmul;

fn add_f16_benchmark(c: &mut Criterion) {
    tensor_dyn::set_num_threads(num_cpus::get_physical());
    tch::set_num_threads(num_cpus::get_physical() as i32);

    let shapes = [
        [96, 96, 96],
        [128, 128, 128],
        [256, 256, 256],
        [512, 512, 512],
    ];

    let mut group = c.benchmark_group("matmul f16 Benchmarks");
    group.warm_up_time(Duration::new(1, 0)).measurement_time(Duration::new(3, 0)).sample_size(10);

    for idx in 0..shapes.len() {
        let a = black_box(Tensor::randn(shapes[idx], (Kind::Half, Device::Cpu)));
        let c = black_box(Tensor::randn(shapes[idx], (Kind::Half, Device::Cpu)));
        let a2 = black_box(_Tensor::<f16>::randn(shapes[idx]).unwrap());
        let c2 = black_box(_Tensor::<f16>::randn(shapes[idx]).unwrap());
        group.bench_with_input(BenchmarkId::new("hpt", idx), &idx, |b, _| {
            b.iter(|| { a2.matmul(&c2).unwrap() });
        });
        group.bench_with_input(BenchmarkId::new("torch", idx), &idx, |b, _| {
            b.iter(|| { a.matmul(&c) });
        });
    }

    group.finish();
}

criterion_group!(benches, add_f16_benchmark);
criterion_main!(benches);
