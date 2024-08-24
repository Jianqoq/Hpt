use std::time::Duration;

use criterion::{ black_box, criterion_group, criterion_main, BenchmarkId, Criterion };
use tch::{ Tensor, Kind, Device };
use tensor_dyn::{ tensor_base::_Tensor, Random };
use half::f16;

fn add_f16_benchmark(c: &mut Criterion) {
    tensor_dyn::set_num_threads(num_cpus::get_physical());
    tch::set_num_threads(num_cpus::get_physical() as i32);
    let shapes = [
        [96, 96, 96, 96],
        [98, 98, 98, 98],
        [100, 100, 100, 100],
        [102, 102, 102, 102],
        [104, 104, 104, 104],
    ];

    let mut group = c.benchmark_group("add f16 Benchmarks");
    group.warm_up_time(Duration::new(1, 0)).measurement_time(Duration::new(3, 0)).sample_size(10);
    for idx in 0..shapes.len() {
        let a = black_box(Tensor::randn(shapes[idx], (Kind::Half, Device::Cpu)));
        let c = black_box(Tensor::randn(shapes[idx], (Kind::Half, Device::Cpu)));
        let a2 = black_box(_Tensor::<f16>::randn(shapes[idx]).unwrap());
        let c2 = black_box(_Tensor::<f16>::randn(shapes[idx]).unwrap());
        group.bench_with_input(BenchmarkId::new("torch", idx), &shapes[idx], |b, _| {
            b.iter(|| { &a + &c });
        });
        group.bench_with_input(BenchmarkId::new("hpt", idx), &shapes[idx], |b, _| {
            b.iter(|| { &a2 + &c2 });
        });
    }

    group.finish();
}

criterion_group!(benches, add_f16_benchmark);
criterion_main!(benches);
