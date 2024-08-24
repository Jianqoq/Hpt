use criterion::{ black_box, criterion_group, criterion_main, BenchmarkId, Criterion };
use tch::{ Tensor, Kind, Device };
use tensor_dyn::{ tensor_base::_Tensor, Random };
use half::f16;

fn random_f16_benchmark(c: &mut Criterion) {
    tch::set_num_threads(tensor_dyn::get_num_threads() as i32);
    let shapes = [
        [8, 8, 8],
        [16, 16, 16],
        [32, 32, 32],
        [64, 64, 64],
        [96, 96, 96],
        [128, 128, 128],
        [256, 256, 256],
    ];

    let mut group = c.benchmark_group("randn f16 Benchmarks");
    for idx in 0..shapes.len() {
        group.bench_with_input(BenchmarkId::new("torch", idx), &idx, |b, idx| {
            b.iter(|| black_box(Tensor::randn(shapes[*idx], (Kind::Half, Device::Cpu))));
        });
        group.bench_with_input(BenchmarkId::new("hpt", idx), &idx, |b, idx| {
            b.iter(|| black_box(_Tensor::<f16>::randn(shapes[*idx]).unwrap()));
        });
    }

    group.finish();
}

criterion_group!(benches, random_f16_benchmark);
criterion_main!(benches);
