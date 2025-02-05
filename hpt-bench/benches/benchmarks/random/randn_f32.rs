use criterion::{ black_box, criterion_group, criterion_main, BenchmarkId, Criterion };
use tch::{ Tensor, Kind, Device };
use hpt_core::{ tensor_base::_Tensor, Random };

fn random_f32_benchmark(c: &mut Criterion) {
    tch::set_num_threads(hpt_core::get_num_threads() as i32);
    let shapes = [
        [8, 8, 8],
        [16, 16, 16],
        [32, 32, 32],
        [64, 64, 64],
        [96, 96, 96],
        [128, 128, 128],
        [256, 256, 256],
    ];

    let mut group = c.benchmark_group("randn f32 Benchmarks");
    for idx in 0..shapes.len() {
        group.bench_with_input(BenchmarkId::new("torch", idx), &idx, |b, idx| {
            b.iter(|| black_box(Tensor::randn(shapes[*idx], (Kind::Float, Device::Cpu))));
        });
        group.bench_with_input(BenchmarkId::new("hpt", idx), &idx, |b, idx| {
            b.iter(|| black_box(_Tensor::<f32>::randn(shapes[*idx]).unwrap()));
        });
    }

    group.finish();
}

criterion_group!(benches, random_f32_benchmark);
