use std::time::Duration;
use tensor_dyn::TensorCreator;
use criterion::{ black_box, criterion_group, criterion_main, BenchmarkId, Criterion };
use tch::{ Tensor, Kind, Device };
use tensor_dyn::{ tensor_base::_Tensor, Random };
use tensor_dyn::ShapeManipulate;
use tensor_dyn::TensorInfo;

fn assert_eq(a: &Tensor, b: &_Tensor<i64>) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const i64, b.size()) };
    let b_raw = b.as_raw();
    for i in 0..b.size() {
        assert_eq!(a_raw[i], b_raw[i]);
    }
}

fn softmax_benchmark(c: &mut Criterion) {
    tensor_dyn::set_num_threads(num_cpus::get_physical());
    tch::set_num_threads(num_cpus::get_physical() as i32);
    let shapes = [
        [15, 2048, 2048],
        [1024, 1024, 1024],
        [2048, 2048, 2048],
        [4096, 4096, 4096],
        [8192, 8192, 8192],
    ];
    let axes = [0, 1, 2];

    let mut group = c.benchmark_group("softmax Benchmarks");
    group.warm_up_time(Duration::new(1, 0)).measurement_time(Duration::new(3, 0)).sample_size(10);
    for idx in 0..shapes.len() {
        let shape = shapes[idx];
        let a = black_box(Tensor::randn(shape, (Kind::Float, Device::Cpu)));
        let a2 = black_box(_Tensor::<f32>::randn(shape).unwrap());
        for (i, axis) in axes.iter().enumerate() {
            group.bench_with_input(
                BenchmarkId::new("torch", format!("tch {}.{}", idx, i)),
                &shapes[idx],
                |b, _| {
                    b.iter(|| { a.softmax(*axis, Kind::Float) });
                }
            );
            group.bench_with_input(
                BenchmarkId::new("hpt", format!("hpt {}.{}", idx, i)),
                &shapes[idx],
                |b, _| {
                    b.iter(|| { a2.softmax(*axis).unwrap() });
                }
            );
        }
    }

    group.finish();
}

criterion_group!(benches, softmax_benchmark);
criterion_main!(benches);
