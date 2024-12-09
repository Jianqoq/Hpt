use std::time::Duration;
use criterion::{ black_box, criterion_group, criterion_main, BenchmarkId, Criterion };
use tch::{ Tensor as TchTensor, Kind, Device };
use tensor_dyn::{ Tensor, Random, TensorCreator };
use tensor_dyn::TensorInfo;
use tensor_dyn::TensorLike;
use crate::benchmarks::unary::float_cmp::assert_eq;

pub fn softmax_benchmark(c: &mut Criterion) {
    tensor_dyn::set_num_threads(num_cpus::get_physical());
    tch::set_num_threads(num_cpus::get_physical() as i32);
    let shapes = [[512, 512, 512, 512]];
    let axes = [0, 1, 2, 3];

    let mut group = c.benchmark_group("softmax Benchmarks");
    group.warm_up_time(Duration::new(1, 0)).measurement_time(Duration::new(3, 0)).sample_size(10);
    for idx in 0..shapes.len() {
        let shape = shapes[idx];
        let a = black_box(TchTensor::randn(shape, (Kind::Float, Device::Cpu)));
        let a2 = black_box(Tensor::<f32>::randn(shape).unwrap());
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
            let a = black_box(TchTensor::randn(shape, (Kind::Double, Device::Cpu)));
            let mut a2 = black_box(Tensor::<f64>::empty(shape)).unwrap();
            let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const f64, a2.size()) };
            let a2_raw_mut = a2.as_raw_mut();
            a2_raw_mut.copy_from_slice(a_raw);
            let a_sin = a.softmax(*axis, Kind::Double);
            let a2_sin = a2.softmax(*axis).unwrap();
            assert_eq(&a_sin, &a2_sin);
        }
    }

    group.finish();
}

criterion_group!(softmax_benches, softmax_benchmark);
criterion_main!(softmax_benches);
