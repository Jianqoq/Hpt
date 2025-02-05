use std::time::Duration;
use hpt_core::TensorCreator;
use criterion::{ black_box, criterion_group, criterion_main, BenchmarkId, Criterion };
use tch::{ Tensor, Kind, Device };
use hpt_core::{ tensor_base::_Tensor, Random };
use hpt_core::ShapeManipulate;
use hpt_core::TensorInfo;

fn assert_eq(a: &Tensor, b: &_Tensor<i64>) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const i64, b.size()) };
    let b_raw = b.as_raw();
    for i in 0..b.size() {
        assert_eq!(a_raw[i], b_raw[i]);
    }
}

fn topk_benchmark(c: &mut Criterion) {
    hpt_core::set_num_threads(num_cpus::get_physical());
    tch::set_num_threads(num_cpus::get_physical() as i32);
    let shapes = [
        [15, 2048, 2048],
    ];
    let axes = [0, 1, 2];

    let mut group = c.benchmark_group("topk Benchmarks");
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
                    b.iter(|| { a.topk(10, *axis, false, true) });
                }
            );
            group.bench_with_input(
                BenchmarkId::new("hpt", format!("hpt {}.{}", idx, i)),
                &shapes[idx],
                |b, _| {
                    b.iter(|| { a2.topk(10, *axis, false, true).unwrap() });
                }
            );
            let (a, a_idx) = black_box(
                Tensor::arange(shape.iter().product::<i64>(), (Kind::Int64, Device::Cpu)).reshape(
                    shape
                )
            ).topk(10, 1, false, true);
            let (a2_idx, a2) = black_box(
                _Tensor::<i64>
                    ::arange(0, shape.iter().product::<i64>())
                    .unwrap()
                    .reshape(shape)
                    .unwrap()
            )
                .topk(10, 1, false, true)
                .unwrap();
            println!("a: {}", a);
            println!("a2: {}", a2);
            assert_eq(&a, &a2);
            assert_eq(&a_idx, &a2_idx);
        }
    }

    group.finish();
}

criterion_group!(benches, topk_benchmark);
criterion_main!(benches);
