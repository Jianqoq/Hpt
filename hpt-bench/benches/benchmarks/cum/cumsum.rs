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
    assert_eq!(a_raw, b_raw);
}

fn cumsum_benchmark(c: &mut Criterion) {
    hpt_core::set_num_threads(num_cpus::get_physical());
    tch::set_num_threads(num_cpus::get_physical() as i32);
    let shapes = [
        [8096, 2048, 8],
        [2048, 512, 256],
        [512, 128, 512],
        [128, 32, 1024],
        [32, 8, 2048],
        [8, 2, 4096],
        [2, 1, 8192],
    ];
    let axes = [0, 1, 2];

    let mut group = c.benchmark_group("cumsum Benchmarks");
    group.warm_up_time(Duration::new(1, 0)).measurement_time(Duration::new(3, 0)).sample_size(10);
    for idx in 0..shapes.len() {
        let shape = shapes[idx];
        let a = black_box(Tensor::randn(shape, (Kind::Float, Device::Cpu)));
        let a2 = black_box(_Tensor::<f32>::randn(shape).unwrap());
        for (i, axis) in axes.iter().enumerate() {
            // group.bench_with_input(
            //     BenchmarkId::new("torch", format!("tch {}.{}", idx, i)),
            //     &shapes[idx],
            //     |b, _| {
            //         b.iter(|| { a.cumsum(*axis, Kind::Float) });
            //     }
            // );
            // group.bench_with_input(
            //     BenchmarkId::new("hpt", format!("hpt {}.{}", idx, i)),
            //     &shapes[idx],
            //     |b, _| {
            //         b.iter(|| { a2.cumsum(Some(*axis)) });
            //     }
            // );
            let a = black_box(
                Tensor::arange(shape.iter().product::<i64>(), (Kind::Int64, Device::Cpu)).reshape(
                    shape
                )
            ).cumsum(*axis, Kind::Int64);
            let a2 = black_box(
                _Tensor::<i64>
                    ::arange(0, shape.iter().product::<i64>())
                    .unwrap()
                    .reshape(shape)
                    .unwrap()
            )
                .cumsum(Some(*axis))
                .unwrap();
            println!("{}", a);
            println!("{}", a2);
            // assert_eq(&a, &a2);
        }
    }

    group.finish();
}

criterion_group!(benches, cumsum_benchmark);
criterion_main!(benches);
