use std::time::Duration;
use tensor_dyn::TensorCreator;
use tensor_dyn::ShapeManipulate;
use tensor_dyn::TensorInfo;
use criterion::{ black_box, criterion_group, criterion_main, BenchmarkId, Criterion };
use tch::{ Tensor, Kind, Device };
use tensor_dyn::{ tensor_base::_Tensor, Random };
use half::f16;

fn assert_eq(a: &Tensor, b: &_Tensor<i64>) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const i64, b.size()) };
    let b_raw = b.as_raw();
    for i in 0..b.size() {
        assert_eq!(a_raw[i], b_raw[i]);
    }
}

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

    let mut group = c.benchmark_group("add f32 Benchmarks");
    group.warm_up_time(Duration::new(1, 0)).measurement_time(Duration::new(3, 0)).sample_size(10);
    for idx in 0..shapes.len() {
        let shape = shapes[idx];
        let a = black_box(Tensor::randn(shape, (Kind::Float, Device::Cpu)));
        let c = black_box(Tensor::randn(shape, (Kind::Float, Device::Cpu)));
        let a2 = black_box(_Tensor::<f32>::randn(shape).unwrap());
        let c2 = black_box(_Tensor::<f32>::randn(shape).unwrap());
        group.bench_with_input(BenchmarkId::new("torch", idx), &shape, |b, _| {
            b.iter(|| { &a + &c });
        });
        group.bench_with_input(BenchmarkId::new("hpt", idx), &shape, |b, _| {
            b.iter(|| { &a2 + &c2 });
        });
        let a = black_box(
            Tensor::arange(shape.iter().product::<i64>(), (Kind::Int64, Device::Cpu)).reshape(
                shape
            )
        );
        let a2 = black_box(
            _Tensor::<i64>
                ::arange(0, shape.iter().product::<i64>())
                .unwrap()
                .reshape(shape)
                .unwrap()
        );
        assert_eq(&a, &a2);
    }

    group.finish();
}

criterion_group!(benches, add_f16_benchmark);