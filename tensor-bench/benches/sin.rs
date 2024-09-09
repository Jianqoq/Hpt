use std::time::Duration;
use tensor_dyn::TensorCreator;
use criterion::{ black_box, criterion_group, criterion_main, BenchmarkId, Criterion };
use tch::{ Tensor, Kind, Device };
use tensor_dyn::{ tensor_base::_Tensor, Random };
use tensor_dyn::ShapeManipulate;
use tensor_dyn::TensorInfo;

fn assert_eq(a: &Tensor, b: &_Tensor<f64>) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const f64, b.size()) };
    let b_raw = b.as_raw();
    for i in 0..b.size() {
        if a_raw[i] < b_raw[i] - 1e-10 || a_raw[i] > b_raw[i] + 1e-10 {
            println!("{} != {}", a_raw[i], b_raw[i]);
        }
    }
    let abs_error = a_raw.iter().zip(b_raw.iter()).map(|(a, b)| (a - b).abs()).collect::<Vec<f64>>();
    let relative_error = a_raw.iter().zip(b_raw.iter()).map(|(a, b)| ((a - b).abs() / b.abs()).max(f64::EPSILON)).collect::<Vec<f64>>();
    let max_abs_error = abs_error.iter().copied().fold(f64::NAN, f64::max);
    let max_relative_error = relative_error.iter().copied().fold(f64::NAN, f64::max);
    println!("Max Abs Error: {}", max_abs_error);
    println!("Max Relative Error: {}", max_relative_error);
}

fn sin_benchmark(c: &mut Criterion) {
    tensor_dyn::set_num_threads(num_cpus::get_physical());
    tch::set_num_threads(num_cpus::get_physical() as i32);
    let shapes = [
        [1024, 2048, 8],
        [2048, 2048, 8],
        [4096, 2048, 8],
        [8192, 2048, 8],
    ];

    let mut group = c.benchmark_group("sin Benchmarks");
    group.warm_up_time(Duration::new(1, 0)).measurement_time(Duration::new(3, 0)).sample_size(10);
    for idx in 0..shapes.len() {
        let shape = shapes[idx];
        let a = black_box(Tensor::randn(shape, (Kind::Float, Device::Cpu)));
        let a2 = black_box(_Tensor::<f32>::randn(shape).unwrap());
        group.bench_with_input(
            BenchmarkId::new("torch", format!("tch {}", idx)),
            &shapes[idx],
            |b, _| {
                b.iter(|| { a.sin() });
            }
        );
        group.bench_with_input(
            BenchmarkId::new("hpt", format!("hpt {}", idx)),
            &shapes[idx],
            |b, _| {
                b.iter(|| { a2.sin().unwrap() });
            }
        );
        let a = black_box(Tensor::randn(shape, (Kind::Double, Device::Cpu)));
        let a2 = black_box(_Tensor::<f64>::empty(shape)).unwrap();
        let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const f64, a2.size()) };
        let a2_raw_mut = a2.as_raw_mut();
        a2_raw_mut.copy_from_slice(a_raw);
        let a_sin = a.sin();
        let a2_sin = a2.sin().unwrap();
        assert_eq(&a_sin, &a2_sin);
    }

    group.finish();
}

criterion_group!(benches, sin_benchmark);
criterion_main!(benches);
