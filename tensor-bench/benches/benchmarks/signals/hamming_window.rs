use std::time::Duration;
use criterion::{ black_box, criterion_group, criterion_main, BenchmarkId, Criterion };
use tch::{ Tensor, Kind, Device };
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::TensorInfo;
use tensor_dyn::TensorLike;

fn assert_eq(a: &Tensor, b: &_Tensor<f64>) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const f64, b.size()) };
    let b_raw = b.as_raw();
    for i in 0..b.size() {
        if a_raw[i] < b_raw[i] - 1e-6 || a_raw[i] > b_raw[i] + 1e-6 {
            panic!("{} != {}", a_raw[i], b_raw[i]);
        }
    }
}

fn hamming_window_benchmark(c: &mut Criterion) {
    tensor_dyn::set_num_threads(num_cpus::get_physical());
    tch::set_num_threads(num_cpus::get_physical() as i32);
    let lens = [
        2048, 10240, 20480, 40960, 81920, 163840, 327680, 655360, 1310720, 2621440, 5242880,
    ];

    let mut group = c.benchmark_group("hamming_window Benchmarks");
    group.warm_up_time(Duration::new(1, 0)).measurement_time(Duration::new(3, 0)).sample_size(10);
    for idx in 0..lens.len() {
        let lens = lens[idx];
        group.bench_with_input(BenchmarkId::new("torch", format!("tch {}", idx)), &lens, |b, _| {
            b.iter(|| { Tensor::hamming_window_periodic(lens, false, (Kind::Float, Device::Cpu)) });
        });
        group.bench_with_input(BenchmarkId::new("hpt", format!("hpt {}", idx)), &lens, |b, _| {
            b.iter(|| { tensor_dyn::tensor::Tensor::<f32>::hamming_window(lens, false).unwrap() });
        });
        let a = black_box(Tensor::hamming_window_periodic(lens, false, (Kind::Double, Device::Cpu)));
        let a2 = black_box(tensor_dyn::tensor::Tensor::<f64>::hamming_window(lens, false).unwrap());
        assert_eq(&a, &a2);
    }

    group.finish();
}

criterion_group!(benches, hamming_window_benchmark);
criterion_main!(benches);
