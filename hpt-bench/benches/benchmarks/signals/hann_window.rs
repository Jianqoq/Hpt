use std::time::Duration;
use criterion::{ black_box, criterion_group, criterion_main, BenchmarkId, Criterion };
use tch::{ Tensor, Kind, Device };
use hpt::tensor_base::_Tensor;
use hpt::TensorInfo;

fn assert_eq(a: &Tensor, b: &_Tensor<f64>) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const f64, b.size()) };
    let b_raw = b.as_raw();
    for i in 0..b.size() {
        if a_raw[i] < b_raw[i] - 1e-6 || a_raw[i] > b_raw[i] + 1e-6 {
            panic!("{} != {}", a_raw[i], b_raw[i]);
        }
    }
}

fn hann_window_benchmark(c: &mut Criterion) {
    hpt::set_num_threads(num_cpus::get_physical());
    tch::set_num_threads(num_cpus::get_physical() as i32);
    let lens = [
        2048, 10240, 20480, 40960, 81920, 163840, 327680, 655360, 1310720, 2621440, 5242880,
    ];

    let mut group = c.benchmark_group("hann_window Benchmarks");
    group.warm_up_time(Duration::new(1, 0)).measurement_time(Duration::new(3, 0)).sample_size(10);
    for idx in 0..lens.len() {
        let lens = lens[idx];
        group.bench_with_input(BenchmarkId::new("torch", format!("tch {}", idx)), &lens, |b, _| {
            b.iter(|| { Tensor::hann_window_periodic(lens, false, (Kind::Float, Device::Cpu)) });
        });
        group.bench_with_input(BenchmarkId::new("hpt", format!("hpt {}", idx)), &lens, |b, _| {
            b.iter(|| { _Tensor::<f32>::hann_window(lens, false).unwrap() });
        });
        let a = black_box(Tensor::hann_window_periodic(lens, false, (Kind::Double, Device::Cpu)));
        let a2 = black_box(_Tensor::<f64>::hann_window(lens, false).unwrap());
        assert_eq(&a, &a2);
    }

    group.finish();
}

criterion_group!(benches, hann_window_benchmark);
criterion_main!(benches);
