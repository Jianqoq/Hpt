#![cfg(feature = "cat")]
use std::time::Duration;
use criterion::{ black_box, criterion_group, BenchmarkId, Criterion };
use tch::{ Tensor, Kind, Device };
use tensor_dyn::{ tensor_base::_Tensor, Random };
use tensor_dyn::TensorInfo;

fn concat_benchmark(c: &mut Criterion) {
    tensor_dyn::set_num_threads(num_cpus::get_physical());
    tch::set_num_threads(num_cpus::get_physical() as i32);
    let axes = [0, 1, 2];

    let mut group = c.benchmark_group("concat Benchmarks");
    group.warm_up_time(Duration::new(1, 0)).measurement_time(Duration::new(3, 0)).sample_size(10);
    for (i, axis) in axes.iter().enumerate() {
        let a = black_box(Tensor::randn([128, 128, 128], (Kind::Float, Device::Cpu)));
        let a1 = black_box(Tensor::randn([128, 128, 128], (Kind::Float, Device::Cpu)));
        let a2 = black_box(Tensor::randn([128, 128, 128], (Kind::Float, Device::Cpu)));
        group.bench_function(BenchmarkId::new("torch", format!("tch {}", i)), |b| {
            b.iter(|| { Tensor::cat(&[&a, &a1, &a2], *axis) });
        });
        let a = black_box(_Tensor::<f32>::randn([128, 128, 128]).unwrap());
        let a1 = black_box(_Tensor::<f32>::randn([128, 128, 128]).unwrap());
        let a2 = black_box(_Tensor::<f32>::randn([128, 128, 128]).unwrap());
        group.bench_function(BenchmarkId::new("hpt", format!("hpt {}", i)), |b| {
            b.iter(|| { _Tensor::<f32>::concat(vec![&a, &a1, &a2], *axis as usize, false) });
        });
        let tch_a = black_box(Tensor::randn([128, 128, 128], (Kind::Float, Device::Cpu)));
        let tch_a1 = black_box(Tensor::randn([128, 128, 128], (Kind::Float, Device::Cpu)));
        let tch_a2 = black_box(Tensor::randn([128, 128, 128], (Kind::Float, Device::Cpu)));
        let tch_concat = Tensor::cat(&[&tch_a, &tch_a1, &tch_a2], *axis);
        a.as_raw_mut().copy_from_slice(
            unsafe { std::slice::from_raw_parts(tch_a.data_ptr() as *const f32, a.size()) }
        );
        a1.as_raw_mut().copy_from_slice(
            unsafe { std::slice::from_raw_parts(tch_a1.data_ptr() as *const f32, a1.size()) }
        );
        a2.as_raw_mut().copy_from_slice(
            unsafe { std::slice::from_raw_parts(tch_a2.data_ptr() as *const f32, a2.size()) }
        );
        let concat = _Tensor::<f32>::concat(vec![&a, &a1, &a2], *axis as usize, false).unwrap();
        let raw = concat.as_raw();
        let tch_raw = unsafe { std::slice::from_raw_parts(tch_concat.data_ptr() as *const f32, concat.size()) };
        if raw != tch_raw {
            panic!("{} != {}", concat, tch_concat);
        }
    }

    group.finish();
}

criterion_group!(cat_benches, concat_benchmark);
