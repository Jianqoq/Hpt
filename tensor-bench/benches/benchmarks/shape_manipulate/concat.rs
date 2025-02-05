#![cfg(feature = "cat")]
use criterion::{black_box, criterion_group, BenchmarkId, Criterion};
use std::time::Duration;
use tch::{Device, Kind, Tensor as TchTensor};
use tensor_dyn::TensorLike;
use tensor_dyn::{Random, Tensor};
use tensor_dyn::{ShapeManipulate, TensorInfo};

fn concat_benchmark(c: &mut Criterion) {
    tensor_dyn::set_num_threads(num_cpus::get_physical());
    tch::set_num_threads(num_cpus::get_physical() as i32);
    let axes = [0, 1, 2];

    let mut group = c.benchmark_group("concat Benchmarks");
    group
        .warm_up_time(Duration::new(1, 0))
        .measurement_time(Duration::new(3, 0))
        .sample_size(10);
    for (i, axis) in axes.iter().enumerate() {
        let a = black_box(TchTensor::randn(
            [128, 128, 128],
            (Kind::Float, Device::Cpu),
        ));
        let a1 = black_box(TchTensor::randn(
            [128, 128, 128],
            (Kind::Float, Device::Cpu),
        ));
        let a2 = black_box(TchTensor::randn(
            [128, 128, 128],
            (Kind::Float, Device::Cpu),
        ));
        group.bench_function(BenchmarkId::new("torch", format!("tch {}", i)), |b| {
            b.iter(|| TchTensor::cat(&[&a, &a1, &a2], *axis));
        });
        let mut a = black_box(Tensor::<f32>::randn([128, 128, 128]).unwrap());
        let mut a1 = black_box(Tensor::<f32>::randn([128, 128, 128]).unwrap());
        let mut a2 = black_box(Tensor::<f32>::randn([128, 128, 128]).unwrap());
        group.bench_function(BenchmarkId::new("hpt", format!("hpt {}", i)), |b| {
            b.iter(|| {
                Tensor::<f32>::concat(
                    vec![a.clone(), a1.clone(), a2.clone()],
                    *axis as usize,
                    false,
                )
            });
        });
        let tch_a = black_box(TchTensor::randn(
            [128, 128, 128],
            (Kind::Float, Device::Cpu),
        ));
        let tch_a1 = black_box(TchTensor::randn(
            [128, 128, 128],
            (Kind::Float, Device::Cpu),
        ));
        let tch_a2 = black_box(TchTensor::randn(
            [128, 128, 128],
            (Kind::Float, Device::Cpu),
        ));
        let tch_concat = TchTensor::cat(&[&tch_a, &tch_a1, &tch_a2], *axis);
        let a_size = a.size();
        let a1_size = a1.size();
        let a2_size = a2.size();
        a.as_raw_mut().copy_from_slice(unsafe {
            std::slice::from_raw_parts(tch_a.data_ptr() as *const f32, a_size)
        });
        a1.as_raw_mut().copy_from_slice(unsafe {
            std::slice::from_raw_parts(tch_a1.data_ptr() as *const f32, a1_size)
        });
        a2.as_raw_mut().copy_from_slice(unsafe {
            std::slice::from_raw_parts(tch_a2.data_ptr() as *const f32, a2_size)
        });
        let concat = Tensor::<f32>::concat(
            vec![a.clone(), a1.clone(), a2.clone()],
            *axis as usize,
            false,
        )
        .unwrap();
        let raw = concat.as_raw();
        let tch_raw = unsafe {
            std::slice::from_raw_parts(tch_concat.data_ptr() as *const f32, concat.size())
        };
        if raw != tch_raw {
            panic!("{} != {}", concat, tch_concat);
        }
    }

    group.finish();
}

criterion_group!(cat_benches, concat_benchmark);
