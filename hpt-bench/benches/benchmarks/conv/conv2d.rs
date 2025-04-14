// use candle_core::Tensor as CandleTensor;
use criterion::{black_box, criterion_group, BenchmarkId, Criterion};
use half::f16;
use hpt::common::cpu::TensorLike;
use hpt::common::TensorInfo;
use hpt::ops::*;
use hpt::Tensor;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::time::Duration;
use tch::{Device, Kind, Tensor as TchTensor};

#[allow(unused)]
fn assert_eq_i64(a: &TchTensor, b: &Tensor<i64>) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const i64, b.size()) };
    let b_raw = b.as_raw();
    a_raw.par_iter().zip(b_raw.par_iter()).for_each(|(a, b)| {
        assert_eq!(a, b);
    });
}

fn conv2d_benchmark(c: &mut Criterion) {
    tch::set_num_threads(num_cpus::get_physical() as i32);
    hpt::utils::set_num_threads(num_cpus::get_physical());
    let shapes = [
        // (
        //     [
        //         1,   /* batch */
        //         64,  /* in channel */
        //         256, /* height */
        //         256, /* width */
        //     ],
        //     [64 /* in channel */, 128 /* out channel */],
        // ),
        ([1, 128, 256, 256], [128, 64]),
        ([1, 128, 256, 256], [128, 96]),
        ([1, 128, 256, 256], [128, 128]),
        ([1, 128, 256, 256], [128, 160]),
        ([1, 128, 256, 256], [128, 192]),
        ([1, 128, 256, 256], [128, 224]),
        ([1, 128, 256, 256], [128, 256]),
        ([1, 128, 256, 256], [128, 288]),
        ([1, 128, 256, 256], [128, 320]),
        ([1, 128, 256, 256], [128, 352]),
        ([1, 128, 256, 256], [128, 384]),
        ([1, 128, 256, 256], [128, 416]),
        ([1, 128, 256, 256], [128, 448]),
        ([1, 128, 256, 256], [128, 480]),
        ([1, 128, 256, 256], [128, 512]),
        ([1, 128, 256, 256], [128, 544]),
        ([1, 128, 256, 256], [128, 576]),
        ([1, 128, 256, 256], [128, 608]),
        ([1, 128, 256, 256], [128, 640]),
        ([1, 128, 256, 256], [128, 672]),
        ([1, 128, 256, 256], [128, 704]),
        ([1, 128, 256, 256], [128, 736]),
        ([1, 128, 256, 256], [128, 768]),

        // ([1, 64, 256, 256], [64, 128]),
        // ([1, 96, 256, 256], [96, 128]),
        // ([1, 128, 256, 256], [128, 128]),
        // ([1, 160, 256, 256], [160, 128]),
        // ([1, 192, 256, 256], [192, 128]),
        // ([1, 224, 256, 256], [224, 128]),
        // ([1, 256, 256, 256], [256, 128]),
        // ([1, 288, 256, 256], [288, 128]),
        // ([1, 320, 256, 256], [320, 128]),
        // ([1, 352, 256, 256], [352, 128]),
        // ([1, 384, 256, 256], [384, 128]),
        // ([1, 416, 256, 256], [416, 128]),
        // ([1, 448, 256, 256], [448, 128]),
        // ([1, 480, 256, 256], [480, 128]),
        // ([1, 512, 256, 256], [512, 128]),
        // ([1, 544, 256, 256], [544, 128]),
        // ([1, 576, 256, 256], [576, 128]),
        // ([1, 608, 256, 256], [608, 128]),
        // ([1, 640, 256, 256], [640, 128]),
        // ([1, 672, 256, 256], [672, 128]),
        // ([1, 704, 256, 256], [704, 128]),
        // ([1, 736, 256, 256], [736, 128]),
        // ([1, 768, 256, 256], [768, 128]),

        // ([1, 128, 64, 4], [128, 128]),
        // ([1, 128, 64, 8], [128, 128]),
        // ([1, 128, 64, 16], [128, 128]),
        // ([1, 128, 64, 32], [128, 128]),
        // ([1, 128, 64, 64], [128, 128]),
        // ([1, 128, 96, 96], [128, 128]),
        // ([1, 128, 128, 128], [128, 128]),
        // ([1, 128, 160, 160], [128, 128]),
        // ([1, 128, 192, 192], [128, 128]),
        // ([1, 128, 224, 224], [128, 128]),
        // ([1, 128, 256, 256], [128, 128]),
        // ([1, 128, 288, 288], [128, 128]),
        // ([1, 128, 320, 320], [128, 128]),
        // ([1, 128, 352, 352], [128, 128]),
        // ([1, 128, 384, 384], [128, 128]),
        // ([1, 128, 416, 416], [128, 128]),
        // ([1, 128, 448, 448], [128, 128]),
        // ([1, 128, 480, 480], [128, 128]),
        // ([1, 128, 512, 512], [128, 128]),
        // ([1, 128, 544, 544], [128, 128]),
        // ([1, 128, 576, 576], [128, 128]),
        // ([1, 128, 608, 608], [128, 128]),
        // ([1, 128, 640, 640], [128, 128]),
        // ([1, 128, 672, 672], [128, 128]),
        // ([1, 128, 704, 704], [128, 128]),
        // ([1, 128, 736, 736], [128, 128]),
        // ([1, 128, 768, 768], [128, 128]),
    ];
    let mut group = c.benchmark_group(concat!("conv2d", " Benchmarks"));
    group
        .warm_up_time(Duration::new(1, 0))
        .measurement_time(Duration::new(3, 0))
        .sample_size(10);
    for idx in 0..shapes.len() {
        let (inp_shape, [in_channels, out_channels]) = shapes[idx];
        let a = black_box(TchTensor::randn(inp_shape, (Kind::Float, Device::Cpu)));
        let a_kernel = black_box(TchTensor::randn(
            [out_channels, in_channels, 3, 3],
            (Kind::Float, Device::Cpu),
        ));
        let a2 = black_box(
            Tensor::<f32>::randn([inp_shape[0], inp_shape[2], inp_shape[3], inp_shape[1]]).unwrap(),
        );
        let a2_kernel = black_box(Tensor::<f32>::randn([3, 3, in_channels, out_channels]).unwrap());
        // let a3 = black_box(
        //     CandleTensor::randn(
        //         0f32,
        //         1f32,
        //         &[
        //             inp_shape[0] as usize,
        //             inp_shape[1] as usize,
        //             inp_shape[2] as usize,
        //             inp_shape[3] as usize,
        //         ],
        //         &candle_core::Device::Cpu,
        //     )
        //     .unwrap(),
        // );
        // let a3_kernel = black_box(
        //     CandleTensor::randn(
        //         0f32,
        //         1f32,
        //         &[out_channels as usize, in_channels as usize, 3, 3],
        //         &candle_core::Device::Cpu,
        //     )
        //     .unwrap(),
        // );
        // group.bench_with_input(
        //     BenchmarkId::new("torch", format!("tch {}", idx)),
        //     &shapes[idx],
        //     |b, _| {
        //         b.iter(|| a.conv2d(&a_kernel, None::<TchTensor>, [1, 1], [0, 0], [1, 1], 1));
        //     },
        // );
        group.bench_with_input(
            BenchmarkId::new("hpt", format!("hpt {}", idx)),
            &shapes[idx],
            |b, _| {
                b.iter(|| {
                    a2.conv2d(&a2_kernel, None, [1, 1], [(0, 0), (0, 0)], [1, 1], None, None)
                        .unwrap()
                });
            },
        );
        // group.bench_with_input(
        //     BenchmarkId::new("candle", format!("candle {}", idx)),
        //     &shapes[idx],
        //     |b, _| {
        //         b.iter(|| a3.conv2d(&a3_kernel, 0, 1, 1, 1).unwrap());
        //     },
        // );
    }
    group.finish();
}

criterion_group!(conv2d_benches, conv2d_benchmark);
