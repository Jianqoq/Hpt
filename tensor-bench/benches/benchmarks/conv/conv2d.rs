use std::time::Duration;
use rayon::iter::{ IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator };
use tensor_dyn::ops::cpu::conv_config::{ Conv2dConfig, KernelParamAlgo };
use criterion::{ black_box, criterion_group, BenchmarkId, Criterion };
use tch::{ Tensor, Kind, Device };
use tensor_dyn::{ tensor_base::_Tensor, Random };
use tensor_dyn::TensorInfo;
use tensor_dyn::TensorLike;

#[allow(unused)]
fn assert_eq_i64(a: &Tensor, b: &_Tensor<i64>) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const i64, b.size()) };
    let b_raw = b.as_raw();
    a_raw
        .par_iter()
        .zip(b_raw.par_iter())
        .for_each(|(a, b)| {
            assert_eq!(a, b);
        });
}

fn conv2d_benchmark(c: &mut Criterion) {
    tch::set_num_threads(num_cpus::get_physical() as i32);
    let shapes = [
        (
            [1 /* batch */, 64 /* in channel */, 256 /* height */, 256 /* width */],
            [64 /* in channel */, 128 /* out channel */],
        ),
        ([1, 16, 256, 256], [16, 16]),
        ([1, 128, 256, 256], [128, 256]),
        ([1, 256, 256, 256], [256, 512]),
        ([1, 64, 56, 56], [64, 128]),
        ([1, 128, 28, 28], [128, 256]),
        ([1, 256, 14, 14], [256, 512]),
        ([1, 512, 7, 7], [512, 512]),
    ];
    let mut group = c.benchmark_group(concat!("conv2d", " Benchmarks"));
    group.warm_up_time(Duration::new(1, 0)).measurement_time(Duration::new(3, 0)).sample_size(10);
    for idx in 0..shapes.len() {
        let (inp_shape, [in_channels, out_channels]) = shapes[idx];
        let a = black_box(Tensor::randn(inp_shape, (Kind::Float, Device::Cpu)));
        let a_kernel = black_box(
            Tensor::randn([out_channels, in_channels, 3, 3], (Kind::Float, Device::Cpu))
        );
        let a2 = black_box(
            _Tensor::<f32>::randn([inp_shape[0], inp_shape[2], inp_shape[3], inp_shape[1]]).unwrap()
        );
        let a2_kernel = black_box(
            _Tensor::<f32>::randn([3, 3, in_channels, out_channels]).unwrap()
        );
        let config = Conv2dConfig::<f32>::new(
            out_channels,
            in_channels,
            [3, 3],
            KernelParamAlgo::Greedy
        );
        group.bench_with_input(
            BenchmarkId::new("torch", format!("tch {}", idx)),
            &shapes[idx],
            |b, _| {
                b.iter(|| { a.conv2d(&a_kernel, None::<Tensor>, [1, 1], [0, 0], [1, 1], 1) });
            }
        );
        group.bench_with_input(
            BenchmarkId::new("hpt", format!("hpt {}", idx)),
            &shapes[idx],
            |b, _| {
                b.iter(|| {
                    a2.conv2d(
                        &a2_kernel,
                        [1, 1],
                        [
                            (0, 0),
                            (0, 0),
                        ],
                        [1, 1],
                        Some(&config)
                    ).unwrap()
                });
            }
        );
    }
    group.finish();
}

criterion_group!(conv2d_benches, conv2d_benchmark);
