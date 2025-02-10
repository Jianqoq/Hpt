use candle_core::Tensor as CandleTensor;
use criterion::{black_box, criterion_group, BenchmarkId, Criterion};
use hpt_core::ShapeManipulate;
use hpt_core::TensorInfo;
use hpt_core::TensorLike;
use hpt_core::{NormalPooling, Random, Tensor};
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

fn maxpool_benchmark(c: &mut Criterion) {
    tch::set_num_threads(num_cpus::get_physical() as i32);
    let ic_sets = [
        64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608,
        640, 672, 704, 736, 768,
    ];
    let kh_sets = [4];
    let kw_sets = [4];
    let h_sets = [256];
    let w_sets = [256];
    let mut group = c.benchmark_group(concat!("maxpool", " Benchmarks"));
    group
        .warm_up_time(Duration::new(1, 0))
        .measurement_time(Duration::new(3, 0))
        .sample_size(10);
    let mut idx = 0;
    for ic in ic_sets {
        for kh in kh_sets {
            for kw in kw_sets {
                for h in h_sets {
                    for w in w_sets {
                        let a = black_box(
                            TchTensor::randn([1, ic, h, w], (Kind::Float, Device::Cpu)).to_mkldnn(),
                        );
                        let a2 = black_box(
                            Tensor::<f32>::randn([1, ic, h, w])
                                .unwrap()
                                .permute([0, 2, 3, 1])
                                .unwrap()
                                .contiguous()
                                .unwrap(),
                        );
                        let a3 = black_box(
                            CandleTensor::randn(
                                0f32,
                                1f32,
                                &[1 as usize, ic as usize, h as usize, w as usize],
                                &candle_core::Device::Cpu,
                            )
                            .unwrap(),
                        );
                        group.bench_with_input(
                            BenchmarkId::new("torch", format!("tch {}", idx)),
                            &[ic, kh, kw, h, w],
                            |b, _| {
                                b.iter(|| {
                                    a.mkldnn_max_pool2d(&[kh, kw], [1, 1], [0, 0], [1, 1], false)
                                });
                            },
                        );
                        group.bench_with_input(
                            BenchmarkId::new("hpt", format!("hpt {}", idx)),
                            &[ic, kh, kw, h, w],
                            |b, _| {
                                b.iter(|| a2.maxpool2d([kh, kw], [1, 1], [(0, 0), (0, 0)], [1, 1]));
                            },
                        );
                        group.bench_with_input(
                            BenchmarkId::new("candle", format!("candle {}", idx)),
                            &[ic, kh, kw, h, w],
                            |b, _| {
                                b.iter(|| a3.max_pool2d((kh as usize, kw as usize)));
                            },
                        );
                        idx += 1;
                    }
                }
            }
        }
    }
    group.finish();
}

criterion_group!(maxpool_benches, maxpool_benchmark);
