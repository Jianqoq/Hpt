use std::time::Duration;
use tensor_dyn::ops::cpu::convolutions::conv_config::{ Conv2dConfig, KernelParamAlgo };
use tensor_dyn::TensorCreator;
use criterion::{ black_box, criterion_group, BenchmarkId, Criterion };
use tch::{ Tensor, Kind, Device };
use tensor_dyn::{ tensor_base::_Tensor, Random };
use tensor_dyn::ShapeManipulate;
use tensor_dyn::TensorInfo;

fn assert_eq_i64(a: &Tensor, b: &_Tensor<i64>) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const i64, b.size()) };
    let b_raw = b.as_raw();
    assert_eq!(a_raw, b_raw);
}

fn conv2d_benchmark(c: &mut Criterion) {
    tensor_dyn::set_num_threads(num_cpus::get_physical());
    tch::set_num_threads(num_cpus::get_physical() as i32);
    let shapes = [
        // ([1, 64, 256, 256], [64, 128]),
        // ([1, 128, 256, 256], [128, 256]),
        ([1, 256, 256, 256], [256, 512]),
        // ([1, 64, 56, 56], [64, 128]),
        // ([1, 128, 28, 28], [128, 256]),
        // ([1, 256, 14, 14], [256, 512]),
        // ([1, 512, 7, 7], [512, 512]),
    ];
    let mut group = c.benchmark_group(concat!("sum", " Benchmarks"));
    group.warm_up_time(Duration::new(1, 0)).measurement_time(Duration::new(3, 0)).sample_size(10);
    for idx in 0..shapes.len() {
        let (inp_shape, [in_channels, out_channels]) = shapes[idx];
        let a = black_box(Tensor::randn(inp_shape, (Kind::Float, Device::Cpu)).to_mkldnn());
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
        println!("config: {:?}", config);
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
                    a2.conv2d_ex(
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
        let a = black_box(
            Tensor::arange(inp_shape.iter().product::<i64>(), (Kind::Int64, Device::Cpu)).reshape(
                inp_shape
            )
        );
        let a_kernel = black_box(
            Tensor::arange(out_channels * in_channels * 3 * 3, (Kind::Int64, Device::Cpu)).reshape([
                out_channels,
                in_channels,
                3,
                3,
            ])
        );
        // let a2 = black_box(
        //     _Tensor::<i64>
        //         ::arange(0, inp_shape.iter().product::<i64>())
        //         .unwrap()
        //         .reshape(inp_shape)
        //         .unwrap()
        //         .permute([0, 2, 3, 1])
        //         .unwrap()
        //         .contiguous()
        //         .unwrap()
        // );
        // let a2_kernel = black_box(
        //     _Tensor::<i64>
        //         ::arange(0, out_channels * in_channels * 3 * 3)
        //         .unwrap()
        //         .reshape([out_channels, in_channels, 3, 3])
        //         .unwrap()
        //         .permute([2, 3, 1, 0])
        //         .unwrap()
        //         .contiguous()
        //         .unwrap()
        // );
        // let res = a.conv2d(&a_kernel, None::<Tensor>, [1, 1], [0, 0], [1, 1], 1);
        // let res2 = a2
        //     .conv2d_ex(
        //         &a2_kernel,
        //         [1, 1],
        //         [
        //             (0, 0),
        //             (0, 0),
        //         ],
        //         [1, 1],
        //         None
        //     )
        //     .unwrap()
        //     .permute([0, 3, 1, 2])
        //     .unwrap()
        //     .contiguous()
        //     .unwrap();
        // println!("{}", res);
        // println!("{}", res2);
        // assert_eq_i64(&res, &res2);
    }
    group.finish();
}

criterion_group!(conv2d_benches, conv2d_benchmark);
