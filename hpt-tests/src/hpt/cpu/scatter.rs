#![allow(unused_imports)]
use hpt::common::cpu::TensorLike;
use hpt::common::TensorInfo;
use hpt::ops::*;
use hpt::Tensor;
use hpt_common::slice;
use hpt_macros::select;
use rand::Rng;

#[allow(unused)]
fn assert_eq(b: &Tensor<f32>, a: &tch::Tensor) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const f32, b.size()) };
    let b_raw = b.as_raw();

    for i in 0..b.size() {
        if a_raw[i] != b_raw[i] {
            panic!(
                "{} != {}, bytes: {:?}, {:?}",
                a_raw[i],
                b_raw[i],
                a_raw[i].to_ne_bytes(),
                b_raw[i].to_ne_bytes()
            );
        }
    }
}

#[test]
fn test_basic_scatter() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    for _ in 0..100 {
        let ndim = rng.random_range(1..=3);
        let shape = (0..ndim).map(|_| 10).collect::<Vec<_>>();

        let tch_src = tch::Tensor::randn(&shape, (tch::Kind::Float, tch::Device::Cpu));
        let mut src = Tensor::<f32>::empty(&shape)?;
        src.as_raw_mut().copy_from_slice(unsafe {
            std::slice::from_raw_parts(tch_src.data_ptr() as *mut f32, tch_src.numel())
        });

        for dim in 0..ndim {
            let indices_shape = shape.clone();
            let tch_dst = tch::Tensor::zeros(&indices_shape, (tch::Kind::Float, tch::Device::Cpu));
            let dst = Tensor::<f32>::zeros(&indices_shape)?;
            let mut indices = Tensor::<i64>::empty(&indices_shape)?;
            let tch_indices = tch::Tensor::randint_low(
                0,
                10,
                &indices_shape,
                (tch::Kind::Int64, tch::Device::Cpu),
            )
            .reshape(&indices_shape);
            indices.as_raw_mut().copy_from_slice(unsafe {
                std::slice::from_raw_parts(tch_indices.data_ptr() as *mut i64, tch_indices.numel())
            });

            let result = dst.scatter(&indices, dim as i64, &src)?;
            let tch_result = tch_dst.scatter(dim as i64, &tch_indices, &tch_src);

            assert_eq(&result, &tch_result);
        }
    }
    Ok(())
}
