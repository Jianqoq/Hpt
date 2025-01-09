#![allow(unused)]
use rand::Rng;
use tch;
use tensor_dyn::set_num_threads;
use tensor_dyn::ShapeManipulate;
use tensor_dyn::TensorLike;
use tensor_dyn::{ CommonBounds, TensorInfo };
use tensor_dyn::{ Tensor, TensorCreator };
use tensor_types::convertion::{ Convertor, FromScalar };
use tensor_types::into_scalar::IntoScalar;
use tensor_types::type_promote::NormalOut;

fn common_input<T>([batch, in_channel, height, width]: [i64; 4])
    -> anyhow::Result<(Tensor<T>, tch::Tensor)>
    where
        T: Convertor + FromScalar<i64> + NormalOut<T, Output = T> + CommonBounds,
        usize: IntoScalar<T>,
        i64: IntoScalar<T>
{
    let a = Tensor::<T>
        ::arange(0, batch * in_channel * height * width)?
        .reshape([batch, in_channel, height, width])?
        .permute([0, 2, 3, 1])?
        .contiguous()?;
    let tch_a = tch::Tensor
        ::arange(batch * in_channel * height * width, (tch::Kind::Double, tch::Device::Cpu))
        .reshape(&[batch, in_channel, height, width]);
    Ok((a, tch_a))
}

#[track_caller]
fn assert_eq(a: &Tensor<f64>, b: &tch::Tensor) -> anyhow::Result<()> {
    let res = a.adaptive_avgpool2d([2, 2])?.permute([0, 3, 1, 2])?.contiguous()?;
    let res2 = b.adaptive_avg_pool2d([2, 2]);
    let res_slice = res.as_raw();
    let res2 = unsafe { std::slice::from_raw_parts(res2.data_ptr() as *const f64, res.size()) };
    res_slice
        .iter()
        .zip(res2.iter())
        .for_each(|(a, b)| {
            let abs_diff = (*a - *b).abs();
            let rel_diff = if *a == 0.0 && *b == 0.0 {
                0.0
            } else {
                abs_diff / (a.abs() + b.abs() + f64::EPSILON)
            };

            if rel_diff > 0.05 {
                panic!("{} != {} (relative_diff: {})", *a, *b, rel_diff);
            }
        });
    Ok(())
}

#[test]
fn test_case0() -> anyhow::Result<()> {
    let mut rng = rand::thread_rng();

    for _ in 0..1000 {
        let batch = rng.gen_range(1..=4);
        let channel = rng.gen_range(1..=32);
        let height = rng.gen_range(8..=64);
        let width = rng.gen_range(8..=64);
        let (a, tch_a) = common_input([batch, channel, height, width])?;
        assert_eq(&a, &tch_a)?;
    }

    Ok(())
}