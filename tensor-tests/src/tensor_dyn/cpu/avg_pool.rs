#![allow(unused)]
use tch;
use tensor_dyn::set_num_threads;
use tensor_dyn::ShapeManipulate;
use tensor_dyn::TensorLike;
use tensor_dyn::{ CommonBounds, TensorInfo };
use tensor_dyn::{ Tensor, TensorCreator };
use tensor_types::convertion::{ Convertor, FromScalar };
use tensor_types::into_scalar::IntoScalar;
use tensor_types::type_promote::NormalOut;

fn common_input<T>([batch, in_channel, kernel_height, kernel_width, height, width]: [
    i64;
    6
])
    -> anyhow::Result<(Tensor<T>, Tensor<T>, tch::Tensor, tch::Tensor)>
    where
        T: Convertor + FromScalar<i64> + NormalOut<T, Output = T> + CommonBounds,
        usize: IntoScalar<T>,
        i64: IntoScalar<T>
{
    let kernel = Tensor::<T>
        ::arange(0, kernel_height * kernel_width)?
        .reshape([kernel_height, kernel_width])?;
    let a = Tensor::<T>
        ::arange(0, batch * in_channel * height * width)?
        .reshape([batch, in_channel, height, width])?
        .permute([0, 2, 3, 1])?
        .contiguous()?;

    let tch_kernel = tch::Tensor
        ::arange(kernel_height * kernel_width, (tch::Kind::Double, tch::Device::Cpu))
        .reshape(&[kernel_height, kernel_width]);
    let tch_a = tch::Tensor
        ::arange(batch * in_channel * height * width, (tch::Kind::Double, tch::Device::Cpu))
        .reshape(&[batch, in_channel, height, width]);
    Ok((kernel, a, tch_kernel, tch_a))
}

#[track_caller]
fn assert_eq(
    a: &Tensor<f64>,
    a_kernel: &Tensor<f64>,
    b: &tch::Tensor,
    b_kernel: &tch::Tensor,
) -> anyhow::Result<()> {
    let res = a
        .avgpool2d(
            &a_kernel.shape(),
            [1, 1],
            [
                (0, 0),
                (0, 0),
            ],
            [1, 1]
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let res2 = b.avg_pool2d(&b_kernel.size(), &[1, 1], &[0, 0], false, true, None);
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

#[track_caller]
fn assert_eq_pad(
    a: &Tensor<f64>,
    a_kernel: &Tensor<f64>,
    b: &tch::Tensor,
    b_kernel: &tch::Tensor
) -> anyhow::Result<()> {
    let res = a
        .avgpool2d(
            &a_kernel.shape(),
            [1, 1],
            [
                (2, 2),
                (2, 2),
            ],
            [1, 1]
        )?
        .permute([0, 3, 1, 2])?
        .contiguous()?;
    let res2 = b.avg_pool2d(&b_kernel.size(), &[1, 1], &[2, 2], false, true, None);
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
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 3, 4, 4, 16, 16])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;

    Ok(())
}

#[test]
fn test_case1() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 3, 4, 4, 20, 20])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 3, 4, 4, 5, 5])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    Ok(())
}

#[test]
fn test_case2() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 3, 4, 4, 20, 20])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 3, 4, 4, 5, 5])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    Ok(())
}

#[test]
fn test_case3() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 3, 4, 4, 20, 20])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 3, 4, 4, 5, 5])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    Ok(())
}

#[test]
fn test_case4() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 3, 4, 4, 20, 20])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 3, 4, 4, 5, 5])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    Ok(())
}

#[test]
fn test_case5() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 3, 4, 4, 19, 19])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 3, 4, 4, 5, 5])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    Ok(())
}

#[test]
fn test_case6() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 3, 4, 4, 20, 20])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 3, 4, 4, 5, 5])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    Ok(())
}

#[test]
fn test_case7() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 3, 4, 4, 20, 20])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 3, 4, 4, 5, 5])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    Ok(())
}

#[test]
fn test_case8() -> anyhow::Result<()> {
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 300, 4, 4, 20, 20])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    let (kernel, a, tch_kernel, tch_a) = common_input([1, 300, 4, 4, 5, 5])?;
    assert_eq(&a, &kernel, &tch_a, &tch_kernel)?;
    assert_eq_pad(&a, &kernel, &tch_a, &tch_kernel)?;
    Ok(())
}
