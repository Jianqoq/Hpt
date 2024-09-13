#![allow(unused)]

use backend::Cpu;
use tensor_base::_Tensor;
use tensor_dyn::*;

fn assert_eq(a: &_Tensor<i64>, b: &tch::Tensor) {
    let raw = a.as_raw();
    let tch_raw = unsafe { core::slice::from_raw_parts(b.data_ptr() as *const i64, a.size()) };
    raw.iter()
        .zip(tch_raw.iter())
        .for_each(|(a, b)| assert_eq!(a, b));
}

fn common_input<const N: usize>(
    end: i64,
    shape: [i64; N]
) -> anyhow::Result<(_Tensor<i64, Cpu>, tch::Tensor)> {
    let a = _Tensor::<i64, Cpu>::arange(0, end)?.reshape(&shape)?;
    let tch_a = tch::Tensor::arange(end, (tch::Kind::Int64, tch::Device::Cpu)).reshape(&shape);
    Ok((a, tch_a))
}

#[test]
fn test_sum() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(8 * 8096 * 2048, [8, 8096, 2048])?;
    let sum = a.sum(0, false)?;
    let tch_sum = tch_a.sum_dim_intlist(0, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum(1, false)?;
    let tch_sum = tch_a.sum_dim_intlist(1, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum(2, false)?;
    let tch_sum = tch_a.sum_dim_intlist(2, false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum([0, 1], false)?;
    let tch_sum = tch_a.sum_dim_intlist(&[0, 1][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum([0, 2], false)?;
    let tch_sum = tch_a.sum_dim_intlist(&[0, 2][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum([1, 2], false)?;
    let tch_sum = tch_a.sum_dim_intlist(&[1, 2][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    let sum = a.sum([0, 1, 2], false)?;
    let tch_sum = tch_a.sum_dim_intlist(&[0, 1, 2][..], false, tch::Kind::Int64);
    assert_eq(&sum, &tch_sum);
    Ok(())
}
