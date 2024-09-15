// use std::ffi::CStr;
// use std::hint::black_box;

use backend::Cpu;
// use half::bf16;
use ops::cpu::convolutions::conv_config::{ Conv2dConfig, KernelParamAlgo };
// use tch::{ Device, Kind, Tensor };
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

fn common_input<const N: usize>(
    end: i64,
    shape: [i64; N]
) -> anyhow::Result<(_Tensor<i64, Cpu>, tch::Tensor)> {
    let a = _Tensor::<i64, Cpu>::arange(0, end)?.reshape(&shape)?;
    let tch_a = tch::Tensor::arange(end, (tch::Kind::Int64, tch::Device::Cpu)).reshape(&shape);
    Ok((a, tch_a))
}
fn main() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let a = a.permute([1, 2, 0])?;
    println!("{}", a);
    let tch_a = tch_a.permute(&[1, 2, 0][..]);
    let sum = a.sum(2, false)?;
    let tch_sum = tch_a.sum_dim_intlist(2, false, tch::Kind::Int64);
    println!("{}", sum);
    println!("{}", tch_sum);
    Ok(())
}
