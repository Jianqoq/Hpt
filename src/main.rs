// use std::ffi::CStr;
// use std::hint::black_box;

use backend::Cpu;
// use half::bf16;
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
    let (a, tch_a) = common_input(8 * 8 * 8, [8, 8, 8])?;
    let sum = a.prod(2, false)?;
    Ok(())
}
