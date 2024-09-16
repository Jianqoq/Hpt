use tensor_common::slice;
use tensor_common::slice::Slice;
use tensor_dyn::match_selection;
use tensor_dyn::slice::SliceOps;
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::ShapeManipulate;
use tensor_dyn::TensorCreator;

fn main() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::arange(100, (tch::Kind::Int, tch::Device::Cpu)).reshape(&[10, 10]);
    let a = _Tensor::<i32>::arange(0, 100)?.reshape(&[10, 10])?;
    let a = slice!(a[1:9, 1:9])?.contiguous()?;
    let tch_a = tch_a.slice(1, 1, 9, 1).slice(0, 1, 9, 1).contiguous();

    let a = slice!(a[2:9, 2:9])?.contiguous()?;
    let tch_a = tch_a.slice(1, 2, 9, 1).slice(0, 2, 9, 1).contiguous();

    Ok(())
}
