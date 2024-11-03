use tch::Tensor;
use tensor_dyn::{ backend::Cpu, TensorCreator };
use tensor_dyn::{ set_num_threads, ShapeManipulate };

fn common_input<const N: usize>(
    end: i64,
    shape: [i64; N]
) -> anyhow::Result<(tensor_dyn::tensor::Tensor<i64, Cpu>, Tensor)> {
    let a = tensor_dyn::tensor::Tensor::<i64, Cpu>::arange(0, end)?.reshape(&shape)?;
    let tch_a = Tensor::arange(end, (tch::Kind::Int64, tch::Device::Cpu)).reshape(&shape);
    Ok((a, tch_a))
}

#[test]
fn test_sum() -> anyhow::Result<()> {
    let (a, tch_a) = common_input(2 * 5 * 10, [2, 5, 10])?;
    let res = a.softmax(2)?;
    let tch_res = tch_a.softmax(2, tch::Kind::Float);
    Ok(())
}
