use tensor_dyn::tensor_base::_Tensor;
use tensor_traits::ops::binary::Matmul;
use tensor_traits::random::Random;
use tensor_traits::tensor::TensorInfo;

pub fn main() -> anyhow::Result<()> {
    let a = _Tensor::<f32>::randn(&[1, 3])?;
    let b = _Tensor::<f32>::randn(&[3, 1])?;
    let c = a.matmul(b.clone())?;

    println!("{:?}", c);

    let d = a
        .iter()
        .zip(b.iter())
        .strided_map(|(x, y)| x * y)
        .collect::<_Tensor<f32>>();

    println!("{}", d);
    Ok(())
}
