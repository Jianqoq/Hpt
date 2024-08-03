use tensor_dyn::tensor::Tensor;
use tensor_traits::tensor::TensorCreator;
use tensor_traits::shape_manipulate::ShapeManipulate;

fn main() -> anyhow::Result<()> {
    let a = Tensor::<f32>::arange(0.0, 30.0)?.reshape(&[2, 3, 5])?;
    let b = Tensor::<f32>::arange(30.0, 80.0)?.reshape(&[2, 5, 5])?;
    let c = Tensor::<f32>::arange(60.0, 90.0)?.reshape(&[2, 3, 5])?;

    let d = Tensor::stack(&[a, b, c], 1, false)?;
    println!("{:?}", d);
    Ok(())
}
