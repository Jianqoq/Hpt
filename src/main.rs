use hpt::{Tensor, TensorError, TensorWhere};

fn main() -> Result<(), TensorError> {
    let condition = Tensor::<bool>::new(&[true, false, true]);
    let x = Tensor::<f64>::new(&[1., 2., 3.]);
    let y = Tensor::<f64>::new(&[4., 5., 6.]);
    
    let result = Tensor::tensor_where(&condition, &x, &y)?;
    println!("{}", result);  // [1., 5., 3.]
    
    Ok(())
}