use tensor_dyn::{Tensor, TensorError, TensorCreator};
fn main() -> Result<(), TensorError> {
    // Create 4 points from 1 to 1000
    let a = Tensor::<f32>::geomspace(1.0, 1000.0, 4, true)?;
    println!("{}", a);
    // [1.0, 10.0, 100.0, 1000.0]

    // Create 3 points from 1 to 100 (exclusive)
    let b = Tensor::<f32>::geomspace(1.0, 100.0, 3, false)?;
    println!("{}", b);
    // [1.0, 4.6416, 21.5443]

    // Create 5 points between 1 and 32
    let c = Tensor::<f32>::geomspace(1.0, 32.0, 5, true)?;
    println!("{}", c);
    // [1.0, 2.3784, 5.6569, 13.4543, 32.0000]

    Ok(())
}