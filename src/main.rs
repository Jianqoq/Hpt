use hpt::{select, ShapeManipulate, Slice, Tensor, TensorCreator, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::arange(0, 4 * 4 * 4 * 4)?.reshape(&[4, 4, 4, 4])?;

    // Select rows 1:3, full range except the last dim, last dim is 3:4
    let b = a.slice(&select![1:3, .., 3:4])?;

    println!("{}", b);
    Ok(())
}
