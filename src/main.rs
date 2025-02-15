use hpt::{CumulativeOps, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // 1D tensor
    let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0]);
    let cum_a = a.cumprod(None)?;
    println!("{}", cum_a);
    // [1.0, 2.0, 6.0]

    // 2D tensor
    let b = Tensor::<f32>::new(&[[1.0, 2.0], [3.0, 4.0]]);

    // Cumprod along axis 0
    let cum_b0 = b.cumprod(0)?;
    println!("{}", cum_b0);
    // [[1.0, 2.0],
    //  [3.0, 8.0]]

    // Cumprod along axis 1
    let cum_b1 = b.cumprod(1)?;
    println!("{}", cum_b1);
    // [[1.0, 2.0],
    //  [3.0, 12.0]]

    Ok(())
}
