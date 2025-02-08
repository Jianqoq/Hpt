use hpt_core::{Tensor, TensorCreator, TensorDot, TensorError};

fn main() -> Result<(), TensorError> {
    // Matrix multiplication (2D tensordot)
    let a = Tensor::new(&[[1., 2.], [3., 4.]]);
    let b = Tensor::new(&[[5., 6.], [7., 8.]]);
    let c = a.tensordot(&b, ([1], [0]))?;  // Contract last axis of a with first axis of b
    println!("Matrix multiplication:\n{}", c);

    // Higher dimensional example
    let d = Tensor::<f32>::ones(&[2, 3, 4])?;
    let e = Tensor::<f32>::ones(&[4, 3, 2])?;
    let f = d.tensordot(&e, ([1, 2], [1, 0]))?;  // Contract axes 1,2 of d with axes 1,0 of e
    println!("Higher dimensional result:\n{}", f);
    
    Ok(())
}