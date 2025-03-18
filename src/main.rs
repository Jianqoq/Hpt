use hpt::{
    error::TensorError,
    ops::{Gemm, TensorCreator},
    Tensor,
};

fn main() -> Result<(), TensorError> {
    // 2D matrix multiplication
    let a = Tensor::<f64>::new(&[[1., 2.], [3., 4.]]);
    let b = Tensor::<f64>::new(&[[5., 6.], [7., 8.]]);
    let c = a.gemm(&b, 0.0, 1.0, false, false, false)?;
    println!("2D result:\n{}", c);

    // 3D batch matrix multiplication
    let d = Tensor::<f64>::ones(&[2, 2, 3])?; // 2 matrices of shape 2x3
    let e = Tensor::<f64>::ones(&[2, 3, 2])?; // 2 matrices of shape 3x2
    let f = d.gemm(&e, 0.0, 1.0, false, false, false)?; // 2 matrices of shape 2x2
    println!("3D result:\n{}", f);

    Ok(())
}
