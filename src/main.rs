use hpt::{
    backend::Cuda,
    error::TensorError,
    ops::{Matmul, TensorCreator},
    Tensor,
};

fn main() -> Result<(), TensorError> {
    // 2D matrix multiplication
    let a = Tensor::<f64>::new(&[[1., 2.], [3., 4.]]).to_cuda::<0>()?;
    let b = Tensor::<f64>::new(&[[5., 6.], [7., 8.]]).to_cuda::<0>()?;
    let c = a.matmul(&b)?;
    println!("2D result:\n{}", c);

    // 3D batch matrix multiplication
    let d = Tensor::<f64, Cuda>::ones(&[2, 2, 3])?; // 2 matrices of shape 2x3
    let e = Tensor::<f64, Cuda>::ones(&[2, 3, 2])?; // 2 matrices of shape 3x2
    let f = d.matmul(&e)?; // 2 matrices of shape 2x2
    println!("3D result:\n{}", f);

    Ok(())
}
