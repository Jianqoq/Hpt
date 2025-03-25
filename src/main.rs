use hpt::{
    backend::Cuda,
    common::TensorInfo,
    error::TensorError,
    ops::{CudaConvBatchNorm, Random, TensorCreator},
    Tensor,
};

fn main() -> Result<(), TensorError> {
    // [batch_size, height, width, in_channels]
    let input = Tensor::<f32, Cuda>::randn([1, 32, 32, 3])?;

    // [out_channels, kernel_height, kernel_width, in_channels]
    let kernels = Tensor::<f32, Cuda>::randn([16, 3, 3, 3])?;

    // Batch normalization parameters
    let mean = Tensor::<f32, Cuda>::zeros([16])?;
    let var = Tensor::<f32, Cuda>::ones([16])?;
    let gamma = Tensor::<f32, Cuda>::ones([16])?;
    let beta = Tensor::<f32, Cuda>::zeros([16])?;

    // Optional convolution bias
    let bias = Tensor::<f32, Cuda>::zeros([16])?;

    // Perform fused convolution with batch normalization
    let output = input.batchnorm_conv2d(
        &kernels,
        &mean,
        &var,
        &gamma,
        &beta,
        Some(&bias),
        1e-5,   // epsilon
        [1, 1], // stride
        [1, 1], // padding
        [1, 1], // dilation
        None,   // auto select algo
    )?;

    println!("Output shape: {:?}", output.shape()); // [1, 32, 32, 16]
    Ok(())
}
