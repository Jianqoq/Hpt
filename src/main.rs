use candle_core::{self, backend::BackendDevice};
use tensor_dyn::TensorCreator;
use tensor_dyn::NormalReduce;

fn main() -> anyhow::Result<()> {
    let device = candle_core::CudaDevice::new(0)?;
    let a = candle_core::Tensor::ones(
        &[1000, 1000, 1000],
        candle_core::DType::F32,
        &candle_core::Device::Cuda(device),
    )?;

    let now = std::time::Instant::now();
    let res = a.sum((0, 2))?;
    res.device().synchronize()?;
    println!("{:?}", now.elapsed());
    // println!("{}", res);

    let a = tensor_dyn::Tensor::<f32, tensor_dyn::Cuda, 0>::ones([1000, 1000, 1000])?;
    let res = a.sum([0, 1, 2], false)?;
    res.device().synchronize()?;

    let now = std::time::Instant::now();
    let res = a.sum([0, 2], false)?;
    res.device().synchronize()?;
    println!("{:?}", now.elapsed());
    Ok(())
}
