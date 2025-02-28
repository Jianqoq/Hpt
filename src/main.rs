use hpt::{set_cuda_seed, Cuda, NormalReduce, Random, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    set_cuda_seed(1000);
    let shape = [256, 256, 16];
    let a = Tensor::<f32, Cuda>::randn(&shape)?;
    let now = std::time::Instant::now();
    for _ in 0..1 {
        let _ = a.sum([0, 1], true)?;
    }
    println!("{:?}", now.elapsed() / 1);
    let sum = a.sum([0, 1], true)?.to_cpu::<0>()?;
    let sum_cpu = a.to_cpu::<0>()?.sum([0, 1], true)?;
    println!("{} {}", sum, sum_cpu);
    Ok(())
}
