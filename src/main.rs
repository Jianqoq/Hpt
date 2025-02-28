use hpt::{set_cuda_seed, Cuda, NormalReduce, Random, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    set_cuda_seed(1000);
    let shape = [1024, 1024, 80];
    let a = Tensor::<f32, Cuda>::randn(&shape)?;
    let now = std::time::Instant::now();
    for _ in 0..1 {
        let _ = a.sum([1, 2], true)?;
    }
    println!("{:?}", now.elapsed() / 1);
    let sum = a.sum([1, 2], true)?;
    println!("{}", sum);
    let sum_cpu = a.to_cpu::<0>()?.sum([1, 2], true)?;
    println!("{}", sum_cpu);
    Ok(())
}
