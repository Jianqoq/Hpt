use hpt::{set_cuda_seed, Cuda, NormalReduce, Random, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    set_cuda_seed(1000);
    let shape = [512, 512, 80];
    let a = Tensor::<f32, Cuda>::randn(&shape)?;
    let now = std::time::Instant::now();
    for _ in 0..1000 {
        let _ = a.sum([0, 2], true)?;
    }
    a.device().synchronize().expect("Synchronize failed");
    println!("{:?}", now.elapsed() / 1000);
    // let sum = a.sum([0, 2], true)?;
    // println!("{}", sum);
    // let sum_cpu = a.to_cpu::<0>()?.sum([0, 2], true)?;
    // println!("{}", sum_cpu);
    Ok(())
}
