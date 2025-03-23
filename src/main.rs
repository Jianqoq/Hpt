use hpt::ops::*;
use hpt::utils::resize_cpu_lru_cache;
use hpt::Tensor;

fn main() -> anyhow::Result<()> {
    resize_cpu_lru_cache(1, 0);

    let a = Tensor::<f32>::randn([1024, 1])?;
    let b = Tensor::<f32>::randn([1, 1024])?;
    let now = std::time::Instant::now();
    for _ in 0..1000 {
        let c = a.matmul(&b)?;
    }
    let duration = now.elapsed();
    println!("Time taken: {:?}", duration / 1000);
    let now = std::time::Instant::now();
    for _ in 0..1000 {
        let c = a.gemm(&b, 0.0, 1.0, false, false, false)?;
    }
    let duration = now.elapsed();
    println!("Time taken: {:?}", duration / 1000);

    Ok(())
}
