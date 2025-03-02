use hpt::{
    match_selection, set_seed, slice, Cpu, Cuda, NormalReduce, RandomInt, ShapeManipulate, Slice,
    Tensor, TensorCreator, TensorError, TensorInfo, TensorLike,
};

fn main() -> Result<(), TensorError> {
    set_seed::<Cuda>(1000);
    let shape = [124, 75, 78];
    let a = Tensor::<i64, Cpu>::arange(0, 124 * 75 * 78)?
        .reshape(&shape)?
        .to_cuda::<0>()?;
    let a = a.slice(&match_selection!(80:86,51:52,3:4))?;
    println!("a: {}", a);
    // let now = std::time::Instant::now();
    // for _ in 0..1 {
    //     let _ = a.sum([0, 1, 2], false)?;
    // }
    // println!("{:?}", now.elapsed() / 1);
    let sum = a.sum([0, 1, 2], false)?.to_cpu::<0>()?;
    let sum_cpu = a.to_cpu::<0>()?.sum([0, 1, 2], false)?;
    println!("sum: {}", sum);
    println!("sum_cpu: {}", sum_cpu);
    // if !sum.allclose(&sum_cpu) {
    //     sum.as_raw().iter().zip(sum_cpu.as_raw().iter()).for_each(|(a, b)| {
    //         if a != b {
    //             panic!("{} {}", a, b);
    //         }
    //     });
    // }
    Ok(())
}
