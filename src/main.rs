use hpt::backend::Cuda;
use hpt::ops::*;
use hpt::types::f16;
use hpt::utils::{set_display_elements, set_seed};
use hpt::{error::TensorError, Tensor};
fn main() -> Result<(), TensorError> {
    let cols = 13;
    set_display_elements(3);
    set_seed::<Cuda>(1024);
    // 2D matrix multiplication
    let a = Tensor::<f32, Cuda>::randn([1, cols])?.astype::<f16>()?;
    let a_cpu = a.to_cpu::<0>()?;
    for _ in 0..1 {
        let b = a.log_softmax(1)?;
        let b_cpu = a_cpu.log_softmax(1)?;
        println!("{}", b);
        println!("{}", b_cpu);
    }

    Ok(())
}
