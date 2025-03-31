use hpt::backend::Cuda;
use hpt::ops::*;
use hpt::utils::{set_display_elements, set_seed};
use hpt::{error::TensorError, Tensor};
use hpt::types::f16;
fn main() -> Result<(), TensorError> {
    set_display_elements(3);
    set_seed::<Cuda>(1024);
    // 2D matrix multiplication
    let a = Tensor::<f32, Cuda>::randn([65536, 1024])?.astype::<f16>()?;
    // let a_cpu = a.to_cpu::<0>()?;
    // println!("{}", a);
    for _ in 0..10 {
        let b = a.layernorm([1024], None, None, f16::from_f32(1e-5))?;
        // let b_cpu = a_cpu.layernorm([1024], None, None, 1e-5)?;
        // println!("{}", b);
        // println!("{}", b_cpu);
    }

    Ok(())
}
