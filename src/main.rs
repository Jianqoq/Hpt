use tensor_dyn::{ set_global_display_precision, TensorCreator };
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    set_global_display_precision(7);
    let a = _Tensor::<i8>::arange(0, 100)?.reshape([10, 10])?;
    let res = a.lp_normalization(2, 0)?;
    Ok(())
}
