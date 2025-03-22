use hpt::backend::Cuda;
use hpt::ops::*;
use hpt::Tensor;
fn main() -> anyhow::Result<()> {
    let m = 48;
    let n = 348;
    let k = 31;
    let dim0 = 1;
    let dim1 = 1;
    let a = Tensor::<f32, Cuda>::randn(&[dim0, dim1, m, k])?;
    let layout = std::alloc::Layout::from_size_align(m * n * 4, 64).unwrap();

    let raw = unsafe { std::alloc::alloc(layout) };
    let c = unsafe { Tensor::<f32, Cuda>::from_raw(raw as *mut f32, &[dim0, dim1, m, n]) }?;
    // assert!(c.allclose(&c2, 1.0e-3, 1.0e-3));
    unsafe { std::alloc::dealloc(raw, layout) };
    Ok(())
}
