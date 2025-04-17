use hpt::Tensor;

use hpt::common::cpu::TensorLike;
use hpt::common::{CommonBounds, TensorInfo};

pub(crate) fn copy_from_tch<T: CommonBounds>(
    a: &mut Tensor<T>,
    tch_a: &tch::Tensor,
) -> anyhow::Result<()> {
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const T, a_size)
    });
    Ok(())
}
