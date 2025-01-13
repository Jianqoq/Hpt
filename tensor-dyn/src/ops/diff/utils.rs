use tensor_common::error::base::TensorError;
use tensor_traits::CommonBounds;

use crate::{tensor::DiffTensor, Cpu, Tensor};

pub(crate) fn handle_grad<T: CommonBounds, const DEVICE: usize>(
    tensor: &mut DiffTensor<T, Cpu, DEVICE>,
    grad: Tensor<T, Cpu, DEVICE>,
) -> Result<(), TensorError> {
    if *tensor.out_degree.borrow() > 0 {
        if let Some(mut stored_grad) = core::mem::take(&mut tensor.grad) {
            stored_grad += grad;
            tensor.grad = Some(stored_grad);
        } else {
            tensor.grad = Some(grad);
        }
        *tensor.out_degree.borrow_mut() -= 1;
    } else {
        if let Some(mut stored_grad) = core::mem::take(&mut tensor.grad) {
            stored_grad += grad;
            tensor.backward.borrow_mut()(stored_grad)?;
        } else {
            tensor.backward.borrow_mut()(grad)?;
        }
    }
    Ok(())
}
