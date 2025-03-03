use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_common::error::base::TensorError;
use hpt_traits::{CommonBounds, NormalReduce};
use hpt_types::into_scalar::Cast;

use crate::{tensor::DiffTensor, Cpu, Tensor};

pub(crate) fn handle_grad<T, const DEVICE: usize, A>(
    tensor: &mut DiffTensor<T, Cpu, DEVICE, A>,
    mut grad: Tensor<T, Cpu, DEVICE, A>,
    broadcast_axes: &[usize],
) -> Result<(), TensorError>
where
    T: CommonBounds + Cast<T>,
    A: Allocator + 'static + Send + Sync,
    A::Output: AllocatorOutputRetrive,
{
    if !broadcast_axes.is_empty() {
        grad = grad.sum(broadcast_axes, false)?;
    }
    if *tensor.out_degree.borrow() > 1 {
        let taked = tensor.grad.borrow_mut().take();
        if let Some(mut stored_grad) = taked {
            stored_grad += grad;
            tensor.grad.borrow_mut().replace(stored_grad);
        } else {
            tensor.grad.borrow_mut().replace(grad);
        }
        *tensor.out_degree.borrow_mut() -= 1;
    } else {
        let taked = tensor.grad.borrow_mut().take();
        if let Some(mut stored_grad) = taked {
            stored_grad += grad;
            let res = tensor.backward.borrow_mut()(stored_grad.clone())?;
            if res {
                tensor.grad.borrow_mut().replace(stored_grad);
            }
        } else {
            let res = tensor.backward.borrow_mut()(grad.clone())?;
            if res {
                tensor.grad.borrow_mut().replace(grad);
            }
        }
    }
    Ok(())
}
