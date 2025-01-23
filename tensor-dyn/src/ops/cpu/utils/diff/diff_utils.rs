use tensor_common::error::base::TensorError;
use tensor_traits::{CommonBounds, NormalReduce};
use tensor_types::{convertion::Convertor, into_scalar::IntoScalar};

use crate::{tensor::DiffTensor, Cpu, Tensor};

pub(crate) fn handle_grad<T, const DEVICE: usize>(
    tensor: &mut DiffTensor<T, Cpu, DEVICE>,
    mut grad: Tensor<T, Cpu, DEVICE>,
    broadcast_axes: &[usize],
) -> Result<(), TensorError>
where
    T: CommonBounds + IntoScalar<T> + Convertor,
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
