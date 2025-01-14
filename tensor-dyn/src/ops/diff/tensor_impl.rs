use tensor_common::error::base::TensorError;

use crate::{tensor::DiffTensor, Cpu, Tensor};


impl<T: Clone, const DEVICE: usize> DiffTensor<T, Cpu, DEVICE> {
    /// Backward the gradient of the tensor
    pub fn backward(&mut self, grad: Tensor<T, Cpu, DEVICE>) -> Result<(), TensorError> {
        if let Ok(true) = self.backward.borrow_mut()(grad.clone()) {
            self.grad.borrow_mut().replace(grad);
        }
        Ok(())
    }

    /// Get the gradient of the tensor
    pub fn grad(&self) -> Option<Tensor<T, Cpu, DEVICE>> {
        self.grad.borrow().as_ref().cloned()
    }
}
