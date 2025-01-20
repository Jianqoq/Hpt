use std::{cell::RefCell, rc::Rc};

use crate::{
    tensor::{DiffTensor, Tensor},
    Cpu,
};
use num::Integer;
use rayon::iter::ParallelIterator;
use tensor_common::{error::base::TensorError, slice::Slice};
use tensor_iterator::{iterator_traits::ParStridedIteratorZip, TensorIterator};
use tensor_traits::{CommonBounds, TensorCreator};
impl<T, const DEVICE: usize> Tensor<T, Cpu, DEVICE>
where
    T: CommonBounds,
{
    /// Extracts a slice of the tensor based on the provided indices.
    ///
    /// This method creates a new tensor that represents a slice of the original tensor.
    /// It slices the tensor according to the specified indices and returns a new tensor
    /// without copying the underlying data, but instead adjusting the shape and strides.
    ///
    /// # Arguments
    ///
    /// * `index` - A reference to a slice of `Slice` structs that define how to slice the tensor along each axis.
    ///   The `Slice` type allows for specifying ranges, single elements, and other slicing options.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the sliced tensor as a new tensor. If any slicing error occurs
    /// (e.g., out-of-bounds access), an error message is returned.
    pub fn slice(&self, index: &[Slice]) -> Result<Tensor<T, Cpu, DEVICE>, TensorError> {
        Ok(self.inner.slice(index)?.into())
    }
}

impl<T, const DEVICE: usize> DiffTensor<T, Cpu, DEVICE>
where
    T: CommonBounds,
{
    /// Extracts a slice of the tensor based on the provided indices.
    ///
    /// This method creates a new tensor that represents a slice of the original tensor.
    /// It slices the tensor according to the specified indices and returns a new tensor
    /// without copying the underlying data, but instead adjusting the shape and strides.
    ///
    /// # Arguments
    ///
    /// * `index` - A reference to a slice of `Slice` structs that define how to slice the tensor along each axis.
    ///   The `Slice` type allows for specifying ranges, single elements, and other slicing options.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the sliced tensor as a new tensor. If any slicing error occurs
    /// (e.g., out-of-bounds access), an error message is returned.
    pub fn slice(&self, index: &[Slice]) -> Result<DiffTensor<T, Cpu, DEVICE>, TensorError> {
        let res = self.inner.slice(index)?;
        let lhs = self.clone();
        if let None = lhs.grad() {
            lhs.grad.replace(Some(self.inner.zeros_like()?));
        }
        lhs.out_degree.borrow_mut().inc();
        let index = index.to_vec();
        Ok(DiffTensor {
            inner: res,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |grad: Tensor<T, Cpu, DEVICE>| {
                let taked = lhs.grad.take();
                if let Some(tmp) = taked {
                    let sliced = tmp.slice(&index)?;
                    sliced
                        .inner
                        .par_iter_mut()
                        .zip(grad.inner.par_iter())
                        .for_each(|(a, b)| {
                            *a = a._add(b);
                        });
                    if *lhs.out_degree.borrow() > 1 {
                        *lhs.out_degree.borrow_mut() -= 1;
                    } else {
                        lhs.backward.borrow_mut()(grad.clone())?;
                    }
                } else {
                    panic!("Gradient is not set for slice");
                }
                Ok(false)
            })),
        })
    }
}
