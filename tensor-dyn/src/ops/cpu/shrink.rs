use std::ops::{Add, Neg, Sub};

use crate::{tensor::Tensor, tensor_base::_Tensor};
use tensor_iterator::TensorIterator;
use tensor_traits::CommonBounds;
use tensor_types::traits::{Init, SimdSelect};
use tensor_types::type_promote::{NormalOut, NormalOutUnary};
use tensor_types::type_promote::SimdCmp;

impl<T> _Tensor<T>
where
    T: CommonBounds + PartialOrd + Sub<Output = T> + Neg<Output = T> + Add<Output = T>,
    T::Vec: SimdCmp + NormalOut<Output = T::Vec>,
    <T::Vec as SimdCmp>::Output: SimdSelect<T::Vec>,
{
    /// Applies the shrinkage operation to the input tensor.
    ///
    /// The shrinkage operation is typically used in regularization techniques such as soft-thresholding.
    /// This method reduces the magnitude of the tensor's elements based on the `lambda` parameter, while
    /// applying a bias. Specifically, the elements are shrunk towards zero by `lambda`, with any values
    /// smaller than `lambda` in magnitude being set to zero.
    ///
    /// The formula for shrinkage is:
    /// - If `x > bias + lambda`, `x = x - lambda`
    /// - If `x < bias - lambda`, `x = x + lambda`
    /// - Otherwise, `x = bias`
    ///
    /// # Arguments
    ///
    /// * `bias` - A bias value that is applied before shrinkage.
    /// * `lambda` - The shrinkage threshold. Elements with magnitudes smaller than `lambda` will be reduced to the bias value.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a new tensor with the shrinkage operation applied.
    #[allow(unused)]
    pub fn shrink(&self, bias: T, lambda: T) -> anyhow::Result<_Tensor<T>> {
        let lambda_vec = T::Vec::splat(lambda);
        let neg_lambda_vec = lambda_vec._neg();
        let bias_vec = T::Vec::splat(bias);

        Ok(self
            .par_iter_simd()
            .strided_map_simd(
                |(x, y)| {
                    *x = if y > lambda {
                        y - bias
                    } else if y < -lambda {
                        y + bias
                    } else {
                        T::ZERO
                    };
                },
                |(x, y)| {
                    let gt_mask = y._gt(lambda_vec);
                    let lt_mask = y._lt(neg_lambda_vec);
                    let sub_bias = y._sub(bias_vec);
                    let add_bias = y._add(bias_vec);
                    let zero = T::Vec::splat(T::ZERO);
                    let res = gt_mask.select(sub_bias, zero);
                    *x = lt_mask.select(add_bias, res);
                },
            )
            .collect())
    }
}

impl<T> Tensor<T>
where
    T: CommonBounds + PartialOrd + Sub<Output = T> + Neg<Output = T> + Add<Output = T>,
    T::Vec: SimdCmp + NormalOut<Output = T::Vec>,
    <T::Vec as SimdCmp>::Output: SimdSelect<T::Vec>,
{
        /// Applies the shrinkage operation to the input tensor.
    ///
    /// The shrinkage operation is typically used in regularization techniques such as soft-thresholding.
    /// This method reduces the magnitude of the tensor's elements based on the `lambda` parameter, while
    /// applying a bias. Specifically, the elements are shrunk towards zero by `lambda`, with any values
    /// smaller than `lambda` in magnitude being set to zero.
    ///
    /// The formula for shrinkage is:
    /// - If `x > bias + lambda`, `x = x - lambda`
    /// - If `x < bias - lambda`, `x = x + lambda`
    /// - Otherwise, `x = bias`
    ///
    /// # Arguments
    ///
    /// * `bias` - A bias value that is applied before shrinkage.
    /// * `lambda` - The shrinkage threshold. Elements with magnitudes smaller than `lambda` will be reduced to the bias value.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a new tensor with the shrinkage operation applied.
    pub fn shrink(&self, bias: T, lambda: T) -> anyhow::Result<Tensor<T>> {
        Ok(_Tensor::shrink(self, bias, lambda)?.into())
    }
}
