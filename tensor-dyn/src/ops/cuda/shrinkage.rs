use std::ops::{Add, Neg, Sub};

use crate::Cuda;
use crate::{tensor::Tensor, tensor_base::_Tensor};
use cudarc::driver::DeviceRepr;
use tensor_traits::CommonBounds;
use tensor_types::cuda_types::scalar::Scalar;
use tensor_types::traits::SimdSelect;
use tensor_types::type_promote::SimdCmp;
use tensor_types::type_promote::{NormalOut, NormalOutUnary};

use super::unary::uary_fn_with_out_simd;

impl<T, const DEVICE_ID: usize> _Tensor<T, Cuda, DEVICE_ID>
where
    T: CommonBounds + PartialOrd + Sub<Output = T> + Neg<Output = T> + Add<Output = T> + DeviceRepr,
    T::Vec: SimdCmp + NormalOut<Output = T::Vec>,
    <T::Vec as SimdCmp>::Output: SimdSelect<T::Vec>,
    Scalar<T>: NormalOutUnary<Base = Scalar<T>> + NormalOut<Output = Scalar<T>>,
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
    pub fn shrinkage(&self, bias: T, lambda: T) -> anyhow::Result<_Tensor<T, Cuda, DEVICE_ID>> {
        uary_fn_with_out_simd(
            self,
            "shrinkage",
            |out, x| {
                let sign = x.clone()._sign();
                let abs = x._abs();
                let zero = Scalar::<T>::new_from_val(T::ZERO);
                let lambda = Scalar::<T>::new_from_val(lambda);
                let bias = Scalar::<T>::new_from_val(bias);
                let res = sign._mul(abs._sub(lambda)._max(zero))._add(bias);
                println!("{}", res.val());
                out.assign(res)
            },
            None::<_Tensor<T, Cuda, DEVICE_ID>>,
        )
    }
}

impl<T, const DEVICE_ID: usize> Tensor<T, Cuda, DEVICE_ID>
where
    T: CommonBounds + PartialOrd + Sub<Output = T> + Neg<Output = T> + Add<Output = T> + DeviceRepr,
    T::Vec: SimdCmp + NormalOut<Output = T::Vec>,
    <T::Vec as SimdCmp>::Output: SimdSelect<T::Vec>,
    Scalar<T>: NormalOutUnary<Base = Scalar<T>> + NormalOut<Output = Scalar<T>>,
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
    pub fn shrinkage(&self, bias: T, lambda: T) -> anyhow::Result<Tensor<T, Cuda, DEVICE_ID>> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID>::shrinkage(self.inner.as_ref(), bias, lambda)?.into())
    }
}
