use crate::{ backend::Cpu, tensor_base::_Tensor, Cuda, Tensor };
use cudarc::driver::DeviceRepr;
use rand_distr::Distribution;
use rayon::iter::ParallelIterator;
use tensor_iterator::{ iterator_traits::ParStridedIteratorSimdZip, TensorIterator };
use tensor_traits::{ CommonBounds, TensorCreator, TensorInfo };
use tensor_types::{ into_scalar::IntoScalar, type_promote::NormalOut };

impl<T, const DEVICE_ID: usize> _Tensor<T, Cuda, DEVICE_ID>
    where
        T: CommonBounds + NormalOut<bool, Output = T> + NormalOut<T, Output = T> + DeviceRepr,
        f64: IntoScalar<T>
{
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn dropout(&self, rate: f64) -> anyhow::Result<_Tensor<T, Cuda, DEVICE_ID>> {
        let ret = _Tensor::<T, Cuda, DEVICE_ID>::empty(self.shape())?;
        let scale = (1.0 / (1.0 - rate)) as f32;
        Ok(ret)
    }
}

impl<T, const DEVICE_ID: usize> Tensor<T, Cuda, DEVICE_ID>
    where
        T: CommonBounds + NormalOut<bool, Output = T> + NormalOut<T, Output = T> + DeviceRepr,
        f64: IntoScalar<T>
{
    /// Applies dropout to the tensor during training.
    ///
    /// This method randomly drops out elements from the tensor based on the specified dropout rate,
    /// which is a common regularization technique in neural networks. During training, elements are
    /// zeroed out with a probability equal to `rate`, and the remaining elements are scaled by
    /// a factor of `1 / (1 - rate)` to maintain the expected sum. This function uses parallel
    /// iteration for performance optimization.
    ///
    /// # Arguments
    ///
    /// * `rate` - The dropout rate, specified as a floating-point number between 0 and 1.
    ///   It represents the probability that each element will be dropped (set to zero).
    ///   For example, a rate of 0.5 means 50% of the elements will be dropped out.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a new tensor with dropout applied.
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn dropout(&self, rate: f64) -> anyhow::Result<Tensor<T, Cuda, DEVICE_ID>> {
        Ok(self.inner.dropout(rate)?.into())
    }
}
