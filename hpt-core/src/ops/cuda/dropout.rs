use crate::ops::cuda::cuda_utils::load_ptx_and_get_data;
use crate::CUDA_SEED;
use crate::{tensor_base::_Tensor, Cuda, Tensor};
use cudarc::driver::{DeviceRepr, LaunchAsync};
use hpt_cudakernels::DROPOUT;
use hpt_traits::{CommonBounds, TensorCreator, TensorInfo};
use hpt_types::{cast::Cast, type_promote::NormalOut};

use super::cuda_utils::compute_kernel_launch_config;

impl<T, const DEVICE_ID: usize> _Tensor<T, Cuda, DEVICE_ID>
where
    T: CommonBounds + NormalOut<bool, Output = T> + NormalOut<T, Output = T> + DeviceRepr,
    f64: Cast<T>,
{
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn dropout(&self, rate: f64) -> anyhow::Result<_Tensor<T, Cuda, DEVICE_ID>> {
        let ret = _Tensor::<T, Cuda, DEVICE_ID>::empty(self.shape())?;
        let scale: T = (1.0 / (1.0 - rate)).cast();
        if self.is_contiguous() {
            let (kernel, reg_info) = load_ptx_and_get_data(
                "dropout",
                &format!("dropout_{}", T::ID),
                self.device(),
                self.device_cap(),
                &DROPOUT,
            )
            .unwrap();
            let cfg = compute_kernel_launch_config(self.device(), &reg_info, self.size());
            unsafe {
                kernel.clone().launch(
                    cfg,
                    (
                        ret.cuda_slice(),
                        self.cuda_slice(),
                        rate as f32,
                        scale,
                        CUDA_SEED.load(std::sync::atomic::Ordering::Relaxed),
                        self.size(),
                    ),
                )
            }
            .unwrap();
        } else {
            let (kernel, reg_info) = load_ptx_and_get_data(
                "dropout",
                &format!("dropout_uncontiguous_{}", T::ID),
                self.device(),
                self.device_cap(),
                &DROPOUT,
            )
            .unwrap();
            let cfg = compute_kernel_launch_config(self.device(), &reg_info, self.size());
            unsafe {
                kernel.clone().launch(
                    cfg,
                    (
                        ret.cuda_slice(),
                        self.cuda_slice(),
                        rate as f32,
                        scale,
                        CUDA_SEED.load(std::sync::atomic::Ordering::Relaxed),
                        &self.cuda_shape()?,
                        &self.cuda_strides()?,
                        self.ndim(),
                        self.size(),
                    ),
                )
            }
            .unwrap();
        }
        Ok(ret)
    }
}

impl<T, const DEVICE_ID: usize> Tensor<T, Cuda, DEVICE_ID>
where
    T: CommonBounds + NormalOut<bool, Output = T> + NormalOut<T, Output = T> + DeviceRepr,
    f64: Cast<T>,
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
