use crate::ops::cpu::pooling::common::pooling_template;
use crate::tensor_base::_Tensor;
use crate::Tensor;
use hpt_common::error::base::TensorError;
use hpt_common::shape::shape::Shape;
use hpt_traits::CommonBounds;
use hpt_types::into_scalar::Cast;
use hpt_types::type_promote::FloatOutBinary;
use hpt_types::type_promote::NormalOut;
use hpt_types::vectors::traits::*;

use super::common::adaptive_pooling_template;

impl<T> _Tensor<T>
where
    T: CommonBounds + Cast<T> + NormalOut<Output = T> + FloatOutBinary<T, Output = T>,
    T::Vec: VecTrait<T>
        + Copy
        + Send
        + Sync
        + NormalOut<Output = T::Vec>
        + FloatOutBinary<T::Vec, Output = T::Vec>,
    bool: Cast<T>,
    i64: Cast<T>,
{
    /// Performs a 2D avg pooling operation on the input tensor.
    ///
    /// This method applies a 2D avg pooling operation on the tensor using the specified kernel,
    /// strides (steps), padding, and dilation factors.
    ///
    /// # Arguments
    ///
    /// * `kernels` - A reference to the tensor representing the convolution kernels (filters).
    ///   The size of the kernel tensor determines the spatial dimensions of the convolution operation.
    /// * `steps` - A 2-element array specifying the stride (step size) of the convolution along the height and width dimensions.
    /// * `padding` - A 2-element array of tuples representing the padding for the height and width dimensions.
    ///   Each tuple specifies the amount of padding added before and after the data along the respective axis.
    /// * `dilation` - A 2-element array specifying the dilation factor for the convolution along the height and width dimensions.
    ///   Dilation allows the kernel to be applied to inputs with gaps, increasing the receptive field of the kernel.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing the output tensor after applying the 2D avg pooling operation.
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn avgpool2d(
        &self,
        kernels_shape: &Shape,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
    ) -> Result<_Tensor<T>, TensorError> {
        let kernel_size: T = kernels_shape.size().cast();
        let kernel_size_vec = T::Vec::splat(kernel_size);
        pooling_template(
            self,
            kernels_shape,
            steps,
            padding,
            dilation,
            |a, b| a._add(b),
            |a, b| a._add(b),
            |a| a._div(kernel_size),
            |a| a._div(kernel_size_vec),
        )
    }

    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn adaptive_avgpool2d(&self, output_size: [i64; 2]) -> Result<_Tensor<T>, TensorError> {
        adaptive_pooling_template(
            self,
            output_size,
            |a, b| a._add(b),
            |a, b| a._add(b),
            |a, kernel_size| a._div(kernel_size),
            |a, kernel_size_vec| a._div(kernel_size_vec),
        )
    }
}

impl<T> Tensor<T>
where
    T: CommonBounds + Cast<T> + NormalOut<Output = T> + FloatOutBinary<T, Output = T>,
    T::Vec: VecTrait<T>
        + Copy
        + Send
        + Sync
        + NormalOut<Output = T::Vec>
        + FloatOutBinary<T::Vec, Output = T::Vec>,
    bool: Cast<T>,
    i64: Cast<T>,
{
    /// Performs a 2D avg pooling operation on the input tensor.
    ///
    /// This method applies a 2D avg pooling operation on the tensor using the specified kernel,
    /// strides (steps), padding, and dilation factors.
    ///
    /// # Arguments
    ///
    /// * `kernels` - A reference to the tensor representing the convolution kernels (filters).
    ///   The size of the kernel tensor determines the spatial dimensions of the convolution operation.
    /// * `steps` - A 2-element array specifying the stride (step size) of the convolution along the height and width dimensions.
    /// * `padding` - A 2-element array of tuples representing the padding for the height and width dimensions.
    ///   Each tuple specifies the amount of padding added before and after the data along the respective axis.
    /// * `dilation` - A 2-element array specifying the dilation factor for the convolution along the height and width dimensions.
    ///   Dilation allows the kernel to be applied to inputs with gaps, increasing the receptive field of the kernel.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing the output tensor after applying the 2D avg pooling operation.
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn avgpool2d(
        &self,
        kernels_shape: &Shape,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
    ) -> Result<Tensor<T>, TensorError> {
        Ok(self
            .inner
            .avgpool2d(&kernels_shape, steps, padding, dilation)?
            .into())
    }

    /// Performs a adaptive avg pooling operation on the input tensor.
    ///
    /// This method applies a adaptive avg pooling operation on the tensor using the specified output size.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing the output tensor after applying the adaptive avg pooling operation.
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn adaptive_avgpool2d(&self, output_size: [i64; 2]) -> Result<Tensor<T>, TensorError> {
        Ok(self.inner.adaptive_avgpool2d(output_size)?.into())
    }
}
