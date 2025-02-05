use crate::ops::cpu::pooling::common::pooling_template;
use crate::tensor_base::_Tensor;
use crate::Tensor;
use tensor_common::error::base::TensorError;
use tensor_common::shape::shape::Shape;
use tensor_traits::CommonBounds;
use tensor_types::into_scalar::Cast;
use tensor_types::type_promote::NormalOut;
use tensor_types::vectors::traits::*;

use super::common::adaptive_pooling_template;

impl<T> _Tensor<T>
where
    T: CommonBounds + Cast<T> + NormalOut<Output = T>,
    T::Vec: VecTrait<T> + Copy + Send + Sync + NormalOut<Output = T::Vec>,
    bool: Cast<T>,
{
    /// Performs a 2D max pooling operation on the input tensor.
    ///
    /// This method applies a 2D max pooling operation on the tensor using the specified kernel,
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
    /// This function returns a `Result` containing the output tensor after applying the 2D max pooling operation.
    #[cfg_attr(feature = "track_caller", track_caller)]
    #[inline(never)]
    pub fn maxpool2d(
        &self,
        kernels_shape: &Shape,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
    ) -> std::result::Result<_Tensor<T>, TensorError> {
        pooling_template(
            self,
            kernels_shape,
            steps,
            padding,
            dilation,
            |a, b| a._max(b),
            |a, b| a._max(b),
            |a| a,
            |a| a,
        )
    }

    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn adaptive_maxpool2d(
        &self,
        output_size: [i64; 2],
    ) -> std::result::Result<_Tensor<T>, TensorError>
    where
        i64: Cast<T>,
    {
        adaptive_pooling_template(
            self,
            output_size,
            |a, b| a._max(b),
            |a, b| a._max(b),
            |a, _| a,
            |a, _| a,
        )
    }
}

impl<T> Tensor<T>
where
    T: CommonBounds + Cast<T> + NormalOut<Output = T>,
    T::Vec: VecTrait<T> + Copy + Send + Sync + NormalOut<Output = T::Vec>,
    bool: Cast<T>,
{
    /// Performs a 2D max pooling operation on the input tensor.
    ///
    /// This method applies a 2D max pooling operation on the tensor using the specified kernel,
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
    /// This function returns a `Result` containing the output tensor after applying the 2D convolution operation.
    #[cfg_attr(feature = "track_caller", track_caller)]
    #[inline(never)]
    pub fn maxpool2d(
        &self,
        kernels_shape: &Shape,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
    ) -> std::result::Result<Tensor<T>, TensorError> {
        Ok(self
            .inner
            .maxpool2d(&kernels_shape, steps, padding, dilation)?
            .into())
    }

    /// Performs a adaptive max pooling operation on the input tensor.
    ///
    /// This method applies a adaptive max pooling operation on the tensor using the specified output size.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing the output tensor after applying the adaptive max pooling operation.
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn adaptive_maxpool2d(
        &self,
        output_size: [i64; 2],
    ) -> std::result::Result<Tensor<T>, TensorError>
    where
        i64: Cast<T>,
    {
        Ok(self.inner.adaptive_maxpool2d(output_size)?.into())
    }
}
