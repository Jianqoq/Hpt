use crate::ops::cpu::pooling::common::pooling_template;
use crate::tensor_base::_Tensor;
use crate::Tensor;
use tensor_common::error::base::TensorError;
use tensor_common::shape::shape::Shape;
use tensor_traits::CommonBounds;
use tensor_types::into_scalar::IntoScalar;
use tensor_types::type_promote::FloatOutBinary;
use tensor_types::type_promote::NormalOut;
use tensor_types::vectors::traits::*;

impl<T> _Tensor<T>
    where
        T: CommonBounds + IntoScalar<T> + NormalOut<Output = T> + FloatOutBinary<T, Output = T>,
        T::Vec: VecTrait<T> +
            Copy +
            Send +
            Sync +
            NormalOut<Output = T::Vec> +
            FloatOutBinary<T::Vec, Output = T::Vec>,
        bool: IntoScalar<T>,
        i64: IntoScalar<T>
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
        dilation: [i64; 2]
    ) -> Result<_Tensor<T>, TensorError> {
        let kernel_size: T = kernels_shape.size().into_scalar();
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
            |a| a._div(kernel_size_vec)
        )
    }
}

impl<T> Tensor<T>
    where
        T: CommonBounds + IntoScalar<T> + NormalOut<Output = T> + FloatOutBinary<T, Output = T>,
        T::Vec: VecTrait<T> +
            Copy +
            Send +
            Sync +
            NormalOut<Output = T::Vec> +
            FloatOutBinary<T::Vec, Output = T::Vec>,
        bool: IntoScalar<T>,
        i64: IntoScalar<T>
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
    #[inline(never)]
    pub fn avgpool2d(
        &self,
        kernels_shape: &Shape,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2]
    ) -> Result<Tensor<T>, TensorError> {
        Ok(self.inner.avgpool2d(&kernels_shape, steps, padding, dilation)?.into())
    }
}
