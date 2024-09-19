use std::borrow::Borrow;

use anyhow::Result;
use tensor_types::{dtype::TypeCommon, type_promote::NormalOut};

use crate::tensor::CommonBounds;

pub trait FloatUaryOps {
    type Output;
    type InplaceOutput;
    type OutputMeta: Send;
    /// Computes the element-wise sine of the tensor.
    ///
    /// The `sin` function calculates the sine of each element in the tensor, returning a new tensor
    /// containing the results. If the `simd` feature is enabled, it utilizes SIMD instructions to
    /// perform the computation more efficiently.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: The sine is computed for each element individually.
    /// - **Data Type Conversion**: The input tensor's data type `T` is converted to a floating-point type suitable for the sine operation.
    ///
    /// # See Also
    ///
    /// - [`cos`]: Computes the element-wise cosine of the tensor.
    /// - [`tan`]: Computes the element-wise tangent of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn sin(&self) -> Result<Self::Output>;

    /// Computes the element-wise cos of the tensor.
    ///
    /// The `sin` function calculates the cos of each element in the tensor, returning a new tensor
    /// containing the results. If the `simd` feature is enabled, it utilizes SIMD instructions to
    /// perform the computation more efficiently.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: The cos is computed for each element individually.
    /// - **Data Type Conversion**: The input tensor's data type `T` is converted to a floating-point type suitable for the cos operation.
    ///
    /// # See Also
    ///
    /// - [`cos`]: Computes the element-wise cosine of the tensor.
    /// - [`tan`]: Computes the element-wise tangent of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn cos(&self) -> Result<Self::Output>;

    /// Computes the element-wise tan of the tensor.
    ///
    /// The `tan` function calculates the tan of each element in the tensor, returning a new tensor
    /// containing the results. If the `simd` feature is enabled, it utilizes SIMD instructions to
    /// perform the computation more efficiently.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: The tan is computed for each element individually.
    /// - **Data Type Conversion**: The input tensor's data type `T` is converted to a floating-point type suitable for the tan operation.
    ///
    /// # See Also
    ///
    /// - [`cos`]: Computes the element-wise cosine of the tensor.
    /// - [`tan`]: Computes the element-wise tangent of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn tan(&self) -> Result<Self::Output>;

    /// Computes the element-wise asin of the tensor.
    ///
    /// The `asin` function calculates the asin of each element in the tensor, returning a new tensor
    /// containing the results. If the `simd` feature is enabled, it utilizes SIMD instructions to
    /// perform the computation more efficiently.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: The asin is computed for each element individually.
    /// - **Data Type Conversion**: The input tensor's data type `T` is converted to a floating-point type suitable for the asin operation.
    ///
    /// # See Also
    ///
    /// - [`cos`]: Computes the element-wise cosine of the tensor.
    /// - [`tan`]: Computes the element-wise tangent of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn asin(&self) -> Result<Self::Output>;

    /// Computes the element-wise acos of the tensor.
    ///
    /// The `acos` function calculates the acos of each element in the tensor, returning a new tensor
    /// containing the results. If the `simd` feature is enabled, it utilizes SIMD instructions to
    /// perform the computation more efficiently.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: The acos is computed for each element individually.
    /// - **Data Type Conversion**: The input tensor's data type `T` is converted to a floating-point type suitable for the acos operation.
    ///
    /// # See Also
    ///
    /// - [`cos`]: Computes the element-wise cosine of the tensor.
    /// - [`tan`]: Computes the element-wise tangent of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn acos(&self) -> Result<Self::Output>;

    /// Computes the element-wise atan of the tensor.
    ///
    /// The `atan` function calculates the atan of each element in the tensor, returning a new tensor
    /// containing the results. If the `simd` feature is enabled, it utilizes SIMD instructions to
    /// perform the computation more efficiently.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: The atan is computed for each element individually.
    /// - **Data Type Conversion**: The input tensor's data type `T` is converted to a floating-point type suitable for the atan operation.
    ///
    /// # See Also
    ///
    /// - [`cos`]: Computes the element-wise cosine of the tensor.
    /// - [`tan`]: Computes the element-wise tangent of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn atan(&self) -> Result<Self::Output>;

    /// Computes the element-wise sinh of the tensor.
    ///
    /// The `sinh` function calculates the sinh of each element in the tensor, returning a new tensor
    /// containing the results. If the `simd` feature is enabled, it utilizes SIMD instructions to
    /// perform the computation more efficiently.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: The sinh is computed for each element individually.
    /// - **Data Type Conversion**: The input tensor's data type `T` is converted to a floating-point type suitable for the sinh operation.
    ///
    /// # See Also
    ///
    /// - [`cos`]: Computes the element-wise cosine of the tensor.
    /// - [`tan`]: Computes the element-wise tangent of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn sinh(&self) -> Result<Self::Output>;

    /// Computes the element-wise cosh of the tensor.
    ///
    /// The `cosh` function calculates the cosh of each element in the tensor, returning a new tensor
    /// containing the results. If the `simd` feature is enabled, it utilizes SIMD instructions to
    /// perform the computation more efficiently.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: The cosh is computed for each element individually.
    /// - **Data Type Conversion**: The input tensor's data type `T` is converted to a floating-point type suitable for the cosh operation.
    ///
    /// # See Also
    ///
    /// - [`cos`]: Computes the element-wise cosine of the tensor.
    /// - [`tan`]: Computes the element-wise tangent of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn cosh(&self) -> Result<Self::Output>;

    /// Computes the element-wise tanh of the tensor.
    ///
    /// The `tanh` function calculates the tanh of each element in the tensor, returning a new tensor
    /// containing the results. If the `simd` feature is enabled, it utilizes SIMD instructions to
    /// perform the computation more efficiently.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: The tanh is computed for each element individually.
    /// - **Data Type Conversion**: The input tensor's data type `T` is converted to a floating-point type suitable for the tanh operation.
    ///
    /// # See Also
    ///
    /// - [`cos`]: Computes the element-wise cosine of the tensor.
    /// - [`tan`]: Computes the element-wise tangent of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn tanh(&self) -> Result<Self::Output>;

    /// Computes the element-wise asinh of the tensor.
    ///
    /// The `asinh` function calculates the asinh of each element in the tensor, returning a new tensor
    /// containing the results. If the `simd` feature is enabled, it utilizes SIMD instructions to
    /// perform the computation more efficiently.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: The asinh is computed for each element individually.
    /// - **Data Type Conversion**: The input tensor's data type `T` is converted to a floating-point type suitable for the asinh operation.
    ///
    /// # See Also
    ///
    /// - [`cos`]: Computes the element-wise cosine of the tensor.
    /// - [`tan`]: Computes the element-wise tangent of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn asinh(&self) -> Result<Self::Output>;

    /// Computes the element-wise acosh of the tensor.
    ///
    /// The `acosh` function calculates the acosh of each element in the tensor, returning a new tensor
    /// containing the results. If the `simd` feature is enabled, it utilizes SIMD instructions to
    /// perform the computation more efficiently.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: The acosh is computed for each element individually.
    /// - **Data Type Conversion**: The input tensor's data type `T` is converted to a floating-point type suitable for the acosh operation.
    ///
    /// # See Also
    ///
    /// - [`cos`]: Computes the element-wise cosine of the tensor.
    /// - [`tan`]: Computes the element-wise tangent of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn acosh(&self) -> Result<Self::Output>;

    /// Computes the element-wise atanh of the tensor.
    ///
    /// The `atanh` function calculates the atanh of each element in the tensor, returning a new tensor
    /// containing the results. If the `simd` feature is enabled, it utilizes SIMD instructions to
    /// perform the computation more efficiently.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: The atanh is computed for each element individually.
    /// - **Data Type Conversion**: The input tensor's data type `T` is converted to a floating-point type suitable for the atanh operation.
    ///
    /// # See Also
    ///
    /// - [`cos`]: Computes the element-wise cosine of the tensor.
    /// - [`tan`]: Computes the element-wise tangent of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn atanh(&self) -> Result<Self::Output>;

    /// Inplace Version of sin.
    ///
    /// # See Also
    ///
    /// - [`sin`]: Computes the element-wise sine of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn sin_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Inplace Version of cos.
    ///
    /// # See Also
    ///
    /// - [`cos`]: Computes the element-wise cosine of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn cos_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Inplace Version of tan.
    ///
    /// # See Also
    ///
    /// - [`tan`]: Computes the element-wise tangent of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn tan_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Inplace Version of asin.
    ///
    /// # See Also
    ///
    /// - [`asin`]: Computes the element-wise asin of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn asin_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Inplace Version of acos.
    ///
    /// # See Also
    ///
    /// - [`acos`]: Computes the element-wise acos of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn acos_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Inplace Version of atan.
    ///
    /// # See Also
    ///
    /// - [`atan`]: Computes the element-wise atan of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn atan_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Inplace Version of sinh.
    ///
    /// # See Also
    ///
    /// - [`sinh`]: Computes the element-wise sinh of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn sinh_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Inplace Version of cosh.
    ///
    /// # See Also
    ///
    /// - [`cosh`]: Computes the element-wise cosh of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn cosh_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Inplace Version of tanh.
    ///
    /// # See Also
    ///
    /// - [`tanh`]: Computes the element-wise tanh of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn tanh_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Inplace Version of asinh.
    ///
    /// # See Also
    ///
    /// - [`asinh`]: Computes the element-wise asinh of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn asinh_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Inplace Version of acosh.
    ///
    /// # See Also
    ///
    /// - [`acosh`]: Computes the element-wise acosh of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn acosh_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Inplace Version of atanh.
    ///
    /// # See Also
    ///
    /// - [`atanh`]: Computes the element-wise atanh of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn atanh_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    #[cfg_attr(feature = "track_caller", track_caller)]
    fn exp(&self) -> Result<Self::Output>;

    /// Computes the element-wise exponential of the tensor.
    ///
    /// The `exp` function calculates the exponential (e<sup>x</sup>) of each element in the tensor, returning a new tensor
    /// containing the results. If the `simd` feature is enabled, it utilizes SIMD instructions to
    /// perform the computation more efficiently.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system.
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: The exponential is computed for each element individually.
    /// - **Data Type Conversion**: The input tensor's data type `T` is converted to a floating-point type suitable for the exponential operation.
    ///
    /// # See Also
    ///
    /// - [`log`]: Computes the element-wise natural logarithm of the tensor.
    /// - [`pow`]: Raises each element of the tensor to a specified power.

    #[cfg_attr(feature = "track_caller", track_caller)]
    fn exp_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise base-2 exponential of the tensor.
    ///
    /// The `exp2` function calculates 2<sup>x</sup> for each element in the tensor, returning a new tensor
    /// containing the results. If the `simd` feature is enabled, it utilizes SIMD instructions to
    /// perform the computation more efficiently.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system.
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: The base-2 exponential is computed for each element individually.
    /// - **Data Type Conversion**: The input tensor's data type `T` is converted to a floating-point type suitable for the base-2 exponential operation.
    ///
    /// # See Also
    ///
    /// - [`exp`]: Computes the element-wise natural exponential of the tensor.
    /// - [`log2`]: Computes the element-wise base-2 logarithm of the tensor.
    /// - [`pow`]: Raises each element of the tensor to a specified power.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn exp2(&self) -> Result<Self::Output>;

    /// in-place version of exp2
    ///
    /// # See Also
    ///
    /// - [`exp2`]: Computes the element-wise base-2 exponential of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn exp2_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise square root of the tensor.
    ///
    /// The `sqrt` function calculates the square root of each element in the tensor, returning a new tensor
    /// containing the results. If the `simd` feature is enabled, it utilizes SIMD instructions to
    /// perform the computation more efficiently.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system.
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: The square root is computed for each element individually.
    /// - **Data Type Conversion**: The input tensor's data type `T` is converted to a floating-point type suitable for the square root operation.
    ///
    /// # See Also
    ///
    /// - [`pow`]: Raises each element of the tensor to a specified power.
    /// - [`exp`]: Computes the element-wise natural exponential of the tensor.
    /// - [`log`]: Computes the element-wise natural logarithm of the tensor.

    #[cfg_attr(feature = "track_caller", track_caller)]
    fn sqrt(&self) -> Result<Self::Output>;

    /// Inplace Version of sqrt.
    ///
    /// # See Also
    ///
    /// - [`sqrt`]: Computes the element-wise square root of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn sqrt_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise reciprocal of the tensor.
    ///
    /// The `recip` function calculates the reciprocal (1/x) of each element in the tensor, returning a new tensor
    /// containing the results. If the `simd` feature is enabled, it utilizes SIMD instructions to
    /// perform the computation more efficiently.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system.
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: The reciprocal is computed for each element individually.
    /// - **Data Type Conversion**: The input tensor's data type `T` is converted to a floating-point type suitable for the reciprocal operation.
    /// - **Error Handling**: If an element is zero, computing its reciprocal will result in an error.
    ///
    /// # See Also
    ///
    /// - [`mul`]: Computes the element-wise multiplication of the tensor.
    /// - [`div`]: Computes the element-wise division of the tensor by another tensor or scalar.
    /// - [`pow`]: Raises each element of the tensor to a specified power.

    #[cfg_attr(feature = "track_caller", track_caller)]
    fn recip(&self) -> Result<Self::Output>;

    /// Inplace Version of recip.
    ///
    /// # See Also
    ///
    /// - [`recip`]: Computes the element-wise reciprocal of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn recip_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise natural logarithm (ln) of the tensor.
    ///
    /// The `ln` function calculates the natural logarithm (base *e*) of each element in the tensor, returning a new tensor
    /// containing the results. If the `simd` feature is enabled, it utilizes SIMD instructions to
    /// perform the computation more efficiently.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system.
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: The natural logarithm is computed for each element individually.
    /// - **Data Type Conversion**: The input tensor's data type `T` is converted to a floating-point type suitable for the natural logarithm operation.
    /// - **Error Handling**: The logarithm is only defined for positive numbers; attempting to compute the logarithm of zero or negative numbers will result in an error.
    ///
    /// # See Also
    ///
    /// - [`log2`]: Computes the element-wise base-2 logarithm of the tensor.
    /// - [`log10`]: Computes the element-wise base-10 logarithm of the tensor.
    /// - [`exp`]: Computes the element-wise natural exponential of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn ln(&self) -> Result<Self::Output>;

    /// Inplace Version of ln.
    ///
    /// # See Also
    ///
    /// - [`ln`]: Computes the element-wise natural logarithm of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn ln_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise base-2 logarithm (log2) of the tensor.
    ///
    /// The `log2` function calculates the base-2 logarithm of each element in the tensor, returning a new tensor
    /// containing the results. If the `simd` feature is enabled, it utilizes SIMD instructions to
    /// perform the computation more efficiently.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system.
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: The base-2 logarithm is computed for each element individually.
    /// - **Data Type Conversion**: The input tensor's data type `T` is converted to a floating-point type suitable for the base-2 logarithm operation.
    /// - **Error Handling**: The logarithm is only defined for positive numbers; attempting to compute the logarithm of zero or negative numbers will result in an error.
    ///
    /// # See Also
    ///
    /// - [`ln`]: Computes the element-wise natural logarithm (base *e*) of the tensor.
    /// - [`log10`]: Computes the element-wise base-10 logarithm of the tensor.
    /// - [`exp2`]: Computes the element-wise base-2 exponential of the tensor.

    #[cfg_attr(feature = "track_caller", track_caller)]
    fn log2(&self) -> Result<Self::Output>;

    /// Inplace Version of log2.
    ///
    /// # See Also
    ///
    /// - [`log2`]: Computes the element-wise base-2 logarithm of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn log2_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise base-10 logarithm (log10) of the tensor.
    ///
    /// The `log10` function calculates the base-10 logarithm of each element in the tensor, returning a new tensor
    /// containing the results. If the `simd` feature is enabled, it utilizes SIMD instructions to
    /// perform the computation more efficiently.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system.
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: The base-10 logarithm is computed for each element individually.
    /// - **Data Type Conversion**: The input tensor's data type `T` is converted to a floating-point type suitable for the base-10 logarithm operation.
    /// - **Error Handling**: The logarithm is only defined for positive numbers; attempting to compute the logarithm of zero or negative numbers will result in an error.
    ///
    /// # See Also
    ///
    /// - [`ln`]: Computes the element-wise natural logarithm (base *e*) of the tensor.
    /// - [`log2`]: Computes the element-wise base-2 logarithm of the tensor.
    /// - [`exp`]: Computes the element-wise natural exponential of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn log10(&self) -> Result<Self::Output>;

    /// Inplace Version of log10.
    ///
    /// # See Also
    ///
    /// - [`log10`]: Computes the element-wise base-10 logarithm of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn log10_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise Continuously Differentiable Exponential Linear Unit (CELU) of the tensor.
    ///
    /// The `celu` function applies the CELU activation function to each element in the tensor. This is defined as:
    /// - `x` if `x > 0`
    /// - `alpha * (exp(x / alpha) - 1)` if `x <= 0`
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system.
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: CELU is computed for each element individually.
    /// - **Parameter**: `alpha` controls the saturation point of the negative inputs. It must be greater than zero.
    ///
    /// # See Also
    ///
    /// - [`elu`]: Computes the element-wise Exponential Linear Unit (ELU).
    /// - [`relu`]: Computes the element-wise Rectified Linear Unit (ReLU).
    /// - [`sigmoid`]: Computes the element-wise sigmoid function of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn celu(&self, alpha: Self::OutputMeta) -> Result<Self::Output>;

    /// Inplace Version of celu.
    ///
    /// # See Also
    ///
    /// - [`celu`]: Computes the element-wise Continuously Differentiable Exponential Linear Unit (CELU) of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn celu_<U>(&self, alpha: Self::OutputMeta, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise sigmoid function of the tensor.
    ///
    /// The `sigmoid` function calculates the sigmoid of each element in the tensor, defined as:
    /// `1 / (1 + exp(-x))`, returning a new tensor containing the results. If the `simd` feature is enabled,
    /// it utilizes SIMD instructions to perform the computation more efficiently.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system.
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: The sigmoid function is computed for each element individually.
    ///
    /// # See Also
    ///
    /// - [`tanh`]: Computes the element-wise hyperbolic tangent of the tensor.
    /// - [`relu`]: Computes the element-wise Rectified Linear Unit (ReLU).
    /// - [`celu`]: Computes the element-wise Continuously Differentiable Exponential Linear Unit (CELU).
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn sigmoid(&self) -> Result<Self::Output>;

    /// Inplace Version of sigmoid.
    ///
    /// # See Also
    ///
    /// - [`sigmoid`]: Computes the element-wise sigmoid function of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn sigmoid_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise Exponential Linear Unit (ELU) of the tensor.
    ///
    /// The `elu` function applies the ELU activation function to each element in the tensor. This is defined as:
    /// - `x` if `x > 0`
    /// - `alpha * (exp(x) - 1)` if `x <= 0`
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system.
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: ELU is computed for each element individually.
    /// - **Parameter**: `alpha` controls the saturation point of the negative inputs. It must be greater than zero.
    ///
    /// # See Also
    ///
    /// - [`celu`]: Computes the element-wise Continuously Differentiable Exponential Linear Unit (CELU).
    /// - [`relu`]: Computes the element-wise Rectified Linear Unit (ReLU).
    /// - [`sigmoid`]: Computes the element-wise sigmoid function of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn elu(&self, alpha: Self::OutputMeta) -> Result<Self::Output>;

    /// Inplace Version of elu.
    ///
    /// # See Also
    ///
    /// - [`elu`]: Computes the element-wise Exponential Linear Unit (ELU) of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn elu_<U>(&self, alpha: Self::OutputMeta, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    fn relu(&self) -> Result<Self::Output>;
    fn relu_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    fn erf(&self) -> Result<Self::Output>;

    fn fast_hard_sigmoid(&self) -> Result<Self::Output>;

    /// Computes the element-wise Leaky Rectified Linear Unit (Leaky ReLU) of the tensor.
    ///
    /// The `leaky_relu` function applies the Leaky ReLU activation function to each element in the tensor. This is defined as:
    /// - `x` if `x > 0`
    /// - `alpha * x` if `x <= 0`
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system.
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: Leaky ReLU is computed for each element individually.
    /// - **Parameter**: `alpha` controls the slope for negative inputs. It must be greater than zero.
    ///
    /// # See Also
    ///
    /// - [`relu`]: Computes the element-wise Rectified Linear Unit (ReLU).
    /// - [`elu`]: Computes the element-wise Exponential Linear Unit (ELU).
    /// - [`mish`]: Computes the element-wise mish activation function of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn leaky_relu(&self, alpha: Self::OutputMeta) -> Result<Self::Output>;

    /// Inplace Version of leaky_relu.
    ///
    /// # See Also
    ///
    /// - [`leaky_relu`]: Computes the element-wise Leaky Rectified Linear Unit (Leaky ReLU) of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn leaky_relu_<U>(&self, alpha: Self::OutputMeta, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise Gaussian Error Linear Unit (GELU) of the tensor.
    ///
    /// The `gelu` function applies the GELU activation function to each element in the tensor, defined as:
    /// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system.
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: GELU is computed for each element individually.
    ///
    /// # See Also
    ///
    /// - [`relu`]: Computes the element-wise Rectified Linear Unit (ReLU).
    /// - [`mish`]: Computes the element-wise mish activation function of the tensor.
    /// - [`sigmoid`]: Computes the element-wise sigmoid function of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn gelu(&self) -> Result<Self::Output>;

    /// Inplace Version of gelu.
    ///
    /// # See Also
    ///
    /// - [`gelu`]: Computes the element-wise Gaussian Error Linear Unit (GELU) of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn gelu_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise Scaled Exponential Linear Unit (SELU) of the tensor.
    ///
    /// The `selu` function applies the SELU activation function to each element in the tensor, defined as:
    /// - `scale * x` if `x > 0`
    /// - `scale * alpha * (exp(x) - 1)` if `x <= 0`
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system.
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: SELU is computed for each element individually.
    /// - **Parameters**: `scale` and `alpha` are predefined constants for SELU activation.
    ///
    /// # See Also
    ///
    /// - [`celu`]: Computes the element-wise Continuously Differentiable Exponential Linear Unit (CELU).
    /// - [`elu`]: Computes the element-wise Exponential Linear Unit (ELU).
    /// - [`relu`]: Computes the element-wise Rectified Linear Unit (ReLU).
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn selu(
        &self,
        alpha: Option<Self::OutputMeta>,
        gamma: Option<Self::OutputMeta>,
    ) -> Result<Self::Output>;

    /// Inplace Version of selu.
    ///
    /// # See Also
    ///
    /// - [`selu`]: Computes the element-wise Scaled Exponential Linear Unit (SELU) of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn selu_<U>(
        &self,
        alpha: Option<Self::OutputMeta>,
        gamma: Option<Self::OutputMeta>,
        out: U,
    ) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise hard sigmoid of the tensor.
    ///
    /// The `hard_sigmoid` function applies a piecewise linear approximation of the sigmoid function to each element in the tensor:
    /// - `0` if `x < -2.5`
    /// - `1` if `x > 2.5`
    /// - `0.2 * x + 0.5` otherwise
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system.
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: Hard sigmoid is computed for each element individually.
    ///
    /// # See Also
    ///
    /// - [`sigmoid`]: Computes the element-wise sigmoid function of the tensor.
    /// - [`hard_swish`]: Computes the element-wise hard swish activation function of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn hard_sigmoid(&self) -> Result<Self::Output>;

    /// Inplace Version of hard_sigmoid.
    ///
    /// # See Also
    ///
    /// - [`hard_sigmoid`]: Computes the element-wise hard sigmoid of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn hard_sigmoid_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise hard swish of the tensor.
    ///
    /// The `hard_swish` function applies a piecewise approximation of the swish activation function to each element in the tensor:
    /// - `x * (0.2 * x + 0.5)` if `-2.5 <= x <= 2.5`
    /// - `0` if `x < -2.5`
    /// - `x` if `x > 2.5`
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system.
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: Hard swish is computed for each element individually.
    ///
    /// # See Also
    ///
    /// - [`swish`]: Computes the element-wise swish activation function.
    /// - [`relu`]: Computes the element-wise Rectified Linear Unit (ReLU).
    /// - [`hard_sigmoid`]: Computes the element-wise hard sigmoid of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn hard_swish(&self) -> Result<Self::Output>;

    /// Inplace Version of hard_swish.
    ///
    /// # See Also
    ///
    /// - [`hard_swish`]: Computes the element-wise hard swish activation function of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn hard_swish_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;
    /// Computes the element-wise ReLU6 activation function of the tensor.
    ///
    /// The `relu6` function applies the ReLU6 activation function, which is a modified ReLU capped at 6:
    /// - `min(max(0, x), 6)`
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system.
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: ReLU6 is computed for each element individually.
    ///
    /// # See Also
    ///
    /// - [`relu`]: Computes the element-wise Rectified Linear Unit (ReLU).
    /// - [`leaky_relu`]: Computes the element-wise Leaky Rectified Linear Unit (Leaky ReLU).
    /// - [`relu6`]: Computes the element-wise ReLU6 activation.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn relu6(&self) -> Result<Self::Output>;

    /// Inplace Version of relu6.
    ///
    /// # See Also
    ///
    /// - [`relu6`]: Computes the element-wise ReLU6 activation function of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn relu6_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise softplus of the tensor.
    ///
    /// The `softplus` function applies the softplus activation function to each element in the tensor, defined as:
    /// `log(1 + exp(x))`
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system.
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: Softplus is computed for each element individually.
    ///
    /// # See Also
    ///
    /// - [`softsign`]: Computes the element-wise softsign activation function of the tensor.
    /// - [`relu`]: Computes the element-wise Rectified Linear Unit (ReLU).
    /// - [`sigmoid`]: Computes the element-wise sigmoid function of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn softplus(&self) -> Result<Self::Output>;

    /// Inplace Version of softplus.
    ///
    /// # See Also
    ///
    /// - [`softplus`]: Computes the element-wise softplus of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn softplus_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise softsign of the tensor.
    ///
    /// The `softsign` function applies the softsign activation function to each element in the tensor, defined as:
    /// `x / (1 + |x|)`
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system.
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: Softsign is computed for each element individually.
    ///
    /// # See Also
    ///
    /// - [`softplus`]: Computes the element-wise softplus activation function of the tensor.
    /// - [`sigmoid`]: Computes the element-wise sigmoid function of the tensor.
    /// - [`tanh`]: Computes the element-wise hyperbolic tangent of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn softsign(&self) -> Result<Self::Output>;

    /// Inplace Version of softsign.
    ///
    /// # See Also
    ///
    /// - [`softsign`]: Computes the element-wise softsign of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn softsign_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise mish activation function of the tensor.
    ///
    /// The `mish` function applies the mish activation function to each element in the tensor, defined as:
    /// `x * tanh(softplus(x))`
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system.
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: Mish is computed for each element individually.
    ///
    /// # See Also
    ///
    /// - [`swish`]: Computes the element-wise swish activation function.
    /// - [`relu`]: Computes the element-wise Rectified Linear Unit (ReLU).
    /// - [`gelu`]: Computes the element-wise Gaussian Error Linear Unit (GELU).
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn mish(&self) -> Result<Self::Output>;

    /// Inplace Version of mish.
    ///
    /// # See Also
    ///
    /// - [`mish`]: Computes the element-wise mish activation function of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn mish_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;
}

pub trait NormalUaryOps
where
    Self: Sized,
{
    type Output;
    type InplaceOutput;
    type OutputMeta;

    #[cfg_attr(feature = "track_caller", track_caller)]
    fn floor(&self) -> Result<Self::Output>;

    #[cfg_attr(feature = "track_caller", track_caller)]
    fn floor_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise square of the tensor.
    ///
    /// The `square` function squares each element in the tensor, returning a new tensor
    /// containing the results.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system.
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: The square is computed for each element individually.
    ///
    /// # See Also
    ///
    /// - [`sqrt`]: Computes the element-wise square root of the tensor.
    /// - [`pow`]: Raises each element of the tensor to a specified power.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn square(&self) -> Result<Self::Output>;

    /// Inplace Version of square.
    ///
    /// # See Also
    ///
    /// - [`square`]: Computes the element-wise square of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn square_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise absolute value of the tensor.
    ///
    /// The `abs` function returns the absolute value of each element in the tensor, returning a new tensor
    /// containing the results.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system.
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: The absolute value is computed for each element individually.
    ///
    /// # See Also
    ///
    /// - [`sign`]: Computes the element-wise sign of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn abs(&self) -> Result<Self>;

    /// Inplace Version of abs.
    ///
    /// # See Also
    ///
    /// - [`abs`]: Computes the element-wise absolute value of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn abs_<U>(&self, out: U) -> Result<Self>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise ceiling (rounding up) of the tensor.
    ///
    /// The `ceil` function rounds each element in the tensor up to the nearest integer, returning a new tensor
    /// containing the results.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system.
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: The ceiling is computed for each element individually.
    ///
    /// # See Also
    ///
    /// - [`floor`]: Computes the element-wise floor (rounding down) of the tensor.
    /// - [`round`]: Rounds each element in the tensor to the nearest integer.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn ceil(&self) -> Result<Self::Output>;

    /// Inplace Version of ceil.
    ///
    /// # See Also
    ///
    /// - [`ceil`]: Computes the element-wise ceiling (rounding up) of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn ceil_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;
    /// Computes the element-wise sign of the tensor.
    ///
    /// The `sign` function returns the sign of each element in the tensor:
    /// - `1` if the element is positive
    /// - `-1` if the element is negative
    /// - `0` if the element is zero
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system.
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: The sign is computed for each element individually.
    ///
    /// # See Also
    ///
    /// - [`abs`]: Computes the element-wise absolute value of the tensor.
    /// - [`neg`]: Negates each element in the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn sign(&self) -> Result<Self::Output>;

    /// Inplace Version of sign.
    ///
    /// # See Also
    ///
    /// - [`sign`]: Computes the element-wise sign of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn sign_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Clips (limits) the values of the tensor to a specified range.
    ///
    /// The `clip` function limits each element in the tensor to lie within the specified range [`min`, `max`].
    /// Any values lower than `min` are replaced by `min`, and any values higher than `max` are replaced by `max`.
    ///
    /// # Parameters
    ///
    /// - `min`: The lower bound of the range.
    /// - `max`: The upper bound of the range.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system.
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: Clipping is applied to each element individually.
    ///
    /// # See Also
    ///
    /// - [`max`]: Returns the element-wise maximum of two tensors.
    /// - [`min`]: Returns the element-wise minimum of two tensors.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn clip(&self, min: Self::OutputMeta, max: Self::OutputMeta) -> Result<Self::Output>;

    /// Inplace Version of clip.
    ///
    /// # See Also
    ///
    /// - [`clip`]: Clips (limits) the values of the tensor to a specified range.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn clip_<U>(
        &self,
        min: Self::OutputMeta,
        max: Self::OutputMeta,
        out: U,
    ) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;

    /// Rounds each element in the tensor to the nearest integer.
    ///
    /// The `round` function rounds each element in the tensor to the nearest integer, returning a new tensor
    /// containing the results. If the element is halfway between two integers, it rounds away from zero.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system.
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: Rounding is computed for each element individually.
    ///
    /// # See Also
    ///
    /// - [`ceil`]: Rounds each element in the tensor up to the nearest integer.
    /// - [`floor`]: Rounds each element in the tensor down to the nearest integer.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn round(&self) -> Result<Self::Output>;

    /// Inplace Version of round.
    ///
    /// # See Also
    ///
    /// - [`round`]: Rounds each element in the tensor to the nearest integer.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn round_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;
}

pub trait Cum
where
    Self: Sized,
    <<Self as Cum>::Meta as TypeCommon>::Vec: Send + Sync,
{
    type Meta: CommonBounds;

    /// Computes the cumulative sum of the elements in the tensor along a specified axis.
    ///
    /// The `cumsum` function computes the cumulative sum of the elements in the tensor along a specified axis,
    /// returning a new tensor containing the results. Each element in the result is the sum of all preceding
    /// elements along the axis.
    ///
    /// # Parameters
    ///
    /// - `axis`: The axis along which to compute the cumulative sum.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system.
    ///
    /// # Notes
    ///
    /// - **Cumulative Operation**: The sum is computed cumulatively along the specified axis.
    /// - **Axis Specification**: The cumulative sum can be computed along any valid axis of the tensor.
    ///
    /// # See Also
    ///
    /// - [`cumprod`]: Computes the cumulative product of the elements in the tensor.
    /// - [`sum`]: Computes the sum of all elements in the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn cumsum(&self, axis: Option<i64>) -> Result<Self>
    where
        Self::Meta: NormalOut<Self::Meta, Output = Self::Meta>;

    /// Computes the cumulative product of the elements in the tensor along a specified axis.
    ///
    /// The `cumprod` function computes the cumulative product of the elements in the tensor along a specified axis,
    /// returning a new tensor containing the results. Each element in the result is the product of all preceding
    /// elements along the axis.
    ///
    /// # Parameters
    ///
    /// - `axis`: The axis along which to compute the cumulative product.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system.
    ///
    /// # Notes
    ///
    /// - **Cumulative Operation**: The product is computed cumulatively along the specified axis.
    /// - **Axis Specification**: The cumulative product can be computed along any valid axis of the tensor.
    ///
    /// # See Also
    ///
    /// - [`cumsum`]: Computes the cumulative sum of the elements in the tensor.
    /// - [`prod`]: Computes the product of all elements in the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn cumprod(&self, axis: Option<i64>) -> Result<Self>
    where
        Self::Meta: NormalOut<Self::Meta, Output = Self::Meta>;
}

pub trait Neg {
    type Output;
    type InplaceOutput;
    type OutputMeta;

    /// Computes the element-wise negation (multiplying by -1) of the tensor.
    ///
    /// The `neg` function negates each element in the tensor, returning a new tensor containing the results.
    ///
    /// # Returns
    ///
    /// - `anyhow::Result<_Tensor<FloatUnaryType<T>>>`: A floating type based on type promote system.
    ///
    /// # Notes
    ///
    /// - **Element-wise Operation**: The negation is applied to each element individually.
    ///
    /// # See Also
    ///
    /// - [`abs`]: Computes the element-wise absolute value of the tensor.
    /// - [`sign`]: Computes the element-wise sign of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn neg(&self) -> Result<Self::Output>;

    /// Inplace Version of neg.
    ///
    /// # See Also
    ///
    /// - [`neg`]: Computes the element-wise negation of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn neg_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>;
}
