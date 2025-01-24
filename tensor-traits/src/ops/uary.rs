use std::borrow::Borrow;
use tensor_common::error::base::TensorError;
use tensor_types::{ dtype::TypeCommon, type_promote::NormalOut };

use crate::tensor::CommonBounds;

/// A trait for tensor unary operations, the output must be a floating point tensor
pub trait FloatUnaryOps {
    /// output tensor type
    type Output;
    /// output tensor type for inplace operation
    type InplaceOutput;
    /// output tensor data type
    type OutputMeta: Send;
    /// Computes the element-wise sine of the tensor.
    ///
    /// This function calculates the sine of each element in the tensor, returning a new tensor
    /// where each element is the sine of the corresponding element in the original tensor.
    /// The sine function is defined as:
    /// ```text
    /// sin(x)
    /// ```
    /// # Arguments
    /// This function takes no arguments.
    /// # Returns
    /// * A new tensor where each element is the sine of the corresponding element in the original tensor.
    /// # Panics
    /// * This function may panic if the tensor contains invalid values for the sine function, such as `NaN` values.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn sin(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Computes the element-wise cos of the tensor.
    ///
    /// This function calculates the cos of each element in the tensor, returning a new tensor
    /// where each element is the cos of the corresponding element in the original tensor.
    /// The cos function is defined as:
    /// ```text
    /// cos(x)
    /// ```
    /// # Arguments
    /// This function takes no arguments.
    /// # Returns
    /// * A new tensor where each element is the cos of the corresponding element in the original tensor.
    /// # Panics
    /// * This function may panic if the tensor contains invalid values for the cos function, such as `NaN` values.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn cos(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Computes the element-wise tan of the tensor.
    ///
    /// This function calculates the tan of each element in the tensor, returning a new tensor
    /// where each element is the tan of the corresponding element in the original tensor.
    /// The tan function is defined as:
    /// ```text
    /// tan(x)
    /// ```
    /// # Arguments
    /// This function takes no arguments.
    /// # Returns
    /// * A new tensor where each element is the tan of the corresponding element in the original tensor.
    /// # Panics
    /// * This function may panic if the tensor contains invalid values for the tan function, such as `NaN` values.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn tan(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Computes the element-wise asin of the tensor.
    ///
    /// This function calculates the asin of each element in the tensor, returning a new tensor
    /// where each element is the asin of the corresponding element in the original tensor.
    /// The asin function is defined as:
    /// ```text
    /// asin(x)
    /// ```
    /// # Arguments
    /// This function takes no arguments.
    /// # Returns
    /// * A new tensor where each element is the asin of the corresponding element in the original tensor.
    /// # Panics
    /// * This function may panic if the tensor contains invalid values for the asin function, such as `NaN` values.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn asin(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Computes the element-wise acos of the tensor.
    ///
    /// This function calculates the acos of each element in the tensor, returning a new tensor
    /// where each element is the acos of the corresponding element in the original tensor.
    /// The acos function is defined as:
    /// ```text
    /// acos(x)
    /// ```
    /// # Arguments
    /// This function takes no arguments.
    /// # Returns
    /// * A new tensor where each element is the acos of the corresponding element in the original tensor.
    /// # Panics
    /// * This function may panic if the tensor contains invalid values for the acos function, such as `NaN` values.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn acos(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Computes the element-wise atan of the tensor.
    ///
    /// This function calculates the atan of each element in the tensor, returning a new tensor
    /// where each element is the atan of the corresponding element in the original tensor.
    /// The atan function is defined as:
    /// ```text
    /// atan(x)
    /// ```
    /// # Arguments
    /// This function takes no arguments.
    /// # Returns
    /// * A new tensor where each element is the atan of the corresponding element in the original tensor.
    /// # Panics
    /// * This function may panic if the tensor contains invalid values for the atan function, such as `NaN` values.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn atan(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Computes the element-wise sinh of the tensor.
    ///
    /// This function calculates the sinh of each element in the tensor, returning a new tensor
    /// where each element is the sinh of the corresponding element in the original tensor.
    /// The sinh function is defined as:
    /// ```text
    /// sinh(x)
    /// ```
    /// # Arguments
    /// This function takes no arguments.
    /// # Returns
    /// * A new tensor where each element is the sinh of the corresponding element in the original tensor.
    /// # Panics
    /// * This function may panic if the tensor contains invalid values for the sinh function, such as `NaN` values.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn sinh(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Computes the element-wise cosh of the tensor.
    ///
    /// This function calculates the cosh of each element in the tensor, returning a new tensor
    /// where each element is the cosh of the corresponding element in the original tensor.
    /// The cosh function is defined as:
    /// ```text
    /// cosh(x)
    /// ```
    /// # Arguments
    /// This function takes no arguments.
    /// # Returns
    /// * A new tensor where each element is the cosh of the corresponding element in the original tensor.
    /// # Panics
    /// * This function may panic if the tensor contains invalid values for the cosh function, such as `NaN` values.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn cosh(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Computes the element-wise tanh of the tensor.
    ///
    /// This function calculates the tanh of each element in the tensor, returning a new tensor
    /// where each element is the tanh of the corresponding element in the original tensor.
    /// The tanh function is defined as:
    /// ```text
    /// tanh(x)
    /// ```
    /// # Arguments
    /// This function takes no arguments.
    /// # Returns
    /// * A new tensor where each element is the tanh of the corresponding element in the original tensor.
    /// # Panics
    /// * This function may panic if the tensor contains invalid values for the tanh function, such as `NaN` values.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn tanh(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Computes the element-wise asinh of the tensor.
    ///
    /// This function calculates the asinh of each element in the tensor, returning a new tensor
    /// where each element is the asinh of the corresponding element in the original tensor.
    /// The asinh function is defined as:
    /// ```text
    /// asinh(x)
    /// ```
    /// # Arguments
    /// This function takes no arguments.
    /// # Returns
    /// * A new tensor where each element is the asinh of the corresponding element in the original tensor.
    /// # Panics
    /// * This function may panic if the tensor contains invalid values for the asinh function, such as `NaN` values.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn asinh(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Computes the element-wise acosh of the tensor.
    ///
    /// This function calculates the acosh of each element in the tensor, returning a new tensor
    /// where each element is the acosh of the corresponding element in the original tensor.
    /// The acosh function is defined as:
    /// ```text
    /// acosh(x)
    /// ```
    /// # Arguments
    /// This function takes no arguments.
    /// # Returns
    /// * A new tensor where each element is the acosh of the corresponding element in the original tensor.
    /// # Panics
    /// * This function may panic if the tensor contains invalid values for the acosh function, such as `NaN` values.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn acosh(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Computes the element-wise atanh of the tensor.
    ///
    /// This function calculates the atanh of each element in the tensor, returning a new tensor
    /// where each element is the atanh of the corresponding element in the original tensor.
    /// The atanh function is defined as:
    /// ```text
    /// atanh(x)
    /// ```
    /// # Arguments
    /// This function takes no arguments.
    /// # Returns
    /// * A new tensor where each element is the atanh of the corresponding element in the original tensor.
    /// # Panics
    /// * This function may panic if the tensor contains invalid values for the atanh function, such as `NaN` values.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn atanh(&self) -> std::result::Result<Self::Output, TensorError>;

    /// sin method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`sin`]: Computes the element-wise sine of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn sin_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// cos method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`cos`]: Computes the element-wise cosine of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn cos_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// tan method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`tan`]: Computes the element-wise tangent of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn tan_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// asin method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`asin`]: Computes the element-wise asin of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn asin_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// acos method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`acos`]: Computes the element-wise acos of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn acos_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// atan method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`atan`]: Computes the element-wise atan of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn atan_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// sinh method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`sinh`]: Computes the element-wise sinh of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn sinh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// cosh method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`cosh`]: Computes the element-wise cosh of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn cosh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// tanh method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`tanh`]: Computes the element-wise tanh of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn tanh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// asinh method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`asinh`]: Computes the element-wise asinh of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn asinh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// acosh method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`acosh`]: Computes the element-wise acosh of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn acosh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// atanh method with output tensor, this method will write the result to the output tensor
    ///
    /// # See Also
    ///
    /// - [`atanh`]: Computes the element-wise atanh of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn atanh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise exponential of the tensor.
    ///
    /// This function calculates the exponential (base `e`) of each element in the tensor.
    /// The exponential function is defined as:
    ///
    /// exp(x) = e<sup>x</sup>
    ///
    /// # Arguments
    /// This function takes no arguments.
    /// # Returns
    /// * A new tensor where each element is the exponential of the corresponding element in the original tensor.
    /// # Panics
    /// * This function may panic if the tensor contains values that would result in an overflow when calculating the exponential.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn exp(&self) -> std::result::Result<Self::Output, TensorError>;

    /// exp method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`exp`]: Computes the element-wise exponential of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn exp_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise base-2 exponential of the tensor.
    ///
    /// This function calculates the base-2 exponential (2<sup>x</sup>) of each element in the tensor.
    /// The base-2 exponential function is defined as:
    ///
    /// exp2(x) = 2<sup>x</sup>
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// * A new tensor where each element is the base-2 exponential of the corresponding element in the original tensor.
    ///
    /// # Panics
    ///
    /// * This function may panic if the tensor contains values that would result in an overflow when calculating the base-2 exponential.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn exp2(&self) -> std::result::Result<Self::Output, TensorError>;

    /// exp2 method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`exp2`]: Computes the element-wise base-2 exponential of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn exp2_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise square root of the tensor.
    ///
    /// This function calculates the square root of each element in the tensor.
    /// The square root function is defined as:
    ///
    /// sqrt(x) = √x
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// * A new tensor where each element is the square root of the corresponding element in the original tensor.
    ///
    /// # Panics
    ///
    /// * This function will panic if the tensor contains negative values, as the square root is not defined for negative numbers in real numbers.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn sqrt(&self) -> std::result::Result<Self::Output, TensorError>;

    /// sqrt method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`sqrt`]: Computes the element-wise square root of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn sqrt_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise reciprocal of the tensor.
    ///
    /// This function calculates the reciprocal of each element in the tensor. The reciprocal of a number is defined as:
    ///
    /// recip(x) = 1 / x
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// * A new tensor where each element is the reciprocal of the corresponding element in the original tensor.
    ///
    /// # Panics
    ///
    /// * This function will panic if the tensor contains zeros, as the reciprocal of zero is undefined.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn recip(&self) -> std::result::Result<Self::Output, TensorError>;

    /// recip method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`recip`]: Computes the element-wise reciprocal of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn recip_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise natural logarithm of the tensor.
    ///
    /// This function calculates the natural logarithm (`ln`, base `e`) of each element in the tensor.
    /// The natural logarithm is defined as:
    ///
    /// `ln(x) = log_e(x)`
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// * A new tensor where each element is the natural logarithm of the corresponding element in the original tensor.
    ///
    /// # Panics
    ///
    /// * This function will panic if the tensor contains values less than or equal to zero, as the logarithm is not defined for such values.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn ln(&self) -> std::result::Result<Self::Output, TensorError>;

    /// ln method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`ln`]: Computes the element-wise natural logarithm of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn ln_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise base-2 logarithm of the tensor.
    ///
    /// This function calculates the base-2 logarithm (`log2`) of each element in the tensor.
    /// The base-2 logarithm is defined as:
    ///
    /// `log2(x) = log_2(x)`
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// * A new tensor where each element is the base-2 logarithm of the corresponding element in the original tensor.
    ///
    /// # Panics
    ///
    /// * This function will panic if the tensor contains values less than or equal to zero, as the logarithm is not defined for such values.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn log2(&self) -> std::result::Result<Self::Output, TensorError>;

    /// log2 method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`log2`]: Computes the element-wise base-2 logarithm of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn log2_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise base-10 logarithm of the tensor.
    ///
    /// This function calculates the base-10 logarithm (`log10`) of each element in the tensor.
    /// The base-10 logarithm is defined as:
    ///
    /// `log10(x) = log_10(x)`
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// * A new tensor where each element is the base-10 logarithm of the corresponding element in the original tensor.
    ///
    /// # Panics
    ///
    /// * This function will panic if the tensor contains values less than or equal to zero, as the logarithm is not defined for such values.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn log10(&self) -> std::result::Result<Self::Output, TensorError>;

    /// log10 method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`log10`]: Computes the element-wise base-10 logarithm of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn log10_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise Continuously Differentiable Exponential Linear Unit (CELU) activation function.
    ///
    /// This function applies the CELU activation function with the given `alpha` parameter:
    ///
    /// `celu(x) = max(0, x) + min(0, alpha * (exp(x / alpha) - 1))`
    ///
    /// # Arguments
    ///
    /// * `alpha` - A parameter controlling the saturation of negative values. This value is applied element-wise.
    ///
    /// # Returns
    ///
    /// * A new tensor where the CELU activation function has been applied to each element.
    ///
    /// # Panics
    ///
    /// * This function should not panic under normal conditions.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn celu(&self, alpha: Self::OutputMeta) -> std::result::Result<Self::Output, TensorError>;

    /// celu method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`celu`]: Computes the element-wise Continuously Differentiable Exponential Linear Unit (CELU).
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn celu_<U>(&self, alpha: Self::OutputMeta, out: U) -> std::result::Result<Self::Output, TensorError>
        where U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise sigmoid activation function of the tensor.
    ///
    /// The sigmoid function is defined as:
    ///
    /// sigmoid(x) = 1 / (1 + e<sup>-x</sup>)
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// * A new tensor where the sigmoid function has been applied to each element.
    ///
    /// # Panics
    ///
    /// * This function should not panic under normal conditions.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn sigmoid(&self) -> std::result::Result<Self::Output, TensorError>;

    /// sigmoid method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`sigmoid`]: Computes the element-wise sigmoid function of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn sigmoid_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise Exponential Linear Unit (ELU) activation function.
    ///
    /// This function applies the ELU activation function with the given `alpha` parameter:
    ///
    /// elu(x) = x if x > 0 else alpha * (e<sup>x</sup> - 1)
    ///
    /// # Arguments
    ///
    /// * `alpha` - A parameter controlling the saturation of negative values.
    ///
    /// # Returns
    ///
    /// * A new tensor where the ELU activation function has been applied to each element.
    ///
    /// # Panics
    ///
    /// * This function should not panic under normal conditions.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn elu(&self, alpha: Self::OutputMeta) -> std::result::Result<Self::Output, TensorError>;

    /// elu method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`elu`]: Computes the element-wise Exponential Linear Unit (ELU).
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn elu_<U>(&self, alpha: Self::OutputMeta, out: U) -> std::result::Result<Self::Output, TensorError>
        where U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise error function (erf) of the tensor.
    ///
    /// The error function is used in probability, statistics, and partial differential equations.
    /// It is defined as:
    ///
    /// erf(x) = 2/sqrt(pi) * integral from 0 to x of exp(-t<sup>2</sup>) dt
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// * A new tensor where the error function has been applied to each element.
    ///
    /// # Panics
    ///
    /// * This function should not panic under normal conditions.
    fn erf(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Computes the element-wise Gaussian Error Linear Unit (GELU) activation function.
    ///
    /// The GELU activation is defined as:
    ///
    /// `gelu(x) = x * P(X <= x)` where `P` is the cumulative distribution function of a Gaussian distribution.
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// * A new tensor where the GELU activation function has been applied to each element.
    ///
    /// # Panics
    ///
    /// * This function should not panic under normal conditions.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn gelu(&self) -> std::result::Result<Self::Output, TensorError>;

    /// gelu method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`gelu`]: Computes the element-wise Gaussian Error Linear Unit (GELU).
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn gelu_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise Scaled Exponential Linear Unit (SELU) activation function.
    ///
    /// This function applies the SELU activation function with optional `alpha` and `gamma` parameters.
    /// By default, SELU uses specific values for `alpha` and `gamma`:
    ///
    /// selu(x) = gamma * (x if x > 0 else alpha * (e<sup>x</sup> - 1))
    ///
    /// # Arguments
    ///
    /// * `alpha` - The scaling parameter for negative inputs. Defaults to a specific constant value if `None`.
    /// * `gamma` - The scaling parameter for all inputs. Defaults to a specific constant value if `None`.
    ///
    /// # Returns
    ///
    /// * A new tensor where the SELU activation function has been applied to each element.
    ///
    /// # Panics
    ///
    /// * This function should not panic under normal conditions.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn selu<U>(&self, alpha: U, gamma: U) -> std::result::Result<Self::Output, TensorError>
        where U: Into<Option<Self::OutputMeta>>;

    /// selu method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`selu`]: Computes the element-wise Scaled Exponential Linear Unit (SELU).
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn selu_<U>(
        &self,
        alpha: Option<Self::OutputMeta>,
        gamma: Option<Self::OutputMeta>,
        out: U
    ) -> std::result::Result<Self::Output, TensorError>
        where U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise Hard Sigmoid activation function.
    ///
    /// The Hard Sigmoid function is an efficient approximation of the sigmoid function:
    ///
    /// `hard_sigmoid(x) = max(0, min(1, 0.2 * x + 0.5))`
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// * A new tensor where the Hard Sigmoid activation function has been applied to each element.
    ///
    /// # Panics
    ///
    /// * This function should not panic under normal conditions.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn hard_sigmoid(&self) -> std::result::Result<Self::Output, TensorError>;

    /// hard_sigmoid method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`hard_sigmoid`]: Computes the element-wise Hard Sigmoid.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn hard_sigmoid_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise Hard Swish activation function.
    ///
    /// The Hard Swish function is an approximation of the Swish activation function:
    ///
    /// `hard_swish(x) = x * max(0, min(1, 0.2 * x + 0.5))`
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// * A new tensor where the Hard Swish activation function has been applied to each element.
    ///
    /// # Panics
    ///
    /// * This function should not panic under normal conditions.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn hard_swish(&self) -> std::result::Result<Self::Output, TensorError>;

    /// hard_swish method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`hard_swish`]: Computes the element-wise Hard Swish.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn hard_swish_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise Softplus activation function.
    ///
    /// The Softplus function is defined as:
    ///
    /// softplus(x) = ln(1 + e<sup>x</sup>)
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// * A new tensor where the Softplus activation function has been applied to each element.
    ///
    /// # Panics
    ///
    /// * This function should not panic under normal conditions.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn softplus(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Softplus method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`softplus`]: Computes the element-wise softplus of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn softplus_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise Softsign activation function.
    ///
    /// The Softsign function is defined as:
    ///
    /// `softsign(x) = x / (1 + |x|)`
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// * A new tensor where the Softsign activation function has been applied to each element.
    ///
    /// # Panics
    ///
    /// * This function should not panic under normal conditions.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn softsign(&self) -> std::result::Result<Self::Output, TensorError>;

    /// softsign method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`softsign`]: Computes the element-wise softsign of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn softsign_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise Mish activation function.
    ///
    /// The Mish function is defined as:
    ///
    /// mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e<sup>x</sup>))
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// * A new tensor where the Mish activation function has been applied to each element.
    ///
    /// # Panics
    ///
    /// * This function should not panic under normal conditions.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn mish(&self) -> std::result::Result<Self::Output, TensorError>;

    /// mish method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`mish`]: Computes the element-wise Mish of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn mish_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise cube root of the tensor.
    ///
    /// This function returns a new tensor where each element is the cube root of the corresponding element in the original tensor.
    ///
    /// # Returns
    ///
    /// * A new tensor where each element is the cube root of the corresponding element in the original tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn cbrt(&self) -> std::result::Result<Self::Output, TensorError>;

    /// cbrt method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`cbrt`]: Computes the element-wise cube root of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn cbrt_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;
}

/// A trait for unary operations, the output must be the same type as the input.
pub trait NormalUaryOps where Self: Sized {
    /// The output type of the unary operation.
    type Output;
    /// The output type of the inplace unary operation.
    type InplaceOutput;
    /// The output type of the unary operation.
    type OutputMeta;

    /// Computes the element-wise floor of the tensor.
    ///
    /// This function rounds each element in the tensor down to the nearest integer, returning a new tensor
    /// where each element is the largest integer less than or equal to the corresponding element in the original tensor.
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// * A new tensor where each element is the floor of the corresponding element in the original tensor.
    ///
    /// # Panics
    ///
    /// * This function should not panic under normal conditions.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn floor(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Floor method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`floor`]: Computes the element-wise floor of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn floor_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise square of the tensor.
    ///
    /// This function returns a new tensor where each element is the square of the corresponding element in the original tensor:
    ///
    /// `square(x) = x^2`
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// * A new tensor where each element is squared.
    ///
    /// # Panics
    ///
    /// * This function should not panic under normal conditions.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn square(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Square method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`square`]: Computes the element-wise square of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn square_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise absolute value of the tensor.
    ///
    /// This function returns a new tensor where each element is the absolute value of the corresponding element in the original tensor:
    ///
    /// `abs(x) = |x|`
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// * A new tensor where each element is the absolute value of the corresponding element in the original tensor.
    ///
    /// # Panics
    ///
    /// * This function should not panic under normal conditions.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn abs(&self) -> std::result::Result<Self::Output, TensorError>;

    /// abs method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`abs`]: Computes the element-wise absolute value of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn abs_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise ceiling of the tensor.
    ///
    /// This function rounds each element in the tensor up to the nearest integer, returning a new tensor
    /// where each element is the smallest integer greater than or equal to the corresponding element in the original tensor.
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// * A new tensor where each element is the ceiling of the corresponding element in the original tensor.
    ///
    /// # Panics
    ///
    /// * This function should not panic under normal conditions.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn ceil(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Ceil method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`ceil`]: Computes the element-wise ceiling of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn ceil_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise sign of the tensor.
    ///
    /// This function returns a new tensor where each element represents the sign of the corresponding element in the original tensor:
    ///
    /// * `1` for positive values
    /// * `0` for zero
    /// * `-1` for negative values
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// * A new tensor where each element is the sign of the corresponding element in the original tensor.
    ///
    /// # Panics
    ///
    /// * This function should not panic under normal conditions.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn sign(&self) -> std::result::Result<Self::Output, TensorError>;

    /// sign method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`sign`]: Computes the element-wise sign of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn sign_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// Clamps (limits) the values of the tensor between the specified `min` and `max`.
    ///
    /// This function returns a new tensor where each element is clipped to be within the range `[min, max]`. If an element is less than `min`, it is set to `min`. If it is greater than `max`, it is set to `max`.
    ///
    /// # Arguments
    ///
    /// * `min` - The minimum allowed value for each element.
    /// * `max` - The maximum allowed value for each element.
    ///
    /// # Returns
    ///
    /// * A new tensor where each element is clipped to the specified range.
    ///
    /// # Panics
    ///
    /// * This function will panic if `min` is greater than `max`.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn clamp(&self, min: Self::OutputMeta, max: Self::OutputMeta) -> std::result::Result<Self::Output, TensorError>;

    /// clamp method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`clamp`]: Clamps (limits) the values of the tensor between the specified `min` and `max`.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn clamp_<U>(&self, min: Self::OutputMeta, max: Self::OutputMeta, out: U) -> std::result::Result<Self::Output, TensorError>
        where U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise rounding of the tensor.
    ///
    /// This function rounds each element in the tensor to the nearest integer, returning a new tensor
    /// where each element is the nearest integer to the corresponding element in the original tensor.
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// * A new tensor where each element is rounded to the nearest integer.
    ///
    /// # Panics
    ///
    /// * This function should not panic under normal conditions.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn round(&self) -> std::result::Result<Self::Output, TensorError>;

    /// round method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`round`]: Computes the element-wise rounding of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn round_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

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
    fn neg(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Inplace Version of neg.
    ///
    /// # See Also
    ///
    /// - [`neg`]: Computes the element-wise negation of the tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn neg_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise Rectified Linear Unit (ReLU) activation function.
    ///
    /// The ReLU function is defined as:
    ///
    /// `relu(x) = max(0, x)`
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// * A new tensor where the ReLU activation function has been applied to each element.
    ///
    /// # Panics
    ///
    /// * This function should not panic under normal conditions.
    fn relu(&self) -> std::result::Result<Self::Output, TensorError>;

    /// relu method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`relu`]: Computes the element-wise Rectified Linear Unit (ReLU).
    fn relu_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise Leaky Rectified Linear Unit (Leaky ReLU) activation function.
    ///
    /// This function applies the Leaky ReLU activation function with the given `alpha` parameter:
    ///
    /// `leaky_relu(x) = x if x > 0 else alpha * x`
    ///
    /// # Arguments
    ///
    /// * `alpha` - A parameter controlling the slope for negative input values. This value is applied element-wise.
    ///
    /// # Returns
    ///
    /// * A new tensor where the Leaky ReLU activation function has been applied to each element.
    ///
    /// # Panics
    ///
    /// * This function should not panic under normal conditions.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn leaky_relu(&self, alpha: Self::OutputMeta) -> std::result::Result<Self::Output, TensorError>;

    /// leaky_relu method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`leaky_relu`]: Computes the element-wise Leaky Rectified Linear Unit (Leaky ReLU).
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn leaky_relu_<U>(&self, alpha: Self::OutputMeta, out: U) -> std::result::Result<Self::Output, TensorError>
        where U: Borrow<Self::InplaceOutput>;

    /// Computes the element-wise Rectified Linear Unit 6 (ReLU6) activation function.
    ///
    /// The ReLU6 function is a variant of the ReLU function, defined as:
    ///
    /// `relu6(x) = min(max(0, x), 6)`
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// * A new tensor where the ReLU6 activation function has been applied to each element.
    ///
    /// # Panics
    ///
    /// * This function should not panic under normal conditions.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn relu6(&self) -> std::result::Result<Self::Output, TensorError>;

    /// relu6 method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`relu6`]: Computes the element-wise Rectified Linear Unit 6 (ReLU6).
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn relu6_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError> where U: Borrow<Self::InplaceOutput>;
}

/// A trait for accumulative operations.
pub trait Cum where Self: Sized, <<Self as Cum>::Meta as TypeCommon>::Vec: Send + Sync {
    /// The output type of the accumulative operation.
    type Meta: CommonBounds;
    /// The output type of the accumulative operation.
    type Output;
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
    fn cumsum(&self, axis: Option<i64>) -> std::result::Result<Self::Output, TensorError>
        where Self::Meta: NormalOut<Self::Meta, Output = Self::Meta>;

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
    fn cumprod(&self, axis: Option<i64>) -> std::result::Result<Self::Output, TensorError>
        where Self::Meta: NormalOut<Self::Meta, Output = Self::Meta>;
}
