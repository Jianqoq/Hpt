use std::borrow::Borrow;

use cudarc::driver::DeviceRepr;
use hpt_common::err_handler::TensorError;
use hpt_traits::{CommonBounds, FloatUnaryOps};
use hpt_types::{
    cast::Cast, cuda_types::scalar::Scalar, dtype::TypeCommon, type_promote::FloatOutUnary,
};

use crate::{
    ops::cpu::tensor_internal::float_out_unary::FloatUnaryType, tensor::Tensor,
    tensor_base::_Tensor, Cuda,
};

impl<T, const DEVICE_ID: usize> FloatUnaryOps for Tensor<T, Cuda, DEVICE_ID>
where
    T: FloatOutUnary<Base = FloatUnaryType<T>> + CommonBounds + DeviceRepr,
    FloatUnaryType<T>: CommonBounds + DeviceRepr,
    f64: Cast<<T as FloatOutUnary>::Output>,
    T::Vec:
        FloatOutUnary<Output = <FloatUnaryType<T> as TypeCommon>::Vec, Base = FloatUnaryType<T>>,
    <FloatUnaryType<T> as TypeCommon>::Vec: Send + Copy + Sync,
    Scalar<T>: FloatOutUnary<Output = Scalar<FloatUnaryType<T>>, Base = Scalar<FloatUnaryType<T>>>,
{
    type Output = Tensor<FloatUnaryType<T>, Cuda, DEVICE_ID>;

    type InplaceOutput = Tensor<FloatUnaryType<T>, Cuda, DEVICE_ID>;

    type OutputMeta = <T as FloatOutUnary>::Base;

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
    /// # Example
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::FloatUnaryOps;
    /// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    /// let b = a.sin().unwrap();
    /// ```
    fn sin(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::sin(self.inner.as_ref())?.into())
    }

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
    /// # Example
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::FloatUnaryOps;
    /// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    /// let b = a.cos().unwrap();
    /// ```
    fn cos(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::cos(self.inner.as_ref())?.into())
    }

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
    /// # Example
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::FloatUnaryOps;
    /// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    /// let b = a.tan().unwrap();
    /// ```
    fn tan(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::tan(self.inner.as_ref())?.into())
    }

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
    /// # Example
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::FloatUnaryOps;
    /// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    /// let b = a.asin().unwrap();
    /// ```
    fn asin(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::asin(self.inner.as_ref())?.into())
    }

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
    /// # Example
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::FloatUnaryOps;
    /// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    /// let b = a.acos().unwrap();
    /// ```
    fn acos(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::acos(self.inner.as_ref())?.into())
    }

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
    /// # Example
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::FloatUnaryOps;
    /// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    /// let b = a.atan().unwrap();
    /// ```
    fn atan(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::atan(self.inner.as_ref())?.into())
    }

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
    /// # Example
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::FloatUnaryOps;
    /// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    /// let b = a.sinh().unwrap();
    /// ```
    fn sinh(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::sinh(self.inner.as_ref())?.into())
    }

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
    /// # Example
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::FloatUnaryOps;
    /// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    /// let b = a.cosh().unwrap();
    /// ```
    fn cosh(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::cosh(self.inner.as_ref())?.into())
    }

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
    /// # Example
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::FloatUnaryOps;
    /// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    /// let b = a.tanh().unwrap();
    /// ```
    fn tanh(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::tanh(self.inner.as_ref())?.into())
    }

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
    /// # Example
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::FloatUnaryOps;
    /// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    /// let b = a.asinh().unwrap();
    /// ```
    fn asinh(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::asinh(self.inner.as_ref())?.into())
    }

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
    /// # Example
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::FloatUnaryOps;
    /// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    /// let b = a.acosh().unwrap();
    /// ```
    fn acosh(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::acosh(self.inner.as_ref())?.into())
    }

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
    /// # Example
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::FloatUnaryOps;
    /// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    /// let b = a.atanh().unwrap();
    /// ```
    fn atanh(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::atanh(self.inner.as_ref())?.into())
    }

    fn sin_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::sin_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn cos_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::cos_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn tan_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::tan_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn asin_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::asin_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn acos_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::acos_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn atan_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::atan_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn sinh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::sinh_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn cosh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::cosh_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn tanh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::tanh_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn asinh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::asinh_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn acosh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::acosh_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn atanh_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::atanh_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

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
    /// # Example
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::FloatUnaryOps;
    /// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    /// let b = a.exp().unwrap();
    /// ```
    fn exp(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::exp(self.inner.as_ref())?.into())
    }

    fn exp_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::exp_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

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
    /// # Example
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::FloatUnaryOps;
    /// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    /// let b = a.exp2().unwrap();
    /// ```
    fn exp2(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::exp2(self.inner.as_ref())?.into())
    }

    fn exp2_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::exp2_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

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
    /// # Example
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::FloatUnaryOps;
    /// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    /// let b = a.sqrt().unwrap();
    /// ```
    fn sqrt(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::sqrt(self.inner.as_ref())?.into())
    }

    fn sqrt_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::sqrt_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

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
    /// # Example
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::FloatUnaryOps;
    /// let a = Tensor::<f64>::new([1.0, 2.0, 3.0, 4.0]);
    /// let b = a.recip().unwrap();
    /// ```
    fn recip(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::recip(self.inner.as_ref())?.into())
    }

    fn recip_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::recip_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

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
    /// # Example
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::FloatUnaryOps;
    /// let a = Tensor::<f64>::new([1.0, 2.0, 3.0, 4.0]);
    /// let b = a.ln().unwrap();
    /// ```
    fn ln(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::ln(self.inner.as_ref())?.into())
    }

    fn ln_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::ln_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

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
    /// # Example
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::FloatUnaryOps;
    /// let a = Tensor::<f64>::new([1.0, 2.0, 3.0, 4.0]);
    /// let b = a.log2().unwrap();
    /// ```
    fn log2(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::log2(self.inner.as_ref())?.into())
    }

    fn log2_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::log2_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

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
    /// # Example
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::FloatUnaryOps;
    /// let a = Tensor::<f64>::new([1.0, 2.0, 3.0, 4.0]);
    /// let b = a.log10().unwrap();
    /// ```
    fn log10(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::log10(self.inner.as_ref())?.into())
    }

    fn log10_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::log10_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

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
    /// # Example
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::FloatUnaryOps;
    /// let a = Tensor::<f64>::new([-1.0, 0.0, 1.0, 2.0]);
    /// let b = a.celu(1.0).unwrap();
    /// ```
    fn celu(&self, alpha: Self::OutputMeta) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::celu(self.inner.as_ref(), alpha)?.into())
    }

    fn celu_<U>(
        &self,
        alpha: Self::OutputMeta,
        out: U,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::celu_(self.inner.as_ref(), alpha, out.borrow().inner.as_ref())?.into())
    }

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
    /// # Example
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::FloatUnaryOps;
    /// let a = Tensor::<f64>::new([-1.0, 0.0, 1.0, 2.0]);
    /// let b = a.sigmoid().unwrap();
    /// ```
    fn sigmoid(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::sigmoid(self.inner.as_ref())?.into())
    }

    fn sigmoid_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::sigmoid_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

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
    /// # Example
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::FloatUnaryOps;
    /// let a = Tensor::<f64>::new([-1.0, 0.0, 1.0, 2.0]);
    /// let b = a.elu(1.0).unwrap();
    /// ```
    fn elu(&self, alpha: Self::OutputMeta) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::elu(self.inner.as_ref(), alpha)?.into())
    }

    fn elu_<U>(
        &self,
        alpha: Self::OutputMeta,
        out: U,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::elu_(self.inner.as_ref(), alpha, out.borrow().inner.as_ref())?.into())
    }
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
    /// # Example
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::FloatUnaryOps;
    /// let a = Tensor::<f64>::new([-1.0, 0.0, 1.0, 2.0]);
    /// let b = a.erf().unwrap();
    /// ```
    fn erf(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::erf(self.inner.as_ref())?.into())
    }

    /// Computes the element-wise fast hard sigmoid activation function.
    ///
    /// This function applies an approximation of the sigmoid function for efficiency, defined as:
    ///
    /// `fast_hard_sigmoid(x) = max(0, min(1, 0.2 * x + 0.5))`
    ///
    /// # Arguments
    ///
    /// This function takes no arguments.
    ///
    /// # Returns
    ///
    /// * A new tensor where the fast hard sigmoid function has been applied to each element.
    ///
    /// # Panics
    ///
    /// * This function should not panic under normal conditions.
    /// # Example
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::FloatUnaryOps;
    /// let a = Tensor::<f64>::new([-1.0, 0.0, 1.0, 2.0]);
    /// let b = a.fast_hard_sigmoid().unwrap();
    /// ```
    fn fast_hard_sigmoid(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::fast_hard_sigmoid(self.inner.as_ref())?.into())
    }

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
    /// # Example
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::FloatUnaryOps;
    /// let a = Tensor::<f64>::new([-1.0, 0.0, 1.0, 2.0]);
    /// let b = a.gelu().unwrap();
    /// ```
    fn gelu(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::gelu(self.inner.as_ref())?.into())
    }

    fn gelu_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::gelu_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

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
    /// # Example
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::FloatUnaryOps;
    /// let a = Tensor::<f64>::new([-1.0, 0.0, 1.0, 2.0]);
    /// let b = a.selu(None, None).unwrap();
    /// ```
    fn selu<U>(&self, alpha: U, gamma: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Into<Option<Self::OutputMeta>>,
    {
        Ok(_Tensor::selu(self.inner.as_ref(), alpha, gamma)?.into())
    }

    fn selu_<U>(
        &self,
        alpha: Option<Self::OutputMeta>,
        gamma: Option<Self::OutputMeta>,
        out: U,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::selu_(
            self.inner.as_ref(),
            alpha,
            gamma,
            out.borrow().inner.as_ref(),
        )?
        .into())
    }

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
    /// # Example
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::FloatUnaryOps;
    /// let a = Tensor::<f64>::new([-1.0, 0.0, 1.0, 2.0]);
    /// let b = a.hard_sigmoid().unwrap();
    /// ```
    fn hard_sigmoid(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::hard_sigmoid(self.inner.as_ref())?.into())
    }

    fn hard_sigmoid_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::hard_sigmoid_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

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
    /// # Example
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::FloatUnaryOps;
    /// let a = Tensor::<f64>::new([-1.0, 0.0, 1.0, 2.0]);
    /// let b = a.hard_swish().unwrap();
    /// ```
    fn hard_swish(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::hard_swish(self.inner.as_ref())?.into())
    }

    fn hard_swish_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::hard_swish_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

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
    /// # Example
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::FloatUnaryOps;
    /// let a = Tensor::<f64>::new([-1.0, 0.0, 1.0, 2.0]);
    /// let b = a.softplus().unwrap();
    /// ```
    fn softplus(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::softplus(self.inner.as_ref())?.into())
    }

    fn softplus_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::softplus_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

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
    /// # Example
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::FloatUnaryOps;
    /// let a = Tensor::<f64>::new([-1.0, 0.0, 1.0, 2.0]);
    /// let b = a.softsign().unwrap();
    /// ```
    fn softsign(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::softsign(self.inner.as_ref())?.into())
    }

    fn softsign_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::softsign_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

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
    /// # Example
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::FloatUnaryOps;
    /// let a = Tensor::<f64>::new([-1.0, 0.0, 1.0, 2.0]);
    /// let b = a.mish().unwrap();
    /// ```
    fn mish(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::mish(self.inner.as_ref())?.into())
    }

    fn mish_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::mish_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }

    fn cbrt(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::cbrt(self.inner.as_ref())?.into())
    }

    fn cbrt_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::cbrt_(self.inner.as_ref(), out.borrow().inner.as_ref())?.into())
    }
}
