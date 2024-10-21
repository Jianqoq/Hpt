use std::borrow::Borrow;

use tensor_traits::{ CommonBounds, FloatUaryOps };
use tensor_types::{ dtype::TypeCommon, into_scalar::IntoScalar, type_promote::FloatOutUnary };

use crate::{
    backend::Cpu,
    ops::cpu::tensor_internal::float_out_unary::FloatUnaryType,
    tensor::Tensor,
    tensor_base::_Tensor,
};
use anyhow::Result;

impl<T> FloatUaryOps
    for Tensor<T>
    where
        T: FloatOutUnary<Base = FloatUnaryType<T>> + CommonBounds,
        FloatUnaryType<T>: CommonBounds,
        f64: IntoScalar<<T as FloatOutUnary>::Output>,
        T::Vec: FloatOutUnary<
            Output = <FloatUnaryType<T> as TypeCommon>::Vec,
            Base = FloatUnaryType<T>
        >,
        <FloatUnaryType<T> as TypeCommon>::Vec: Send + Copy + Sync
{
    type Output = Tensor<FloatUnaryType<T>>;

    type InplaceOutput = _Tensor<FloatUnaryType<T>>;

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
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::FloatUaryOps;
    /// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    /// let b = a.sin().unwrap();
    /// ```
    fn sin(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::sin(self)?.into())
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
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::FloatUaryOps;
    /// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    /// let b = a.cos().unwrap();
    /// ```
    fn cos(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::cos(self)?.into())
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
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::FloatUaryOps;
    /// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    /// let b = a.tan().unwrap();
    /// ```
    fn tan(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::tan(self)?.into())
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
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::FloatUaryOps;
    /// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    /// let b = a.asin().unwrap();
    /// ```
    fn asin(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::asin(self)?.into())
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
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::FloatUaryOps;
    /// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    /// let b = a.acos().unwrap();
    /// ```
    fn acos(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::acos(self)?.into())
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
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::FloatUaryOps;
    /// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    /// let b = a.atan().unwrap();
    /// ```
    fn atan(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::atan(self)?.into())
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
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::FloatUaryOps;
    /// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    /// let b = a.sinh().unwrap();
    /// ```
    fn sinh(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::sinh(self)?.into())
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
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::FloatUaryOps;
    /// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    /// let b = a.cosh().unwrap();
    /// ```
    fn cosh(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::cosh(self)?.into())
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
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::FloatUaryOps;
    /// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    /// let b = a.tanh().unwrap();
    /// ```
    fn tanh(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::tanh(self)?.into())
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
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::FloatUaryOps;
    /// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    /// let b = a.asinh().unwrap();
    /// ```
    fn asinh(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::asinh(self)?.into())
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
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::FloatUaryOps;
    /// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    /// let b = a.acosh().unwrap();
    /// ```
    fn acosh(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::acosh(self)?.into())
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
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::FloatUaryOps;
    /// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    /// let b = a.atanh().unwrap();
    /// ```
    fn atanh(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::atanh(self)?.into())
    }

    fn sin_<U>(&self, out: U) -> Result<Self::Output> where U: Borrow<Self::InplaceOutput> {
        Ok(_Tensor::<T, Cpu>::sin_(self, out)?.into())
    }

    fn cos_<U>(&self, out: U) -> Result<Self::Output> where U: Borrow<Self::InplaceOutput> {
        Ok(_Tensor::<T, Cpu>::cos_(self, out)?.into())
    }

    fn tan_<U>(&self, out: U) -> Result<Self::Output> where U: Borrow<Self::InplaceOutput> {
        Ok(_Tensor::<T, Cpu>::tan_(self, out)?.into())
    }

    fn asin_<U>(&self, out: U) -> Result<Self::Output> where U: Borrow<Self::InplaceOutput> {
        Ok(_Tensor::<T, Cpu>::asin_(self, out)?.into())
    }

    fn acos_<U>(&self, out: U) -> Result<Self::Output> where U: Borrow<Self::InplaceOutput> {
        Ok(_Tensor::<T, Cpu>::acos_(self, out)?.into())
    }

    fn atan_<U>(&self, out: U) -> Result<Self::Output> where U: Borrow<Self::InplaceOutput> {
        Ok(_Tensor::<T, Cpu>::atan_(self, out)?.into())
    }

    fn sinh_<U>(&self, out: U) -> Result<Self::Output> where U: Borrow<Self::InplaceOutput> {
        Ok(_Tensor::<T, Cpu>::sinh_(self, out)?.into())
    }

    fn cosh_<U>(&self, out: U) -> Result<Self::Output> where U: Borrow<Self::InplaceOutput> {
        Ok(_Tensor::<T, Cpu>::cosh_(self, out)?.into())
    }

    fn tanh_<U>(&self, out: U) -> Result<Self::Output> where U: Borrow<Self::InplaceOutput> {
        Ok(_Tensor::<T, Cpu>::tanh_(self, out)?.into())
    }

    fn asinh_<U>(&self, out: U) -> Result<Self::Output> where U: Borrow<Self::InplaceOutput> {
        Ok(_Tensor::<T, Cpu>::asinh_(self, out)?.into())
    }

    fn acosh_<U>(&self, out: U) -> Result<Self::Output> where U: Borrow<Self::InplaceOutput> {
        Ok(_Tensor::<T, Cpu>::acosh_(self, out)?.into())
    }

    fn atanh_<U>(&self, out: U) -> Result<Self::Output> where U: Borrow<Self::InplaceOutput> {
        Ok(_Tensor::<T, Cpu>::atanh_(self, out)?.into())
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
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::FloatUaryOps;
    /// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    /// let b = a.exp().unwrap();
    /// ```
    fn exp(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::exp(self)?.into())
    }

    fn exp_<U>(&self, out: U) -> Result<Self::Output> where U: Borrow<Self::InplaceOutput> {
        Ok(_Tensor::<T, Cpu>::exp_(self, out)?.into())
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
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::FloatUaryOps;
    /// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    /// let b = a.exp2().unwrap();
    /// ```
    fn exp2(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::exp2(self)?.into())
    }

    fn exp2_<U>(&self, out: U) -> Result<Self::Output> where U: Borrow<Self::InplaceOutput> {
        Ok(_Tensor::<T, Cpu>::exp2_(self, out)?.into())
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
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::FloatUaryOps;
    /// let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    /// let b = a.sqrt().unwrap();
    /// ```
    fn sqrt(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::sqrt(self)?.into())
    }

    fn sqrt_<U>(&self, out: U) -> Result<Self::Output> where U: Borrow<Self::InplaceOutput> {
        Ok(_Tensor::<T, Cpu>::sqrt_(self, out)?.into())
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
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::FloatUaryOps;
    /// let a = Tensor::<f64>::new([1.0, 2.0, 3.0, 4.0]);
    /// let b = a.recip().unwrap();
    /// ```
    fn recip(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::recip(self)?.into())
    }

    fn recip_<U>(&self, out: U) -> Result<Self::Output> where U: Borrow<Self::InplaceOutput> {
        Ok(_Tensor::<T, Cpu>::recip_(self, out)?.into())
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
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::FloatUaryOps;
    /// let a = Tensor::<f64>::new([1.0, 2.0, 3.0, 4.0]);
    /// let b = a.ln().unwrap();
    /// ```
    fn ln(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::ln(self)?.into())
    }

    fn ln_<U>(&self, out: U) -> Result<Self::Output> where U: Borrow<Self::InplaceOutput> {
        Ok(_Tensor::<T, Cpu>::ln_(self, out)?.into())
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
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::FloatUaryOps;
    /// let a = Tensor::<f64>::new([1.0, 2.0, 3.0, 4.0]);
    /// let b = a.log2().unwrap();
    /// ```
    fn log2(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::log2(self)?.into())
    }

    fn log2_<U>(&self, out: U) -> Result<Self::Output> where U: Borrow<Self::InplaceOutput> {
        Ok(_Tensor::<T, Cpu>::log2_(self, out)?.into())
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
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::FloatUaryOps;
    /// let a = Tensor::<f64>::new([1.0, 2.0, 3.0, 4.0]);
    /// let b = a.log10().unwrap();
    /// ```
    fn log10(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::log10(self)?.into())
    }

    fn log10_<U>(&self, out: U) -> Result<Self::Output> where U: Borrow<Self::InplaceOutput> {
        Ok(_Tensor::<T, Cpu>::log10_(self, out)?.into())
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
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::FloatUaryOps;
    /// let a = Tensor::<f64>::new([-1.0, 0.0, 1.0, 2.0]);
    /// let b = a.celu(1.0).unwrap();
    /// ```
    fn celu(&self, alpha: Self::OutputMeta) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::celu(self, alpha)?.into())
    }

    fn celu_<U>(&self, alpha: Self::OutputMeta, out: U) -> Result<Self::Output>
        where U: Borrow<Self::InplaceOutput>
    {
        Ok(_Tensor::<T, Cpu>::celu_(self, alpha, out)?.into())
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
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::FloatUaryOps;
    /// let a = Tensor::<f64>::new([-1.0, 0.0, 1.0, 2.0]);
    /// let b = a.sigmoid().unwrap();
    /// ```
    fn sigmoid(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::sigmoid(self)?.into())
    }

    fn sigmoid_<U>(&self, out: U) -> Result<Self::Output> where U: Borrow<Self::InplaceOutput> {
        Ok(_Tensor::<T, Cpu>::sigmoid_(self, out)?.into())
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
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::FloatUaryOps;
    /// let a = Tensor::<f64>::new([-1.0, 0.0, 1.0, 2.0]);
    /// let b = a.elu(1.0).unwrap();
    /// ```
    fn elu(&self, alpha: Self::OutputMeta) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::elu(self, alpha)?.into())
    }

    fn elu_<U>(&self, alpha: Self::OutputMeta, out: U) -> Result<Self::Output>
        where U: Borrow<Self::InplaceOutput>
    {
        Ok(_Tensor::<T, Cpu>::elu_(self, alpha, out)?.into())
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
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::FloatUaryOps;
    /// let a = Tensor::<f64>::new([-1.0, 0.0, 1.0, 2.0]);
    /// let b = a.erf().unwrap();
    /// ```
    fn erf(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::erf(self)?.into())
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
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::FloatUaryOps;
    /// let a = Tensor::<f64>::new([-1.0, 0.0, 1.0, 2.0]);
    /// let b = a.fast_hard_sigmoid().unwrap();
    /// ```
    fn fast_hard_sigmoid(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::fast_hard_sigmoid(self)?.into())
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
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::FloatUaryOps;
    /// let a = Tensor::<f64>::new([-1.0, 0.0, 1.0, 2.0]);
    /// let b = a.gelu().unwrap();
    /// ```
    fn gelu(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::gelu(self)?.into())
    }

    fn gelu_<U>(&self, out: U) -> Result<Self::Output> where U: Borrow<Self::InplaceOutput> {
        Ok(_Tensor::<T, Cpu>::gelu_(self, out)?.into())
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
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::FloatUaryOps;
    /// let a = Tensor::<f64>::new([-1.0, 0.0, 1.0, 2.0]);
    /// let b = a.selu(None, None).unwrap();
    /// ```
    fn selu(
        &self,
        alpha: Option<Self::OutputMeta>,
        gamma: Option<Self::OutputMeta>
    ) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::selu(self, alpha, gamma)?.into())
    }

    fn selu_<U>(
        &self,
        alpha: Option<Self::OutputMeta>,
        gamma: Option<Self::OutputMeta>,
        out: U
    ) -> Result<Self::Output>
        where U: Borrow<Self::InplaceOutput>
    {
        Ok(_Tensor::<T, Cpu>::selu_(self, alpha, gamma, out)?.into())
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
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::FloatUaryOps;
    /// let a = Tensor::<f64>::new([-1.0, 0.0, 1.0, 2.0]);
    /// let b = a.hard_sigmoid().unwrap();
    /// ```
    fn hard_sigmoid(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::hard_sigmoid(self)?.into())
    }

    fn hard_sigmoid_<U>(&self, out: U) -> Result<Self::Output> where U: Borrow<Self::InplaceOutput> {
        Ok(_Tensor::<T, Cpu>::hard_sigmoid_(self, out)?.into())
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
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::FloatUaryOps;
    /// let a = Tensor::<f64>::new([-1.0, 0.0, 1.0, 2.0]);
    /// let b = a.hard_swish().unwrap();
    /// ```
    fn hard_swish(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::hard_swish(self)?.into())
    }

    fn hard_swish_<U>(&self, out: U) -> Result<Self::Output> where U: Borrow<Self::InplaceOutput> {
        Ok(_Tensor::<T, Cpu>::hard_swish_(self, out)?.into())
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
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::FloatUaryOps;
    /// let a = Tensor::<f64>::new([-1.0, 0.0, 1.0, 2.0]);
    /// let b = a.softplus().unwrap();
    /// ```
    fn softplus(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::softplus(self)?.into())
    }

    fn softplus_<U>(&self, out: U) -> Result<Self::Output> where U: Borrow<Self::InplaceOutput> {
        Ok(_Tensor::<T, Cpu>::softplus_(self, out)?.into())
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
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::FloatUaryOps;
    /// let a = Tensor::<f64>::new([-1.0, 0.0, 1.0, 2.0]);
    /// let b = a.softsign().unwrap();
    /// ```
    fn softsign(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::softsign(self)?.into())
    }

    fn softsign_<U>(&self, out: U) -> Result<Self::Output> where U: Borrow<Self::InplaceOutput> {
        Ok(_Tensor::<T, Cpu>::softsign_(self, out)?.into())
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
    /// use tensor_dyn::tensor::Tensor;
    /// use tensor_dyn::FloatUaryOps;
    /// let a = Tensor::<f64>::new([-1.0, 0.0, 1.0, 2.0]);
    /// let b = a.mish().unwrap();
    /// ```
    fn mish(&self) -> Result<Self::Output> {
        Ok(_Tensor::<T, Cpu>::mish(self)?.into())
    }

    fn mish_<U>(&self, out: U) -> Result<Self::Output> where U: Borrow<Self::InplaceOutput> {
        Ok(_Tensor::<T, Cpu>::mish_(self, out)?.into())
    }
}
