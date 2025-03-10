use hpt_common::error::base::TensorError;
use hpt_types::{dtype::TypeCommon, into_scalar::Cast, type_promote::NormalOut};
use std::borrow::BorrowMut;

use crate::tensor::CommonBounds;

/// A trait for tensor unary operations, the output must be a floating point tensor
pub trait FloatUnaryOps {
    /// output tensor type
    type Output;
    /// output tensor type for inplace operation
    type InplaceOutput;
    /// output tensor data type
    type OutputMeta: Send;
    /// Computes sine element-wise.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.sin()?;
    /// ```
    #[track_caller]
    fn sin(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Computes cosine element-wise.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.cos()?;
    /// ```
    #[track_caller]
    fn cos(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Computes sine and cosine element-wise.
    ///
    /// # Example
    #[track_caller]
    fn sincos(&self) -> std::result::Result<(Self::Output, Self::Output), TensorError>;

    /// Computes tangent element-wise.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.tan()?;
    /// ```
    #[track_caller]
    fn tan(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Computes arcsine element-wise.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.asin()?;
    /// ```
    #[track_caller]
    fn asin(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Computes arccosine element-wise.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.acos()?;
    /// ```
    #[track_caller]
    fn acos(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Computes arctangent element-wise.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.atan()?;
    /// ```
    #[track_caller]
    fn atan(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Computes hyperbolic sine element-wise.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.sinh()?;
    /// ```
    #[track_caller]
    fn sinh(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Computes hyperbolic cosine element-wise.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.cosh()?;
    /// ```
    #[track_caller]
    fn cosh(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Computes hyperbolic tangent element-wise.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.tanh()?;
    /// ```
    #[track_caller]
    fn tanh(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Computes the element-wise asinh of the tensor.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.asinh()?;
    /// ```
    #[track_caller]
    fn asinh(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Computes the element-wise acosh of the tensor.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.acosh()?;
    /// ```
    #[track_caller]
    fn acosh(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Computes the element-wise atanh of the tensor.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.atanh()?;
    /// ```
    #[track_caller]
    fn atanh(&self) -> std::result::Result<Self::Output, TensorError>;

    /// sin method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`sin`]
    #[track_caller]
    fn sin_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// cos method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`cos`]
    #[track_caller]
    fn cos_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// tan method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`tan`]
    #[track_caller]
    fn tan_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// asin method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`asin`]
    #[track_caller]
    fn asin_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// acos method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`acos`]
    #[track_caller]
    fn acos_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// atan method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`atan`]
    #[track_caller]
    fn atan_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// sinh method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`sinh`]
    #[track_caller]
    fn sinh_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// cosh method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`cosh`]
    #[track_caller]
    fn cosh_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// tanh method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`tanh`]
    #[track_caller]
    fn tanh_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// asinh method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`asinh`]
    #[track_caller]
    fn asinh_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// acosh method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`acosh`]
    #[track_caller]
    fn acosh_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// atanh method with output tensor, this method will write the result to the output tensor
    ///
    /// # See Also
    /// - [`atanh`]
    #[track_caller]
    fn atanh_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// sincos method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`sincos`]
    #[track_caller]
    fn sincos_<U, O>(
        &self,
        outs: (U, O),
    ) -> std::result::Result<(Self::InplaceOutput, Self::InplaceOutput), TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
        O: BorrowMut<Self::InplaceOutput>;

    /// Computes the element-wise exponential of the tensor.
    #[track_caller]
    fn exp(&self) -> std::result::Result<Self::Output, TensorError>;

    /// exp method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`exp`]
    #[track_caller]
    fn exp_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// Computes the element-wise base-2 exponential of the tensor.
    #[track_caller]
    fn exp2(&self) -> std::result::Result<Self::Output, TensorError>;

    /// exp2 method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`exp2`]
    #[track_caller]
    fn exp2_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// Computes the element-wise base-10 exponential of the tensor.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.exp10()?;
    /// ```
    #[track_caller]
    fn exp10(&self) -> std::result::Result<Self::Output, TensorError>;

    /// exp10 method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`exp10`]
    #[track_caller]
    fn exp10_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// Computes the element-wise square root of the tensor.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.sqrt()?;
    /// ```
    #[track_caller]
    fn sqrt(&self) -> std::result::Result<Self::Output, TensorError>;

    /// sqrt method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`sqrt`]
    #[track_caller]
    fn sqrt_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// Computes the element-wise reciprocal of the tensor.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.recip()?;
    /// ```
    #[track_caller]
    fn recip(&self) -> std::result::Result<Self::Output, TensorError>;

    /// recip method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`recip`]
    #[track_caller]
    fn recip_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// Computes the element-wise natural logarithm of the tensor.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.ln()?;
    /// ```
    #[track_caller]
    fn ln(&self) -> std::result::Result<Self::Output, TensorError>;

    /// ln method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`ln`]
    #[track_caller]
    fn ln_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// Computes the element-wise base-2 logarithm of the tensor.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.log2()?;
    /// ```
    #[track_caller]
    fn log2(&self) -> std::result::Result<Self::Output, TensorError>;

    /// log2 method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`log2`]
    #[track_caller]
    fn log2_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// Computes the element-wise base-10 logarithm of the tensor.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.log10()?;
    /// ```
    #[track_caller]
    fn log10(&self) -> std::result::Result<Self::Output, TensorError>;

    /// log10 method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`log10`]
    #[track_caller]
    fn log10_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// Computes the element-wise Continuously Differentiable Exponential Linear Unit (CELU) activation function.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.celu(1.0)?;
    /// ```
    #[track_caller]
    fn celu<V: Cast<Self::OutputMeta>>(
        &self,
        alpha: V,
    ) -> std::result::Result<Self::Output, TensorError>;

    /// celu method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`celu`]
    #[track_caller]
    fn celu_<V, U>(
        &self,
        alpha: V,
        out: U,
    ) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        V: Cast<Self::OutputMeta>,
        U: BorrowMut<Self::InplaceOutput>;

    /// Computes the element-wise sigmoid activation function of the tensor.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.sigmoid()?;
    /// ```
    #[track_caller]
    fn sigmoid(&self) -> std::result::Result<Self::Output, TensorError>;

    /// sigmoid method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`sigmoid`]
    #[track_caller]
    fn sigmoid_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// Computes the element-wise Exponential Linear Unit (ELU) activation function.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.elu(1.0)?;
    /// ```
    #[track_caller]
    fn elu<V: Cast<Self::OutputMeta>>(
        &self,
        alpha: V,
    ) -> std::result::Result<Self::Output, TensorError>;

    /// elu method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`elu`]
    #[track_caller]
    fn elu_<V, U>(&self, alpha: V, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        V: Cast<Self::OutputMeta>,
        U: BorrowMut<Self::InplaceOutput>;

    /// Computes the element-wise error function (erf) of the tensor.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.erf()?;
    /// ```
    fn erf(&self) -> std::result::Result<Self::Output, TensorError>;

    /// erf method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`erf`]
    #[track_caller]
    fn erf_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// Computes the element-wise Gaussian Error Linear Unit (GELU) activation function.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.gelu()?;
    /// ```
    #[track_caller]
    fn gelu(&self) -> std::result::Result<Self::Output, TensorError>;

    /// gelu method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`gelu`]
    #[track_caller]
    fn gelu_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// Computes the element-wise Scaled Exponential Linear Unit (SELU) activation function.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.selu()?;
    /// ```
    #[track_caller]
    fn selu(&self) -> std::result::Result<Self::Output, TensorError>;

    /// selu method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`selu`]
    #[track_caller]
    fn selu_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// Computes the element-wise Hard Sigmoid activation function.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.hard_sigmoid()?;
    /// ```
    #[track_caller]
    fn hard_sigmoid(&self) -> std::result::Result<Self::Output, TensorError>;

    /// hard_sigmoid method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`hard_sigmoid`]
    #[track_caller]
    fn hard_sigmoid_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// Computes the element-wise Hard Swish activation function.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.hard_swish()?;
    /// ```
    #[track_caller]
    fn hard_swish(&self) -> std::result::Result<Self::Output, TensorError>;

    /// hard_swish method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`hard_swish`]
    #[track_caller]
    fn hard_swish_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// Computes the element-wise Softplus activation function.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.softplus()?;
    /// ```
    #[track_caller]
    fn softplus(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Softplus method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`softplus`]
    #[track_caller]
    fn softplus_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// Computes the element-wise Softsign activation function.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.softsign()?;
    /// ```
    #[track_caller]
    fn softsign(&self) -> std::result::Result<Self::Output, TensorError>;

    /// softsign method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`softsign`]
    #[track_caller]
    fn softsign_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// Computes the element-wise Mish activation function.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.mish()?;
    /// ```
    #[track_caller]
    fn mish(&self) -> std::result::Result<Self::Output, TensorError>;

    /// mish method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`mish`]
    #[track_caller]
    fn mish_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// Computes the element-wise cube root of the tensor.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.cbrt()?;
    /// ```
    #[track_caller]
    fn cbrt(&self) -> std::result::Result<Self::Output, TensorError>;

    /// cbrt method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`cbrt`]
    #[track_caller]
    fn cbrt_<U>(&self, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;
}

/// A trait for unary operations, the output must be the same type as the input.
pub trait NormalUaryOps
where
    Self: Sized,
{
    /// The output type of the unary operation.
    type Output;
    /// The output type of the inplace unary operation.
    type InplaceOutput;
    /// The output type of the unary operation.
    type OutputMeta;

    /// Computes the element-wise floor of the tensor.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.floor()?;
    /// ```
    #[track_caller]
    fn floor(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Floor method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`floor`]: Computes the element-wise floor of the tensor.
    #[track_caller]
    fn floor_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// Computes the element-wise square of the tensor.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.square()?;
    /// ```
    #[track_caller]
    fn square(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Square method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`square`]
    #[track_caller]
    fn square_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// Computes the element-wise absolute value of the tensor.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.abs()?;
    /// ```
    #[track_caller]
    fn abs(&self) -> std::result::Result<Self::Output, TensorError>;

    /// abs method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`abs`]
    #[track_caller]
    fn abs_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// Computes the element-wise ceiling of the tensor.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.ceil()?;
    /// ```
    #[track_caller]
    fn ceil(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Ceil method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`ceil`]
    #[track_caller]
    fn ceil_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// Computes the element-wise sign of the tensor.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.sign()?;
    /// ```
    #[track_caller]
    fn sign(&self) -> std::result::Result<Self::Output, TensorError>;

    /// sign method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`sign`]
    #[track_caller]
    fn sign_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// Clamps (limits) the values of the tensor between the specified `min` and `max`.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.clamp(0.0, 1.0)?;
    /// ```
    #[track_caller]
    fn clamp(
        &self,
        min: Self::OutputMeta,
        max: Self::OutputMeta,
    ) -> std::result::Result<Self::Output, TensorError>;

    /// clamp method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`clamp`]
    #[track_caller]
    fn clamp_<U>(
        &self,
        min: Self::OutputMeta,
        max: Self::OutputMeta,
        out: U,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// Computes the element-wise rounding of the tensor.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.round()?;
    /// ```
    #[track_caller]
    fn round(&self) -> std::result::Result<Self::Output, TensorError>;

    /// round method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`round`]
    #[track_caller]
    fn round_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// Computes the element-wise negation (multiplying by -1) of the tensor.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.neg()?;
    /// ```
    #[track_caller]
    fn neg(&self) -> std::result::Result<Self::Output, TensorError>;

    /// Inplace Version of neg.
    ///
    /// # See Also
    ///
    /// - [`neg`]
    #[track_caller]
    fn neg_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// Computes the element-wise Rectified Linear Unit (ReLU) activation function.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.relu()?;
    /// ```
    #[track_caller]
    fn relu(&self) -> std::result::Result<Self::Output, TensorError>;

    /// relu method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`relu`]
    #[track_caller]
    fn relu_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// Computes the element-wise Leaky Rectified Linear Unit (Leaky ReLU) activation function.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.leaky_relu(0.01)?;
    /// ```
    #[track_caller]
    fn leaky_relu(&self, alpha: Self::OutputMeta)
        -> std::result::Result<Self::Output, TensorError>;

    /// leaky_relu method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`leaky_relu`]
    #[track_caller]
    fn leaky_relu_<U>(
        &self,
        alpha: Self::OutputMeta,
        out: U,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// Computes the element-wise Rectified Linear Unit 6 (ReLU6) activation function.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([10.0]);
    /// let b = a.relu6()?;
    /// ```
    #[track_caller]
    fn relu6(&self) -> std::result::Result<Self::Output, TensorError>;

    /// relu6 method with output tensor, this method will write the result to the output tensor
    /// # See Also
    /// - [`relu6`]
    #[track_caller]
    fn relu6_<U>(&self, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;
}

/// A trait for accumulative operations.
pub trait Cum
where
    Self: Sized,
    <<Self as Cum>::Meta as TypeCommon>::Vec: Send + Sync,
{
    /// The output type of the accumulative operation.
    type Meta: CommonBounds;
    /// The output type of the accumulative operation.
    type Output;
    /// Computes the cumulative sum of the elements in the tensor along a specified axis.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([1.0, 2.0, 3.0]);
    /// let b = a.cumsum(None)?;
    /// ```
    #[track_caller]
    fn cumsum(&self, axis: Option<i64>) -> std::result::Result<Self::Output, TensorError>
    where
        Self::Meta: NormalOut<Self::Meta, Output = Self::Meta>;

    /// Computes the cumulative product of the elements in the tensor along a specified axis.
    ///
    /// # Example
    /// ```rust
    /// let a = Tensor::<f32>::new([1.0, 2.0, 3.0]);
    /// let b = a.cumprod(None)?;
    /// ```
    #[track_caller]
    fn cumprod(&self, axis: Option<i64>) -> std::result::Result<Self::Output, TensorError>
    where
        Self::Meta: NormalOut<Self::Meta, Output = Self::Meta>;
}

/// A trait for contiguous operation
pub trait Contiguous: Sized {
    /// Returns the tensor as a contiguous tensor.
    ///
    /// # Note
    ///
    /// This function will return a contiguous tensor. If the tensor is already contiguous, it will return a clone of the tensor.
    ///
    /// If the tensor is a view tensor, it will return a new tensor with the same data but with a contiguous layout.
    #[track_caller]
    fn contiguous(&self) -> Result<Self, TensorError>;
}
