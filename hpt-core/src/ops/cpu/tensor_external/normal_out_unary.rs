use std::borrow::BorrowMut;

use crate::{
    ops::cpu::tensor_internal::normal_out_unary::NormalType, tensor::Tensor, tensor_base::_Tensor,
    Cpu,
};
use hpt_common::error::base::TensorError;
use hpt_traits::{CommonBounds, NormalUaryOps, TensorLike};
use hpt_types::{into_scalar::Cast, type_promote::NormalOutUnary};

impl<T, const DEVICE: usize> NormalUaryOps for Tensor<T, Cpu, DEVICE>
where
    T: CommonBounds + Cast<T>,
    NormalType<T>: CommonBounds,
    T::Vec: NormalOutUnary,
    T: NormalOutUnary,
    _Tensor<NormalType<T>>: TensorLike<NormalType<T>>,
{
    type Output = Tensor<NormalType<T>, Cpu, DEVICE>;

    type InplaceOutput = Tensor<NormalType<T>, Cpu, DEVICE>;

    type OutputMeta = NormalType<T>;

    /// Computes the element-wise floor of the tensor.
    ///
    /// This function rounds each element in the tensor down to the nearest integer, returning a new tensor
    /// where each element is the largest integer less than or equal to the corresponding element in the original tensor.
    /// # Arguments
    /// This function takes no arguments.
    /// # Returns
    /// * A new tensor where each element is the floor of the corresponding element in the original tensor.
    /// # Panics
    /// * This function should not panic under normal conditions.
    /// # Examples
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::NormalUaryOps;
    /// let a = Tensor::<f64>::new([0.1, 1.5, 2.9, 3.0]);
    /// let b = a.floor().unwrap();
    /// ```
    fn floor(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::floor(self.inner.as_ref())?.into())
    }

    fn floor_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::floor_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    /// Computes the element-wise square of the tensor.
    ///
    /// This function returns a new tensor where each element is the square of the corresponding element in the original tensor:
    ///
    /// `square(x) = x^2`
    /// # Arguments
    /// This function takes no arguments.
    /// # Returns
    /// * A new tensor where each element is squared.
    /// # Panics
    /// * This function should not panic under normal conditions.
    /// # Examples
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::NormalUaryOps;
    /// let a = Tensor::<f64>::new([0.1, 1.5, 2.9, 3.0]);
    /// let b = a.square().unwrap();
    /// ```
    fn square(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::square(self.inner.as_ref())?.into())
    }

    fn square_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::square_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

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
    /// # Examples
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::NormalUaryOps;
    /// let a = Tensor::<f64>::new([-1.0, 1.5, -2.9, 3.0]);
    /// let b = a.abs().unwrap();
    /// ```
    fn abs(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::abs(self.inner.as_ref())?.into())
    }

    fn abs_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::abs_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

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
    /// # Examples
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::NormalUaryOps;
    /// let a = Tensor::<f64>::new([0.1, 1.5, 2.9, 3.0]);
    /// let b = a.ceil().unwrap();
    /// ```
    fn ceil(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::ceil(self.inner.as_ref())?.into())
    }

    fn ceil_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::ceil_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

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
    /// # Examples
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::NormalUaryOps;
    /// let a = Tensor::<f64>::new([-1.0, 1.5, -2.9, 3.0]);
    /// let b = a.sign().unwrap();
    /// ```
    fn sign(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::sign(self.inner.as_ref())?.into())
    }

    fn sign_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::sign_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    /// Clips (limits) the values of the tensor between the specified `min` and `max`.
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
    /// # Examples
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::NormalUaryOps;
    /// let a = Tensor::<f64>::new([-1.0, 1.5, -2.9, 3.0]);
    /// let b = a.clip(-1.0, 1.0).unwrap();
    /// ```
    fn clamp(
        &self,
        min: Self::OutputMeta,
        max: Self::OutputMeta,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::clamp(self.inner.as_ref(), min, max)?.into())
    }

    fn clamp_<U>(
        &self,
        min: Self::OutputMeta,
        max: Self::OutputMeta,
        mut out: U,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::clamp_(
            self.inner.as_ref(),
            min,
            max,
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

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
    /// # Examples
    /// ```
    /// use hpt::tensor::Tensor;
    /// use hpt::NormalUaryOps;
    /// let a = Tensor::<f64>::new([0.1, 1.5, 2.9, 3.0]);
    /// let b = a.round().unwrap();
    /// ```
    fn round(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::round(self.inner.as_ref())?.into())
    }

    fn round_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::round_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn neg(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::neg(self.inner.as_ref())?.into())
    }

    fn neg_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::neg_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn relu(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::relu(self.inner.as_ref())?.into())
    }

    fn relu_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::relu_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn leaky_relu(
        &self,
        alpha: Self::OutputMeta,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::leaky_relu(self.inner.as_ref(), alpha)?.into())
    }

    fn leaky_relu_<U>(
        &self,
        alpha: Self::OutputMeta,
        mut out: U,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::leaky_relu_(
            self.inner.as_ref(),
            alpha,
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }

    fn relu6(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::<T, Cpu, DEVICE>::relu6(self.inner.as_ref())?.into())
    }

    fn relu6_<U>(&self, mut out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
    {
        Ok(_Tensor::<T, Cpu, DEVICE>::relu6_(
            self.inner.as_ref(),
            out.borrow_mut().inner.as_ref().clone(),
        )?
        .into())
    }
}
