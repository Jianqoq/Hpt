use std::{
    borrow::Borrow,
    fmt::{Debug, Display},
    ops::{Add, Deref, Div, Mul, Rem, Sub},
    sync::{atomic::Ordering, Arc},
};

use crate::{
    backend::{BackendTy, Buffer, Cpu},
    ops::cpu::unary::{FloatUnaryType, NormalType},
    tensor_base::_Tensor,
    BoolVector, DISPLAY_LR_ELEMENTS, DISPLAY_PRECISION,
};
use anyhow::Result;
use tensor_common::{layout::Layout, pointer::Pointer, shape::Shape};
use tensor_display::display;
use tensor_iterator::TensorIterator;
use tensor_traits::{
    tensor::{CommonBounds, TensorAlloc, TensorCreator, TensorInfo, TensorLike},
    NormalUaryOps,
};
use tensor_types::{
    convertion::{Convertor, FromScalar},
    dtype::TypeCommon,
    into_scalar::IntoScalar,
    type_promote::{FloatOutUnary, NormalOut},
};

/// A wrapper of `Tensor` for user.
/// This is the main tensor for user.
///
/// # Properties
/// - `basic`: The pointer of `Tensor`.
#[derive(Clone)]
pub struct Tensor<T, B = Cpu>
where
    B: BackendTy + Buffer,
{
    pub(crate) inner: Arc<_Tensor<T, B>>,
}

impl<T, B> Deref for Tensor<T, B>
where
    B: BackendTy + Buffer,
{
    type Target = _Tensor<T, B>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> TensorLike<T> for Tensor<T>
where
    T: CommonBounds,
{
    fn as_raw(&self) -> &[T] {
        self.inner.as_raw()
    }

    fn as_raw_mut(&mut self) -> &mut [T] {
        let slice =
            unsafe { std::slice::from_raw_parts_mut(self.inner.ptr().ptr as *mut T, self.size()) };
        slice
    }

    fn contiguous(&self) -> anyhow::Result<Self> {
        Ok(_Tensor::contiguous(self)?.into())
    }
}

impl<T: CommonBounds> TensorIterator<'_, T> for Tensor<T> {}

impl<T> TensorInfo<T> for Tensor<T>
where
    T: CommonBounds,
{
    fn ptr(&self) -> Pointer<T> {
        self.data.clone()
    }

    fn size(&self) -> usize {
        self.layout.size() as usize
    }

    fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    fn strides(&self) -> &tensor_common::strides::Strides {
        self.layout.strides()
    }

    fn layout(&self) -> &Layout {
        &self.layout
    }

    fn parent(&self) -> Option<Pointer<T>> {
        self.parent.clone()
    }

    fn ndim(&self) -> usize {
        self.layout.ndim()
    }

    fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }
}

impl<T> TensorInfo<T> for &Tensor<T>
where
    T: CommonBounds,
{
    fn ptr(&self) -> Pointer<T> {
        self.data.clone()
    }

    fn size(&self) -> usize {
        self.layout.size() as usize
    }

    fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    fn strides(&self) -> &tensor_common::strides::Strides {
        self.layout.strides()
    }

    fn layout(&self) -> &Layout {
        &self.layout
    }

    fn parent(&self) -> Option<Pointer<T>> {
        self.parent.clone()
    }

    fn ndim(&self) -> usize {
        self.layout.ndim()
    }

    fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }
}

impl<T: CommonBounds> TensorAlloc for Tensor<T> {
    type Meta = T;
    fn _empty<S: Into<Shape>>(shape: S) -> Result<Self>
    where
        Self: Sized,
    {
        Self::empty(shape)
    }
}

impl<T: CommonBounds> Tensor<T> {
    /// Converts the tensor to a new type.
    ///
    /// This method attempts to convert the elements of the tensor to a specified type `U`.
    /// It returns a new tensor with the converted elements.
    ///
    /// # Returns
    /// A `Result` containing the new tensor of type `U` or an error if the conversion fails.
    ///
    /// # Examples
    /// ```
    /// use tensor_core::Tensor;
    /// let tensor = Tensor::<f32>::new([1.0, 2.0, 3.0]);
    /// let converted_tensor = tensor.astype::<i32>().unwrap();
    /// assert!(tensor.allclose(&converted_tensor))
    /// ```
    pub fn astype<U>(&self) -> Result<Tensor<U>>
    where
        U: CommonBounds,
        T: IntoScalar<U>,
    {
        Ok(_Tensor::<T, Cpu>::astype(self)?.into())
    }

    /// Try to cast the tensor to a new type, with an optimization for same-type casting.
    ///
    /// This method checks if the target type `U` is the same as the current type `T`.
    /// If they are the same, it performs a static cast, which is more efficient.
    /// Otherwise, it falls back to the `astype` method.
    /// Note that type checking is done at compile time.
    ///
    /// # Returns
    /// A `Result` containing the new tensor of type `U` or an error if the conversion fails.
    ///
    /// # Examples
    /// ```
    /// use tensor_core::Tensor;
    /// let tensor = Tensor::<f32>::new([1.0, 2.0, 3.0]);
    /// let converted_tensor = tensor.try_astype::<i32>().unwrap();
    /// assert!(tensor.allclose(&converted_tensor))
    /// ```
    pub fn try_astype<U>(&self) -> Result<Tensor<U>>
    where
        U: CommonBounds,
        T: IntoScalar<U>,
    {
        if U::ID == T::ID {
            Ok(self.static_cast()?)
        } else {
            Ok(self.astype::<U>()?)
        }
    }

    /// Performs a static cast of the tensor to a new type without actual data conversion.
    ///
    /// This method is used when the target type `U` is the same as the current type `T`.
    /// It reinterprets the tensor data as the new type without changing the underlying bytes.
    ///
    /// # Returns
    /// A `Result` containing the new tensor of type `U`.
    ///
    /// # Examples
    /// ```
    /// use tensor_core::Tensor;
    /// let tensor = Tensor::<f32>::new([1.0, 2.0, 3.0]);
    /// let static_cast_tensor = tensor.static_cast::<f32>().unwrap();
    /// assert!(tensor.allclose(&static_cast_tensor))
    /// ```
    pub fn static_cast<U>(&self) -> Result<Tensor<U>>
    where
        U: CommonBounds,
    {
        Ok(_Tensor::<T, Cpu>::static_cast(self)?.into())
    }

    /// Checks if all elements of the tensor are close to the elements of another tensor.
    ///
    /// This method computes the absolute difference between each pair of corresponding elements
    /// and checks if they are within a specified tolerance. The default tolerance values
    /// are `1.0e-8` and `1.0e-5`.
    ///
    /// # Arguments
    /// - `other`: The other tensor to compare with.
    ///
    /// # Returns
    /// `true` if all elements are close; otherwise, `false`.
    ///
    /// # Examples
    /// ```
    /// use tensor_core::Tensor;
    /// let tensor1 = Tensor::<f64>::new([1.0, 2.0, 3.0]);
    /// let tensor2 = Tensor::<f64>::new([1.0, 2.0, 3.0]);
    /// assert!(tensor1.allclose(&tensor2));
    /// ```
    pub fn allclose<U>(&self, other: &Tensor<U>) -> bool
    where
        T: Convertor,
        U: Convertor + CommonBounds,
    {
        self.inner.allclose(&other.inner)
    }
}

impl<T: CommonBounds> TensorCreator<T> for Tensor<T> {
    fn empty<S: Into<Shape>>(shape: S) -> Result<Self> {
        Ok(_Tensor::<T, Cpu>::empty(shape)?.into())
    }

    fn zeros<S: Into<Shape>>(shape: S) -> Result<Self> {
        Ok(_Tensor::<T, Cpu>::zeros(shape)?.into())
    }

    fn ones<S: Into<Shape>>(shape: S) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        Ok(_Tensor::<T, Cpu>::ones(shape)?.into())
    }

    fn empty_like(&self) -> Result<Self> {
        Ok(_Tensor::empty_like(self)?.into())
    }

    fn zeros_like(&self) -> Result<Self> {
        Ok(_Tensor::zeros_like(self)?.into())
    }

    fn ones_like(&self) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        Ok(_Tensor::ones_like(self)?.into())
    }

    fn full<S: Into<Shape>>(val: T, shape: S) -> Result<Self> {
        Ok(_Tensor::<T, Cpu>::full(val, shape)?.into())
    }

    fn full_like(&self, val: T) -> Result<Self> {
        Ok(_Tensor::full_like(self, val)?.into())
    }

    fn arange<U>(start: U, end: U) -> Result<Self>
    where
        T: Convertor + FromScalar<U> + NormalOut<T, Output = T>,
        usize: IntoScalar<T>,
        U: Convertor + IntoScalar<T> + Copy,
    {
        Ok(_Tensor::<T, Cpu>::arange(start, end)?.into())
    }

    fn arange_step(start: T, end: T, step: T) -> Result<Self>
    where
        T: Convertor + FromScalar<usize> + NormalOut<T, Output = T>,
    {
        Ok(_Tensor::<T, Cpu>::arange_step(start, end, step)?.into())
    }

    fn eye(n: usize, m: usize, k: usize) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        Ok(_Tensor::<T, Cpu>::eye(n, m, k)?.into())
    }

    fn linspace(start: T, end: T, num: usize, include_end: bool) -> Result<Self>
    where
        T: Convertor + num::Float + NormalOut<T, Output = T>,
        usize: IntoScalar<T>,
        f64: IntoScalar<T>,
    {
        Ok(_Tensor::<T, Cpu>::linspace(start, end, num, include_end)?.into())
    }

    fn logspace(start: T, end: T, num: usize, include_end: bool, base: T) -> Result<Self>
    where
        T: Convertor + num::Float + FromScalar<usize> + FromScalar<f64> + NormalOut<T, Output = T>,
    {
        Ok(_Tensor::<T, Cpu>::logspace(start, end, num, include_end, base)?.into())
    }

    fn geomspace(start: T, end: T, n: usize, include_end: bool) -> Result<Self>
    where
        T: PartialOrd
            + FloatOutUnary
            + NormalOut<T, Output = T>
            + FromScalar<FloatUnaryType<T>>
            + std::ops::Neg<Output = T>,
        FloatUnaryType<T>: Sub<Output = FloatUnaryType<T>>
            + FromScalar<usize>
            + FromScalar<f64>
            + Div<Output = FloatUnaryType<T>>
            + NormalOut<Output = FloatUnaryType<T>>
            + CommonBounds,
    {
        Ok(_Tensor::<T, Cpu>::geomspace(start, end, n, include_end)?.into())
    }

    fn tri(n: usize, m: usize, k: i64, low_triangle: bool) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        Ok(_Tensor::<T, Cpu>::tri(n, m, k, low_triangle)?.into())
    }

    fn tril(&self, k: i64) -> Result<Self>
    where
        T: NormalOut<bool, Output = T> + IntoScalar<T> + TypeCommon,
        <T as TypeCommon>::Vec: NormalOut<BoolVector, Output = <T as TypeCommon>::Vec>,
    {
        Ok(_Tensor::tril(self, k)?.into())
    }

    fn triu(&self, k: i64) -> Result<Self>
    where
        T: NormalOut<bool, Output = T> + IntoScalar<T> + TypeCommon,
        <T as TypeCommon>::Vec: NormalOut<BoolVector, Output = <T as TypeCommon>::Vec>,
    {
        Ok(_Tensor::triu(self, k)?.into())
    }

    fn identity(n: usize) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        Ok(_Tensor::<T, Cpu>::identity(n)?.into())
    }
}

impl<T> NormalUaryOps for Tensor<T>
where
    T: NormalOut<Output = T> + CommonBounds + IntoScalar<T>,
    NormalType<T>: CommonBounds,
    _Tensor<NormalType<T>>: TensorLike<NormalType<T>>,
    T::Vec: NormalOut<Output = T::Vec>,
{
    type Output = Tensor<NormalType<T>>;

    type InplaceOutput = _Tensor<NormalType<T>>;

    type OutputMeta = NormalType<T>;

    fn floor(&self) -> Result<Self::Output> {
        Ok(_Tensor::floor(self)?.into())
    }

    fn floor_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::floor_(self, out)?.into())
    }

    fn square(&self) -> Result<Self::Output> {
        Ok(_Tensor::square(self)?.into())
    }

    fn square_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::square_(self, out)?.into())
    }

    fn abs(&self) -> Result<Self> {
        Ok(_Tensor::abs(self)?.into())
    }

    fn abs_<U>(&self, out: U) -> Result<Self>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::abs_(self, out)?.into())
    }

    fn ceil(&self) -> Result<Self::Output> {
        Ok(_Tensor::ceil(self)?.into())
    }

    fn ceil_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::ceil_(self, out)?.into())
    }

    fn sign(&self) -> Result<Self::Output> {
        Ok(_Tensor::sign(self)?.into())
    }

    fn sign_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::sign_(self, out)?.into())
    }

    fn clip(&self, min: Self::OutputMeta, max: Self::OutputMeta) -> Result<Self::Output> {
        Ok(_Tensor::clip(self, min, max)?.into())
    }

    fn clip_<U>(&self, min: Self::OutputMeta, max: Self::OutputMeta, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::clip_(self, min, max, out)?.into())
    }

    fn round(&self) -> Result<Self::Output> {
        Ok(_Tensor::round(self)?.into())
    }

    fn round_<U>(&self, out: U) -> Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        Ok(_Tensor::round_(self, out)?.into())
    }
}

impl<T> Display for Tensor<T>
where
    T: CommonBounds + Convertor,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let precision = DISPLAY_PRECISION.load(Ordering::Relaxed);
        let lr_element_size = DISPLAY_LR_ELEMENTS.load(Ordering::Relaxed);
        display(self, f, lr_element_size, precision, false)
    }
}

impl<T> Debug for Tensor<T>
where
    T: CommonBounds + Convertor,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let precision = DISPLAY_PRECISION.load(Ordering::Relaxed);
        let lr_element_size = DISPLAY_LR_ELEMENTS.load(Ordering::Relaxed);
        display(self, f, lr_element_size, precision, false)
    }
}

macro_rules! normal_ops_1 {
    ($op:ident, $op2:ident) => {
        impl<T, U> $op<Tensor<U>> for Tensor<T>
        where
            T: CommonBounds + NormalOut<U>,
            U: CommonBounds,
            <T as NormalOut<U>>::Output: CommonBounds,
            <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>,
            T::Vec: NormalOut<U::Vec, Output = <<T as NormalOut<U>>::Output as TypeCommon>::Vec>,
        {
            type Output = Tensor<<T as NormalOut<U>>::Output>;

            fn $op2(self, rhs: Tensor<U>) -> Self::Output {
                (self.inner.as_ref().$op2(rhs.inner.as_ref().clone())).into()
            }
        }
    };
}

macro_rules! normal_ops_2 {
    ($op:ident, $op2:ident) => {
        impl<'a, T, U> std::ops::$op<&'a Tensor<U>> for Tensor<T>
        where
            T: CommonBounds + NormalOut<U>,
            U: CommonBounds,
            <T as NormalOut<U>>::Output: CommonBounds,
            <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>,
            T::Vec: NormalOut<U::Vec, Output = <<T as NormalOut<U>>::Output as TypeCommon>::Vec>,
        {
            type Output = Tensor<<T as NormalOut<U>>::Output>;

            fn $op2(self, rhs: &'a Tensor<U>) -> Self::Output {
                (self.inner.as_ref() + rhs.inner.as_ref().clone()).into()
            }
        }
    };
}

macro_rules! normal_ops_3 {
    ($op:ident, $op2:ident) => {
        impl<'a, T, U> std::ops::$op<&'a Tensor<U>> for &'a Tensor<T>
        where
            T: CommonBounds + NormalOut<U>,
            U: CommonBounds,
            <T as NormalOut<U>>::Output: CommonBounds,
            <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>,
            T::Vec: NormalOut<U::Vec, Output = <<T as NormalOut<U>>::Output as TypeCommon>::Vec>,
        {
            type Output = Tensor<<T as NormalOut<U>>::Output>;

            fn $op2(self, rhs: &'a Tensor<U>) -> Self::Output {
                (self.inner.as_ref() + rhs.inner.as_ref().clone()).into()
            }
        }
    };
}

macro_rules! normal_ops_4 {
    ($op:ident, $op2:ident) => {
        impl<'a, T, U> std::ops::$op<Tensor<U>> for &'a Tensor<T>
        where
            T: CommonBounds + NormalOut<U>,
            U: CommonBounds,
            <T as NormalOut<U>>::Output: CommonBounds,
            <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>,
            T::Vec: NormalOut<U::Vec, Output = <<T as NormalOut<U>>::Output as TypeCommon>::Vec>,
        {
            type Output = Tensor<<T as NormalOut<U>>::Output>;

            fn $op2(self, rhs: Tensor<U>) -> Self::Output {
                (self.inner.as_ref() + rhs.inner.as_ref().clone()).into()
            }
        }
    };
}

normal_ops_1!(Add, add);
normal_ops_1!(Sub, sub);
normal_ops_1!(Mul, mul);
normal_ops_1!(Rem, rem);

normal_ops_2!(Add, add);
normal_ops_2!(Sub, sub);
normal_ops_2!(Mul, mul);
normal_ops_2!(Rem, rem);

normal_ops_3!(Add, add);
normal_ops_3!(Sub, sub);
normal_ops_3!(Mul, mul);
normal_ops_3!(Rem, rem);

normal_ops_4!(Add, add);
normal_ops_4!(Sub, sub);
normal_ops_4!(Mul, mul);
normal_ops_4!(Rem, rem);
