use std::{
    fmt::{Debug, Display},
    ops::{Add, Deref, Mul, Rem, Sub},
    sync::{atomic::Ordering, Arc},
};

use crate::{
    backend::{BackendTy, Buffer, Cpu},
    tensor_base::_Tensor,
    DISPLAY_LR_ELEMENTS, DISPLAY_PRECISION,
};
use anyhow::Result;
use tensor_common::{layout::Layout, pointer::Pointer, shape::Shape};
use tensor_display::display;
use tensor_iterator::TensorIterator;
use tensor_traits::tensor::{CommonBounds, TensorAlloc, TensorCreator, TensorInfo, TensorLike};
use tensor_types::{
    convertion::Convertor, dtype::TypeCommon, into_scalar::IntoScalar, type_promote::NormalOut,
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
