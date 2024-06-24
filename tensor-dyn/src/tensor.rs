use std::{ fmt::{ Debug, Display }, ops::{ Deref, Div, Sub }, sync::Arc };

use tensor_common::{ axis::Axis, layout::Layout, pointer::Pointer, shape::Shape };
use tensor_display::display;
use tensor_iterator::{ strided::Strided, strided_mut::StridedMut };
use tensor_traits::{
    shape_manipulate::ShapeManipulate,
    tensor::{ CommonBounds, TensorAlloc, TensorCreator, TensorInfo, TensorLike },
};
use tensor_types::{
    convertion::{ Convertor, FromScalar },
    into_scalar::IntoScalar,
    type_promote::{ FloatOut, NormalOut },
};
use anyhow::Result;
use crate::{ ops::cpu::reduce::stack, tensor_base::_Tensor };

/// A wrapper of `Tensor` for user.
/// This is the main tensor for user.
///
/// # Properties
/// - `basic`: The pointer of `Tensor`.
pub struct Tensor<T> {
    pub(crate) inner: Arc<_Tensor<T>>,
}

impl<T> Deref for Tensor<T> {
    type Target = _Tensor<T>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T, U> TensorLike<T, U, Tensor<U>>
    for Tensor<T>
    where T: IntoScalar<U> + CommonBounds, U: CommonBounds
{
    type Output = Tensor<U>;
    fn to_raw(&self) -> &[T] {
        self.as_raw()
    }

    fn to_raw_mut(&self) -> &mut [T] {
        self.as_raw_mut()
    }

    fn static_cast(&self) -> Result<Self::Output> {
        self.static_cast()
    }
}

impl<T> TensorInfo<T> for Tensor<T> {
    fn ptr(&self) -> Pointer<T> {
        self.data
    }

    fn size(&self) -> usize {
        self.layout.size()
    }

    fn shape(&self) -> &tensor_common::shape::Shape {
        self.layout.shape()
    }

    fn strides(&self) -> &tensor_common::strides::Strides {
        self.layout.strides()
    }

    fn layout(&self) -> &Layout {
        &self.layout
    }

    fn parent(&self) -> Option<Pointer<T>> {
        self.parent
    }

    fn ndim(&self) -> usize {
        self.layout.ndim()
    }

    fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }
}

impl<T> TensorInfo<T> for &Tensor<T> {
    fn ptr(&self) -> Pointer<T> {
        self.data
    }

    fn size(&self) -> usize {
        self.layout.size()
    }

    fn shape(&self) -> &tensor_common::shape::Shape {
        self.layout.shape()
    }

    fn strides(&self) -> &tensor_common::strides::Strides {
        self.layout.strides()
    }

    fn layout(&self) -> &Layout {
        &self.layout
    }

    fn parent(&self) -> Option<Pointer<T>> {
        self.parent
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

    fn _empty<S: Into<Shape>>(shape: S) -> anyhow::Result<Self> where Self: Sized {
        Self::empty(shape)
    }
}

impl<T: CommonBounds> Tensor<T> {
    /// Converts a tensor to a raw slice representing direct memory access.
    ///
    /// This function provides direct, read-only access to the tensor's underlying memory. It is useful
    /// for interfacing with low-level or external functions that require direct memory access.
    ///
    /// # Returns
    /// `&[T]`: A raw slice providing direct, read-only access to the tensor's memory.
    ///
    /// # Caution
    /// - Direct memory access can lead to undefined behavior if not handled properly.
    /// - This function bypasses Rust's safety checks, so caution must be exercised to avoid data corruption
    ///   or undefined behavior.
    /// - The caller is responsible for ensuring that the memory accessed is valid and not being mutated
    ///   elsewhere concurrently.
    ///
    /// # Examples
    /// ```
    /// let tensor = YourType::new(...);
    /// let direct_memory_access = tensor.as_raw();
    /// // Use direct_memory_access for operations requiring direct memory access
    /// ```
    pub fn as_raw(&self) -> &[T] {
        let ptr = self.data.ptr;
        let size;
        if !self.is_contiguous() {
            size = self.layout.real_size();
        } else {
            size = self.size();
        }
        let slice = unsafe { std::slice::from_raw_parts(ptr as *mut T, size) };
        slice
    }

    /// Converts a tensor to a raw mutable slice representing direct memory access.
    ///
    /// This function provides direct, mutable access to the tensor's underlying memory. It is intended for
    /// advanced use cases where direct memory manipulation is required.
    ///
    /// # Returns
    /// `&mut [T]`: A raw mutable slice providing direct access to the tensor's memory.
    ///
    /// # Caution
    /// - Modifying data through this interface can lead to undefined behavior and should be done with utmost care.
    /// - This method bypasses Rust's safety and concurrency checks.
    /// - The caller must ensure that no other references to the tensor data are being mutated concurrently.
    ///
    /// # Examples
    /// ```
    /// let mut tensor = YourType::new(...);
    /// let direct_memory_access_mut = tensor.as_raw_mut();
    /// // Perform operations requiring direct and mutable memory access
    /// ```
    pub fn as_raw_mut(&self) -> &mut [T] {
        let ptr = self.data.ptr;
        let size;
        if !self.is_contiguous() {
            size = self.layout.real_size();
        } else {
            size = self.size();
        }
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut T, size) };
        slice
    }

    pub fn iter(&self) -> Strided<T> {
        Strided::new(self)
    }

    pub fn iter_mut(&self) -> StridedMut<T> {
        StridedMut::new(self)
    }

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
    pub fn astype<U>(&self) -> Result<Tensor<U>> where U: CommonBounds, T: IntoScalar<U> {
        Ok(_Tensor::astype(self)?.into())
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
    pub fn try_astype<U>(&self) -> Result<Tensor<U>> where U: CommonBounds, T: IntoScalar<U> {
        if U::ID == T::ID {
            return Ok(self.static_cast()?);
        } else {
            return Ok(self.astype::<U>()?);
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
    pub fn static_cast<U>(&self) -> Result<Tensor<U>> where U: CommonBounds {
        Ok(_Tensor::static_cast(self)?.into())
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
    pub fn allclose<U: CommonBounds>(&self, other: &Tensor<U>) -> bool
        where T: Convertor, U: Convertor
    {
        self.inner.allclose(&other.inner)
    }

    /// Create a contiguous copy of the tensor.
    ///
    /// This method returns a new tensor that has the same data as the original tensor,
    /// but with elements laid out in contiguous memory.
    ///
    /// # Returns
    /// A `Result` containing the new contiguous tensor.
    ///
    /// # Examples
    /// ```
    /// use tensor_core::Tensor;
    /// use tensor_trait::{ShapeManipulate, TensorInfo};
    /// let tensor = Tensor::<f32>::new([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
    /// let transposed_tensor = tensor.transpose([1, 0]).unwrap();
    /// let contiguous_tensor = tensor.contiguous().unwrap();
    /// assert!(contiguous_tensor.is_contiguous())
    /// ```
    pub fn contiguous(&self) -> Result<Self> {
        Ok(_Tensor::contiguous(self)?.into())
    }

    /// Stacks a sequence of tensors along a specified axis.
    ///
    /// Given a list of tensors, this function concatenates them along the specified axis.
    /// All tensors must have the same shape, except in the dimension corresponding to `axis`.
    ///
    /// # Arguments
    /// - `tensors`: A vector of tensor references to be stacked.
    /// - `axis`: The axis along which the tensors will be stacked.
    /// - `keepdims`: A boolean indicating whether to keep the dimension of the axis or not.
    ///
    /// # Returns
    /// A `Result` containing the stacked tensor or an error if the operation fails.
    ///
    /// # Examples
    /// ```
    /// use tensor_core::Tensor;
    /// let tensor1 = Tensor::<f32>::new([1.0, 2.0, 3.0]);
    /// let tensor2 = Tensor::<f32>::new([4.0, 5.0, 6.0]);
    /// let stacked_tensor = Tensor::stack(vec![&tensor1, &tensor2], 0, true).unwrap();
    /// assert!(stacked_tensor.allclose(&Tensor::<f64>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])));
    /// ```
    pub fn stack(tensors: Vec<&Tensor<T>>, axis: usize, keepdims: bool) -> Result<Self>
        where T: 'static
    {
        Ok(
            stack(
                tensors
                    .iter()
                    .map(|x| x.inner.as_ref())
                    .collect(),
                axis,
                keepdims
            )?.into()
        )
    }

    /// Vertically stacks a sequence of tensors.
    ///
    /// This is a convenience method for stacking tensors along the first axis (axis=0).
    /// All tensors must have the same number of dimensions and the same shape,
    /// except for the first axis.
    ///
    /// # Arguments
    /// - `tensors`: A vector of tensor references to be vertically stacked.
    ///
    /// # Returns
    /// A `Result` containing the vertically stacked tensor or an error if the operation fails.
    ///
    /// # Examples
    /// ```
    /// use tensor_core::Tensor;
    /// let tensor1 = Tensor::<f32>::new([1.0, 2.0, 3.0]);
    /// let tensor2 = Tensor::<f32>::new([4.0, 5.0, 6.0]);
    /// let vstacked_tensor = Tensor::vstack(vec![&tensor1, &tensor2]).unwrap();
    /// assert!(vstacked_tensor.allclose(&Tensor::<f64>::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])));
    /// ```
    pub fn vstack(tensors: Vec<&Tensor<T>>) -> Result<Tensor<T>> {
        Ok(
            _Tensor
                ::vstack(
                    tensors
                        .into_iter()
                        .map(|x| x.inner.as_ref())
                        .collect()
                )?
                .into()
        )
    }
    /// Horizontally stacks a sequence of tensors.
    ///
    /// This function concatenates tensors along the second axis (axis=1).
    /// It automatically reshapes tensors with fewer dimensions to have an additional axis.
    /// For 1-dimensional tensors, they are reshaped to 2D before stacking.
    /// Scalars are reshaped to 1x1 tensors.
    ///
    /// # Arguments
    /// - `tensors`: A vector of references to the tensors to be horizontally stacked.
    ///
    /// # Returns
    /// A `Result` containing the horizontally stacked tensor or an error if the operation fails.
    ///
    /// # Examples
    /// ```
    /// use tensor_core::Tensor;
    /// let tensor1 = Tensor::<f32>::new([1.0, 2.0, 3.0]);
    /// let tensor2 = Tensor::<f32>::new([4.0, 5.0, 6.0]);
    /// let hstacked_tensor = Tensor::hstack(vec![&tensor1, &tensor2]).unwrap();
    /// assert!(hstacked_tensor.allclose(&Tensor::<f64>::new([1.0, 2.0, 3.0,4.0, 5.0, 6.0])));
    /// ```
    pub fn hstack(mut tensors: Vec<&Tensor<T>>) -> Result<Tensor<T>> {
        Ok(
            stack(
                tensors
                    .iter_mut()
                    .map(|x| x.inner.as_ref())
                    .collect(),
                1,
                false
            )?.into()
        )
    }

    /// Depth-stacks a sequence of tensors.
    ///
    /// This function concatenates tensors along the third axis (axis=2).
    /// It automatically reshapes tensors with fewer dimensions to match the required number of dimensions.
    /// For 1-dimensional tensors, they are reshaped to 1xNx1 before stacking.
    /// For 2-dimensional tensors, they are reshaped to NxMx1.
    /// Scalars are reshaped to 1x1x1 tensors.
    ///
    /// # Arguments
    /// - `tensors`: A vector of references to the tensors to be depth-stacked.
    ///
    /// # Returns
    /// A `Result` containing the depth-stacked tensor or an error if the operation fails.
    ///
    /// # Examples
    /// ```
    /// use tensor_core::Tensor;
    /// let tensor1 = Tensor::<f32>::new([1.0, 2.0, 3.0]);
    /// let tensor2 = Tensor::<f32>::new([4.0, 5.0, 6.0]);
    /// let dstacked_tensor = Tensor::dstack(vec![&tensor1, &tensor2]).unwrap();
    /// assert!(dstacked_tensor.allclose(&Tensor::<f64>::new([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]])));
    /// ```
    pub fn dstack(mut tensors: Vec<&Tensor<T>>) -> Result<Tensor<T>> {
        Ok(
            stack(
                tensors
                    .iter_mut()
                    .map(|x| x.inner.as_ref())
                    .collect(),
                2,
                false
            )?.into()
        )
    }
}

impl<T: CommonBounds> TensorCreator<T> for Tensor<T> {
    type StridedIter = Strided<T>;

    type Mask = Tensor<bool>;

    type Basic = Tensor<T>;

    fn empty<S: Into<Shape>>(shape: S) -> anyhow::Result<Self> {
        Ok(_Tensor::empty(shape)?.into())
    }

    fn zeros<S: Into<Shape>>(shape: S) -> anyhow::Result<Self> {
        Ok(_Tensor::zeros(shape)?.into())
    }

    fn ones<S: Into<Shape>>(shape: S) -> anyhow::Result<Self> where u8: IntoScalar<T> {
        Ok(_Tensor::ones(shape)?.into())
    }

    fn empty_like(&self) -> anyhow::Result<Self> {
        Ok(_Tensor::empty_like(self)?.into())
    }

    fn zeros_like(&self) -> anyhow::Result<Self> {
        Ok(_Tensor::zeros_like(self)?.into())
    }

    fn ones_like(&self) -> anyhow::Result<Self> where u8: IntoScalar<T> {
        Ok(_Tensor::ones_like(self)?.into())
    }

    fn full<S: Into<Shape>>(val: T, shape: S) -> anyhow::Result<Self> {
        Ok(_Tensor::full(val, shape)?.into())
    }

    fn full_like(&self, val: T) -> anyhow::Result<Self> {
        Ok(_Tensor::full_like(self, val)?.into())
    }

    fn arange<U>(start: U, end: U) -> anyhow::Result<Self>
        where T: Convertor + FromScalar<usize> + FromScalar<U> + NormalOut<T, Output = T>
    {
        Ok(_Tensor::arange(start, end)?.into())
    }

    fn arange_step(start: T, end: T, step: T) -> anyhow::Result<Self>
        where T: Convertor + FromScalar<usize> + NormalOut<T, Output = T>
    {
        Ok(_Tensor::arange_step(start, end, step)?.into())
    }

    fn eye(n: usize, m: usize, k: usize) -> anyhow::Result<Self> where u8: IntoScalar<T> {
        Ok(_Tensor::eye(n, m, k)?.into())
    }

    fn linspace(start: T, end: T, num: usize, include_end: bool) -> anyhow::Result<Self>
        where
            T: Convertor +
                num::Float +
                FromScalar<usize> +
                FromScalar<f64> +
                NormalOut<T, Output = T>
    {
        Ok(_Tensor::linspace(start, end, num, include_end)?.into())
    }

    fn logspace(start: T, end: T, num: usize, include_end: bool, base: T) -> anyhow::Result<Self>
        where
            T: Convertor +
                num::Float +
                FromScalar<usize> +
                FromScalar<f64> +
                NormalOut<T, Output = T>
    {
        Ok(_Tensor::logspace(start, end, num, include_end, base)?.into())
    }

    fn geomspace(start: T, end: T, n: usize, include_end: bool) -> anyhow::Result<Self>
        where
            T: PartialOrd +
                FloatOut<T> +
                NormalOut<T, Output = T> +
                FromScalar<<T as FloatOut>::Output> +
                std::ops::Neg<Output = T>,
            <T as FloatOut>::Output: Sub<Output = <T as FloatOut>::Output> +
                FromScalar<usize> +
                FromScalar<f64> +
                Div<Output = <T as FloatOut>::Output> +
                NormalOut<Output = <T as FloatOut>::Output> +
                CommonBounds
    {
        Ok(_Tensor::geomspace(start, end, n, include_end)?.into())
    }

    fn tri(n: usize, m: usize, k: i64, low_triangle: bool) -> anyhow::Result<Self>
        where u8: IntoScalar<T>
    {
        Ok(_Tensor::tri(n, m, k, low_triangle)?.into())
    }

    fn tril(&self, k: i64) -> anyhow::Result<Self>
        where T: NormalOut<bool, Output = T> + IntoScalar<T>
    {
        Ok(_Tensor::tril(self, k)?.into())
    }

    fn triu(&self, k: i64) -> anyhow::Result<Self>
        where T: NormalOut<bool, Output = T> + IntoScalar<T>
    {
        Ok(_Tensor::triu(self, k)?.into())
    }

    fn identity(n: usize) -> anyhow::Result<Self> where u8: IntoScalar<T> {
        Ok(_Tensor::identity(n)?.into())
    }
}

impl<T: CommonBounds> ShapeManipulate for Tensor<T> {
    type Meta = T;
    fn squeeze<A: Into<Axis>>(&self, axes: A) -> Result<Tensor<T>> {
        Ok(_Tensor::squeeze(self, axes)?.into())
    }

    fn unsqueeze<A: Into<Axis>>(&self, axes: A) -> Result<Tensor<T>> {
        Ok(_Tensor::unsqueeze(self, axes)?.into())
    }

    fn reshape<S: Into<Shape>>(&self, shape: S) -> Result<Tensor<T>> {
        Ok(_Tensor::reshape(self, shape)?.into())
    }

    fn transpose(&self, axis1: i64, axis2: i64) -> Result<Tensor<T>> {
        Ok(_Tensor::transpose(self, axis1, axis2)?.into())
    }

    fn permute<A: Into<Axis>>(&self, axes: A) -> Result<Tensor<T>> {
        Ok(_Tensor::permute(self, axes)?.into())
    }

    fn expand<S: Into<Shape>>(&self, shape: S) -> Result<Tensor<T>> {
        Ok(_Tensor::expand(self, shape)?.into())
    }

    fn t(&self) -> Result<Self> {
        Ok(_Tensor::t(self)?.into())
    }

    fn mt(&self) -> Result<Self> {
        Ok(_Tensor::mt(self)?.into())
    }

    fn flip<A: Into<Axis>>(&self, axes: A) -> Result<Self> {
        Ok(_Tensor::flip(self, axes)?.into())
    }

    fn fliplr(&self) -> Result<Self> {
        Ok(_Tensor::fliplr(self)?.into())
    }

    fn flipud(&self) -> Result<Self> {
        Ok(_Tensor::flipud(self)?.into())
    }

    fn tile<S: Into<Axis>>(&self, repeats: S) -> Result<Self> {
        Ok(_Tensor::tile(self, repeats)?.into())
    }

    fn trim_zeros(&self, trim: &str) -> Result<Self> where Self::Meta: PartialEq {
        Ok(_Tensor::trim_zeros(self, trim)?.into())
    }

    fn repeat(&self, repeats: usize, axes: i16) -> Result<Tensor<T>> {
        Ok(_Tensor::repeat(self, repeats, axes)?.into())
    }

    fn split(&self, indices_or_sections: &[i64], axis: i64) -> Result<Vec<Self>> {
        Ok(
            _Tensor
                ::split(self, indices_or_sections, axis)?
                .into_iter()
                .map(|x| x.into())
                .collect()
        )
    }

    fn dsplit(&self, indices: &[i64]) -> Result<Vec<Self>> {
        Ok(
            _Tensor
                ::dsplit(self, indices)?
                .into_iter()
                .map(|x| x.into())
                .collect()
        )
    }

    fn hsplit(&self, indices: &[i64]) -> Result<Vec<Self>> {
        Ok(
            _Tensor
                ::hsplit(self, indices)?
                .into_iter()
                .map(|x| x.into())
                .collect()
        )
    }

    fn vsplit(&self, indices: &[i64]) -> Result<Vec<Self>> {
        Ok(
            _Tensor
                ::vsplit(self, indices)?
                .into_iter()
                .map(|x| x.into())
                .collect()
        )
    }

    fn swap_axes(&self, axis1: i64, axis2: i64) -> Result<Self> {
        Ok(_Tensor::swap_axes(self, axis1, axis2)?.into())
    }
}

impl<T> Display for Tensor<T> where T: CommonBounds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        display(self, f, 1000, 20, 6, 12, 4, false)
    }
}

impl<T> Debug for Tensor<T> where T: CommonBounds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        display(self, f, 1000, 20, 6, 12, 4, false)
    }
}
