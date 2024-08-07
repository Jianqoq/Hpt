use std::{ fmt::{ Debug, Display }, ops::{ Add, Deref, Div, Mul, Rem, Sub }, sync::Arc };

use tensor_common::{ axis::Axis, layout::Layout, pointer::Pointer, shape::Shape };
use tensor_display::display;
use tensor_iterator::{ strided::Strided, strided_mut::StridedMut };
use tensor_traits::{
    ops::uary::FloatUaryOps,
    shape_manipulate::ShapeManipulate,
    tensor::{ CommonBounds, NormalReduce, TensorAlloc, TensorCreator, TensorInfo, TensorLike },
    BaseTensor,
    FloatReduce,
    NormalUaryOps,
};
use tensor_types::{
    convertion::{ Convertor, FromScalar },
    into_scalar::IntoScalar,
    type_promote::{ Cmp, Eval, FloatOut, NormalOut },
};
use anyhow::Result;
use crate::{
    backend::Cpu,
    ops::cpu::{ stack::stack, uary::{ FloatType, NormalType } },
    tensor_base::_Tensor,
};

/// A wrapper of `Tensor` for user.
/// This is the main tensor for user.
///
/// # Properties
/// - `basic`: The pointer of `Tensor`.
#[derive(Clone)]
pub struct Tensor<T, B = Cpu> {
    pub(crate) inner: Arc<_Tensor<T, B>>,
}

impl<T, B> Deref for Tensor<T, B> {
    type Target = _Tensor<T, B>;

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

    fn to_raw_mut(&mut self) -> &mut [T] {
        self.as_raw_mut()
    }

    fn static_cast(&self) -> Result<Self::Output> {
        self.static_cast()
    }
}

impl<T> TensorInfo<T> for Tensor<T> where T: CommonBounds {
    fn ptr(&self) -> Pointer<T> {
        self.data
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
        self.parent
    }

    fn ndim(&self) -> usize {
        self.layout.ndim()
    }

    fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }
}

impl<T> TensorInfo<T> for &Tensor<T> where T: CommonBounds {
    fn ptr(&self) -> Pointer<T> {
        self.data
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

    fn _empty<S: Into<Shape>>(shape: S) -> Result<Self> where Self: Sized {
        Self::empty(shape)
    }
}

impl<T: CommonBounds> BaseTensor for Tensor<T> {
    type Output = _Tensor<T>;
    fn base(&self) -> &Self::Output {
        &self.inner
    }
}

impl<T: CommonBounds> BaseTensor for &Tensor<T> {
    type Output = _Tensor<T>;
    fn base(&self) -> &Self::Output {
        &self.inner
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
        let slice = unsafe { std::slice::from_raw_parts(ptr, size) };
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
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, size) };
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
    pub fn try_astype<U>(&self) -> Result<Tensor<U>> where U: CommonBounds, T: IntoScalar<U> {
        if U::ID == T::ID { Ok(self.static_cast()?) } else { Ok(self.astype::<U>()?) }
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
        where T: Convertor, U: Convertor + CommonBounds
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
        Ok(_Tensor::<T, Cpu>::contiguous(self)?.into())
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
    pub fn stack(tensors: &[Tensor<T>], axis: usize, keepdims: bool) -> Result<Self>
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
            _Tensor::<T, Cpu>
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

    fn empty<S: Into<Shape>>(shape: S) -> Result<Self> {
        Ok(_Tensor::<T, Cpu>::empty(shape)?.into())
    }

    fn zeros<S: Into<Shape>>(shape: S) -> Result<Self> {
        Ok(_Tensor::<T, Cpu>::zeros(shape)?.into())
    }

    fn ones<S: Into<Shape>>(shape: S) -> Result<Self> where u8: IntoScalar<T> {
        Ok(_Tensor::<T, Cpu>::ones(shape)?.into())
    }

    fn empty_like(&self) -> Result<Self> {
        Ok(_Tensor::empty_like(self)?.into())
    }

    fn zeros_like(&self) -> Result<Self> {
        Ok(_Tensor::zeros_like(self)?.into())
    }

    fn ones_like(&self) -> Result<Self> where u8: IntoScalar<T> {
        Ok(_Tensor::ones_like(self)?.into())
    }

    fn full<S: Into<Shape>>(val: T, shape: S) -> Result<Self> {
        Ok(_Tensor::<T, Cpu>::full(val, shape)?.into())
    }

    fn full_like(&self, val: T) -> Result<Self> {
        Ok(_Tensor::full_like(self, val)?.into())
    }

    fn arange<U>(start: U, end: U) -> Result<Self>
        where T: Convertor + FromScalar<U> + NormalOut<T, Output = T>, usize: IntoScalar<T>
    {
        Ok(_Tensor::<T, Cpu>::arange(start, end)?.into())
    }

    fn arange_step(start: T, end: T, step: T) -> Result<Self>
        where T: Convertor + FromScalar<usize> + NormalOut<T, Output = T>
    {
        Ok(_Tensor::<T, Cpu>::arange_step(start, end, step)?.into())
    }

    fn eye(n: usize, m: usize, k: usize) -> Result<Self> where u8: IntoScalar<T> {
        Ok(_Tensor::<T, Cpu>::eye(n, m, k)?.into())
    }

    fn linspace(start: T, end: T, num: usize, include_end: bool) -> Result<Self>
        where
            T: Convertor + num::Float + NormalOut<T, Output = T>,
            usize: IntoScalar<T>,
            f64: IntoScalar<T>
    {
        Ok(_Tensor::<T, Cpu>::linspace(start, end, num, include_end)?.into())
    }

    fn logspace(start: T, end: T, num: usize, include_end: bool, base: T) -> Result<Self>
        where
            T: Convertor +
                num::Float +
                FromScalar<usize> +
                FromScalar<f64> +
                NormalOut<T, Output = T>
    {
        Ok(_Tensor::<T, Cpu>::logspace(start, end, num, include_end, base)?.into())
    }

    fn geomspace(start: T, end: T, n: usize, include_end: bool) -> Result<Self>
        where
            T: PartialOrd +
                FloatOut<T> +
                NormalOut<T, Output = T> +
                FromScalar<FloatType<T>> +
                std::ops::Neg<Output = T>,
            FloatType<T>: Sub<Output = FloatType<T>> +
                FromScalar<usize> +
                FromScalar<f64> +
                Div<Output = FloatType<T>> +
                NormalOut<Output = FloatType<T>> +
                CommonBounds
    {
        Ok(_Tensor::<T, Cpu>::geomspace(start, end, n, include_end)?.into())
    }

    fn tri(n: usize, m: usize, k: i64, low_triangle: bool) -> Result<Self> where u8: IntoScalar<T> {
        Ok(_Tensor::<T, Cpu>::tri(n, m, k, low_triangle)?.into())
    }

    fn tril(&self, k: i64) -> Result<Self> where T: NormalOut<bool, Output = T> + IntoScalar<T> {
        Ok(_Tensor::tril(self, k)?.into())
    }

    fn triu(&self, k: i64) -> Result<Self> where T: NormalOut<bool, Output = T> + IntoScalar<T> {
        Ok(_Tensor::triu(self, k)?.into())
    }

    fn identity(n: usize) -> Result<Self> where u8: IntoScalar<T> {
        Ok(_Tensor::<T, Cpu>::identity(n)?.into())
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

    fn flatten<A>(&self, axis: A) -> Result<Self> where A: Into<Option<usize>> {
        Ok(_Tensor::flatten(self, axis)?.into())
    }
}

impl<T: CommonBounds + NormalOut<Output = T> + Eval<Output = bool> + Cmp> NormalReduce<T>
for Tensor<T> {
    type Output = Tensor<T>;

    type BoolOutput = Tensor<bool>;

    fn sum<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output> {
        Ok(_Tensor::sum(self, axis, keep_dims)?.into())
    }

    fn sum_<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
        init_out: bool,
        out: Self::Output
    ) -> Result<Self::Output> {
        Ok(_Tensor::sum_(self, axis, keep_dims, init_out, out.inner.as_ref().clone())?.into())
    }

    fn sum_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool
    ) -> Result<Self::Output> {
        Ok(_Tensor::sum_with_init(self, init_val, axes, keep_dims)?.into())
    }

    fn nansum<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output> {
        Ok(_Tensor::nansum(self, axis, keep_dims)?.into())
    }

    fn nansum_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool
    ) -> Result<Self::Output> {
        Ok(_Tensor::nansum_with_init(self, init_val, axes, keep_dims)?.into())
    }

    fn prod<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output> {
        Ok(_Tensor::prod(self, axis, keep_dims)?.into())
    }

    fn prod_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool
    ) -> Result<Self::Output> {
        Ok(_Tensor::prod_with_init(self, init_val, axes, keep_dims)?.into())
    }

    fn nanprod<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::Output> {
        Ok(_Tensor::nanprod(self, axis, keep_dims)?.into())
    }

    fn nanprod_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool
    ) -> Result<Self::Output> {
        Ok(_Tensor::nanprod_with_init(self, init_val, axes, keep_dims)?.into())
    }

    fn min<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self> {
        Ok(_Tensor::min(self, axis, keep_dims)?.into())
    }

    fn min_with_init<S: Into<Axis>>(&self, init_val: T, axes: S, keep_dims: bool) -> Result<Self> {
        Ok(_Tensor::min_with_init(self, init_val, axes, keep_dims)?.into())
    }

    fn max<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self> {
        Ok(_Tensor::max(self, axis, keep_dims)?.into())
    }

    fn max_with_init<S: Into<Axis>>(&self, init_val: T, axes: S, keep_dims: bool) -> Result<Self> {
        Ok(_Tensor::max_with_init(self, init_val, axes, keep_dims)?.into())
    }

    fn all<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::BoolOutput> {
        Ok(_Tensor::all(self, axis, keep_dims)?.into())
    }

    fn any<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> Result<Self::BoolOutput> {
        Ok(_Tensor::any(self, axis, keep_dims)?.into())
    }

    fn reducel1<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        Ok(_Tensor::reducel1(self, axis, keep_dims)?.into())
    }
    
    fn sum_square<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        Ok(_Tensor::sum_square(self, axis, keep_dims)?.into())
    }
}

impl<T> FloatReduce<T>
    for Tensor<T>
    where
        T: CommonBounds                                                                                 // prettier-ignore
        + NormalOut<T, Output = T>                                                                                  // prettier-ignore
        + NormalOut<FloatType<T>, Output = FloatType<T>>                          // prettier-ignore
        + FloatOut + Cmp + IntoScalar<T>, // prettier-ignore
        FloatType<T>: CommonBounds                                                           // prettier-ignore
        + NormalOut<T, Output = FloatType<T>>
        + FloatOut<Output = FloatType<T>>
        + NormalOut<FloatType<T>, Output = FloatType<T>> // prettier-ignore
        + FromScalar<usize> + IntoScalar<FloatType<T>>, // prettier-ignore
        f64: IntoScalar<<T as NormalOut>::Output>, // prettier-ignore
        f64: IntoScalar<FloatType<T>>, // prettier-ignore
        _Tensor<FloatType<T>>: TensorLike<
        FloatType<T>,
        Output = _Tensor<FloatType<T>>
    > // prettier-ignore
{
    type Output = Tensor<FloatType<T>>;
    fn mean<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        Ok(_Tensor::mean(self, axis, keep_dims)?.into())
    }
    fn reducel2<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        Ok(_Tensor::reducel2(self, axis, keep_dims)?.into())
    }
    fn reducel3<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        Ok(_Tensor::reducel3(self, axis, keep_dims)?.into())
    }
    fn logsumexp<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        Ok(_Tensor::logsumexp(self, axis, keep_dims)?.into())
    }
}

impl<T> FloatUaryOps
    for Tensor<T>
    where
        T: FloatOut + CommonBounds,
        FloatType<T>: CommonBounds,
        f64: IntoScalar<FloatType<T>>,
        FloatType<T>: IntoScalar<FloatType<T>>
{
    type Output = Tensor<FloatType<T>>;

    type InplaceOutput = _Tensor<FloatType<T>>;

    type OutputMeta = FloatType<T>;

    fn sin(&self) -> Result<Self::Output> {
        Ok(_Tensor::sin(self)?.into())
    }

    fn cos(&self) -> Result<Self::Output> {
        Ok(_Tensor::cos(self)?.into())
    }

    fn tan(&self) -> Result<Self::Output> {
        Ok(_Tensor::tan(self)?.into())
    }

    fn asin(&self) -> Result<Self::Output> {
        Ok(_Tensor::asin(self)?.into())
    }

    fn acos(&self) -> Result<Self::Output> {
        Ok(_Tensor::acos(self)?.into())
    }

    fn atan(&self) -> Result<Self::Output> {
        Ok(_Tensor::atan(self)?.into())
    }

    fn sinh(&self) -> Result<Self::Output> {
        Ok(_Tensor::sinh(self)?.into())
    }

    fn cosh(&self) -> Result<Self::Output> {
        Ok(_Tensor::cosh(self)?.into())
    }

    fn tanh(&self) -> Result<Self::Output> {
        Ok(_Tensor::tanh(self)?.into())
    }

    fn asinh(&self) -> Result<Self::Output> {
        Ok(_Tensor::asinh(self)?.into())
    }

    fn acosh(&self) -> Result<Self::Output> {
        Ok(_Tensor::acosh(self)?.into())
    }

    fn atanh(&self) -> Result<Self::Output> {
        Ok(_Tensor::atanh(self)?.into())
    }

    fn sin_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::sin_(self, out.base().clone())?.into())
    }

    fn cos_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::cos_(self, out.base().clone())?.into())
    }

    fn tan_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::tan_(self, out.base().clone())?.into())
    }

    fn asin_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::asin_(self, out.base().clone())?.into())
    }

    fn acos_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::acos_(self, out.base().clone())?.into())
    }

    fn atan_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::atan_(self, out.base().clone())?.into())
    }

    fn sinh_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::sinh_(self, out.base().clone())?.into())
    }

    fn cosh_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::cosh_(self, out.base().clone())?.into())
    }

    fn tanh_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::tanh_(self, out.base().clone())?.into())
    }

    fn asinh_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::asinh_(self, out.base().clone())?.into())
    }

    fn acosh_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::acosh_(self, out.base().clone())?.into())
    }

    fn atanh_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::atanh_(self, out.base().clone())?.into())
    }

    fn exp(&self) -> Result<Self::Output> {
        Ok(_Tensor::exp(self)?.into())
    }

    fn exp_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::exp_(self, out.base().clone())?.into())
    }

    fn exp2(&self) -> Result<Self::Output> {
        Ok(_Tensor::exp2(self)?.into())
    }

    fn exp2_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::exp2_(self, out.base().clone())?.into())
    }

    fn sqrt(&self) -> Result<Self::Output> {
        Ok(_Tensor::sqrt(self)?.into())
    }

    fn sqrt_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::sqrt_(self, out.base().clone())?.into())
    }

    fn recip(&self) -> Result<Self::Output> {
        Ok(_Tensor::recip(self)?.into())
    }

    fn recip_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::recip_(self, out.base().clone())?.into())
    }

    fn ln(&self) -> Result<Self::Output> {
        Ok(_Tensor::ln(self)?.into())
    }

    fn ln_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::ln_(self, out.base().clone())?.into())
    }

    fn log2(&self) -> Result<Self::Output> {
        Ok(_Tensor::log2(self)?.into())
    }

    fn log2_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::log2_(self, out.base().clone())?.into())
    }

    fn log10(&self) -> Result<Self::Output> {
        Ok(_Tensor::log10(self)?.into())
    }

    fn log10_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::log10_(self, out.base().clone())?.into())
    }

    fn celu(&self, alpha: Self::OutputMeta) -> anyhow::Result<Self::Output> {
        Ok(_Tensor::celu(self, alpha)?.into())
    }

    fn celu_<U>(&self, alpha: Self::OutputMeta, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::celu_(self, alpha, out.base().clone())?.into())
    }

    fn sigmoid(&self) -> Result<Self::Output> {
        Ok(_Tensor::sigmoid(self)?.into())
    }

    fn sigmoid_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::sigmoid_(self, out.base().clone())?.into())
    }

    fn elu(&self, alpha: Self::OutputMeta) -> anyhow::Result<Self::Output> {
        Ok(_Tensor::elu(self, alpha)?.into())
    }

    fn elu_<U>(&self, alpha: Self::OutputMeta, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::elu_(self, alpha, out.base().clone())?.into())
    }

    fn leaky_relu(&self, alpha: Self::OutputMeta) -> anyhow::Result<Self::Output> {
        Ok(_Tensor::leaky_relu(self, alpha)?.into())
    }

    fn leaky_relu_<U>(&self, alpha: Self::OutputMeta, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::leaky_relu_(self, alpha, out.base().clone())?.into())
    }

    fn gelu(&self) -> anyhow::Result<Self::Output> {
        Ok(_Tensor::gelu(self)?.into())
    }

    fn gelu_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::gelu_(self, out.base().clone())?.into())
    }

    fn selu(
        &self,
        alpha: Option<Self::OutputMeta>,
        gamma: Option<Self::OutputMeta>
    ) -> anyhow::Result<Self::Output> {
        Ok(_Tensor::selu(self, alpha, gamma)?.into())
    }

    fn selu_<U>(
        &self,
        alpha: Option<Self::OutputMeta>,
        gamma: Option<Self::OutputMeta>,
        out: U
    ) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::selu_(self, alpha, gamma, out.base().clone())?.into())
    }

    fn hard_sigmoid(
        &self,
        alpha: Option<Self::OutputMeta>,
        beta: Option<Self::OutputMeta>
    ) -> anyhow::Result<Self::Output> {
        Ok(_Tensor::hard_sigmoid(self, alpha, beta)?.into())
    }

    fn hard_sigmoid_<U>(
        &self,
        alpha: Option<Self::OutputMeta>,
        beta: Option<Self::OutputMeta>,
        out: U
    ) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::hard_sigmoid_(self, alpha, beta, out.base().clone())?.into())
    }

    fn hard_swish(&self) -> anyhow::Result<Self::Output> {
        Ok(_Tensor::hard_swish(self)?.into())
    }

    fn hard_swish_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::hard_swish_(self, out.base().clone())?.into())
    }

    fn relu6(&self) -> anyhow::Result<Self::Output> {
        Ok(_Tensor::relu6(self)?.into())
    }

    fn relu6_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::relu6_(self, out.base().clone())?.into())
    }

    fn softplus(&self) -> anyhow::Result<Self::Output> {
        Ok(_Tensor::softplus(self)?.into())
    }

    fn softplus_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::softplus_(self, out.base().clone())?.into())
    }

    fn softsign(&self) -> anyhow::Result<Self::Output> {
        Ok(_Tensor::softsign(self)?.into())
    }

    fn softsign_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::softsign_(self, out.base().clone())?.into())
    }

    fn mish(&self) -> anyhow::Result<Self::Output> {
        Ok(_Tensor::mish(self)?.into())
    }

    fn mish_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::mish_(self, out.base().clone())?.into())
    }
}

impl<T> NormalUaryOps
    for Tensor<T>
    where
        T: NormalOut + CommonBounds + IntoScalar<T>,
        NormalType<T>: CommonBounds,
        <T as NormalOut>::Output: IntoScalar<<T as NormalOut>::Output>
{
    type Output = Tensor<NormalType<T>>;

    type InplaceOutput = Tensor<NormalType<T>>;

    type OutputMeta = NormalType<T>;

    fn square(&self) -> Result<Self::Output> {
        Ok(_Tensor::square(self)?.into())
    }

    fn square_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::square_(self, out.base().clone())?.into())
    }

    fn abs(&self) -> Result<Self> {
        Ok(_Tensor::abs(self)?.into())
    }

    fn abs_<U>(&self, out: U) -> Result<Self>
        where U: BaseTensor<Output = Self>
    {
        Ok(_Tensor::abs_(self, out.base().clone())?.into())
    }

    fn ceil(&self) -> Result<Self::Output> {
        Ok(_Tensor::ceil(self)?.into())
    }

    fn ceil_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::ceil_(self, out.base().clone())?.into())
    }

    fn sign(&self) -> Result<Self::Output> {
        Ok(_Tensor::sign(self)?.into())
    }

    fn sign_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput> + BaseTensor
    {
        Ok(_Tensor::sign_(self, out.base().clone())?.into())
    }

    fn clip(&self, min: Self::OutputMeta, max: Self::OutputMeta) -> Result<Self::Output> {
        Ok(_Tensor::clip(self, min, max)?.into())
    }

    fn clip_<U>(&self, min: Self::OutputMeta, max: Self::OutputMeta, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        Ok(_Tensor::clip_(self, min, max, out.base().clone())?.into())
    }

    fn round(&self) -> Result<Self::Output> {
        Ok(_Tensor::round(self)?.into())
    }

    fn round_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput> + BaseTensor
    {
        Ok(_Tensor::round_(self, out.base().clone())?.into())
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

macro_rules! normal_ops_1 {
    ($op:ident, $op2:ident) => {
        impl<T, U> $op<Tensor<U>>
        for Tensor<T>
        where
            T: CommonBounds + NormalOut<U>,
            U: CommonBounds,
            <T as NormalOut<U>>::Output: CommonBounds,
            <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>
        {
            type Output = Tensor<<T as NormalOut<U>>::Output>;

            fn $op2(self, rhs: Tensor<U>) -> Self::Output {
                (self.inner.as_ref().$op2(rhs.inner.as_ref())).into()
            }
        }
    };
}

macro_rules! normal_ops_2 {
    ($op:ident, $op2:ident) => {
        impl<'a, T, U> std::ops::$op<&'a Tensor<U>>
        for Tensor<T>
        where
            T: CommonBounds + NormalOut<U>,
            U: CommonBounds,
            <T as NormalOut<U>>::Output: CommonBounds,
            <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>
        {
            type Output = Tensor<<T as NormalOut<U>>::Output>;

            fn $op2(self, rhs: &'a Tensor<U>) -> Self::Output {
                (self.inner.as_ref() + rhs.inner.as_ref()).into()
            }
        }
    };
}

macro_rules! normal_ops_3 {
    ($op:ident, $op2:ident) => {
        impl<'a, T, U> std::ops::$op<&'a Tensor<U>>
        for &'a Tensor<T>
        where
            T: CommonBounds + NormalOut<U>,
            U: CommonBounds,
            <T as NormalOut<U>>::Output: CommonBounds,
            <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>
        {
            type Output = Tensor<<T as NormalOut<U>>::Output>;

            fn $op2(self, rhs: &'a Tensor<U>) -> Self::Output {
                (self.inner.as_ref() + rhs.inner.as_ref()).into()
            }
        }
    };
}

macro_rules! normal_ops_4 {
    ($op:ident, $op2:ident) => {
        impl<'a, T, U> std::ops::$op<Tensor<U>>
        for &'a Tensor<T>
        where
            T: CommonBounds + NormalOut<U>,
            U: CommonBounds,
            <T as NormalOut<U>>::Output: CommonBounds,
            <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>
        {
            type Output = Tensor<<T as NormalOut<U>>::Output>;

            fn $op2(self, rhs: Tensor<U>) -> Self::Output {
                (self.inner.as_ref() + rhs.inner.as_ref()).into()
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
