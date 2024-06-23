use std::sync::Arc;

use tensor_traits::tensor::{CommonBounds, TensorLike};
use tensor_types::into_scalar::IntoScalar;
use anyhow::Result;
use crate::tensor_base::_Tensor;


/// A wrapper of `_Tensor` for user.
/// This is the main tensor for user.
///
/// # Properties
/// - `basic`: The pointer of `_Tensor`.
pub struct Tensor<T> {
    pub(crate) inner: Arc<_Tensor<T>>,
}

impl<T, U> TensorLike<T, U, Tensor<U>>
    for Tensor<T>
    where T: IntoScalar<U> + CommonBounds, U: CommonBounds
{
    type Output = _Tensor<U>;
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

impl<T> TensorInfo<T> for _Tensor<T> {
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

impl<T> TensorInfo<T> for &_Tensor<T> {
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

impl<T: CommonBounds> TensorAlloc for _Tensor<T> {
    type Meta = T;

    fn _empty<S: Into<Shape>>(shape: S) -> anyhow::Result<Self> where Self: Sized {
        Self::empty(shape)
    }
}

impl<T: CommonBounds> _Tensor<T> {
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
    /// use tensor_core::_Tensor;
    /// let tensor = _Tensor::<f32>::new([1.0, 2.0, 3.0]);
    /// let converted_tensor = tensor.astype::<i32>().unwrap();
    /// assert!(tensor.allclose(&converted_tensor))
    /// ```
    pub fn astype<U>(&self) -> Result<_Tensor<U>> where U: CommonBounds, T: IntoScalar<U> {
        // Create an empty tensor of the new type with the same shape.
        let ret: _Tensor<U> = _Tensor::<U>::empty(self.layout.shape().clone())?;

        // Parallel iteration to convert and copy each element to the new tensor.
        ret.as_raw_mut()
            .par_iter_mut()
            .zip(self.as_raw().par_iter())
            .for_each(|(a, &b)| {
                *a = b.into_scalar();
            });
        Ok(ret)
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
    /// use tensor_core::_Tensor;
    /// let tensor = _Tensor::<f32>::new([1.0, 2.0, 3.0]);
    /// let converted_tensor = tensor.try_astype::<i32>().unwrap();
    /// assert!(tensor.allclose(&converted_tensor))
    /// ```
    pub fn try_astype<U>(&self) -> Result<_Tensor<U>> where U: CommonBounds, T: IntoScalar<U> {
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
    /// use tensor_core::_Tensor;
    /// let tensor = _Tensor::<f32>::new([1.0, 2.0, 3.0]);
    /// let static_cast_tensor = tensor.static_cast::<f32>().unwrap();
    /// assert!(tensor.allclose(&static_cast_tensor))
    /// ```
    pub fn static_cast<U>(&self) -> Result<_Tensor<U>> where U: CommonBounds {
        assert_eq!(U::ID, T::ID);
        match self.parent {
            Some(parent) => {
                let new_parent = Pointer::new(parent.ptr as *mut U);
                return Ok(_Tensor {
                    data: Pointer::new(self.data.ptr as *mut U),
                    parent: Some(new_parent),
                    mem_layout: self.mem_layout.clone(),
                    layout: self.layout.clone(),
                });
            }
            None => {
                let new_parent = Pointer::new(self.data.ptr as *mut U);
                return Ok(_Tensor {
                    data: Pointer::new(self.data.ptr as *mut U),
                    parent: Some(new_parent),
                    mem_layout: self.mem_layout.clone(),
                    layout: self.layout.clone(),
                });
            }
        }
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
    /// use tensor_core::_Tensor;
    /// let tensor1 = _Tensor::<f64>::new([1.0, 2.0, 3.0]);
    /// let tensor2 = _Tensor::<f64>::new([1.0, 2.0, 3.0]);
    /// assert!(tensor1.allclose(&tensor2));
    /// ```
    pub fn allclose<U: CommonBounds>(&self, other: &_Tensor<U>) -> bool
        where T: Convertor, U: Convertor
    {
        if self.shape() != other.shape() {
            return false;
        }
        let folder = self
            .iter()
            .zip(other.iter())
            .fold(
                || true,
                |acc, (a, b)| {
                    let a_val: f64 = a.to_f64();
                    let b_val: f64 = b.to_f64();
                    let abs_diff: f64 = (a_val - b_val).abs();
                    let torlerance: f64 = 1.0e-8 + 1.0e-5 * b_val.abs();
                    acc && abs_diff <= torlerance
                }
            );
        return folder.reduce(
            || true,
            |a, b| a && b
        );
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
    /// use tensor_core::_Tensor;
    /// use tensor_trait::{ShapeManipulate, TensorInfo};
    /// let tensor = _Tensor::<f32>::new([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
    /// let transposed_tensor = tensor.transpose([1, 0]).unwrap();
    /// let contiguous_tensor = tensor.contiguous().unwrap();
    /// assert!(contiguous_tensor.is_contiguous())
    /// ```
    pub fn contiguous(&self) -> Result<Self> {
        let res = self
            .iter()
            .strided_map(|x| { x })
            .collect();
        Ok(res)
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
    /// use tensor_core::_Tensor;
    /// let tensor1 = _Tensor::<f32>::new([1.0, 2.0, 3.0]);
    /// let tensor2 = _Tensor::<f32>::new([4.0, 5.0, 6.0]);
    /// let stacked_tensor = _Tensor::stack(vec![&tensor1, &tensor2], 0, true).unwrap();
    /// assert!(stacked_tensor.allclose(&_Tensor::<f64>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])));
    /// ```
    pub fn stack(tensors: Vec<&_Tensor<T>>, axis: usize, keepdims: bool) -> Result<Self>
        where T: 'static
    {
        stack(tensors, axis, keepdims)
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
    /// use tensor_core::_Tensor;
    /// let tensor1 = _Tensor::<f32>::new([1.0, 2.0, 3.0]);
    /// let tensor2 = _Tensor::<f32>::new([4.0, 5.0, 6.0]);
    /// let vstacked_tensor = _Tensor::vstack(vec![&tensor1, &tensor2]).unwrap();
    /// assert!(vstacked_tensor.allclose(&_Tensor::<f64>::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])));
    /// ```
    pub fn vstack(tensors: Vec<&_Tensor<T>>) -> Result<_Tensor<T>> {
        return stack(tensors, 0, false);
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
    /// use tensor_core::_Tensor;
    /// let tensor1 = _Tensor::<f32>::new([1.0, 2.0, 3.0]);
    /// let tensor2 = _Tensor::<f32>::new([4.0, 5.0, 6.0]);
    /// let hstacked_tensor = _Tensor::hstack(vec![&tensor1, &tensor2]).unwrap();
    /// assert!(hstacked_tensor.allclose(&_Tensor::<f64>::new([1.0, 2.0, 3.0,4.0, 5.0, 6.0])));
    /// ```
    pub fn hstack(mut tensors: Vec<&_Tensor<T>>) -> Result<_Tensor<T>> {
        for tensor in tensors.iter_mut() {
            if tensor.shape().len() < 2 {
                if tensor.shape().len() == 1 {
                    return stack(tensors, 0, false);
                } else {
                    // scalar
                    let mut tensors_ref = Vec::with_capacity(tensors.len());
                    let mut tensors_holder = Vec::with_capacity(tensors.len());
                    for tensor in tensors {
                        tensors_holder.push(tensor.reshape(vec![1])?);
                    }
                    for tensor in tensors_holder.iter() {
                        tensors_ref.push(tensor);
                    }
                    return stack(tensors_ref, 0, false);
                }
            }
        }
        return stack(tensors, 1, false);
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
    /// use tensor_core::_Tensor;
    /// let tensor1 = _Tensor::<f32>::new([1.0, 2.0, 3.0]);
    /// let tensor2 = _Tensor::<f32>::new([4.0, 5.0, 6.0]);
    /// let dstacked_tensor = _Tensor::dstack(vec![&tensor1, &tensor2]).unwrap();
    /// assert!(dstacked_tensor.allclose(&_Tensor::<f64>::new([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]])));
    /// ```
    pub fn dstack(mut tensors: Vec<&_Tensor<T>>) -> Result<_Tensor<T>> {
        let mut new_tensors = Vec::with_capacity(tensors.len());
        for tensor in tensors.iter_mut() {
            if tensor.shape().len() < 3 {
                if tensor.shape().len() == 1 {
                    new_tensors.push(tensor.reshape(vec![1, tensor.shape()[0], 1])?);
                } else if tensor.shape().len() == 0 {
                    new_tensors.push(tensor.reshape(vec![1, 1, 1])?);
                } else {
                    new_tensors.push(
                        tensor.reshape(vec![tensor.shape()[0], tensor.shape()[1], 1])?
                    );
                }
            } else {
                new_tensors.push(tensor.clone());
            }
        }
        let mut tensors_ref = Vec::with_capacity(new_tensors.len());
        for tensor in new_tensors.iter() {
            tensors_ref.push(tensor);
        }
        return stack(tensors_ref, 2, false);
    }
}

impl<T: CommonBounds> TensorCreator<T> for _Tensor<T> {
    type StridedIter = Strided<T>;

    type Mask = _Tensor<bool>;

    type Basic = _Tensor<T>;

    fn empty<S: Into<Shape>>(shape: S) -> anyhow::Result<Self> {
        let _shape = shape.into();
        let res_shape = Shape::from(_shape);
        let mut size = 1;
        let mut strides = vec![0; res_shape.len()];
        for i in (0..res_shape.len()).rev() {
            let tmp = res_shape[i];
            strides[i] = size as i64;
            size *= tmp as usize;
        }
        let layout = std::alloc::Layout
            ::from_size_align((size as usize) * std::mem::size_of::<T>(), 8)
            .unwrap();
        let ptr = unsafe { CACHE.allocate(layout) };
        return Ok(_Tensor {
            data: Pointer::new(ptr as *mut T),
            parent: None,
            layout: Layout::new(res_shape, strides.into()),
            mem_layout: Arc::new(layout),
        });
    }

    fn zeros<S: Into<Shape>>(shape: S) -> anyhow::Result<Self> {
        let _shape = shape.into();
        let res_shape = Shape::from(_shape);
        let mut size = 1;
        let mut strides = vec![0; res_shape.len()];
        for i in (0..res_shape.len()).rev() {
            let tmp = res_shape[i] as usize;
            strides[i] = size as i64;
            size *= tmp;
        }
        let layout = std::alloc::Layout
            ::from_size_align(size * std::mem::size_of::<T>(), 8)
            .unwrap();
        let ptr = unsafe { CACHE.allocate(layout) };
        unsafe {
            std::ptr::write_bytes(ptr as *mut T, 0, size);
        }
        return Ok(_Tensor {
            data: Pointer::new(ptr as *mut T),
            parent: None,
            layout: Layout::new(res_shape, strides.into()),
            mem_layout: Arc::new(layout),
        });
    }

    fn ones<S: Into<Shape>>(shape: S) -> anyhow::Result<Self> where u8: IntoScalar<T> {
        let _shape = shape.into();
        let res_shape = Shape::from(_shape);
        let mut size = 1;
        let mut strides = vec![0; res_shape.len()];
        for i in (0..res_shape.len()).rev() {
            let tmp = res_shape[i] as usize;
            strides[i] = size as i64;
            size *= tmp;
        }
        let layout = std::alloc::Layout
            ::from_size_align(size * std::mem::size_of::<T>(), 8)
            .unwrap();
        let ptr = unsafe { CACHE.allocate(layout) };
        unsafe {
            std::ptr::write_bytes(ptr as *mut T, 1, size);
        }
        return Ok(_Tensor {
            data: Pointer::new(ptr as *mut T),
            parent: None,
            layout: Layout::new(res_shape, strides.into()),
            mem_layout: Arc::new(layout),
        });
    }

    fn empty_like(&self) -> anyhow::Result<Self> {
        return Self::empty(self.shape());
    }

    fn zeros_like(&self) -> anyhow::Result<Self> {
        return Self::zeros(self.shape());
    }

    fn ones_like(&self) -> anyhow::Result<Self> where u8: IntoScalar<T> {
        return Self::ones(self.shape());
    }

    fn full<S: Into<Shape>>(val: T, shape: S) -> anyhow::Result<Self> {
        let _shape = shape.into();
        let res_shape = Shape::from(_shape);
        let ret = _Tensor::empty(res_shape)?;
        ret.as_raw_mut()
            .into_par_iter()
            .for_each(|x| {
                *x = val;
            });
        return Ok(ret);
    }

    fn full_like(&self, val: T) -> anyhow::Result<Self> {
        _Tensor::full(val, self.shape())
    }

    fn arange<U>(start: U, end: U) -> anyhow::Result<Self>
        where T: Convertor + FromScalar<usize> + FromScalar<U> + NormalOut<T, Output = T>
    {
        let start = T::__from(start);
        let end = T::__from(end);
        let size: i64 = end.to_i64() - start.to_i64();
        if size <= 0 {
            return _Tensor::empty(Arc::new(vec![0]));
        }
        let data: _Tensor<T> = _Tensor::empty(Arc::new(vec![size]))?;

        data.as_raw_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(i, x)| {
                *x = start._add(T::__from(i));
            });
        return Ok(data);
    }

    fn arange_step(start: T, end: T, step: T) -> anyhow::Result<Self>
        where T: Convertor + FromScalar<usize> + NormalOut<T, Output = T>
    {
        let step_float: f64 = step.to_f64();
        let end_usize = end.to_i64();
        let start_usize = start.to_i64();
        let size: usize = ((end_usize - start_usize) as usize) / (step_float.abs() as usize);
        let data = _Tensor::empty(Arc::new(vec![size as i64]))?;
        data.as_raw_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(i, x)| {
                *x = start._add(T::__from(i)._mul(step));
            });
        return Ok(data);
    }

    fn eye(n: usize, m: usize, k: usize) -> anyhow::Result<Self> where u8: IntoScalar<T> {
        let shape = vec![n as i64, m as i64];
        let res = _Tensor::empty(Arc::new(shape))?;
        let _r = res.as_raw_mut();
        let one = (1).into_scalar();
        let zero = (0).into_scalar();
        _r.into_par_iter()
            .enumerate()
            .for_each(|(i, x)| {
                let row = i / m;
                let col = i % m;
                if col == row + k {
                    *x = one;
                } else {
                    *x = zero;
                }
            });
        return Ok(res);
    }

    fn linspace(start: T, end: T, num: usize, include_end: bool) -> anyhow::Result<Self>
        where
            T: Convertor +
                num::Float +
                FromScalar<usize> +
                FromScalar<f64> +
                NormalOut<T, Output = T>
    {
        let _start: f64 = start.to_f64();
        let _end: f64 = end.to_f64();
        let n: f64 = num as f64;
        let step: f64 = if include_end { (_end - _start) / (n - 1.0) } else { (_end - _start) / n };
        let step_t: T = T::__from(step);
        let data = _Tensor::empty(Arc::new(vec![n as i64]))?;
        data.as_raw_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(i, x)| {
                *x = start._add(T::__from(i)._mul(step_t));
            });
        return Ok(data);
    }

    fn logspace(start: T, end: T, num: usize, include_end: bool, base: T) -> anyhow::Result<Self>
        where
            T: Convertor +
                num::Float +
                FromScalar<usize> +
                FromScalar<f64> +
                NormalOut<T, Output = T>
    {
        let _start: f64 = start.to_f64();
        let _end: f64 = end.to_f64();
        let n: f64 = num as f64;
        let step: f64 = if include_end { (_end - _start) / (n - 1.0) } else { (_end - _start) / n };
        let step_t: T = T::__from(step);
        let data = _Tensor::empty(Arc::new(vec![n as i64]))?;
        data.as_raw_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(i, x)| {
                *x = base._pow(start._add(T::__from(i)._mul(step_t)));
            });
        return Ok(data);
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
        let both_negative = start < T::ZERO && end < T::ZERO;
        let float_n = <T as FloatOut>::Output::__from(n);
        let step = if include_end {
            if start > T::ZERO && end > T::ZERO {
                (end._log10() - start._log10()) / (float_n - <T as FloatOut>::Output::__from(1f64))
            } else if start < T::ZERO && end < T::ZERO {
                (end._abs()._log10() - start._abs()._log10()) /
                    (float_n - <T as FloatOut>::Output::__from(1.0))
            } else {
                return Err(anyhow::Error::msg("start and end must have the same sign"));
            }
        } else {
            if start > T::ZERO && end > T::ZERO {
                (end._log10() - start._log10()) / <T as FloatOut>::Output::__from(n)
            } else if start < T::ZERO && end < T::ZERO {
                (end._abs()._log10() - start._abs()._log10()) / float_n
            } else {
                return Err(anyhow::Error::msg("start and end must have the same sign"));
            }
        };
        let data = _Tensor::<T>::empty(Arc::new(vec![n as i64]))?;
        let ten: <T as FloatOut>::Output = <T as FloatOut>::Output::__from(10.0);
        let start = if start > T::ZERO { start._log10() } else { start._abs()._log10() };
        if T::ID == Dtype::F32 || T::ID == Dtype::F64 {
            if both_negative {
                data.as_raw_mut()
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(i, x)| {
                        let val = ten._pow(
                            start._add(<T as FloatOut>::Output::__from(i)._mul(step))
                        );
                        *x = -T::__from(val);
                    });
            } else {
                data.as_raw_mut()
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(i, x)| {
                        let val = ten._pow(
                            start._add(<T as FloatOut>::Output::__from(i)._mul(step))
                        );
                        *x = T::__from(val);
                    });
            }
            return Ok(data);
        } else {
            if both_negative {
                data.as_raw_mut()
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(i, x)| {
                        let val = ten._pow(
                            start._add(<T as FloatOut>::Output::__from(i)._mul(step))
                        );
                        *x = -T::__from(val);
                    });
            } else {
                data.as_raw_mut()
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(i, x)| {
                        let val = ten._pow(
                            start._add(<T as FloatOut>::Output::__from(i)._mul(step))
                        );
                        *x = T::__from(val);
                    });
            }
        }
        return Ok(data);
    }

    fn tri(n: usize, m: usize, k: i64, low_triangle: bool) -> anyhow::Result<Self>
        where u8: IntoScalar<T>
    {
        let shape = vec![n as i64, m as i64];
        let res = _Tensor::empty(Arc::new(shape))?;
        let _r = res.as_raw_mut();
        let one = (1).into_scalar();
        let zero = (0).into_scalar();
        if low_triangle {
            _r.into_par_iter()
                .enumerate()
                .for_each(|(i, x)| {
                    let row = i / m;
                    let col = i % m;
                    if (col as i64) <= (row as i64) + k {
                        *x = one;
                    } else {
                        *x = zero;
                    }
                });
        } else {
            let k = k - 1;
            _r.into_par_iter()
                .enumerate()
                .for_each(|(i, x)| {
                    let row = i / m;
                    let col = i % m;
                    if (col as i64) <= (row as i64) + k {
                        *x = zero;
                    } else {
                        *x = one;
                    }
                });
        }
        return Ok(res);
    }

    fn tril(&self, k: i64) -> anyhow::Result<Self>
        where T: NormalOut<bool, Output = T> + IntoScalar<T>
    {
        if self.shape().len() < 2 {
            let message = format!("_Tensor must have at least 2 dimensions for tril method");
            return Err(anyhow::Error::msg(message));
        }
        let mask: _Tensor<bool> = _Tensor::<bool>::tri(
            self.shape()[self.shape().len() - 2] as usize,
            self.shape()[self.shape().len() - 1] as usize,
            k,
            true
        )?;
        let res: <_Tensor<T> as Mul<_Tensor<bool>>>::Output = self.clone() * mask;
        return Ok(res);
    }

    fn triu(&self, k: i64) -> anyhow::Result<Self>
        where T: NormalOut<bool, Output = T> + IntoScalar<T>
    {
        if self.shape().len() < 2 {
            let message: String = format!(
                "_Tensor must have at least 2 dimensions for tril method"
            );
            return Err(anyhow::Error::msg(message));
        }
        let mask: _Tensor<bool> = _Tensor::<bool>::tri(
            self.shape()[self.shape().len() - 2] as usize,
            self.shape()[self.shape().len() - 1] as usize,
            k,
            false
        )?;
        let res = self.clone() * mask;
        return Ok(res);
    }

    fn identity(n: usize) -> anyhow::Result<Self> where u8: IntoScalar<T> {
        _Tensor::eye(n, n, 0)
    }
}

impl<T: CommonBounds> ShapeManipulate for _Tensor<T> {
    type Meta = T;
    fn squeeze<A: Into<Axis>>(&self, axes: A) -> Result<_Tensor<T>> {
        let axes: Vec<usize> = process_axes(axes, self.ndim())?;
        for i in 0..axes.len() {
            if self.shape()[axes[i]] != 1 {
                return Err(
                    anyhow::anyhow!(
                        "cannot select an axis to squeeze out which has size not equal to one, try to squeeze axis {} with size {} in shape {:?}",
                        axes[i],
                        self.shape()[axes[i]],
                        self.shape()
                    )
                );
            }
        }
        let new_shape: Vec<i64> = self
            .shape()
            .iter()
            .enumerate()
            .filter(|&(i, _)| !axes.contains(&i))
            .map(|(_, &x)| x)
            .collect();
        return self.reshape(new_shape);
    }

    fn unsqueeze<A: Into<Axis>>(&self, axes: A) -> Result<_Tensor<T>> {
        let mut res_shape: Vec<i64> = self.shape().to_vec();
        let axes: Vec<usize> = process_axes(axes, self.ndim())?;
        axes.iter().for_each(|&x| {
            res_shape = yield_one_before(&res_shape, x);
        });
        return self.reshape(res_shape);
    }

    fn reshape<S: Into<Shape>>(&self, shape: S) -> Result<_Tensor<T>> {
        let shape = shape.into();
        ErrHandler::check_size_match(&shape, self.shape())?;
        if let Ok(new_layout) = self.layout.inplace_reshape(&shape) {
            return Ok(_Tensor {
                data: self.data.clone(),
                parent: self.parent.clone(),
                mem_layout: self.mem_layout.clone(),
                layout: new_layout,
            });
        } else {
            return self.contiguous()?.reshape(shape);
        }
    }

    fn transpose(&self, axis1: i64, axis2: i64) -> Result<_Tensor<T>> {
        if self.ndim() < 2 {
            return Err(
                anyhow::Error::msg(
                    "_Tensor with less than 2 dimensions for `transpose` method is not allowed"
                )
            );
        } else {
            self.permute(vec![axis1, axis2])
        }
    }

    fn permute<A: Into<Axis>>(&self, axes: A) -> Result<_Tensor<T>> {
        let permuted_layout = self.layout.permute(axes)?;
        return Ok(_Tensor {
            data: self.data.clone(),
            layout: permuted_layout,
            parent: self.parent,
            mem_layout: self.mem_layout.clone(),
        });
    }

    fn expand<S: Into<Shape>>(&self, shape: S) -> Result<_Tensor<T>> {
        let res_shape = Shape::from(shape.into());
        ErrHandler::check_expand_dims(self.shape(), &res_shape).unwrap();
        let res_strides = self.layout.expand_strides(&res_shape);
        return Ok(Self {
            data: self.data.clone(),
            parent: self.parent.clone(),
            mem_layout: self.mem_layout.clone(),
            layout: Layout::new(res_shape, res_strides.into()),
        });
    }

    fn t(&self) -> Result<Self> {
        if self.ndim() < 2 {
            return Err(
                anyhow::Error::msg(
                    "_Tensor with less than 2 dimensions for `t` method is not allowed"
                )
            );
        } else if self.ndim() > 2 {
            let mut axes = (0..self.ndim() as i64).collect::<Vec<i64>>();
            axes.swap(self.ndim() - 1, self.ndim() - 2);
            return self.permute(axes);
        }
        return self.transpose(1, 0);
    }

    fn mt(&self) -> Result<Self> {
        return self.permute((0..self.ndim() as i64).rev().collect::<Vec<i64>>());
    }

    fn flip<A: Into<Axis>>(&self, axes: A) -> Result<Self> {
        let axes = process_axes(axes, self.ndim())?;
        let mut new_strides = self.strides().to_vec();
        let mut ptr = self.ptr();
        for &i in axes.iter() {
            new_strides[i as usize] = -new_strides[i as usize];
            ptr.offset(self.strides()[i as usize]);
        }
        if self.parent.is_none() {
            return Ok(Self {
                data: ptr,
                parent: Some(self.data),
                mem_layout: self.mem_layout.clone(),
                layout: Layout::new(self.shape().clone(), new_strides.into()),
            });
        } else {
            return Ok(Self {
                data: ptr,
                parent: self.parent.clone(),
                mem_layout: self.mem_layout.clone(),
                layout: Layout::new(self.shape().clone(), new_strides.into()),
            });
        }
    }

    fn fliplr(&self) -> Result<Self> {
        if self.ndim() < 2 {
            return Err(anyhow::Error::msg("_Tensor must have at least 2 dimensions for fliplr"));
        }
        return self.flip(1);
    }

    fn flipud(&self) -> Result<Self> {
        if self.ndim() < 1 {
            return Err(anyhow::Error::msg("_Tensor must have at least 1 dimensions for flipud"));
        }
        return self.flip(0);
    }

    fn tile<S: Into<Axis>>(&self, repeats: S) -> Result<Self> {
        let repeats: Vec<usize> = process_axes(repeats, self.ndim())?;
        let repeats: Vec<i64> = repeats
            .into_iter()
            .map(|x| x as i64)
            .collect::<Vec<i64>>();
        let final_repeats;
        let mut final_shape;
        if repeats.len() > self.ndim() {
            final_shape = try_pad_shape(self.shape().as_ref(), repeats.len());
            final_repeats = repeats.clone();
        } else {
            final_shape = self.shape().to_vec();
            final_repeats = try_pad_shape(repeats.as_ref(), self.ndim());
        }
        let mut res = self.reshape(&final_shape)?;
        let mut cnt = 0;
        for (idx, &i) in final_repeats.iter().enumerate() {
            if i == 1 {
                continue;
            } else {
                let tmp_shape = yield_one_before(res.shape().as_ref(), idx);
                res = res.reshape(tmp_shape)?;
                res = res.repeat(i as usize, (idx + cnt) as i16)?;
                final_shape[idx] *= i;
                cnt += 1;
            }
        }
        return res.reshape(final_shape);
    }

    fn trim_zeros(&self, trim: &str) -> Result<Self> where Self::Meta: PartialEq {
        if !(trim == "fb" || trim == "f" || trim == "b") {
            return Err(anyhow::Error::msg("trim must be one of 'fb', 'f', 'b'"));
        }
        if self.ndim() > 1 {
            return Err(
                anyhow::Error::msg("_Tensor must have at most 1 dimension for trim_zeros method")
            );
        }
        let stride = self.strides()[0] as isize;
        let raw = self.as_raw();
        let mut ptr = raw.as_ptr();
        let mut left_len = 0;
        if trim.contains('f') {
            unsafe {
                for i in 0..raw.len() as isize {
                    if *ptr.offset(i * stride) != T::ZERO {
                        break;
                    } else {
                        left_len += 1;
                    }
                }
            }
        }
        let mut right_len = raw.len() as i64;
        if trim.contains('b') {
            unsafe {
                ptr = raw.as_ptr().offset(((raw.len() - 1) as isize) * stride);
                let stride = -stride;
                for i in 0..raw.len() as isize {
                    if *ptr.offset(i * stride) != T::ZERO {
                        break;
                    } else {
                        right_len -= 1;
                    }
                }
            }
        }
        return slice!(self[left_len:right_len]);
    }

    fn repeat(&self, repeats: usize, axes: i16) -> Result<_Tensor<T>> {
        let mut val: usize = axes as usize;
        if axes < 0 {
            val = self.shape().len() + (axes as usize);
        }
        let mut new_shape = yield_one_after(&self.shape(), val);
        let mut new_tensor: _Tensor<T> = self.reshape(&new_shape)?;
        new_shape[val + 1] *= repeats as i64;
        new_tensor = new_tensor.expand(new_shape)?;
        new_shape = self.shape().to_vec();
        new_shape[val] *= repeats as i64;
        return Ok(new_tensor.contiguous()?.reshape(new_shape)?);
    }

    fn split(&self, indices_or_sections: &[i64], axis: i64) -> Result<Vec<Self>> {
        let mut new_axis = axis;
        if axis < 0 {
            new_axis = (self.ndim() as i64) + axis;
        }
        assert!(new_axis >= 0);
        let mut reses = vec![];
        let mut tmp: Vec<Slice> = Vec::with_capacity(self.ndim());
        for _ in 0..self.ndim() {
            tmp.push(Slice::Full);
        }
        let mut prev = 0;
        for &i in indices_or_sections.iter() {
            tmp[axis as usize] = Slice::Range((prev, i));
            prev = i;
            reses.push(self.slice(&tmp)?);
        }
        let last = *indices_or_sections.last().unwrap();
        let remain = self.slice([Slice::RangeFrom(last)])?;
        reses.push(remain);
        return Ok(reses);
    }

    fn dsplit(&self, indices: &[i64]) -> Result<Vec<Self>> {
        if self.shape().len() < 3 {
            return Err(
                anyhow::Error::msg("_Tensor must have at least 3 dimensions for dsplit method")
            );
        }
        return self.split(indices, 2);
    }

    fn hsplit(&self, indices: &[i64]) -> Result<Vec<Self>> {
        if self.shape().len() < 2 {
            return Err(
                anyhow::Error::msg("_Tensor must have at least 2 dimensions for hsplit method")
            );
        }
        return self.split(indices, 1);
    }

    fn vsplit(&self, indices: &[i64]) -> Result<Vec<Self>> {
        if self.shape().len() < 1 {
            return Err(
                anyhow::Error::msg("_Tensor must have at least 1 dimensions for vsplit method")
            );
        }
        return self.split(indices, 0);
    }

    fn swap_axes(&self, mut axis1: i64, mut axis2: i64) -> Result<Self> {
        if axis1 < 0 {
            while axis1 < 0 {
                axis1 += self.ndim() as i64;
            }
        }
        if axis2 < 0 {
            while axis2 < 0 {
                axis2 += self.ndim() as i64;
            }
        }
        ErrHandler::check_index_in_range(self.ndim(), axis1 as usize)?;
        ErrHandler::check_index_in_range(self.ndim(), axis2 as usize)?;
        let mut new_shape = self.shape().to_vec();
        let mut new_strides = self.strides().to_vec();
        new_shape.swap(axis1 as usize, axis2 as usize);
        new_strides.swap(axis1 as usize, axis2 as usize);
        let layout = Layout::new(Shape::from(new_shape), new_strides.into());
        return Ok(Self {
            data: self.data.clone(),
            layout,
            parent: self.parent.clone(),
            mem_layout: self.mem_layout.clone(),
        });
    }
}

impl<T> Display for _Tensor<T> where T: CommonBounds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        display(self, f, 1000, 20, 6, 12, 4, false)
    }
}

impl<T> Debug for _Tensor<T> where T: CommonBounds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        display(self, f, 1000, 20, 6, 12, 4, false)
    }
}