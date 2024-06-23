use std::{ ops::{ Div, Mul, Sub }, sync::Arc };

use tensor_allocator::CACHE;
use tensor_common::{ layout::Layout, pointer::Pointer, shape::Shape };
use tensor_iterator::{strided::Strided, strided_mut::StridedMut};
use tensor_traits::tensor::{ CommonBounds, TensorAlloc, TensorCreator, TensorInfo };
use anyhow::Result;
use tensor_types::{
    convertion::{ Convertor, FromScalar },
    dtype::Dtype,
    into_scalar::IntoScalar,
    type_promote::{ FloatOut, NormalOut },
};
use rayon::iter::{
    IndexedParallelIterator,
    IntoParallelIterator,
    IntoParallelRefIterator,
    IntoParallelRefMutIterator,
    ParallelIterator,
};
/// This struct is the heart of the `DiffTensors` and `BasicTensors`. Both of them are just `wrappers` around this struct.
///
/// All the operations are happen on this struct.
///
/// The `DiffTensors` and `BasicTensors` are just call `_Tensor` methods and add extra logic.
///
/// # Properties
/// - `data`: The pointer to the data.
/// - `layout`: The layout of the tensor. We can get strides, shape, ndim, size from it.
/// - `parent`: The parent tensor of the tensor.
///
///  If the tensor is a view of another tensor, the parent tensor will be the original tensor.
/// - `mem_layout`: std::alloc::layout, use for deallocate the memory.
#[derive(Clone)]
pub struct _Tensor<T> {
    data: Pointer<T>,
    parent: Option<Pointer<T>>,
    layout: Layout,
    mem_layout: Arc<std::alloc::Layout>,
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

impl<T> TensorAlloc for _Tensor<T> {
    type Meta = T;

    fn _empty<S: Into<Shape>>(shape: S) -> anyhow::Result<Self> where Self: Sized {
        todo!()
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

    fn tril(&self, k: i64) -> anyhow::Result<Self> {
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

    fn triu(&self, k: i64) -> anyhow::Result<Self> {
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
