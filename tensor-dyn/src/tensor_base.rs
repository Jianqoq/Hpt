use anyhow::Result;
use rand_distr::{
    uniform::SampleUniform, Distribution, Exp1, Open01, OpenClosed01, Standard, StandardNormal,
};
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
        IntoParallelRefMutIterator, ParallelIterator,
    },
    slice::ParallelSliceMut,
};
use std::{
    fmt::{Debug, Display},
    ops::{Div, Sub},
    panic::Location,
    sync::{atomic::Ordering, Arc},
};
use tensor_allocator::CACHE;
use tensor_common::shape_utils::try_pad_shape;
use tensor_common::slice;
use tensor_common::{
    axis::{process_axes, Axis},
    err_handler::ErrHandler,
    layout::Layout,
    pointer::Pointer,
    shape::Shape,
    shape_utils::{yield_one_after, yield_one_before},
    slice::Slice,
};
use tensor_display::display;
use tensor_iterator::{iterator_traits::ParStridedIteratorZip, TensorIterator};
use tensor_macros::match_selection;
use tensor_traits::{
    random::Random,
    shape_manipulate::ShapeManipulate,
    tensor::{CommonBounds, TensorAlloc, TensorCreator, TensorInfo, TensorLike},
    RandomInt,
};
use tensor_types::{
    convertion::{Convertor, FromScalar},
    dtype::Dtype,
    into_scalar::IntoScalar,
    type_promote::NormalOut,
};
use tensor_types::{dtype::TypeCommon, type_promote::FloatOutUnary};

use crate::{
    backend::{Backend, BackendTy, Buffer, Cpu},
    ops::cpu::concat::concat,
    tensor::Tensor,
    BoolVector, ALIGN, DISPLAY_LR_ELEMENTS, DISPLAY_PRECISION,
};

/// This struct is the heart of the `DiffTensors` and `BasicTensors`. Both of them are just `wrappers` around this struct.
///
/// All the operations are happen on this struct.
///
/// # Properties
/// - `data`: The pointer to the data.
/// - `layout`: The layout of the tensor. We can get strides, shape, ndim, size from it.
/// - `parent`: The parent tensor of the tensor. parent is always the root tensor (`not a view`).
/// - `mem_layout`: std::alloc::layout, use for deallocate the memory and find cache in the allocator.
#[derive(Clone)]
pub struct _Tensor<T, B = Cpu>
where
    B: BackendTy + Buffer,
{
    pub(crate) data: Pointer<T>,
    pub(crate) parent: Option<Pointer<T>>,
    pub(crate) layout: Layout,
    pub(crate) mem_layout: Arc<std::alloc::Layout>,
    pub(crate) _backend: Backend<B>,
}

impl<T, B> Drop for _Tensor<T, B>
where
    B: BackendTy + Buffer,
{
    fn drop(&mut self) {
        match B::ID {
            0 => CACHE.deallocate(
                self._backend._backend.get_ptr() as *mut u8,
                &self.mem_layout,
            ),
            _ => {
                panic!("Invalid Backend ID")
            }
        }
    }
}

impl<T> TensorLike<T> for _Tensor<T>
where
    T: CommonBounds,
{
    fn as_raw(&self) -> &[T] {
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

    fn as_raw_mut(&mut self) -> &mut [T] {
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

    fn contiguous(&self) -> anyhow::Result<Self> {
        use tensor_types::traits::VecTrait;
        let res = self
            .par_iter_simd()
            .strided_map_simd(
                |(res, x)| {
                    *res = x;
                },
                |(res, x)| {
                    // possibily a rust bug when we use sse vector,
                    // so we have to use ptr directly or hope rust is able to inline the `write_unaligned`

                    // let ptr = res.as_mut_ptr() as *mut T::Vec;
                    // unsafe {
                    //     ptr.write_unaligned(x);
                    // }
                    res.write_unaligned(x);
                },
            )
            .collect();
        Ok(res)
    }
}

impl<T> TensorInfo<T> for _Tensor<T>
where
    T: CommonBounds,
{
    #[inline(always)]
    fn ptr(&self) -> Pointer<T> {
        self.data.clone()
    }

    #[inline(always)]
    fn size(&self) -> usize {
        self.layout.size() as usize
    }
    #[inline(always)]
    fn shape(&self) -> &Shape {
        self.layout.shape()
    }
    #[inline(always)]
    fn strides(&self) -> &tensor_common::strides::Strides {
        self.layout.strides()
    }
    #[inline(always)]
    fn layout(&self) -> &Layout {
        &self.layout
    }
    #[inline(always)]
    fn parent(&self) -> Option<Pointer<T>> {
        self.parent.clone()
    }
    #[inline(always)]
    fn ndim(&self) -> usize {
        self.layout.ndim()
    }
    #[inline(always)]
    fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }
}

impl<T> TensorInfo<T> for &_Tensor<T>
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

impl<T: CommonBounds> TensorAlloc for _Tensor<T> {
    type Meta = T;
    fn _empty<S: Into<Shape>>(shape: S) -> Result<Self>
    where
        Self: Sized,
    {
        Self::empty(shape)
    }
}

impl<T: CommonBounds> TensorIterator<'_, T> for _Tensor<T> {}

impl<T: CommonBounds> _Tensor<T> {
    /// copy the data from the other tensor to this tensor
    pub fn assign(&mut self, other: &_Tensor<T>) {
        self.par_iter_mut_simd()
            .zip(other.par_iter_simd())
            .for_each(|(a, b)| {
                *a = b;
            });
    }

    /// cast the tensor to the new type
    pub fn astype<U>(&self) -> Result<_Tensor<U>>
    where
        U: CommonBounds,
        T: IntoScalar<U>,
    {
        // Create an empty tensor of the new type with the same shape.
        let mut ret: _Tensor<U> = _Tensor::<U>::empty(self.layout.shape().clone())?;

        // Parallel iteration to convert and copy each element to the new tensor.
        ret.as_raw_mut()
            .par_iter_mut()
            .zip(self.as_raw().par_iter())
            .for_each(|(a, &b)| {
                *a = b.into_scalar();
            });
        Ok(ret)
    }

    /// try to cast the tensor to the new type, if the type is the same, return the tensor itself, otherwise return the new tensor
    pub fn try_astype<U>(&self) -> Result<_Tensor<U>>
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

    /// bitcast the tensor to the new type, the user must ensure the size of the new type is the same as the old type
    pub fn static_cast<Dst>(&self) -> Result<_Tensor<Dst>>
    where
        Dst: CommonBounds,
    {
        if T::ID == Dst::ID {
            match self.parent.clone() {
                Some(parent) => {
                    #[cfg(feature = "bound_check")]
                    let new_parent = Pointer::new(parent.ptr as *mut Dst, parent.layout.clone());
                    #[cfg(not(feature = "bound_check"))]
                    let new_parent = Pointer::new(parent.ptr as *mut Dst);
                    Ok(_Tensor {
                        #[cfg(feature = "bound_check")]
                        data: Pointer::new(self.data.ptr as *mut Dst, self.layout.clone()),
                        #[cfg(not(feature = "bound_check"))]
                        data: Pointer::new(self.data.ptr as *mut Dst),
                        parent: Some(new_parent),
                        mem_layout: self.mem_layout.clone(),
                        layout: self.layout.clone(),
                        _backend: self._backend.clone(),
                    })
                }
                None => Ok(_Tensor {
                    #[cfg(feature = "bound_check")]
                    data: Pointer::new(self.data.ptr as *mut Dst, self.layout.clone()),
                    #[cfg(not(feature = "bound_check"))]
                    data: Pointer::new(self.data.ptr as *mut Dst),
                    parent: None,
                    mem_layout: self.mem_layout.clone(),
                    layout: self.layout.clone(),
                    _backend: self._backend.clone(),
                }),
            }
        } else {
            panic!("Cannot cast tensor to different type")
        }
    }

    /// check if two tensors are close to each other
    pub fn allclose<U: CommonBounds>(&self, other: &_Tensor<U>) -> bool
    where
        T: Convertor,
        U: Convertor,
    {
        if self.shape() != other.shape() {
            return false;
        }
        let folder = self.par_iter().zip(other.par_iter()).fold(
            || true,
            |acc, (a, b)| {
                let a_val: f64 = a.to_f64();
                let b_val: f64 = b.to_f64();
                let abs_diff: f64 = (a_val - b_val).abs();
                let torlerance: f64 = 1.0e-8 + 1.0e-5 * b_val.abs();
                acc && abs_diff <= torlerance
            },
        );
        folder.reduce(|| true, |a, b| a && b)
    }
}

impl<T: CommonBounds> TensorCreator<T> for _Tensor<T> {
    fn empty<S: Into<Shape>>(shape: S) -> Result<Self> {
        let _shape = shape.into();
        let res_shape = Shape::from(_shape);
        let size = res_shape
            .iter()
            .try_fold(1i64, |acc, &num| acc.checked_mul(num).or(Some(i64::MAX)))
            .unwrap_or(i64::MAX) as usize;
        let layout = std::alloc::Layout::from_size_align(
            size.checked_mul(size_of::<T>())
                .or(Some(usize::MAX))
                .unwrap_or(usize::MAX),
            ALIGN,
        )?;
        let ptr = CACHE.allocate(layout);
        let ly = Layout::from(res_shape.clone());
        Ok(_Tensor {
            #[cfg(feature = "bound_check")]
            data: Pointer::new(ptr as *mut T, ly.clone()),
            #[cfg(not(feature = "bound_check"))]
            data: Pointer::new(ptr as *mut T),
            parent: None,
            layout: ly,
            mem_layout: Arc::new(layout),
            _backend: Backend::new(ptr as u64),
        })
    }

    fn zeros<S: Into<Shape>>(shape: S) -> Result<Self> {
        Self::full(T::ZERO, shape)
    }

    fn ones<S: Into<Shape>>(shape: S) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        Self::full(T::ONE, shape)
    }

    fn empty_like(&self) -> Result<Self> {
        Self::empty(self.shape())
    }

    fn zeros_like(&self) -> Result<Self> {
        Self::zeros(self.shape())
    }

    fn ones_like(&self) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        Self::ones(self.shape())
    }

    fn full<S: Into<Shape>>(val: T, shape: S) -> Result<Self> {
        let empty = Self::empty(shape)?;
        let ptr = empty.ptr().ptr;
        let size = empty.size();
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut T, size) };
        slice.into_par_iter().for_each(|x|{
            *x = val;
        });
        Ok(empty)
    }

    fn full_like(&self, val: T) -> Result<Self> {
        _Tensor::full(val, self.shape())
    }

    fn arange<U>(start: U, end: U) -> Result<Self>
    where
        T: Convertor + FromScalar<U> + NormalOut<T, Output = T>,
        usize: IntoScalar<T>,
        U: Convertor + IntoScalar<T> + Copy,
    {
        let size = end.to_i64() - start.to_i64();
        let start = start.into_scalar();
        if size <= 0 {
            return _Tensor::<T, Cpu>::empty(Arc::new(vec![0]));
        }
        let mut data: _Tensor<T> = _Tensor::<T, Cpu>::empty(Arc::new(vec![size]))?;

        data.as_raw_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(i, x)| {
                *x = start._add(i.into_scalar());
            });
        Ok(data)
    }

    fn arange_step(start: T, end: T, step: T) -> Result<Self>
    where
        T: Convertor + FromScalar<usize> + NormalOut<T, Output = T>,
    {
        let step_float = step.to_f64();
        let end_usize = end.to_i64();
        let start_usize = start.to_i64();
        let size = ((end_usize - start_usize) as usize) / (step_float.abs() as usize);
        let mut data = _Tensor::<T, Cpu>::empty(Arc::new(vec![size as i64]))?;
        data.as_raw_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(i, x)| {
                *x = start._add(T::__from(i)._mul(step));
            });
        Ok(data)
    }

    fn eye(n: usize, m: usize, k: usize) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        let shape = vec![n as i64, m as i64];
        let mut res = _Tensor::<T, Cpu>::empty(Arc::new(shape))?;
        res.as_raw_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(i, x)| {
                let row = i / m;
                let col = i % m;
                if col == row + k {
                    *x = T::ONE;
                } else {
                    *x = T::ZERO;
                }
            });
        Ok(res)
    }

    fn linspace(start: T, end: T, num: usize, include_end: bool) -> Result<Self>
    where
        T: Convertor + num::Float + NormalOut<T, Output = T>,
        usize: IntoScalar<T>,
        f64: IntoScalar<T>,
    {
        let _start: f64 = start.to_f64();
        let _end: f64 = end.to_f64();
        let n: f64 = num as f64;
        let step: f64 = if include_end {
            (_end - _start) / (n - 1.0)
        } else {
            (_end - _start) / n
        };
        let step_t: T = step.into_scalar();
        let mut data = _Tensor::<T, Cpu>::empty(Arc::new(vec![n as i64]))?;
        data.as_raw_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(i, x)| {
                *x = start._add(i.into_scalar()._mul(step_t));
            });
        Ok(data)
    }

    fn logspace(start: T, end: T, num: usize, include_end: bool, base: T) -> Result<Self>
    where
        T: Convertor + num::Float + FromScalar<usize> + FromScalar<f64> + NormalOut<T, Output = T>,
    {
        let _start = start.to_f64();
        let _end = end.to_f64();
        let n = num as f64;
        let step = if include_end {
            (_end - _start) / (n - 1.0)
        } else {
            (_end - _start) / n
        };
        let step_t = T::__from(step);
        let mut data = _Tensor::<T, Cpu>::empty(Arc::new(vec![n as i64]))?;
        data.as_raw_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(i, x)| {
                *x = base._pow(start._add(T::__from(i)._mul(step_t)));
            });
        Ok(data)
    }

    fn geomspace(start: T, end: T, n: usize, include_end: bool) -> Result<Self>
    where
        T: PartialOrd
            + FloatOutUnary
            + NormalOut<T, Output = T>
            + FromScalar<<T as FloatOutUnary>::Output>
            + std::ops::Neg<Output = T>,
        <T as FloatOutUnary>::Output: Sub<Output = <T as FloatOutUnary>::Output>
            + FromScalar<usize>
            + FromScalar<f64>
            + Div<Output = <T as FloatOutUnary>::Output>
            + NormalOut<Output = <T as FloatOutUnary>::Output>
            + CommonBounds,
    {
        let both_negative = start < T::ZERO && end < T::ZERO;
        let float_n = <T as FloatOutUnary>::Output::__from(n);
        let step = if include_end {
            if start > T::ZERO && end > T::ZERO {
                (end._log10() - start._log10())
                    / (float_n - <T as FloatOutUnary>::Output::__from(1f64))
            } else if start < T::ZERO && end < T::ZERO {
                (end._abs()._log10() - start._abs()._log10())
                    / (float_n - <T as FloatOutUnary>::Output::__from(1.0))
            } else {
                return Err(anyhow::Error::msg("start and end must have the same sign"));
            }
        } else if start > T::ZERO && end > T::ZERO {
            (end._log10() - start._log10()) / <T as FloatOutUnary>::Output::__from(n)
        } else if start < T::ZERO && end < T::ZERO {
            (end._abs()._log10() - start._abs()._log10()) / float_n
        } else {
            return Err(anyhow::Error::msg("start and end must have the same sign"));
        };
        let mut data = _Tensor::<T>::empty(Arc::new(vec![n as i64]))?;
        let ten: <T as FloatOutUnary>::Output = <T as FloatOutUnary>::Output::__from(10.0);
        let start = if start > T::ZERO {
            start._log10()
        } else {
            start._abs()._log10()
        };
        if T::ID == Dtype::F32 || T::ID == Dtype::F64 {
            if both_negative {
                data.as_raw_mut()
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(i, x)| {
                        let val = ten
                            ._pow(start._add(<T as FloatOutUnary>::Output::__from(i)._mul(step)));
                        *x = -T::__from(val);
                    });
            } else {
                data.as_raw_mut()
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(i, x)| {
                        let val = ten
                            ._pow(start._add(<T as FloatOutUnary>::Output::__from(i)._mul(step)));
                        *x = T::__from(val);
                    });
            }
            return Ok(data);
        } else if both_negative {
            data.as_raw_mut()
                .into_par_iter()
                .enumerate()
                .for_each(|(i, x)| {
                    let val =
                        ten._pow(start._add(<T as FloatOutUnary>::Output::__from(i)._mul(step)));
                    *x = -T::__from(val);
                });
        } else {
            data.as_raw_mut()
                .into_par_iter()
                .enumerate()
                .for_each(|(i, x)| {
                    let val =
                        ten._pow(start._add(<T as FloatOutUnary>::Output::__from(i)._mul(step)));
                    *x = T::__from(val);
                });
        }
        Ok(data)
    }

    fn tri(n: usize, m: usize, k: i64, low_triangle: bool) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        let shape = vec![n as i64, m as i64];
        let mut res = _Tensor::<T, Cpu>::empty(Arc::new(shape))?;
        if low_triangle {
            res.as_raw_mut()
                .into_par_iter()
                .enumerate()
                .for_each(|(i, x)| {
                    let row = i / m;
                    let col = i % m;
                    if (col as i64) <= (row as i64) + k {
                        *x = T::ONE;
                    } else {
                        *x = T::ZERO;
                    }
                });
        } else {
            let k = k - 1;
            res.as_raw_mut()
                .into_par_iter()
                .enumerate()
                .for_each(|(i, x)| {
                    let row = i / m;
                    let col = i % m;
                    if (col as i64) <= (row as i64) + k {
                        *x = T::ZERO;
                    } else {
                        *x = T::ONE;
                    }
                });
        }
        Ok(res)
    }

    fn tril(&self, k: i64) -> Result<Self>
    where
        T: NormalOut<bool, Output = T> + IntoScalar<T>,
        <T as TypeCommon>::Vec: NormalOut<BoolVector, Output = <T as TypeCommon>::Vec>,
    {
        if self.shape().len() < 2 {
            return Err(
                ErrHandler::NdimNotEnough(2, self.shape().len(), Location::caller()).into(),
            );
        }
        let mask: _Tensor<bool> = _Tensor::<bool>::tri(
            self.shape()[self.shape().len() - 2] as usize,
            self.shape()[self.shape().len() - 1] as usize,
            k,
            true,
        )?;
        let res: _Tensor<T> = self.clone() * mask;
        Ok(res)
    }

    fn triu(&self, k: i64) -> Result<Self>
    where
        T: NormalOut<bool, Output = T> + IntoScalar<T>,
        <T as TypeCommon>::Vec: NormalOut<BoolVector, Output = <T as TypeCommon>::Vec>,
    {
        if self.shape().len() < 2 {
            return Err(
                ErrHandler::NdimNotEnough(2, self.shape().len(), Location::caller()).into(),
            );
        }
        let mask: _Tensor<bool> = _Tensor::<bool>::tri(
            self.shape()[self.shape().len() - 2] as usize,
            self.shape()[self.shape().len() - 1] as usize,
            k,
            false,
        )?;
        let res = self.clone() * mask;
        Ok(res)
    }

    fn identity(n: usize) -> Result<Self>
    where
        u8: IntoScalar<T>,
    {
        _Tensor::eye(n, n, 0)
    }
}

impl<T: CommonBounds> ShapeManipulate for _Tensor<T> {
    type Meta = T;
    fn concat(tensors: Vec<&_Tensor<T>>, axis: usize, keepdims: bool) -> Result<Self>
    where
        T: 'static,
    {
        concat(tensors, axis, keepdims)
    }
    fn dstack(mut tensors: Vec<&_Tensor<T>>) -> Result<_Tensor<T>> {
        let mut new_tensors = Vec::with_capacity(tensors.len());
        for tensor in tensors.iter_mut() {
            if tensor.shape().len() < 3 {
                if tensor.shape().len() == 1 {
                    new_tensors.push(tensor.reshape(vec![1, tensor.shape()[0], 1])?);
                } else if tensor.shape().len() == 0 {
                    new_tensors.push(tensor.reshape(vec![1, 1, 1])?);
                } else {
                    new_tensors.push(tensor.reshape(vec![
                        tensor.shape()[0],
                        tensor.shape()[1],
                        1,
                    ])?);
                }
            } else {
                new_tensors.push(tensor.clone());
            }
        }
        let mut tensors_ref = Vec::with_capacity(new_tensors.len());
        for tensor in new_tensors.iter() {
            tensors_ref.push(tensor);
        }
        concat(tensors_ref, 2, false)
    }
    fn vstack(tensors: Vec<&_Tensor<T>>) -> Result<_Tensor<T>> {
        concat(tensors, 0, false)
    }
    fn hstack(mut tensors: Vec<&_Tensor<T>>) -> Result<_Tensor<T>> {
        for tensor in tensors.iter_mut() {
            if tensor.shape().len() < 2 {
                return if tensor.shape().len() == 1 {
                    concat(tensors, 0, false)
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
                    concat(tensors_ref, 0, false)
                };
            }
        }
        concat(tensors, 1, false)
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn squeeze<A: Into<Axis>>(&self, axes: A) -> Result<_Tensor<T>> {
        let axes: Vec<usize> = process_axes(axes, self.ndim())?;
        for i in 0..axes.len() {
            if self.shape()[axes[i]] != 1 {
                return Err(ErrHandler::SqueezeError(
                    axes[i],
                    self.shape().clone(),
                    Location::caller(),
                )
                .into());
            }
        }
        let new_shape: Vec<i64> = self
            .shape()
            .iter()
            .enumerate()
            .filter(|&(i, _)| !axes.contains(&i))
            .map(|(_, &x)| x)
            .collect();
        self.reshape(new_shape)
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn unsqueeze<A: Into<Axis>>(&self, axes: A) -> Result<_Tensor<T>> {
        let mut res_shape: Vec<i64> = self.shape().to_vec();
        let axes: Vec<usize> = process_axes(axes, self.ndim())?;
        axes.iter().for_each(|&x| {
            res_shape = yield_one_before(&res_shape, x);
        });
        self.reshape(res_shape)
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn reshape<S: Into<Shape>>(&self, shape: S) -> Result<_Tensor<T>> {
        let shape: Shape = shape.into();
        if shape.size() != (self.size() as i64) {
            return Err(ErrHandler::ReshapeError(
                self.shape().clone(),
                shape.clone(),
                self.size(),
                shape.size() as usize,
                Location::caller(),
            )
            .into());
        }
        if let Ok(new_layout) = self.layout.inplace_reshape(&shape) {
            Ok(_Tensor {
                data: self.data.clone(),
                parent: self.parent.clone(),
                mem_layout: self.mem_layout.clone(),
                layout: new_layout,
                _backend: self._backend.clone(),
            })
        } else {
            self.contiguous()?.reshape(shape)
        }
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn transpose(&self, axis1: i64, axis2: i64) -> Result<_Tensor<T>> {
        if self.ndim() < 2 {
            Err(
                ErrHandler::TransposeError(self.shape().clone(), self.ndim(), Location::caller())
                    .into(),
            )
        } else {
            self.permute(vec![axis1, axis2])
        }
    }

    #[cfg_attr(feature = "track_caller", track_caller)]
    fn permute<A: Into<Axis>>(&self, axes: A) -> Result<_Tensor<T>> {
        let permuted_layout = self.layout.permute(axes)?;
        Ok(_Tensor {
            data: self.data.clone(),
            layout: permuted_layout,
            parent: self.parent.clone(),
            mem_layout: self.mem_layout.clone(),
            _backend: self._backend.clone(),
        })
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn permute_inv<A: Into<Axis>>(&self, axes: A) -> Result<Self> {
        let permuted_layout = self.layout.permute_inv(axes)?;
        Ok(_Tensor {
            data: self.data.clone(),
            layout: permuted_layout,
            parent: self.parent.clone(),
            mem_layout: self.mem_layout.clone(),
            _backend: self._backend.clone(),
        })
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn expand<S: Into<Shape>>(&self, shape: S) -> Result<_Tensor<T>> {
        let res_shape = Shape::from(shape.into());
        let res_strides = self.layout.expand_strides(&res_shape);
        Ok(Self {
            data: self.data.clone(),
            parent: self.parent.clone(),
            mem_layout: self.mem_layout.clone(),
            layout: Layout::new(res_shape, res_strides),
            _backend: self._backend.clone(),
        })
    }

    #[cfg_attr(feature = "track_caller", track_caller)]
    fn t(&self) -> Result<Self> {
        if self.ndim() > 2 {
            let mut axes = (0..self.ndim() as i64).collect::<Vec<i64>>();
            axes.swap(self.ndim() - 1, self.ndim() - 2);
            return self.permute(axes);
        }
        self.transpose(1, 0)
    }
    fn mt(&self) -> Result<Self> {
        self.permute((0..self.ndim() as i64).rev().collect::<Vec<i64>>())
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn flip<A: Into<Axis>>(&self, axes: A) -> Result<Self> {
        let axes = process_axes(axes, self.ndim())?;
        let mut new_strides = self.strides().to_vec();
        let mut ptr = self.ptr();
        for &i in axes.iter() {
            new_strides[i] = -new_strides[i];
            ptr.offset(self.strides()[i]);
        }
        if self.parent.is_none() {
            Ok(Self {
                data: ptr,
                parent: Some(self.data.clone()),
                mem_layout: self.mem_layout.clone(),
                layout: Layout::new(self.shape().clone(), new_strides),
                _backend: self._backend.clone(),
            })
        } else {
            Ok(Self {
                data: ptr,
                parent: self.parent.clone(),
                mem_layout: self.mem_layout.clone(),
                layout: Layout::new(self.shape().clone(), new_strides),
                _backend: self._backend.clone(),
            })
        }
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn fliplr(&self) -> Result<Self> {
        if self.ndim() < 2 {
            return Err(ErrHandler::NdimNotEnough(2, self.ndim(), Location::caller()).into());
        }
        self.flip(1)
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn flipud(&self) -> Result<Self> {
        if self.ndim() < 1 {
            return Err(ErrHandler::NdimNotEnough(1, self.ndim(), Location::caller()).into());
        }
        self.flip(0)
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn tile<S: Into<Axis>>(&self, repeats: S) -> Result<Self> {
        let repeats: Axis = repeats.into();
        ErrHandler::check_index_in_range(self.ndim(), (repeats.axes.len() - 1) as i64)?;
        let repeats: Vec<i64> = repeats
            .axes
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
        res.reshape(final_shape)
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn trim_zeros(&self, trim: &str) -> Result<Self>
    where
        Self::Meta: PartialEq,
    {
        if !(trim == "fb" || trim == "f" || trim == "b") {
            return Err(anyhow::Error::msg("trim must be one of 'fb', 'f', 'b'"));
        }
        if self.ndim() > 1 {
            return Err(ErrHandler::NdimExceed(1, self.ndim(), Location::caller()).into());
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
        slice!(self[left_len:right_len])
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
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
        Ok(new_tensor.contiguous()?.reshape(new_shape)?)
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
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
        tmp[axis as usize] = Slice::Range((last, self.shape()[axis as usize]));
        let remain = self.slice(&tmp)?;
        reses.push(remain);
        Ok(reses)
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn dsplit(&self, indices: &[i64]) -> Result<Vec<Self>> {
        if self.shape().len() < 3 {
            return Err(
                ErrHandler::NdimNotEnough(3, self.shape().len(), Location::caller()).into(),
            );
        }
        self.split(indices, 2)
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn hsplit(&self, indices: &[i64]) -> Result<Vec<Self>> {
        if self.shape().len() < 2 {
            return Err(
                ErrHandler::NdimNotEnough(2, self.shape().len(), Location::caller()).into(),
            );
        }
        self.split(indices, 1)
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn vsplit(&self, indices: &[i64]) -> Result<Vec<Self>> {
        if self.shape().len() < 1 {
            return Err(
                ErrHandler::NdimNotEnough(1, self.shape().len(), Location::caller()).into(),
            );
        }
        self.split(indices, 0)
    }

    #[cfg_attr(feature = "track_caller", track_caller)]
    fn swap_axes(&self, mut axis1: i64, mut axis2: i64) -> Result<Self> {
        ErrHandler::check_index_in_range_mut(self.ndim(), &mut axis1)?;
        ErrHandler::check_index_in_range_mut(self.ndim(), &mut axis2)?;
        let mut new_shape = self.shape().to_vec();
        let mut new_strides = self.strides().to_vec();
        new_shape.swap(axis1 as usize, axis2 as usize);
        new_strides.swap(axis1 as usize, axis2 as usize);
        let layout = Layout::new(new_shape, new_strides);
        Ok(Self {
            data: self.data.clone(),
            layout,
            parent: self.parent.clone(),
            mem_layout: self.mem_layout.clone(),
            _backend: self._backend.clone(),
        })
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn flatten<A>(&self, start_dim: A, end_dim: A) -> Result<Self>
    where
        A: Into<Option<usize>>,
    {
        let start = start_dim.into().unwrap_or(0);
        let end = end_dim.into().unwrap_or(self.ndim() - 1);
        let shape = self.shape();
        ErrHandler::check_index_in_range(self.ndim(), start as i64)?;
        ErrHandler::check_index_in_range(self.ndim(), end as i64)?;
        let flattened_dim = shape[start..=end].iter().product::<i64>();
        let mut new_shape = Vec::new();
        for (i, &dim) in shape.iter().enumerate() {
            if i < start {
                new_shape.push(dim);
            } else if i == start {
                new_shape.push(flattened_dim);
            } else if i > end {
                new_shape.push(dim);
            }
        }
        self.reshape(new_shape)
    }
}

impl<T> Random for Tensor<T>
where
    T: CommonBounds + SampleUniform + num::Float + rand_distr::num_traits::FloatConst,
    <T as SampleUniform>::Sampler: Sync,
    StandardNormal: Distribution<T>,
    Open01: Distribution<T>,
    Exp1: Distribution<T>,
    OpenClosed01: Distribution<T>,
    Standard: Distribution<T>,
{
    type Meta = T;

    fn randn<S: Into<Shape>>(shape: S) -> Result<Self> {
        Ok(_Tensor::<T>::randn(shape)?.into())
    }

    fn randn_like(&self) -> Result<Self> {
        Ok(_Tensor::<T>::randn_like(self)?.into())
    }

    fn rand<S: Into<Shape>>(shape: S, low: Self::Meta, high: Self::Meta) -> Result<Self> {
        Ok(_Tensor::<T>::rand(shape, low, high)?.into())
    }

    fn rand_like(&self) -> Result<Self> {
        Ok(_Tensor::<T>::rand_like(self)?.into())
    }

    fn beta<S: Into<Shape>>(a: Self::Meta, b: Self::Meta, shape: S) -> Result<Self> {
        Ok(_Tensor::<T>::beta(a, b, shape)?.into())
    }

    fn beta_like(&self, a: Self::Meta, b: Self::Meta) -> Result<Self> {
        Ok(_Tensor::<T>::beta_like(self, a, b)?.into())
    }

    fn chisquare<S: Into<Shape>>(df: Self::Meta, shape: S) -> Result<Self> {
        Ok(_Tensor::<T>::chisquare(df, shape)?.into())
    }

    fn chisquare_like(&self, df: Self::Meta) -> Result<Self> {
        Ok(_Tensor::<T>::chisquare_like(self, df)?.into())
    }

    fn exponential<S: Into<Shape>>(lambda: Self::Meta, shape: S) -> Result<Self> {
        Ok(_Tensor::<T>::exponential(lambda, shape)?.into())
    }

    fn exponential_like(&self, lambda: Self::Meta) -> Result<Self> {
        Ok(_Tensor::<T>::exponential_like(self, lambda)?.into())
    }

    fn gamma<S: Into<Shape>>(gamm_shape: Self::Meta, scale: Self::Meta, shape: S) -> Result<Self> {
        Ok(_Tensor::<T>::gamma(gamm_shape, scale, shape)?.into())
    }

    fn gamma_like(&self, shape: Self::Meta, scale: Self::Meta) -> Result<Self> {
        Ok(_Tensor::<T>::gamma_like(self, shape, scale)?.into())
    }

    fn gumbel<S: Into<Shape>>(mu: Self::Meta, beta: Self::Meta, shape: S) -> Result<Self> {
        Ok(_Tensor::<T>::gumbel(mu, beta, shape)?.into())
    }

    fn gumbel_like(&self, mu: Self::Meta, beta: Self::Meta) -> Result<Self> {
        Ok(_Tensor::<T>::gumbel_like(self, mu, beta)?.into())
    }

    fn lognormal<S: Into<Shape>>(mean: Self::Meta, std: Self::Meta, shape: S) -> Result<Self> {
        Ok(_Tensor::<T>::lognormal(mean, std, shape)?.into())
    }

    fn lognormal_like(&self, mean: Self::Meta, std: Self::Meta) -> Result<Self> {
        Ok(_Tensor::<T>::lognormal_like(self, mean, std)?.into())
    }

    fn normal_gaussian<S: Into<Shape>>(
        mean: Self::Meta,
        std: Self::Meta,
        shape: S,
    ) -> Result<Self> {
        Ok(_Tensor::<T>::normal_gaussian(mean, std, shape)?.into())
    }

    fn normal_gaussian_like(&self, mean: Self::Meta, std: Self::Meta) -> Result<Self> {
        Ok(_Tensor::<T>::normal_gaussian_like(self, mean, std)?.into())
    }

    fn pareto<S: Into<Shape>>(pareto_shape: Self::Meta, a: Self::Meta, shape: S) -> Result<Self> {
        Ok(_Tensor::<T>::pareto(pareto_shape, a, shape)?.into())
    }

    fn pareto_like(&self, pareto_shape: Self::Meta, a: Self::Meta) -> Result<Self> {
        Ok(_Tensor::<T>::pareto_like(self, pareto_shape, a)?.into())
    }

    fn poisson<S: Into<Shape>>(lambda: Self::Meta, shape: S) -> Result<Self> {
        Ok(_Tensor::<T>::poisson(lambda, shape)?.into())
    }

    fn poisson_like(&self, lambda: Self::Meta) -> Result<Self> {
        Ok(_Tensor::<T>::poisson_like(self, lambda)?.into())
    }

    fn weibull<S: Into<Shape>>(a: Self::Meta, b: Self::Meta, shape: S) -> Result<Self> {
        Ok(_Tensor::<T>::weibull(a, b, shape)?.into())
    }

    fn weibull_like(&self, a: Self::Meta, b: Self::Meta) -> Result<Self> {
        Ok(_Tensor::<T>::weibull_like(self, a, b)?.into())
    }

    fn zipf<S: Into<Shape>>(n: u64, a: Self::Meta, shape: S) -> Result<Self> {
        Ok(_Tensor::<T>::zipf(n, a, shape)?.into())
    }

    fn zipf_like(&self, n: u64, a: Self::Meta) -> Result<Self> {
        Ok(_Tensor::<T>::zipf_like(self, n, a)?.into())
    }

    fn triangular<S: Into<Shape>>(
        low: Self::Meta,
        high: Self::Meta,
        mode: Self::Meta,
        shape: S,
    ) -> Result<Self> {
        Ok(_Tensor::<T>::triangular(low, high, mode, shape)?.into())
    }

    fn triangular_like(&self, low: Self::Meta, high: Self::Meta, mode: Self::Meta) -> Result<Self> {
        Ok(_Tensor::<T>::triangular_like(self, low, high, mode)?.into())
    }

    fn bernoulli<S: Into<Shape>>(shape: S, p: Self::Meta) -> Result<Self>
    where
        T: IntoScalar<f64>,
        bool: IntoScalar<T>,
    {
        Ok(_Tensor::<T>::bernoulli(shape, p)?.into())
    }
}

impl<T> RandomInt for Tensor<T>
where
    T: CommonBounds + SampleUniform,
{
    type Meta = T;

    fn randint<S: Into<Shape>>(low: Self::Meta, high: Self::Meta, shape: S) -> Result<Self>
    where
        <T as SampleUniform>::Sampler: Sync,
    {
        Ok(_Tensor::<T>::randint(low, high, shape)?.into())
    }

    fn randint_like(&self, low: Self::Meta, high: Self::Meta) -> Result<Self>
    where
        <T as SampleUniform>::Sampler: Sync,
    {
        Ok(_Tensor::<T>::randint_like(self, low, high)?.into())
    }
}

impl<T> Display for _Tensor<T>
where
    T: CommonBounds + Convertor,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let precision = DISPLAY_PRECISION.load(Ordering::Relaxed);
        let lr_element_size = DISPLAY_LR_ELEMENTS.load(Ordering::Relaxed);
        display(self, f, lr_element_size, precision, false)
    }
}

impl<T> Debug for _Tensor<T>
where
    T: CommonBounds + Convertor,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let precision = DISPLAY_PRECISION.load(Ordering::Relaxed);
        let lr_element_size = DISPLAY_LR_ELEMENTS.load(Ordering::Relaxed);
        display(self, f, lr_element_size, precision, false)
    }
}

impl<T> Into<Tensor<T>> for _Tensor<T> {
    fn into(self) -> Tensor<T> {
        Tensor { inner: self.into() }
    }
}

impl<T> Into<_Tensor<T>> for &_Tensor<T>
where
    T: CommonBounds,
{
    fn into(self) -> _Tensor<T> {
        _Tensor {
            data: self.data.clone(),
            parent: self.parent.clone(),
            layout: self.layout.clone(),
            mem_layout: self.mem_layout.clone(),
            _backend: self._backend.clone(),
        }
    }
}

impl<T> Into<Tensor<T>> for &Tensor<T> {
    fn into(self) -> Tensor<T> {
        Tensor {
            inner: self.inner.clone(),
        }
    }
}

impl<'a, T> Into<_Tensor<T>> for &'a [T] {
    fn into(self) -> _Tensor<T> {
        let shape = vec![self.len() as i64];
        let strides = vec![1];
        let layout = Layout::new(shape, strides);
        let mem_layout =
            std::alloc::Layout::from_size_align(self.len() * size_of::<T>(), ALIGN).unwrap();
        let ptr = CACHE.allocate(mem_layout.clone());
        unsafe {
            std::ptr::copy_nonoverlapping(self.as_ptr(), ptr as *mut T, self.len());
        }
        _Tensor {
            #[cfg(feature = "bound_check")]
            data: Pointer::new(ptr as *mut T, layout.clone()),
            #[cfg(not(feature = "bound_check"))]
            data: Pointer::new(ptr as *mut T),
            parent: None,
            layout,
            mem_layout: Arc::new(mem_layout),
            _backend: Backend::new(ptr as u64),
        }
    }
}
