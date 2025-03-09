use std::borrow::Borrow;

use crate::backend::Cpu;
use crate::backends::cpu::utils::reduce::reduce::{reduce, reduce_with_post};
use crate::tensor_base::_Tensor;
use crate::BoolVector;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_common::axis::axis::{process_axes, Axis};
use hpt_common::error::base::TensorError;
use hpt_traits::ops::reduce::{EvalReduce, FloatReduce, NormalEvalReduce, NormalReduce};
use hpt_traits::tensor::{CommonBounds, TensorInfo};
use hpt_types::into_vec::IntoVec;
use hpt_types::type_promote::NormalOutUnary;
use hpt_types::vectors::traits::VecTrait;
use hpt_types::{
    dtype::TypeCommon,
    into_scalar::Cast,
    type_promote::{Eval, FloatOutBinary, FloatOutUnary, NormalOut},
    vectors::traits::SimdSelect,
};

type FloatBinaryType<T> = <T as FloatOutBinary>::Output;

impl<T, const DEVICE: usize, A> NormalReduce<T> for _Tensor<T, Cpu, DEVICE, A>
where
    T: CommonBounds,
    A: Allocator + 'static + Send + Sync,
    A::Output: AllocatorOutputRetrive,
{
    type Output = Self;

    fn sum<S: Into<Axis>>(
        &self,
        axes: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes = process_axes(axes, self.ndim())?;
        reduce(
            self,
            |a| a,
            |a| a,
            |a, b| a._add(b),
            |a| a,
            |a| a,
            |a, b| a._add(b),
            &axes,
            T::ZERO,
            keep_dims,
            false,
            None,
        )
    }

    fn sum_<S: Into<Axis>, O>(
        &self,
        axes: S,
        keep_dims: bool,
        init_out: bool,
        out: O,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        O: Borrow<Self::Output>,
    {
        let axes = process_axes(axes, self.ndim())?;
        reduce(
            self,
            |a| a,
            |a| a,
            |a, b| a._add(b),
            |a| a,
            |a| a,
            |a, b| a._add(b),
            &axes,
            T::ZERO,
            keep_dims,
            init_out,
            Some(out.borrow().clone()),
        )
    }

    fn prod<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes = process_axes(axis, self.ndim())?;
        reduce(
            self,
            |a| a,
            |a| a,
            |a, b| a._mul(b),
            |a| a,
            |a| a,
            |a, b| a._mul(b),
            &axes,
            T::ONE,
            keep_dims,
            false,
            None,
        )
    }

    fn min<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce(
            self,
            |a| a,
            |a| a,
            |a, b| a._min(b),
            |a| a,
            |a| a,
            |a, b| a._min(b),
            &axes,
            T::INF,
            keep_dims,
            false,
            None,
        )
    }

    fn max<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce(
            self,
            |a| a,
            |a| a,
            |a, b| a._max(b),
            |a| a,
            |a| a,
            |a, b| a._max(b),
            &axes,
            T::NEG_INF,
            keep_dims,
            false,
            None,
        )
    }

    fn reducel1<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce(
            self,
            |a| a._abs(),
            |a| a._abs(),
            |a, b| a._add(b),
            |a| a._abs(),
            |a| a._abs(),
            |a, b| a._add(b),
            &axes,
            T::ZERO,
            keep_dims,
            false,
            None,
        )
    }

    fn sum_square<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce(
            self,
            |a| a._square(),
            |a| a._square(),
            |a, b| a._add(b),
            |a| a._square(),
            |a| a._square(),
            |a, b| a._add(b),
            &axes,
            T::ZERO,
            keep_dims,
            false,
            None,
        )
    }
}

impl<T, const DEVICE: usize, Al> EvalReduce for _Tensor<T, Cpu, DEVICE, Al>
where
    T: CommonBounds + Eval<Output = bool> + Cast<bool>,
    T::Vec: IntoVec<BoolVector>,
    Al: Allocator + 'static + Send + Sync,
    Al::Output: AllocatorOutputRetrive,
{
    type BoolOutput = _Tensor<bool, Cpu, DEVICE, Al>;
    fn all<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::BoolOutput, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce(
            self,
            |a| a._is_true(),
            |a| a,
            |a, b| a & b,
            |a| a.into_vec(),
            |a| a,
            |a, b| a & b,
            &axes,
            true,
            keep_dims,
            false,
            None,
        )
    }

    fn any<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::BoolOutput, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce(
            self,
            |a| a._is_true(),
            |a| a,
            |a, b| a | b,
            |a| a.into_vec(),
            |a| a,
            |a, b| a | b,
            &axes,
            false,
            keep_dims,
            false,
            None,
        )
    }
}

impl<T, const DEVICE: usize, Al> NormalEvalReduce<T> for _Tensor<T, Cpu, DEVICE, Al>
where
    T: CommonBounds + Eval<Output = bool> + Cast<bool>,
    T::Vec: Eval,
    <T::Vec as Eval>::Output: SimdSelect<T::Vec>,
    Al: Allocator + 'static + Send + Sync,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = Self;

    fn nansum<S: Into<Axis>>(
        &self,
        axes: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes = process_axes(axes, self.ndim())?;
        reduce(
            self,
            |a| if a._is_nan() { T::ZERO } else { a },
            |a| if a._is_nan() { T::ZERO } else { a },
            |a, b| a._add(b),
            |a| a._is_nan().select(T::Vec::splat(T::ZERO), a),
            |a| a._is_nan().select(T::Vec::splat(T::ZERO), a),
            |a, b| a._add(b),
            &axes,
            T::ZERO,
            keep_dims,
            false,
            None,
        )
    }

    fn nansum_<S: Into<Axis>, O>(
        &self,
        axes: S,
        keep_dims: bool,
        init_out: bool,
        out: O,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        O: Borrow<Self::Output>,
    {
        let axes = process_axes(axes, self.ndim())?;
        reduce(
            self,
            |a| if a._is_nan() { T::ZERO } else { a },
            |a| if a._is_nan() { T::ZERO } else { a },
            |a, b| a._add(b),
            |a| a._is_nan().select(T::Vec::splat(T::ZERO), a),
            |a| a._is_nan().select(T::Vec::splat(T::ZERO), a),
            |a, b| a._add(b),
            &axes,
            T::ZERO,
            keep_dims,
            init_out,
            Some(out.borrow().clone()),
        )
    }

    fn nanprod<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce(
            self,
            |a| if a._is_nan() { T::ONE } else { a },
            |a| if a._is_nan() { T::ONE } else { a },
            |a, b| a._mul(b),
            |a| a._is_nan().select(T::Vec::splat(T::ONE), a),
            |a| a._is_nan().select(T::Vec::splat(T::ONE), a),
            |a, b| a._mul(b),
            &axes,
            T::ONE,
            keep_dims,
            false,
            None,
        )
    }
}

type FloatBinaryTypeVec<T> = <FloatBinaryType<T> as TypeCommon>::Vec;

impl<T, const DEVICE: usize, Al> FloatReduce<T> for _Tensor<T, Cpu, DEVICE, Al>
where
    T: CommonBounds + Cast<FloatBinaryType<T>>,
    T::Vec: NormalOut<FloatBinaryTypeVec<T>, Output = FloatBinaryTypeVec<T>>
        + IntoVec<FloatBinaryTypeVec<T>>,

    FloatBinaryType<T>: CommonBounds
        + FloatOutUnary<Output = FloatBinaryType<T>>
        + NormalOut<T, Output = FloatBinaryType<T>>
        + NormalOut<<T as FloatOutUnary>::Output, Output = FloatBinaryType<T>>,
    FloatBinaryTypeVec<T>: NormalOut<T::Vec, Output = FloatBinaryTypeVec<T>>
        + FloatOutUnary<Output = FloatBinaryTypeVec<T>>
        + NormalOut<<<T as TypeCommon>::Vec as FloatOutUnary>::Output, Output = FloatBinaryTypeVec<T>>,

    f64: Cast<FloatBinaryType<T>>,
    Al: Allocator + 'static + Send + Sync,
    Al::Output: AllocatorOutputRetrive,

    <T as FloatOutUnary>::Output: std::fmt::Debug,
{
    type Output = _Tensor<FloatBinaryType<T>, Cpu, DEVICE, Al>;

    #[track_caller]
    fn mean<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        let reduce_size: FloatBinaryType<T> = (axes
            .iter()
            .fold(1, |acc, &x| acc * (self.shape()[x] as usize))
            as f64)
            .cast();
        let reduce_vec = FloatBinaryTypeVec::<T>::splat(reduce_size);
        reduce_with_post(
            self,
            |a| a.cast(),
            |a| a,
            |a, b| a._add(b),
            move |a| a._div(reduce_size),
            |a| a.into_vec(),
            |a| a,
            |a, b| a._add(b),
            move |a| a._div(reduce_vec),
            &axes,
            FloatBinaryType::<T>::ZERO,
            keep_dims,
            false,
            None,
        )
    }

    #[allow(unused)]
    #[track_caller]
    fn reducel2<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce_with_post(
            self,
            |a| {
                let casted_a: FloatBinaryType<T> = a.cast();
                casted_a._square()
            },
            |a| a._square(),
            |a, b| a._add(b),
            move |a| a._sqrt(),
            |a| {
                let casted_a: FloatBinaryTypeVec<T> = a.into_vec();
                casted_a._square()
            },
            |a| a._square(),
            |a, b| a._add(b),
            move |a| a._sqrt(),
            &axes,
            FloatBinaryType::<T>::ZERO,
            keep_dims,
            false,
            None,
        )
    }
    #[allow(unused)]
    #[track_caller]
    fn reducel3<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        let three: FloatBinaryType<T> = (3.0).cast();
        let three_vec = FloatBinaryTypeVec::<T>::splat(three);
        let one_third: FloatBinaryType<T> = (1.0f64 / 3.0f64).cast();
        let one_third_vec = FloatBinaryTypeVec::<T>::splat(one_third);
        let mut res = reduce_with_post(
            self,
            move |a| {
                let cast_abs_a: FloatBinaryType<T> = a._abs().cast();
                cast_abs_a._pow(three)
            },
            move |a| a._abs()._pow(three),
            |a, b| a._add(b),
            move |a| a._pow(one_third),
            move |a| {
                let cast_abs_a: FloatBinaryTypeVec<T> = a._abs().into_vec();
                cast_abs_a._pow(three_vec)
            },
            move |a| a._abs()._pow(three_vec),
            |a, b| a._add(b),
            move |a| a._pow(one_third_vec),
            &axes,
            FloatBinaryType::<T>::ZERO,
            keep_dims,
            false,
            None,
        )?;
        Ok(res)
    }
    #[allow(unused)]
    #[track_caller]
    fn logsumexp<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        T: CommonBounds,
    {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce_with_post(
            self,
            |a| {
                let casted_a: FloatBinaryType<T> = a.cast();
                casted_a._exp()
            },
            |a| a._exp(),
            |a, b| a._add(b),
            |a| a._ln(),
            |a| {
                let casted_a: FloatBinaryTypeVec<T> = a.into_vec();
                casted_a._exp()
            },
            |a| a._exp(),
            |a, b| a._add(b),
            |a| a._ln(),
            &axes,
            FloatBinaryType::<T>::ZERO,
            keep_dims,
            false,
            None,
        )
    }
}
