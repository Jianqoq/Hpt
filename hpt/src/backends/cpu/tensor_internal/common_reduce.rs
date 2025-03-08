use std::borrow::Borrow;

use crate::backend::Cpu;
use crate::backends::cpu::utils::reduce::reduce::{reduce, reduce2, reduce3};
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
            |a, b| a._add(b),
            |a, b| a._add(b),
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
            |a, b| a._add(b),
            |a, b| a._add(b),
            |a, b| a._add(b),
            &axes,
            T::ZERO,
            keep_dims,
            init_out,
            Some(out.borrow().clone()),
        )
    }

    // fn sum_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool
    // ) -> anyhow::Result<Self::Output> {
    //     let axes = process_axes(axes, self.ndim())?;
    //     reduce(
    //         self,
    //         |a, b| a._add(b),
    //         |a, b| a._add(b),
    //         |a, b| a._add(b),
    //         &axes,
    //         init_val,
    //         keep_dims,
    //         false,
    //         None
    //     )
    // }

    fn prod<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes = process_axes(axis, self.ndim())?;
        reduce(
            self,
            |a, b| a._mul(b),
            |a, b| a._mul(b),
            |a, b| a._mul(b),
            &axes,
            T::ONE,
            keep_dims,
            false,
            None,
        )
    }

    // fn prod_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool
    // ) -> anyhow::Result<Self::Output> {
    //     let axes = process_axes(axes, self.ndim())?;
    //     reduce(
    //         self,
    //         |a, b| a._mul(b),
    //         |a, b| a._mul(b),
    //         |a, b| a._mul(b),
    //         &axes,
    //         init_val,
    //         keep_dims,
    //         false,
    //         None
    //     )
    // }

    fn min<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce(
            self,
            |a, b| a._min(b),
            |a, b| a._min(b),
            |a, b| a._min(b),
            &axes,
            T::INF,
            keep_dims,
            false,
            None,
        )
    }

    // fn min_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool
    // ) -> anyhow::Result<Self> {
    //     let axes: Vec<usize> = process_axes(axes, self.ndim())?;
    //     reduce(
    //         self,
    //         |a, b| a._min(b),
    //         |a, b| a._min(b),
    //         |a, b| a._min(b),
    //         &axes,
    //         init_val,
    //         keep_dims,
    //         false,
    //         None
    //     )
    // }

    fn max<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce(
            self,
            |a, b| a._max(b),
            |a, b| a._max(b),
            |a, b| a._max(b),
            &axes,
            T::NEG_INF,
            keep_dims,
            false,
            None,
        )
    }

    // fn max_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool
    // ) -> anyhow::Result<Self> {
    //     let axes: Vec<usize> = process_axes(axes, self.ndim())?;
    //     reduce(
    //         self,
    //         |a, b| a._max(b),
    //         |a, b| a._max(b),
    //         |a, b| a._max(b),
    //         &axes,
    //         init_val,
    //         keep_dims,
    //         false,
    //         None
    //     )
    // }

    fn reducel1<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce(
            self,
            |a, b| {
                let b_abs = b._abs();
                a._add(b_abs)
            },
            |a, b| {
                let b_abs = b._abs();
                a._add(b_abs)
            },
            |a, b| a._add(b._abs()),
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
        reduce2(
            self,
            |a, b| a._add(b._square()),
            |a, b| a._add(b._square()),
            |a, b| a._add(b),
            |a, b| a._add(b._square()),
            |a, b| a._add(b._square()),
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
        reduce2(
            self,
            |a, b| b._is_true() & a,
            |a, b| b._is_true() & a,
            |a, b| b & a,
            |a, b| {
                let mask: BoolVector = b.into_vec();
                mask & a
            },
            |a, b| b & a,
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
        reduce2(
            self,
            |a, b| b._is_true() | a,
            |a, b| b._is_true() | a,
            |a, b| b | a,
            |a, b| {
                let mask: BoolVector = b.into_vec();
                mask | a
            },
            |a, b| b | a,
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
            |a, b| {
                if b._is_nan() {
                    a
                } else {
                    b._add(a)
                }
            },
            |a, b| {
                if b._is_nan() {
                    a
                } else {
                    b._add(a)
                }
            },
            |a, b| {
                let mask = b._is_nan();
                mask.select(a, b._add(a))
            },
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
            |a, b| {
                if b._is_nan() {
                    a
                } else {
                    b._add(a)
                }
            },
            |a, b| {
                if b._is_nan() {
                    a
                } else {
                    b._add(a)
                }
            },
            |a, b| {
                let mask = b._is_nan();
                mask.select(a, b._add(a))
            },
            &axes,
            T::ZERO,
            keep_dims,
            init_out,
            Some(out.borrow().clone()),
        )
    }

    // fn nansum_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool
    // ) -> anyhow::Result<Self::Output> {
    //     let axes = process_axes(axes, self.ndim())?;
    //     reduce(
    //         self,
    //         |a, b| {
    //             if b._is_nan() { a } else { b._add(a) }
    //         },
    //         |a, b| {
    //             if b._is_nan() { a } else { b._add(a) }
    //         },
    //         |a, b| {
    //             let mask = b._is_nan();
    //             mask.select(a, b._add(a))
    //         },
    //         &axes,
    //         init_val,
    //         keep_dims,
    //         false,
    //         None
    //     )
    // }

    fn nanprod<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce(
            self,
            |a, b| {
                if b._is_nan() {
                    a
                } else {
                    b._mul(a)
                }
            },
            |a, b| {
                if b._is_nan() {
                    a
                } else {
                    b._mul(a)
                }
            },
            |a, b| {
                let mask = b._is_nan();
                mask.select(a, b._mul(a))
            },
            &axes,
            T::ONE,
            keep_dims,
            false,
            None,
        )
    }

    // fn nanprod_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool
    // ) -> anyhow::Result<Self::Output> {
    //     let axes: Vec<usize> = process_axes(axes, self.ndim())?;
    //     reduce(
    //         self,
    //         |a, b| {
    //             if b._is_nan() { a } else { b._mul(a) }
    //         },
    //         |a, b| {
    //             if b._is_nan() { a } else { b._mul(a) }
    //         },
    //         |a, b| {
    //             let mask = b._is_nan();
    //             mask.select(a, b._mul(a))
    //         },
    //         &axes,
    //         init_val,
    //         keep_dims,
    //         false,
    //         None
    //     )
    // }
}

type FloatBinaryTypeVec<T> = <FloatBinaryType<T> as TypeCommon>::Vec;

impl<T, const DEVICE: usize, Al> FloatReduce<T> for _Tensor<T, Cpu, DEVICE, Al>
where
    T: CommonBounds + Cast<FloatBinaryType<T>>,
    T::Vec: NormalOut<FloatBinaryTypeVec<T>, Output = FloatBinaryTypeVec<T>>,

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
        reduce3(
            self,
            |a, b| a._add(b),
            |a, b| a._add(b),
            |a, b| a._add(b),
            move |a| a._div(reduce_size),
            |a, b| a._add(b),
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
        reduce3(
            self,
            |a, b| a._add(b._square()),
            |a, b| a._add(b._square()),
            |a, b| a._add(b),
            move |a| a._sqrt(),
            |a, b| a._add(b._square()),
            |a, b| a._add(b._square()),
            |a| a._sqrt(),
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
        let mut res = reduce3(
            self,
            move |a, b| {
                let pow = b._abs()._pow(three);
                a._add(pow)
            },
            move |a, b| {
                let pow = b._abs()._pow(three);
                a._add(pow)
            },
            move |a, b| a._add(b),
            move |a| a._pow(one_third),
            move |a, b| {
                let abs = b._abs();
                let pow = abs._pow(three_vec);
                a._add(pow)
            },
            move |a, b| {
                let abs = b._abs();
                let pow = abs._pow(three_vec);
                a._add(pow)
            },
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
        reduce3(
            self,
            |acc, b| {
                let exp = b._exp();
                acc._add(exp)
            },
            |acc, b| {
                let exp = b._exp();
                acc._add(exp)
            },
            |acc, b| {
                let exp = b._exp();
                acc._add(exp)
            },
            move |a| a._ln(),
            |a, b| a._add(b._exp()),
            |a, b| a._add(b._exp()),
            move |a| a._ln(),
            &axes,
            FloatBinaryType::<T>::ZERO,
            keep_dims,
            false,
            None,
        )
    }
}
