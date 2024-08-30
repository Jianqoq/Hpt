use tensor_common::axis::{ process_axes, Axis };
use tensor_traits::{ CommonBounds, TensorInfo };
use tensor_types::{
    convertion::VecConvertor,
    dtype::TypeCommon,
    into_scalar::IntoScalar,
    into_vec::IntoVec,
    type_promote::{ BitWiseOut, Eval, FloatOutBinary, FloatOutUnary, NormalOut },
    vectors::traits::SimdSelect,
};
use tensor_types::vectors::traits::Init;
use crate::tensor_base::_Tensor;

use super::{ reduce::{ reduce, reduce2, reduce3 }, unary::FloatBinaryType };

impl<T> _Tensor<T>
    where
        T: CommonBounds + NormalOut<T, Output = T>,
        <T as TypeCommon>::Vec: NormalOut<<T as TypeCommon>::Vec, Output = <T as TypeCommon>::Vec>
{
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn sum<S: Into<Axis>>(&self, axes: S, keep_dims: bool) -> anyhow::Result<Self> {
        let axes = process_axes(axes, self.ndim())?;
        reduce(
            self,
            |a, b| a._add(b),
            |a, b| a._add(b),
            &axes,
            T::ZERO,
            keep_dims,
            false,
            None
        )
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn sum_<S: Into<Axis>>(
        &self,
        axes: S,
        keep_dims: bool,
        init_out: bool,
        out: &_Tensor<T>
    ) -> anyhow::Result<Self> {
        let axes = process_axes(axes, self.ndim())?;
        reduce(
            self,
            |a, b| a._add(b),
            |a, b| a._add(b),
            &axes,
            T::ZERO,
            keep_dims,
            init_out,
            Some(out.clone())
        )
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn sum_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool
    ) -> anyhow::Result<Self> {
        let axes = process_axes(axes, self.ndim())?;
        reduce(
            self,
            |a, b| a._add(b),
            |a, b| a._add(b),
            &axes,
            init_val,
            keep_dims,
            false,
            None
        )
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn nansum<S: Into<Axis>>(&self, axes: S, keep_dims: bool) -> anyhow::Result<Self>
        where
            T: Eval<Output = bool>,
            <T as TypeCommon>::Vec: Eval,
            <<T as TypeCommon>::Vec as Eval>::Output: SimdSelect<<T as TypeCommon>::Vec>
    {
        let axes = process_axes(axes, self.ndim())?;
        reduce(
            self,
            |a, b| {
                if b._is_nan() { a } else { b._add(a) }
            },
            |a, b| {
                let mask = b._is_nan();
                mask.select(a, b._add(a))
            },
            &axes,
            T::ZERO,
            keep_dims,
            false,
            None
        )
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn nansum_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool
    )
        -> anyhow::Result<Self>
        where
            T: Eval<Output = bool>,
            <T as TypeCommon>::Vec: Eval,
            <<T as TypeCommon>::Vec as Eval>::Output: SimdSelect<<T as TypeCommon>::Vec>
    {
        let axes = process_axes(axes, self.ndim())?;
        reduce(
            self,
            |a, b| {
                if b._is_nan() { a } else { b._add(a) }
            },
            |a, b| {
                let mask = b._is_nan();
                mask.select(a, b._add(a))
            },
            &axes,
            init_val,
            keep_dims,
            false,
            None
        )
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn prod<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self> {
        let axes = process_axes(axis, self.ndim())?;
        reduce(
            self,
            |a, b| a._mul(b),
            |a, b| a._mul(b),
            &axes,
            T::ONE,
            keep_dims,
            false,
            None
        )
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn prod_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool
    ) -> anyhow::Result<Self> {
        let axes = process_axes(axes, self.ndim())?;
        reduce(
            self,
            |a, b| a._mul(b),
            |a, b| a._mul(b),
            &axes,
            init_val,
            keep_dims,
            false,
            None
        )
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn nanprod<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self>
        where
            T: Eval<Output = bool>,
            <T as TypeCommon>::Vec: Eval,
            <<T as TypeCommon>::Vec as Eval>::Output: SimdSelect<<T as TypeCommon>::Vec>
    {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce(
            self,
            |a, b| {
                if b._is_nan() { a } else { b._mul(a) }
            },
            |a, b| {
                let mask = b._is_nan();
                mask.select(a, b._mul(a))
            },
            &axes,
            T::ONE,
            keep_dims,
            false,
            None
        )
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn nanprod_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool
    )
        -> anyhow::Result<Self>
        where
            T: Eval<Output = bool>,
            <T as TypeCommon>::Vec: Eval,
            <<T as TypeCommon>::Vec as Eval>::Output: SimdSelect<<T as TypeCommon>::Vec>
    {
        let axes: Vec<usize> = process_axes(axes, self.ndim())?;
        reduce(
            self,
            |a, b| {
                if b._is_nan() { a } else { b._mul(a) }
            },
            |a, b| {
                let mask = b._is_nan();
                mask.select(a, b._mul(a))
            },
            &axes,
            init_val,
            keep_dims,
            false,
            None
        )
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn min<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce(
            self,
            |a, b| a._min(b),
            |a, b| a._min(b),
            &axes,
            T::INF,
            keep_dims,
            false,
            None
        )
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn min_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool
    ) -> anyhow::Result<Self> {
        let axes: Vec<usize> = process_axes(axes, self.ndim())?;
        reduce(
            self,
            |a, b| a._min(b),
            |a, b| a._min(b),
            &axes,
            init_val,
            keep_dims,
            false,
            None
        )
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn max<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce(
            self,
            |a, b| a._max(b),
            |a, b| a._max(b),
            &axes,
            T::NEG_INF,
            keep_dims,
            false,
            None
        )
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn max_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool
    ) -> anyhow::Result<Self> {
        let axes: Vec<usize> = process_axes(axes, self.ndim())?;
        reduce(
            self,
            |a, b| a._max(b),
            |a, b| a._max(b),
            &axes,
            init_val,
            keep_dims,
            false,
            None
        )
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn all<S: Into<Axis>>(&self, axes: S, keep_dims: bool) -> anyhow::Result<_Tensor<bool>>
        where
            T: IntoScalar<bool> + Eval<Output = bool>,
            <T as TypeCommon>::Vec: Eval + VecConvertor,
            <<T as TypeCommon>::Vec as Eval>::Output: SimdSelect<<T as TypeCommon>::Vec>
    {
        let axes: Vec<usize> = process_axes(axes, self.ndim())?;
        reduce2(
            self,
            |a, b| b._is_true() & a,
            |a, b| b | a,
            |a, b| {
                let mask = b.to_bool();
                mask | a
            },
            &axes,
            true,
            keep_dims,
            false,
            None
        )
    }

    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn any<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<_Tensor<bool>>
        where
            <T as TypeCommon>::Vec: IntoVec<tensor_types::vectors::boolx32::boolx32>,
            T: IntoScalar<bool> + Eval<Output = bool>,
            <T as TypeCommon>::Vec: Eval + BitWiseOut + VecConvertor,
            <<T as TypeCommon>::Vec as Eval>::Output: SimdSelect<<T as TypeCommon>::Vec>
    {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce2(
            self,
            |a, b| b._is_true() | a,
            |a, b| b | a,
            |a, b| {
                let mask = b.to_bool();
                mask | a
            },
            &axes,
            false,
            keep_dims,
            false,
            None
        )
    }

    #[allow(unused)]
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn reducel1<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce(
            self,
            |a, b| a._add(b._abs()),
            |a, b| a._add(b._abs()),
            &axes,
            T::ZERO,
            keep_dims,
            false,
            None
        )
    }
    #[allow(unused)]
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn sum_square<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce(
            self,
            |a, b| a._add(b._square()),
            |a, b| a._add(b._square()),
            &axes,
            T::ZERO,
            keep_dims,
            false,
            None
        )
    }
}

impl<T> _Tensor<T>
    where
        T: FloatOutBinary + CommonBounds + IntoScalar<<T as FloatOutBinary>::Output>,
        <T as FloatOutBinary>::Output: TypeCommon,
        <T as FloatOutBinary>::Output: CommonBounds +
            NormalOut<<T as FloatOutBinary>::Output, Output = <T as FloatOutBinary>::Output> +
            FloatOutUnary<Output = <T as FloatOutBinary>::Output>,
        <<T as FloatOutBinary>::Output as TypeCommon>::Vec: NormalOut<
            <T as TypeCommon>::Vec,
            Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec
        > +
            FloatOutUnary<Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec>
{
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn mean<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool
    )
        -> anyhow::Result<_Tensor<FloatBinaryType<T>>>
        where
            f64: IntoScalar<<T as FloatOutBinary>::Output>,
            <T as FloatOutBinary>::Output: NormalOut<T, Output = <T as FloatOutBinary>::Output> +
                FloatOutBinary<
                    <T as FloatOutBinary>::Output,
                    Output = <T as FloatOutBinary>::Output
                >,
            <<T as FloatOutBinary>::Output as TypeCommon>::Vec: FloatOutBinary<
                <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
                Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec
            >
    {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        let reduce_size: FloatBinaryType<T> = (
            axes.iter().fold(1, |acc, &x| acc * (self.shape()[x] as usize)) as f64
        ).into_scalar();
        let reduce_vec = <FloatBinaryType<T> as TypeCommon>::Vec::splat(reduce_size);
        reduce3(
            self,
            |a, b| a._add(b),
            |a, b| a._add(b),
            move |a| a._div(reduce_size),
            |a, b| a._add(b),
            move |a| a._div(reduce_vec),
            &axes,
            <T as FloatOutBinary>::Output::ZERO,
            keep_dims,
            false,
            None
        )
    }

    #[allow(unused)]
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn reducel2<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool
    )
        -> anyhow::Result<_Tensor<FloatBinaryType<T>>>
        where
            T: NormalOut,
            <T as FloatOutBinary>::Output: NormalOut<T, Output = <T as FloatOutBinary>::Output>
    {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce3(
            self,
            |a, b| {
                let b = b._square();
                a._add(b)
            },
            |a, b| a._add(b),
            move |a| a._sqrt(),
            |a, b| a._add(b),
            |a| a._sqrt(),
            &axes,
            <T as FloatOutBinary>::Output::ZERO,
            keep_dims,
            false,
            None
        )
    }

    #[allow(unused)]
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn reducel3<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool
    )
        -> anyhow::Result<_Tensor<FloatBinaryType<T>>>
        where
            f32: IntoScalar<<T as FloatOutBinary>::Output>,
            T: NormalOut<<T as FloatOutBinary>::Output, Output = <T as FloatOutBinary>::Output>,
            <T as FloatOutBinary>::Output: TypeCommon,
            <T as TypeCommon>::Vec: NormalOut<
                <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
                Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec
            >,
            <<T as FloatOutBinary>::Output as TypeCommon>::Vec: NormalOut<
                <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
                Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec
            >
    {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        let three: <T as FloatOutBinary>::Output = (3.0).into_scalar();
        let three_vec = <<T as FloatOutBinary>::Output as TypeCommon>::Vec::splat(three);
        reduce3(
            self,
            move |a, b| {
                let b = b._abs();
                let pow = b._pow(three);
                a._add(pow)
            },
            move |a, b| a._add(<FloatBinaryType<T> as NormalOut>::_abs(b)._pow(three)),
            move |a| a,
            move |a, b| {
                let abs = b._abs();
                let pow = abs._pow(three_vec);
                a._add(pow)
            },
            |a| a,
            &axes,
            <T as FloatOutBinary>::Output::ZERO,
            keep_dims,
            false,
            None
        )
    }

    #[allow(unused)]
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn logsumexp<S: Into<Axis>>(
        &self,
        _: S,
        _: bool
    )
        -> anyhow::Result<_Tensor<FloatBinaryType<T>>>
        where
            T: CommonBounds + NormalOut<T, Output = T>,
            <T as TypeCommon>::Vec: NormalOut<
                <T as TypeCommon>::Vec,
                Output = <T as TypeCommon>::Vec
            >
    {
        todo!()
        // let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        // let x_max = self.max(axes.as_ref(), true)?;
        // let sub = self - &x_max;
        // let exp = sub.exp()?;
        // let sum_exp = reduce(
        //     &exp,
        //     |a, b| a._add(b),
        //     &axes,
        //     <T as FloatOut>::Output::ZERO,
        //     true,
        //     false,
        //     None
        // )?;
        // let add = x_max + sum_exp.ln()?;
        // if keep_dims {
        //     Ok(add)
        // } else {
        //     Ok(add.squeeze(axes)?)
        // }
    }
}
