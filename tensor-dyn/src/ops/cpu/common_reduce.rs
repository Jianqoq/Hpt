use tensor_common::axis::{ process_axes, Axis };
use tensor_traits::{ CommonBounds, TensorInfo };
use tensor_types::{
    convertion::VecConvertor,
    dtype::TypeCommon,
    into_scalar::IntoScalar,
    into_vec::IntoVec,
    type_promote::{ BitWiseOut, Eval, NormalOut },
    vectors::traits::SimdSelect,
};
use crate::tensor_base::_Tensor;

use super::reduce::{ reduce, reduce2 };

impl<T> _Tensor<T>
    where
        T: CommonBounds + NormalOut<T, Output = T>,
        <T as TypeCommon>::Vec: NormalOut<<T as TypeCommon>::Vec, Output = <T as TypeCommon>::Vec>
{
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
    fn reducel1<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self> {
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
    fn sum_square<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self> {
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
