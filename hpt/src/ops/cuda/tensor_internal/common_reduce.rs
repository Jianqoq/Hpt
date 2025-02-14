use std::borrow::Borrow;
use std::ops::BitAnd;

use crate::ops::cuda::reduce::{reduce, reduce2};
use crate::tensor_base::_Tensor;
use crate::Cuda;
use cudarc::driver::DeviceRepr;
use hpt_common::axis::axis::{process_axes, Axis};
use hpt_common::error::base::TensorError;
use hpt_cudakernels::{ALL, ANY, MAX, MIN, NANPROD, NANSUM, PROD, REDUCEL1, SUM};
use hpt_traits::{CommonBounds, EvalReduce, NormalEvalReduce, NormalReduce, TensorInfo};
use hpt_types::dtype::CudaType;
use hpt_types::into_scalar::Cast;
use hpt_types::traits::SimdSelect;
use hpt_types::type_promote::Eval;

impl<T: CommonBounds + DeviceRepr + CudaType, const DEVICE_ID: usize> NormalReduce<T>
    for _Tensor<T, Cuda, DEVICE_ID>
{
    type Output = Self;

    fn sum<S: Into<Axis>>(
        &self,
        axes: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes = process_axes(axes, self.ndim())?;
        reduce(self, &axes, T::ZERO, keep_dims, false, &SUM, "sum", None)
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
            &axes,
            T::ZERO,
            keep_dims,
            init_out,
            &SUM,
            "sum",
            Some(out.borrow().clone()),
        )
    }

    // fn sum_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool,
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
    //         &SUM,
    //         "sum",
    //         None,
    //     )
    // }

    fn prod<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes = process_axes(axis, self.ndim())?;
        reduce(self, &axes, T::ONE, keep_dims, false, &PROD, "prod", None)
    }

    // fn prod_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool,
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
    //         &PROD,
    //         "prod",
    //         None,
    //     )
    // }

    fn min<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce(self, &axes, T::INF, keep_dims, false, &MIN, "min", None)
    }

    // fn min_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool,
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
    //         &MIN,
    //         "min",
    //         None,
    //     )
    // }

    fn max<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce(self, &axes, T::NEG_INF, keep_dims, false, &MAX, "max", None)
    }

    // fn max_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool,
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
    //         &MAX,
    //         "max",
    //         None,
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
            &axes,
            T::ZERO,
            keep_dims,
            false,
            &REDUCEL1,
            "reducel1",
            None,
        )
    }

    fn sum_square<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce2(self, &axes, T::ZERO, keep_dims, false, &SUM, "sum", None)
    }
}

impl<T, const DEVICE_ID: usize> EvalReduce for _Tensor<T, Cuda, DEVICE_ID>
where
    T: CommonBounds + Eval<Output = bool> + Cast<bool> + DeviceRepr + CudaType,
{
    type BoolOutput = _Tensor<bool, Cuda, DEVICE_ID>;
    fn all<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::BoolOutput, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce2(self, &axes, true, keep_dims, false, &ALL, "all", None)
    }

    fn any<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::BoolOutput, TensorError> {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce2(self, &axes, false, keep_dims, false, &ANY, "any", None)
    }
}

impl<T, const DEVICE_ID: usize> NormalEvalReduce<T> for _Tensor<T, Cuda, DEVICE_ID>
where
    T: CommonBounds + Eval<Output = bool> + Cast<bool> + DeviceRepr + CudaType,
    T::Vec: Eval,
    <T::Vec as Eval>::Output: SimdSelect<T::Vec> + Copy,
    <T::Vec as Eval>::Output: BitAnd<Output = <T::Vec as Eval>::Output>,
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
            &axes,
            T::ZERO,
            keep_dims,
            false,
            &NANSUM,
            "nansum",
            None,
        )
    }

    // fn nansum_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool,
    // ) -> anyhow::Result<Self::Output> {
    //     let axes = process_axes(axes, self.ndim())?;
    //     reduce(
    //         self,
    //         |a, b| {
    //             if b._is_nan() {
    //                 if a._is_nan() {
    //                     T::ZERO
    //                 } else {
    //                     a
    //                 }
    //             } else {
    //                 b._add(a)
    //             }
    //         },
    //         |a, b| {
    //             if b._is_nan() {
    //                 if a._is_nan() {
    //                     T::ZERO
    //                 } else {
    //                     a
    //                 }
    //             } else {
    //                 b._add(a)
    //             }
    //         },
    //         |a, b| {
    //             let mask = b._is_nan();
    //             mask.select(a, b._add(a))
    //         },
    //         &axes,
    //         init_val,
    //         keep_dims,
    //         false,
    //         &NANSUM,
    //         "nansum",
    //         None,
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
            &axes,
            T::ONE,
            keep_dims,
            false,
            &NANPROD,
            "nanprod",
            None,
        )
    }

    fn nansum_<S: Into<hpt_common::axis::axis::Axis>, O>(
        &self,
        _: S,
        _: bool,
        _: bool,
        _: O,
    ) -> Result<Self::Output, hpt_common::error::base::TensorError>
    where
        O: Borrow<Self::Output>,
    {
        todo!()
    }

    // fn nanprod_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool,
    // ) -> anyhow::Result<Self::Output> {
    //     let axes: Vec<usize> = process_axes(axes, self.ndim())?;
    //     reduce(
    //         self,
    //         |a, b| {
    //             if b._is_nan() {
    //                 if a._is_nan() {
    //                     T::ONE
    //                 } else {
    //                     a
    //                 }
    //             } else {
    //                 b._mul(a)
    //             }
    //         },
    //         |a, b| {
    //             if b._is_nan() {
    //                 if a._is_nan() {
    //                     T::ONE
    //                 } else {
    //                     a
    //                 }
    //             } else {
    //                 b._mul(a)
    //             }
    //         },
    //         |a, b| {
    //             let mask = b._is_nan();
    //             mask.select(a, b._mul(a))
    //         },
    //         &axes,
    //         init_val,
    //         keep_dims,
    //         false,
    //         &NANPROD,
    //         "nanprod",
    //         None,
    //     )
    // }
}
