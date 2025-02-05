use std::borrow::Borrow;
use std::ops::BitAnd;

use crate::ops::cuda::reduce::{reduce, reduce2};
use crate::tensor_base::_Tensor;
use crate::Cuda;
use cudarc::driver::DeviceRepr;
use cudarc::types::CudaTypeName;
use tensor_common::axis::{process_axes, Axis};
use tensor_common::err_handler::TensorError;
use tensor_cudakernels::{ALL, ANY, MAX, MIN, NANPROD, NANSUM, PROD, REDUCEL1, SUM};
use tensor_traits::{CommonBounds, EvalReduce, NormalEvalReduce, NormalReduce, TensorInfo};
use tensor_types::convertion::VecConvertor;
use tensor_types::into_scalar::Cast;
use tensor_types::traits::SimdSelect;
use tensor_types::traits::VecTrait;
use tensor_types::type_promote::NormalOutUnary;
use tensor_types::type_promote::{Eval, NormalOut};

impl<T: CommonBounds + DeviceRepr + CudaTypeName, const DEVICE_ID: usize> NormalReduce<T>
    for _Tensor<T, Cuda, DEVICE_ID>
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
            &SUM,
            "sum",
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
        reduce(
            self,
            |a, b| a._mul(b),
            |a, b| a._mul(b),
            |a, b| a._mul(b),
            &axes,
            T::ONE,
            keep_dims,
            false,
            &PROD,
            "prod",
            None,
        )
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
        reduce(
            self,
            |a, b| a._min(b),
            |a, b| a._min(b),
            |a, b| a._min(b),
            &axes,
            T::INF,
            keep_dims,
            false,
            &MIN,
            "min",
            None,
        )
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
        reduce(
            self,
            |a, b| a._max(b),
            |a, b| a._max(b),
            |a, b| a._max(b),
            &axes,
            T::NEG_INF,
            keep_dims,
            false,
            &MAX,
            "max",
            None,
        )
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
            &SUM,
            "sum",
            None,
        )
    }
}

impl<T, const DEVICE_ID: usize> EvalReduce for _Tensor<T, Cuda, DEVICE_ID>
where
    T: CommonBounds + Eval<Output = bool> + Cast<bool> + DeviceRepr + CudaTypeName,
{
    type BoolOutput = _Tensor<bool, Cuda, DEVICE_ID>;
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
                let mask = b.to_bool();
                mask & a
            },
            |a, b| b & a,
            &axes,
            true,
            keep_dims,
            false,
            &ALL,
            "all",
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
                let mask = b.to_bool();
                mask | a
            },
            |a, b| b | a,
            &axes,
            false,
            keep_dims,
            false,
            &ANY,
            "any",
            None,
        )
    }
}

impl<T, const DEVICE_ID: usize> NormalEvalReduce<T> for _Tensor<T, Cuda, DEVICE_ID>
where
    T: CommonBounds + Eval<Output = bool> + Cast<bool> + DeviceRepr + CudaTypeName,
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
            |a, b| {
                if b._is_nan() {
                    if a._is_nan() {
                        T::ZERO
                    } else {
                        a
                    }
                } else {
                    b._add(a)
                }
            },
            |a, b| {
                if b._is_nan() {
                    if a._is_nan() {
                        T::ZERO
                    } else {
                        a
                    }
                } else {
                    b._add(a)
                }
            },
            |a, b| {
                let b_is_nan = b._is_nan();
                let a_is_nan = a._is_nan();
                let both_nan = b_is_nan & a_is_nan;

                // 使用SIMD select操作来高效处理不同情况
                let result = both_nan.select(
                    T::Vec::splat(T::ZERO), // 两个都是NaN
                    b_is_nan.select(
                        a, // 只有b是NaN
                        a_is_nan.select(
                            b,         // 只有a是NaN
                            b._add(a), // 都不是NaN
                        ),
                    ),
                );
                result
            },
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
            |a, b| {
                if b._is_nan() {
                    if a._is_nan() {
                        T::ONE
                    } else {
                        a
                    }
                } else {
                    b._mul(a)
                }
            },
            |a, b| {
                if b._is_nan() {
                    if a._is_nan() {
                        T::ONE
                    } else {
                        a
                    }
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
            &NANPROD,
            "nanprod",
            None,
        )
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
