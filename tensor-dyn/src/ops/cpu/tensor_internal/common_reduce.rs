use std::borrow::Borrow;

use crate::ops::cpu::reduce::{reduce, reduce2, reduce3};
use crate::ops::cpu::tensor_internal::float_out_unary::FloatBinaryType;
use crate::tensor_base::_Tensor;
use tensor_common::axis::axis::{process_axes, Axis};
use tensor_common::error::base::TensorError;
use tensor_iterator::iterator_traits::ParStridedIteratorSimd;
use tensor_iterator::TensorIterator;
use tensor_traits::{CommonBounds, EvalReduce, NormalEvalReduce, NormalReduce, TensorInfo};
use tensor_types::type_promote::NormalOutUnary;
use tensor_types::vectors::traits::VecTrait;
use tensor_types::{
    convertion::{Convertor, VecConvertor},
    dtype::TypeCommon,
    into_scalar::IntoScalar,
    type_promote::{Eval, FloatOutBinary, FloatOutUnary, NormalOut},
    vectors::traits::SimdSelect,
};

impl<T: CommonBounds> NormalReduce<T> for _Tensor<T> {
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

impl<T> EvalReduce for _Tensor<T>
where
    T: CommonBounds + Eval<Output = bool> + IntoScalar<bool>,
{
    type BoolOutput = _Tensor<bool>;
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
            None,
        )
    }
}

impl<T> NormalEvalReduce<T> for _Tensor<T>
where
    T: CommonBounds + Eval<Output = bool> + IntoScalar<bool>,
    T::Vec: Eval,
    <T::Vec as Eval>::Output: SimdSelect<T::Vec>,
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

impl<T> _Tensor<T>
where
    T: FloatOutBinary + CommonBounds + IntoScalar<<T as FloatOutBinary>::Output> + Convertor,
    <T as FloatOutBinary>::Output:
        CommonBounds + FloatOutUnary<Output = <T as FloatOutBinary>::Output>,
    <<T as FloatOutBinary>::Output as TypeCommon>::Vec: NormalOut<T::Vec, Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec>
        + FloatOutUnary<Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec>
        + NormalOut<
            <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
            Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
        >,
{
    /// Computes the mean (average) of elements along a specified axis.
    ///
    /// This method calculates the mean (average) of the tensor's elements along a specified axis,
    /// with an option to retain or collapse the reduced dimension. It returns a tensor with
    /// the same type as the original tensor but transformed to its corresponding floating-point
    /// type, making it suitable for operations that require floating-point precision.
    ///
    /// # Arguments
    ///
    /// * `axis` - Specifies the axis along which to compute the mean. This can be
    ///   passed as a type that implements the `Into<Axis>` trait, allowing flexible input for axes.
    /// * `keep_dims` - A boolean flag indicating whether to keep the reduced dimension in
    ///   the output tensor:
    ///   - If `true`, the reduced dimension will be retained with size 1.
    ///   - If `false`, the reduced dimension will be removed from the output.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor with the mean values along
    /// the specified axis.
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn mean<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<_Tensor<FloatBinaryType<T>>, TensorError>
    where
        f64: IntoScalar<<T as FloatOutBinary>::Output>,
        <T as FloatOutBinary>::Output: NormalOut<T, Output = <T as FloatOutBinary>::Output>,
    {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        let reduce_size: FloatBinaryType<T> = (axes
            .iter()
            .fold(1, |acc, &x| acc * (self.shape()[x] as usize))
            as f64)
            .into_scalar();
        let reduce_vec = <FloatBinaryType<T> as TypeCommon>::Vec::splat(reduce_size);
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
            <T as FloatOutBinary>::Output::ZERO,
            keep_dims,
            false,
            None,
        )
    }
    /// Computes the L2 norm (Euclidean norm) along a specified axis.
    ///
    /// This method calculates the L2 norm of the tensor's elements along a specified axis
    /// by summing the squares of the elements and then taking the square root. It provides
    /// an option to retain or collapse the reduced dimension. The L2 norm is often used
    /// in machine learning and optimization problems where minimizing distances is involved.
    ///
    /// # Arguments
    ///
    /// * `axis` - Specifies the axis along which to compute the L2 norm. This can be
    ///   passed as a type that implements the `Into<Axis>` trait, allowing flexible input for axes.
    /// * `keep_dims` - A boolean flag indicating whether to keep the reduced dimension in
    ///   the output tensor:
    ///   - If `true`, the reduced dimension will be retained with size 1.
    ///   - If `false`, the reduced dimension will be removed from the output.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor of type `_Tensor<FloatBinaryType<T>>`
    /// with the L2 norm values along the specified axis.
    #[allow(unused)]
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn reducel2<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<_Tensor<FloatBinaryType<T>>, TensorError>
    where
        T: NormalOut,
        <T as FloatOutBinary>::Output: NormalOut<T, Output = <T as FloatOutBinary>::Output>,
    {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce3(
            self,
            |a, b| a._add(b._square()),
            |a, b| a._add(b._square()),
            |a, b| a._add(b),
            move |a| a._sqrt(),
            |a, b| a._add(b._mul(b)),
            |a, b| a._add(b._mul(b)),
            |a| a._sqrt(),
            &axes,
            <T as FloatOutBinary>::Output::ZERO,
            keep_dims,
            false,
            None,
        )
    }

    /// Computes the L3 norm along a specified axis.
    ///
    /// This method calculates the L3 norm of the tensor's elements along a specified axis
    /// by summing the cubes of the absolute values of the elements and then taking the cube root.
    /// The L3 norm is less commonly used than L1 or L2 norms but can be applied in specific
    /// mathematical or signal processing contexts. It provides an option to retain or collapse
    /// the reduced dimension.
    ///
    /// # Arguments
    ///
    /// * `axis` - Specifies the axis along which to compute the L3 norm. This can be
    ///   passed as a type that implements the `Into<Axis>` trait, allowing flexible input for axes.
    /// * `keep_dims` - A boolean flag indicating whether to keep the reduced dimension in
    ///   the output tensor:
    ///   - If `true`, the reduced dimension will be retained with size 1.
    ///   - If `false`, the reduced dimension will be removed from the output.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor of type `_Tensor<FloatBinaryType<T>>`
    /// with the L3 norm values along the specified axis.
    #[allow(unused)]
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn reducel3<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<_Tensor<FloatBinaryType<T>>, TensorError>
    where
        f64: IntoScalar<<T as FloatOutBinary>::Output>,
        <T as FloatOutBinary>::Output: TypeCommon,
        T::Vec: NormalOut<
            <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
            Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
        >,
    {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        let three: <T as FloatOutBinary>::Output = (3.0).into_scalar();
        let three_vec = <<T as FloatOutBinary>::Output as TypeCommon>::Vec::splat(three);
        let res = reduce3(
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
            move |a| a,
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
            |a| a,
            &axes,
            <T as FloatOutBinary>::Output::ZERO,
            keep_dims,
            false,
            None,
        )?;
        let one_third: <T as FloatOutBinary>::Output = (1.0f64 / 3.0f64).into_scalar();
        let one_third_vec = <<T as FloatOutBinary>::Output as TypeCommon>::Vec::splat(one_third);
        res.par_iter_mut_simd().for_each(
            |x| {
                *x = x._pow(one_third);
            },
            |mut x| {
                x.write_unaligned(x.read_unaligned()._pow(one_third_vec));
            },
        );
        Ok(res)
    }

    /// Computes the logarithm of the sum of exponentials (LogSumExp) along a specified axis.
    ///
    /// This method computes the LogSumExp function, which is a smooth approximation of the maximum.
    /// It calculates `log(sum(exp(x)))` in a numerically stable way, which is useful in various
    /// statistical and machine learning applications, such as when computing softmax functions.
    /// It provides an option to retain or collapse the reduced dimension.
    ///
    /// # Arguments
    ///
    /// * `axis` - Specifies the axis along which to compute the LogSumExp. This can be
    ///   passed as a type that implements the `Into<Axis>` trait, allowing flexible input for axes.
    /// * `keep_dims` - A boolean flag indicating whether to keep the reduced dimension in
    ///   the output tensor:
    ///   - If `true`, the reduced dimension will be retained with size 1.
    ///   - If `false`, the reduced dimension will be removed from the output.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor of type `_Tensor<FloatBinaryType<T>>`
    /// with the LogSumExp values along the specified axis.
    #[allow(unused)]
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn logsumexp<S: Into<Axis>>(
        &self,
        _: S,
        _: bool,
    ) -> std::result::Result<_Tensor<FloatBinaryType<T>>, TensorError>
    where
        T: CommonBounds,
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
