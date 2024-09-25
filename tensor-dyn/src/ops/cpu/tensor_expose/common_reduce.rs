use std::borrow::Borrow;

use crate::ops::cpu::tensor_internal::float_out_unary::FloatBinaryType;
use crate::tensor::Tensor;
use tensor_common::axis::Axis;
use tensor_traits::{CommonBounds, EvalReduce, NormalEvalReduce, NormalReduce};
use tensor_types::{
    convertion::Convertor,
    dtype::TypeCommon,
    into_scalar::IntoScalar,
    type_promote::{Eval, FloatOutBinary, FloatOutUnary, NormalOut},
    vectors::traits::SimdSelect,
};

impl<T: CommonBounds> NormalReduce<T> for Tensor<T> {
    type Output = Self;

    fn sum<S: Into<Axis>>(&self, axes: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        Ok(self.inner.sum(axes, keep_dims)?.into())
    }

    fn sum_<S: Into<Axis>, O>(
        &self,
        axes: S,
        keep_dims: bool,
        init_out: bool,
        out: O,
    ) -> anyhow::Result<Self::Output>
    where
        O: Borrow<Self::Output>,
    {
        Ok(self
            .inner
            .sum_(axes, keep_dims, init_out, out.borrow())?
            .into())
    }

    fn sum_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool,
    ) -> anyhow::Result<Self::Output> {
        Ok(self.inner.sum_with_init(init_val, axes, keep_dims)?.into())
    }

    fn prod<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        Ok(self.inner.prod(axis, keep_dims)?.into())
    }

    fn prod_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool,
    ) -> anyhow::Result<Self::Output> {
        Ok(self.inner.prod_with_init(init_val, axes, keep_dims)?.into())
    }

    fn min<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self> {
        Ok(self.inner.min(axis, keep_dims)?.into())
    }

    fn min_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool,
    ) -> anyhow::Result<Self> {
        Ok(self.inner.min_with_init(init_val, axes, keep_dims)?.into())
    }

    fn max<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self> {
        Ok(self.inner.max(axis, keep_dims)?.into())
    }

    fn max_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool,
    ) -> anyhow::Result<Self> {
        Ok(self.inner.max_with_init(init_val, axes, keep_dims)?.into())
    }

    fn reducel1<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        Ok(self.inner.reducel1(axis, keep_dims)?.into())
    }

    fn sum_square<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        Ok(self.inner.sum_square(axis, keep_dims)?.into())
    }
}

impl<T> EvalReduce for Tensor<T>
where
    T: CommonBounds + Eval<Output = bool> + IntoScalar<bool>,
{
    type BoolOutput = Tensor<bool>;
    fn all<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::BoolOutput> {
        Ok(self.inner.all(axis, keep_dims)?.into())
    }

    fn any<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::BoolOutput> {
        Ok(self.inner.any(axis, keep_dims)?.into())
    }
}

impl<T> NormalEvalReduce<T> for Tensor<T>
where
    T: CommonBounds + Eval<Output = bool> + IntoScalar<bool>,
    T::Vec: Eval,
    <T::Vec as Eval>::Output: SimdSelect<T::Vec>,
{
    type Output = Self;

    fn nansum<S: Into<Axis>>(&self, axes: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        Ok(self.inner.nansum(axes, keep_dims)?.into())
    }

    fn nansum_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool,
    ) -> anyhow::Result<Self::Output> {
        Ok(self
            .inner
            .nansum_with_init(init_val, axes, keep_dims)?
            .into())
    }

    fn nanprod<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self::Output> {
        Ok(self.inner.nanprod(axis, keep_dims)?.into())
    }

    fn nanprod_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool,
    ) -> anyhow::Result<Self::Output> {
        Ok(self
            .inner
            .nanprod_with_init(init_val, axes, keep_dims)?
            .into())
    }
}

impl<T> Tensor<T>
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
    ) -> anyhow::Result<Tensor<FloatBinaryType<T>>>
    where
        f64: IntoScalar<<T as FloatOutBinary>::Output>,
        <T as FloatOutBinary>::Output: NormalOut<T, Output = <T as FloatOutBinary>::Output>,
    {
        Ok(self.inner.mean(axis, keep_dims)?.into())
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
    ) -> anyhow::Result<Tensor<FloatBinaryType<T>>>
    where
        T: NormalOut,
        <T as FloatOutBinary>::Output: NormalOut<T, Output = <T as FloatOutBinary>::Output>,
    {
        Ok(self.inner.reducel2(axis, keep_dims)?.into())
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
    ) -> anyhow::Result<Tensor<FloatBinaryType<T>>>
    where
        f64: IntoScalar<<T as FloatOutBinary>::Output>,
        <T as FloatOutBinary>::Output: TypeCommon,
        T::Vec: NormalOut<
            <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
            Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
        >,
    {
        Ok(self.inner.reducel3(axis, keep_dims)?.into())
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
    ) -> anyhow::Result<Tensor<FloatBinaryType<T>>>
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
