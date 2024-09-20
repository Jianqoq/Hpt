use crate::tensor_base::_Tensor;
use tensor_common::axis::{process_axes, Axis};
use tensor_traits::{CommonBounds, TensorInfo};
use tensor_types::vectors::traits::Init;
use tensor_types::{
    convertion::{Convertor, VecConvertor},
    dtype::TypeCommon,
    into_scalar::IntoScalar,
    type_promote::{BitWiseOut, Eval, FloatOutBinary, FloatOutUnary, NormalOut},
    vectors::traits::SimdSelect,
};

use super::{
    reduce::{reduce, reduce2, reduce3},
    unary::FloatBinaryType,
};

impl<T> _Tensor<T>
where
    T: CommonBounds + NormalOut<T, Output = T> + Convertor,
    T::Vec: NormalOut<T::Vec, Output = T::Vec>,
{
    /// Sums the elements of the tensor along specified axes.
    ///
    /// This method computes the sum of elements along the given axes, effectively reducing the tensor's dimensionality. The `axes` parameter specifies which dimensions to sum over, and `keep_dims` determines whether the original dimensions are retained or squeezed out.
    ///
    /// # Arguments
    ///
    /// * `axes` - The axes along which to sum the elements. This can be a single axis or a list of axes, specified using the `Into<Axis>` trait.
    /// * `keep_dims` - A boolean flag indicating whether to keep the original dimensions:
    ///  If `true`, the summed dimensions are retained with size 1.
    ///  If `false`, the summed dimensions are removed from the tensor shape.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing the summed tensor.
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
            None,
        )
    }

    /// same as `sum` but but with an output tensor
    ///
    /// # Arguments
    ///
    /// * `axes` - The axes along which to sum the elements. This can be a single axis or a list of axes, specified using the `Into<Axis>` trait.
    ///
    /// * `keep_dims` - A boolean flag indicating whether to keep the original dimensions:  If `true`, the summed dimensions are retained with size 1. If `false`, the summed dimensions are removed from the tensor shape.
    ///
    /// * `init_out` - A boolean flag indicating whether to initialize the output tensor with zeros.
    ///
    /// * `out` - The output tensor
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing the summed tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn sum_<S: Into<Axis>>(
        &self,
        axes: S,
        keep_dims: bool,
        init_out: bool,
        out: &_Tensor<T>,
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
            Some(out.clone()),
        )
    }

    /// same as `sum` but but with an init value
    ///
    /// # Arguments
    ///
    /// * `axes` - The axes along which to sum the elements. This can be a single axis or a list of axes, specified using the `Into<Axis>` trait.
    ///
    /// * `keep_dims` - A boolean flag indicating whether to keep the original dimensions:  If `true`, the summed dimensions are retained with size 1. If `false`, the summed dimensions are removed from the tensor shape.
    ///
    /// * `init_val` - The initial value for the sum
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing the summed tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn sum_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool,
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
            None,
        )
    }

    /// Computes the sum of elements along specified axes, treating NaNs as zero.
    ///
    /// This method calculates the sum of the tensor's elements along one or more axes,
    /// ignoring `NaN` values (treating them as zero). It provides an option to retain
    /// or collapse the reduced dimensions. This function is particularly useful for
    /// handling missing data represented as `NaN` values.
    ///
    /// # Arguments
    ///
    /// * `axes` - Specifies the axis or axes along which to compute the sum. This can be
    ///   passed as a type that implements the `Into<Axis>` trait, allowing flexible input for axes.
    /// * `keep_dims` - A boolean flag indicating whether to keep the reduced dimensions in
    ///   the output tensor:
    ///   - If `true`, the reduced dimensions will be retained with size 1.
    ///   - If `false`, the reduced dimensions will be removed from the output.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor with the summed values
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn nansum<S: Into<Axis>>(&self, axes: S, keep_dims: bool) -> anyhow::Result<Self>
    where
        T: Eval<Output = bool>,
        T::Vec: Eval,
        <T::Vec as Eval>::Output: SimdSelect<T::Vec>,
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

    /// same as `nansum` but but with an output tensor
    ///
    /// # Arguments
    ///
    /// * `axes` - The axes along which to sum the elements. This can be a single axis or a list of axes, specified using the `Into<Axis>` trait.
    ///
    /// * `keep_dims` - A boolean flag indicating whether to keep the original dimensions:  If `true`, the summed dimensions are retained with size 1. If `false`, the summed dimensions are removed from the tensor shape.
    ///
    /// * `init_out` - A boolean flag indicating whether to initialize the output tensor with zeros.
    ///
    /// * `out` - The output tensor
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing the summed tensor.
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn nansum_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool,
    ) -> anyhow::Result<Self>
    where
        T: Eval<Output = bool>,
        T::Vec: Eval,
        <T::Vec as Eval>::Output: SimdSelect<T::Vec>,
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
                let mask = b._is_nan();
                mask.select(a, b._add(a))
            },
            &axes,
            init_val,
            keep_dims,
            false,
            None,
        )
    }

    /// Computes the product of elements along a specified axis.
    ///
    /// This method calculates the product of the tensor's elements along a specified axis,
    /// with an option to retain or collapse the reduced dimension. It is useful for
    /// multiplying values along a specific dimension of a multi-dimensional tensor.
    ///
    /// # Arguments
    ///
    /// * `axis` - Specifies the axis along which to compute the product. This can be
    ///   passed as a type that implements the `Into<Axis>` trait, allowing flexible input for axes.
    /// * `keep_dims` - A boolean flag indicating whether to keep the reduced dimension in
    ///   the output tensor:
    ///   - If `true`, the reduced dimension will be retained with size 1.
    ///   - If `false`, the reduced dimension will be removed from the output.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor with the product of the values
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
            None,
        )
    }

    /// same as `prod` but but with an output tensor
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to compute the product. This can be a single axis or a list of axes, specified using the `Into<Axis>` trait.
    ///
    /// * `keep_dims` - A boolean flag indicating whether to keep the original dimensions:  If `true`, the summed dimensions are retained with size 1. If `false`, the summed dimensions are removed from the tensor shape.
    ///
    /// * `init_out` - A boolean flag indicating whether to initialize the output tensor with zeros.
    ///
    /// * `out` - The output tensor
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor with the product of the values
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn prod_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool,
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
            None,
        )
    }

    /// Computes the product of elements along a specified axis, treating NaNs as one.
    ///
    /// This method calculates the product of the tensor's elements along a specified axis,
    /// ignoring `NaN` values (treating them as one). It provides an option to retain or
    /// collapse the reduced dimension. This function is particularly useful for handling
    /// missing data represented as `NaN` values.
    ///
    /// # Arguments
    ///
    /// * `axis` - Specifies the axis along which to compute the product. This can be
    ///   passed as a type that implements the `Into<Axis>` trait, allowing flexible input for axes.
    /// * `keep_dims` - A boolean flag indicating whether to keep the reduced dimension in
    ///   the output tensor:
    ///   - If `true`, the reduced dimension will be retained with size 1.
    ///   - If `false`, the reduced dimension will be removed from the output.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor with the product of the values
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn nanprod<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<Self>
    where
        T: Eval<Output = bool>,
        T::Vec: Eval,
        <T::Vec as Eval>::Output: SimdSelect<T::Vec>,
    {
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

    /// same as `nanprod` but but with an output tensor
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to compute the product. This can be a single axis or a list of axes, specified using the `Into<Axis>` trait.
    ///
    /// * `keep_dims` - A boolean flag indicating whether to keep the original dimensions:  If `true`, the summed dimensions are retained with size 1. If `false`, the summed dimensions are removed from the tensor shape.
    ///
    /// * `init_out` - A boolean flag indicating whether to initialize the output tensor with zeros.
    ///
    /// * `out` - The output tensor
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor with the product of the values
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn nanprod_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool,
    ) -> anyhow::Result<Self>
    where
        T: Eval<Output = bool>,
        T::Vec: Eval,
        <T::Vec as Eval>::Output: SimdSelect<T::Vec>,
    {
        let axes: Vec<usize> = process_axes(axes, self.ndim())?;
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
                let mask = b._is_nan();
                mask.select(a, b._mul(a))
            },
            &axes,
            init_val,
            keep_dims,
            false,
            None,
        )
    }

    /// Computes the minimum value along a specified axis.
    ///
    /// This method calculates the minimum value of the tensor's elements along a specified axis,
    /// with an option to retain or collapse the reduced dimension. It is useful for finding
    /// the smallest value along a specific dimension of a multi-dimensional tensor.
    ///
    /// # Arguments
    ///
    /// * `axis` - Specifies the axis along which to compute the minimum value. This can be
    ///   passed as a type that implements the `Into<Axis>` trait, allowing flexible input for axes.
    /// * `keep_dims` - A boolean flag indicating whether to keep the reduced dimension in
    ///   the output tensor:
    ///   - If `true`, the reduced dimension will be retained with size 1.
    ///   - If `false`, the reduced dimension will be removed from the output.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor with the minimum values along
    /// the specified axis.
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
            None,
        )
    }

    /// same as `min` but but with an output tensor
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to compute the minimum value. This can be a single axis or a list of axes, specified using the `Into<Axis>` trait.
    ///
    /// * `keep_dims` - A boolean flag indicating whether to keep the original dimensions:  If `true`, the summed dimensions are retained with size 1. If `false`, the summed dimensions are removed from the tensor shape.
    ///
    /// * `init_out` - A boolean flag indicating whether to initialize the output tensor with zeros.
    ///
    /// * `out` - The output tensor
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor with the minimum values along
    /// the specified axis.
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn min_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool,
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
            None,
        )
    }

    /// Computes the maximum value along a specified axis.
    ///
    /// This method calculates the maximum value of the tensor's elements along a specified axis,
    /// with an option to retain or collapse the reduced dimension. It is useful for finding
    /// the largest value along a specific dimension of a multi-dimensional tensor.
    ///
    /// # Arguments
    ///
    /// * `axis` - Specifies the axis along which to compute the maximum value. This can be
    ///   passed as a type that implements the `Into<Axis>` trait, allowing flexible input for axes.
    /// * `keep_dims` - A boolean flag indicating whether to keep the reduced dimension in
    ///   the output tensor:
    ///   - If `true`, the reduced dimension will be retained with size 1.
    ///   - If `false`, the reduced dimension will be removed from the output.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor with the maximum values along
    /// the specified axis.
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
            None,
        )
    }

    /// same as `max` but but with an output tensor
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to compute the maximum value. This can be a single axis or a list of axes, specified using the `Into<Axis>` trait.
    ///
    /// * `keep_dims` - A boolean flag indicating whether to keep the original dimensions:  If `true`, the summed dimensions are retained with size 1. If `false`, the summed dimensions are removed from the tensor shape.
    ///
    /// * `init_out` - A boolean flag indicating whether to initialize the output tensor with zeros.
    ///
    /// * `out` - The output tensor
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor with the maximum values along the specified axis.
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn max_with_init<S: Into<Axis>>(
        &self,
        init_val: T,
        axes: S,
        keep_dims: bool,
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
            None,
        )
    }

    /// Checks if all elements along the specified axes are `true`.
    ///
    /// This method evaluates whether all the elements in the tensor along one or more specified axes
    /// are `true` (non-zero for numerical tensors). It can optionally retain or collapse the
    /// reduced dimensions.
    ///
    /// # Arguments
    ///
    /// * `axes` - Specifies the axis or axes along which to evaluate the condition. This can be
    ///   passed as a type that implements the `Into<Axis>` trait, allowing flexible input for axes.
    /// * `keep_dims` - A boolean flag indicating whether to keep the reduced dimensions in
    ///   the output tensor:
    ///   - If `true`, the reduced dimensions will be retained with size 1.
    ///   - If `false`, the reduced dimensions will be removed from the output.
    ///
    /// # Returns
    ///
    /// This function returns a tensor where each element along the reduced axes is either `true`
    /// or `false`, depending on whether all elements in the corresponding slice were `true`.
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn all<S: Into<Axis>>(&self, axes: S, keep_dims: bool) -> anyhow::Result<_Tensor<bool>>
    where
        T: IntoScalar<bool> + Eval<Output = bool> + Convertor,
        T::Vec: Eval + VecConvertor,
        <T::Vec as Eval>::Output: SimdSelect<T::Vec>,
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
            None,
        )
    }
    /// Checks if any element along the specified axis is `true`.
    ///
    /// This method evaluates whether any of the elements in the tensor along a specified axis
    /// are `true` (non-zero for numerical tensors). It can optionally retain or collapse the
    /// reduced dimension.
    ///
    /// # Arguments
    ///
    /// * `axis` - Specifies the axis along which to evaluate the condition. This can be
    ///   passed as a type that implements the `Into<Axis>` trait, allowing flexible input for axes.
    /// * `keep_dims` - A boolean flag indicating whether to keep the reduced dimension in
    ///   the output tensor:
    ///   - If `true`, the reduced dimension will be retained with size 1.
    ///   - If `false`, the reduced dimension will be removed from the output.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor of boolean values, where each element
    /// is `true` if any element along the specified axis is `true`, and `false` otherwise.
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn any<S: Into<Axis>>(&self, axis: S, keep_dims: bool) -> anyhow::Result<_Tensor<bool>>
    where
        T: IntoScalar<bool> + Eval<Output = bool> + Convertor,
        T::Vec: Eval + BitWiseOut + VecConvertor,
        <T::Vec as Eval>::Output: SimdSelect<T::Vec>,
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
            None,
        )
    }
    /// Computes the L1 norm (sum of absolute values) along a specified axis.
    ///
    /// This method calculates the L1 norm of the tensor along a specified axis by summing
    /// the absolute values of the elements. It provides an option to retain or collapse the
    /// reduced dimension. The L1 norm is commonly used in optimization problems and regularization.
    ///
    /// # Arguments
    ///
    /// * `axis` - Specifies the axis along which to compute the L1 norm. This can be
    ///   passed as a type that implements the `Into<Axis>` trait, allowing flexible input for axes.
    /// * `keep_dims` - A boolean flag indicating whether to keep the reduced dimension in
    ///   the output tensor:
    ///   - If `true`, the reduced dimension will be retained with size 1.
    ///   - If `false`, the reduced dimension will be removed from the output.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor with the L1 norm values along
    /// the specified axis
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
            None,
        )
    }

    /// Computes the sum of squares of elements along a specified axis.
    ///
    /// This method calculates the sum of the squared values of the tensor's elements
    /// along a specified axis. It provides an option to retain or collapse the reduced
    /// dimension. This operation is commonly used in various mathematical and machine
    /// learning applications, including loss calculations.
    ///
    /// # Arguments
    ///
    /// * `axis` - Specifies the axis along which to compute the sum of squares. This can be
    ///   passed as a type that implements the `Into<Axis>` trait, allowing flexible input for axes.
    /// * `keep_dims` - A boolean flag indicating whether to keep the reduced dimension in
    ///   the output tensor:
    ///   - If `true`, the reduced dimension will be retained with size 1.
    ///   - If `false`, the reduced dimension will be removed from the output.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor with the sum of squares of the values
    /// along the specified axis.
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
            None,
        )
    }
}

impl<T> _Tensor<T>
where
    T: FloatOutBinary + CommonBounds + IntoScalar<<T as FloatOutBinary>::Output> + Convertor,
    <T as FloatOutBinary>::Output: CommonBounds
        + NormalOut<<T as FloatOutBinary>::Output, Output = <T as FloatOutBinary>::Output>
        + FloatOutUnary<Output = <T as FloatOutBinary>::Output>,
    <<T as FloatOutBinary>::Output as TypeCommon>::Vec: NormalOut<T::Vec, Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec>
        + FloatOutUnary<Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec>
        + NormalOut<
            <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
            Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
        >,
    T::Vec: NormalOut<
            <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
            Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
        > + NormalOut<T::Vec, Output = T::Vec>,
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
    ) -> anyhow::Result<_Tensor<FloatBinaryType<T>>>
    where
        f64: IntoScalar<<T as FloatOutBinary>::Output>,
        <T as FloatOutBinary>::Output: NormalOut<T, Output = <T as FloatOutBinary>::Output>
            + FloatOutBinary<<T as FloatOutBinary>::Output, Output = <T as FloatOutBinary>::Output>,
        <<T as FloatOutBinary>::Output as TypeCommon>::Vec: FloatOutBinary<
            <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
            Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
        >,
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
            move |a| a._div(reduce_size),
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
    ) -> anyhow::Result<_Tensor<FloatBinaryType<T>>>
    where
        T: NormalOut,
        <T as FloatOutBinary>::Output: NormalOut<T, Output = <T as FloatOutBinary>::Output>,
    {
        let axes: Vec<usize> = process_axes(axis, self.ndim())?;
        reduce3(
            self,
            |a, b| {
                let b = b._square();
                a._add(b)
            },
            |a, b| a._add(b._mul(b)),
            move |a| a._sqrt(),
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
    ) -> anyhow::Result<_Tensor<FloatBinaryType<T>>>
    where
        f32: IntoScalar<<T as FloatOutBinary>::Output>,
        T: NormalOut<<T as FloatOutBinary>::Output, Output = <T as FloatOutBinary>::Output>,
        <T as FloatOutBinary>::Output: TypeCommon,
        T::Vec: NormalOut<
            <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
            Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
        >,
        <<T as FloatOutBinary>::Output as TypeCommon>::Vec: NormalOut<
            <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
            Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
        >,
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
                let abs = <T::Vec as NormalOut>::_abs(b);
                let pow = abs._pow(three_vec);
                a._add(pow)
            },
            |a| a,
            &axes,
            <T as FloatOutBinary>::Output::ZERO,
            keep_dims,
            false,
            None,
        )
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
    ) -> anyhow::Result<_Tensor<FloatBinaryType<T>>>
    where
        T: CommonBounds + NormalOut<T, Output = T>,
        T::Vec: NormalOut<T::Vec, Output = T::Vec>,
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
