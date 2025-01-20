use std::borrow::Borrow;
use std::cell::RefCell;
use std::rc::Rc;

use crate::ops::cpu::utils::diff::diff_utils::handle_grad;
use crate::tensor::{DiffTensor, Tensor};
use crate::tensor_base::_Tensor;
use crate::BoolVector;
use crate::{ops::cpu::tensor_internal::float_out_unary::FloatBinaryType, Cpu};
use rayon::iter::ParallelIterator;
use tensor_common::axis::axis::Axis;
use tensor_common::error::autograd::AutogradError;
use tensor_common::error::base::TensorError;
use tensor_iterator::iterator_traits::ParStridedIteratorZip;
use tensor_iterator::TensorIterator;
use tensor_traits::{
    CommonBounds, EvalReduce, FloatReduce, NormalEvalReduce, NormalReduce, ShapeManipulate,
    TensorCmp, TensorInfo,
};
use tensor_types::into_vec::IntoVec;
use tensor_types::type_promote::{Cmp, NormalOutUnary, SimdCmp};
use tensor_types::{
    convertion::Convertor,
    dtype::TypeCommon,
    into_scalar::IntoScalar,
    type_promote::{Eval, FloatOutBinary, FloatOutUnary, NormalOut},
    vectors::traits::SimdSelect,
};

impl<T: CommonBounds, const DEVICE: usize> NormalReduce<T> for Tensor<T, Cpu, DEVICE> {
    type Output = Self;

    fn sum<S: Into<Axis>>(
        &self,
        axes: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(self.inner.sum(axes, keep_dims)?.into())
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
        Ok(self
            .inner
            .sum_(axes, keep_dims, init_out, out.borrow())?
            .into())
    }

    // fn sum_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool
    // ) -> anyhow::Result<Self::Output> {
    //     Ok(self.inner.sum_with_init(init_val, axes, keep_dims)?.into())
    // }

    fn prod<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(self.inner.prod(axis, keep_dims)?.into())
    }

    // fn prod_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool
    // ) -> anyhow::Result<Self::Output> {
    //     Ok(self.inner.prod_with_init(init_val, axes, keep_dims)?.into())
    // }

    fn min<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self, TensorError> {
        Ok(self.inner.min(axis, keep_dims)?.into())
    }

    // fn min_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool
    // ) -> anyhow::Result<Self> {
    //     Ok(self.inner.min_with_init(init_val, axes, keep_dims)?.into())
    // }

    fn max<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self, TensorError> {
        Ok(self.inner.max(axis, keep_dims)?.into())
    }

    // fn max_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool
    // ) -> anyhow::Result<Self> {
    //     Ok(self.inner.max_with_init(init_val, axes, keep_dims)?.into())
    // }

    fn reducel1<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(self.inner.reducel1(axis, keep_dims)?.into())
    }

    fn sum_square<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(self.inner.sum_square(axis, keep_dims)?.into())
    }
}

impl<T, const DEVICE: usize> EvalReduce for Tensor<T, Cpu, DEVICE>
where
    T: CommonBounds + Eval<Output = bool> + IntoScalar<bool>,
{
    type BoolOutput = Tensor<bool, Cpu, DEVICE>;
    fn all<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::BoolOutput, TensorError> {
        Ok(self.inner.all(axis, keep_dims)?.into())
    }

    fn any<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::BoolOutput, TensorError> {
        Ok(self.inner.any(axis, keep_dims)?.into())
    }
}

impl<T, const DEVICE: usize> NormalEvalReduce<T> for Tensor<T, Cpu, DEVICE>
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
        Ok(self.inner.nansum(axes, keep_dims)?.into())
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
        Ok(self
            .inner
            .nansum_(axes, keep_dims, init_out, out.borrow())?
            .into())
    }

    // fn nansum_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool
    // ) -> anyhow::Result<Self::Output> {
    //     Ok(self.inner.nansum_with_init(init_val, axes, keep_dims)?.into())
    // }

    fn nanprod<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(self.inner.nanprod(axis, keep_dims)?.into())
    }

    // fn nanprod_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool
    // ) -> anyhow::Result<Self::Output> {
    //     Ok(self.inner.nanprod_with_init(init_val, axes, keep_dims)?.into())
    // }
}

impl<T, const DEVICE: usize> FloatReduce<T> for Tensor<T, Cpu, DEVICE>
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
    f64: IntoScalar<<T as FloatOutBinary>::Output>,
    <T as FloatOutBinary>::Output: NormalOut<T, Output = <T as FloatOutBinary>::Output>,
    T::Vec: NormalOut<
        <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
        Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
    >,
{
    type Output = Tensor<FloatBinaryType<T>, Cpu, DEVICE>;
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
    fn mean<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> Result<Tensor<FloatBinaryType<T>, Cpu, DEVICE>, TensorError>
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
    fn reducel2<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> Result<Tensor<FloatBinaryType<T>, Cpu, DEVICE>, TensorError>
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
    fn reducel3<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> Result<Tensor<FloatBinaryType<T>, Cpu, DEVICE>, TensorError>
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
    fn logsumexp<S: Into<Axis>>(
        &self,
        _: S,
        _: bool,
    ) -> Result<Tensor<FloatBinaryType<T>, Cpu, DEVICE>, TensorError>
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

impl<T, const DEVICE: usize> NormalReduce<T> for DiffTensor<T, Cpu, DEVICE>
where
    T: CommonBounds + FloatOutBinary + Cmp<T, Output = bool> + FloatOutBinary<i64>,
    T::Vec: SimdCmp<T::Vec>,
    <T as FloatOutBinary>::Output: CommonBounds + IntoScalar<T>,
    <T as FloatOutBinary<i64>>::Output: CommonBounds + NormalOut<i64>,
    <<T as FloatOutBinary<i64>>::Output as NormalOut<i64>>::Output:
        CommonBounds + IntoScalar<i64> + IntoScalar<T>,
    <T::Vec as SimdCmp<T::Vec>>::Output: IntoVec<BoolVector>,
{
    type Output = Self;

    fn sum<S: Into<Axis>>(
        &self,
        axes: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes: Axis = axes.into();
        let ret = self.inner.sum(axes.clone(), keep_dims)?;
        let original_shape = self.inner.shape().clone();
        let mut lhs = self.clone();
        Ok(DiffTensor {
            inner: ret,
            grad: self.grad.clone(),
            out_degree: self.out_degree.clone(),
            backward: Rc::new(RefCell::new(move |mut grad: Tensor<T, Cpu, DEVICE>| {
                let grad = if keep_dims {
                    grad.expand(original_shape.clone())?
                } else {
                    for &axis in axes.axes.iter().rev() {
                        grad = grad.unsqueeze(axis)?;
                    }
                    grad.expand(&original_shape)?
                };
                handle_grad(&mut lhs, grad, &Vec::<usize>::new())?;
                Ok(false)
            })),
        })
    }

    fn sum_<S: Into<Axis>, O>(
        &self,
        _: S,
        _: bool,
        _: bool,
        _: O,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        O: Borrow<Self::Output>,
    {
        Err(TensorError::Autograd(AutogradError::InplaceCompError {
            op: "sum_",
            location: std::panic::Location::caller(),
        }))
    }

    // fn sum_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool
    // ) -> anyhow::Result<Self::Output> {
    //     Ok(self.inner.sum_with_init(init_val, axes, keep_dims)?.into())
    // }

    fn prod<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes: Axis = axis.into();
        let ret = self.inner.prod(axes.clone(), keep_dims)?;
        let original_shape = self.inner.shape().clone();
        let mut lhs = self.clone();
        let mut prod = ret.clone();
        Ok(DiffTensor {
            inner: ret,
            grad: self.grad.clone(),
            out_degree: self.out_degree.clone(),
            backward: Rc::new(RefCell::new(move |mut grad: Tensor<T, Cpu, DEVICE>| {
                let (grad, prod) = if keep_dims {
                    (grad.expand(original_shape.clone())?, prod.clone())
                } else {
                    for &axis in axes.axes.iter().rev() {
                        grad = grad.unsqueeze(axis)?;
                        prod = prod.unsqueeze(axis)?;
                    }
                    (grad.expand(&original_shape)?, prod.expand(&original_shape)?)
                };
                let grad = grad
                    .inner
                    .par_iter()
                    .zip(prod.inner.par_iter())
                    .zip(lhs.inner.inner.par_iter())
                    .strided_map(|((g, x), y)| g._mul(x)._div(y).into_scalar())
                    .collect::<_Tensor<T, Cpu, DEVICE>>();
                handle_grad(&mut lhs, grad.into(), &Vec::<usize>::new())?;
                Ok(false)
            })),
        })
    }

    // fn prod_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool
    // ) -> anyhow::Result<Self::Output> {
    //     Ok(self.inner.prod_with_init(init_val, axes, keep_dims)?.into())
    // }

    fn min<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self, TensorError> {
        let axes: Axis = axis.into();
        let ret = self.inner.min(axes.clone(), keep_dims)?;
        let original_shape = self.inner.shape().clone();
        let mut lhs = self.clone();

        Ok(DiffTensor {
            inner: ret.clone(),
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |grad: Tensor<T, Cpu, DEVICE>| {
                let mut grad = if keep_dims {
                    grad
                } else {
                    let mut g = grad;
                    for &axis in axes.axes.iter().rev() {
                        g = g.unsqueeze(axis)?;
                    }
                    g
                };

                let min_broadcasted = ret.expand(&original_shape)?;
                grad = grad.expand(&original_shape)?;

                let mask = lhs
                    .inner
                    .tensor_eq(&min_broadcasted)?
                    .astype::<i64>()
                    .expect("Failed to convert tensor to i64");

                let count = mask.sum(axes.clone(), true)?;
                grad.inner
                    .par_iter_mut()
                    .zip(count.inner.par_iter())
                    .zip(mask.inner.par_iter())
                    .for_each(|((g, c), m)| {
                        *g = g._div(c)._mul(m).into_scalar();
                    });

                handle_grad(&mut lhs, grad, &Vec::<usize>::new())?;
                Ok(false)
            })),
        })
    }

    // fn min_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool
    // ) -> anyhow::Result<Self> {
    //     Ok(self.inner.min_with_init(init_val, axes, keep_dims)?.into())
    // }

    fn max<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self, TensorError> {
        let axes: Axis = axis.into();
        let ret = self.inner.max(axes.clone(), keep_dims)?;
        let original_shape = self.inner.shape().clone();
        let mut lhs = self.clone();

        Ok(DiffTensor {
            inner: ret.clone(),
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |grad: Tensor<T, Cpu, DEVICE>| {
                let mut grad = if keep_dims {
                    grad
                } else {
                    let mut g = grad;
                    for &axis in axes.axes.iter().rev() {
                        g = g.unsqueeze(axis)?;
                    }
                    g
                };

                let max_broadcasted = ret.expand(&original_shape)?;
                grad = grad.expand(&original_shape)?;

                let mask = lhs
                    .inner
                    .tensor_eq(&max_broadcasted)?
                    .astype::<i64>()
                    .expect("Failed to convert tensor to i64");

                let count = mask.sum(axes.clone(), true)?;
                grad.inner
                    .par_iter_mut()
                    .zip(count.inner.par_iter())
                    .zip(mask.inner.par_iter())
                    .for_each(|((g, c), m)| {
                        *g = g._div(c)._mul(m).into_scalar();
                    });

                handle_grad(&mut lhs, grad, &Vec::<usize>::new())?;
                Ok(false)
            })),
        })
    }

    // fn max_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool
    // ) -> anyhow::Result<Self> {
    //     Ok(self.inner.max_with_init(init_val, axes, keep_dims)?.into())
    // }

    fn reducel1<S: Into<Axis>>(
        &self,
        axes: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes: Axis = axes.into();
        let ret = self.inner.reducel1(axes.clone(), keep_dims)?;
        let original_shape = self.inner.shape().clone();
        let mut lhs = self.clone();
        Ok(DiffTensor {
            inner: ret,
            grad: self.grad.clone(),
            out_degree: self.out_degree.clone(),
            backward: Rc::new(RefCell::new(move |mut grad: Tensor<T, Cpu, DEVICE>| {
                let grad = if keep_dims {
                    grad.expand(original_shape.clone())?
                } else {
                    for &axis in axes.axes.iter().rev() {
                        grad = grad.unsqueeze(axis)?;
                    }
                    grad.expand(&original_shape)?
                };
                grad.inner
                    .par_iter_mut()
                    .zip(lhs.inner.inner.par_iter())
                    .for_each(|(g, inp)| {
                        *g = g._mul(inp._signum());
                    });

                handle_grad(&mut lhs, grad, &Vec::<usize>::new())?;
                Ok(false)
            })),
        })
    }

    fn sum_square<S: Into<Axis>>(
        &self,
        axes: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes: Axis = axes.into();
        let ret = self.inner.sum_square(axes.clone(), keep_dims)?;
        let original_shape = self.inner.shape().clone();
        let mut lhs = self.clone();
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |grad: Tensor<T, Cpu, DEVICE>| {
                let mut grad = if keep_dims {
                    grad
                } else {
                    let mut g = grad;
                    for &axis in axes.axes.iter().rev() {
                        g = g.unsqueeze(axis)?;
                    }
                    g
                };
                grad = grad.expand(&original_shape)?;
                grad.inner
                    .par_iter_mut()
                    .zip(lhs.inner.inner.par_iter())
                    .for_each(|(g, inp)| {
                        *g = g._mul(inp._mul(T::TWO));
                    });

                handle_grad(&mut lhs, grad, &Vec::<usize>::new())?;
                Ok(false)
            })),
        })
    }
}

impl<T, const DEVICE: usize> EvalReduce for DiffTensor<T, Cpu, DEVICE>
where
    T: CommonBounds + Eval<Output = bool> + IntoScalar<bool>,
{
    type BoolOutput = DiffTensor<bool, Cpu, DEVICE>;
    fn all<S: Into<Axis>>(
        &self,
        axes: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::BoolOutput, TensorError> {
        let ret = self.inner.all(axes, keep_dims)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| {
                Err(TensorError::Autograd(AutogradError::UnsupportOpError {
                    op: "all",
                    location: std::panic::Location::caller(),
                }))
            })),
        })
    }

    fn any<S: Into<Axis>>(
        &self,
        axes: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::BoolOutput, TensorError> {
        let ret = self.inner.any(axes, keep_dims)?;
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| {
                Err(TensorError::Autograd(AutogradError::UnsupportOpError {
                    op: "any",
                    location: std::panic::Location::caller(),
                }))
            })),
        })
    }
}

impl<T, const DEVICE: usize> NormalEvalReduce<T> for DiffTensor<T, Cpu, DEVICE>
where
    T: CommonBounds + Eval<Output = bool> + IntoScalar<bool>,
    T::Vec: Eval,
    <T::Vec as Eval>::Output: SimdSelect<T::Vec>,
    <T as FloatOutBinary>::Output: IntoScalar<T>,
{
    type Output = Self;

    fn nansum<S: Into<Axis>>(
        &self,
        axes: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes: Axis = axes.into();
        let ret = self.inner.nansum(axes.clone(), keep_dims)?;
        let original_shape = self.inner.shape().clone();
        let mut lhs = self.clone();
        Ok(DiffTensor {
            inner: ret,
            grad: self.grad.clone(),
            out_degree: self.out_degree.clone(),
            backward: Rc::new(RefCell::new(move |grad: Tensor<T, Cpu, DEVICE>| {
                let mut grad = if keep_dims {
                    grad
                } else {
                    let mut g = grad;
                    for &axis in axes.axes.iter().rev() {
                        g = g.unsqueeze(axis)?;
                    }
                    g
                };
                grad = grad.expand(&original_shape)?;
                grad.inner
                    .par_iter_mut()
                    .zip(lhs.inner.inner.par_iter())
                    .for_each(|(g, inp)| {
                        if inp._is_nan() {
                            *g = T::ZERO;
                        }
                    });
                handle_grad(&mut lhs, grad, &Vec::<usize>::new())?;
                Ok(false)
            })),
        })
    }

    fn nansum_<S: Into<Axis>, O>(
        &self,
        _: S,
        _: bool,
        _: bool,
        _: O,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        O: Borrow<Self::Output>,
    {
        Err(TensorError::Autograd(AutogradError::InplaceCompError {
            op: "nansum_",
            location: std::panic::Location::caller(),
        }))
    }

    // fn nansum_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool
    // ) -> anyhow::Result<Self::Output> {
    //     Ok(self.inner.nansum_with_init(init_val, axes, keep_dims)?.into())
    // }

    fn nanprod<S: Into<Axis>>(
        &self,
        axis: S,
        keep_dims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        let axes: Axis = axis.into();
        let ret = self.inner.nanprod(axes.clone(), keep_dims)?;
        let original_shape = self.inner.shape().clone();
        let mut lhs = self.clone();
        let mut prod = ret.clone();

        Ok(DiffTensor {
            inner: ret,
            grad: self.grad.clone(),
            out_degree: self.out_degree.clone(),
            backward: Rc::new(RefCell::new(move |mut grad: Tensor<T, Cpu, DEVICE>| {
                let (grad, prod) = if keep_dims {
                    (grad.expand(original_shape.clone())?, prod.clone())
                } else {
                    for &axis in axes.axes.iter().rev() {
                        grad = grad.unsqueeze(axis)?;
                        prod = prod.unsqueeze(axis)?;
                    }
                    (grad.expand(&original_shape)?, prod.expand(&original_shape)?)
                };

                grad.inner
                    .par_iter_mut()
                    .zip(prod.inner.par_iter())
                    .zip(lhs.inner.inner.par_iter())
                    .for_each(|((g, p), x)| {
                        if x._is_nan() {
                            *g = T::ZERO;
                        } else {
                            *g = g._mul(p)._div(x).into_scalar();
                        }
                    });

                handle_grad(&mut lhs, grad, &Vec::<usize>::new())?;
                Ok(false)
            })),
        })
    }

    // fn nanprod_with_init<S: Into<Axis>>(
    //     &self,
    //     init_val: T,
    //     axes: S,
    //     keep_dims: bool
    // ) -> anyhow::Result<Self::Output> {
    //     Ok(self.inner.nanprod_with_init(init_val, axes, keep_dims)?.into())
    // }
}

impl<T, const DEVICE: usize> FloatReduce<T> for DiffTensor<T, Cpu, DEVICE>
where
    T: FloatOutBinary + CommonBounds + IntoScalar<<T as FloatOutBinary>::Output> + Convertor,
    <T as FloatOutBinary>::Output:
        CommonBounds + FloatOutUnary<Output = <T as FloatOutBinary>::Output> + IntoScalar<T>,
    <<T as FloatOutBinary>::Output as TypeCommon>::Vec: NormalOut<T::Vec, Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec>
        + FloatOutUnary<Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec>
        + NormalOut<
            <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
            Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
        >,
    f64: IntoScalar<<T as FloatOutBinary>::Output>,
    i64: IntoScalar<<T as FloatOutBinary>::Output>,
    <T as FloatOutBinary>::Output: NormalOut<T, Output = <T as FloatOutBinary>::Output>,
    T::Vec: NormalOut<
        <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
        Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
    >,
{
    type Output = DiffTensor<FloatBinaryType<T>, Cpu, DEVICE>;
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
    fn mean<S: Into<Axis>>(
        &self,
        axes: S,
        keep_dims: bool,
    ) -> Result<DiffTensor<FloatBinaryType<T>, Cpu, DEVICE>, TensorError> {
        let axes: Axis = axes.into();
        let ret = self.inner.mean(axes.clone(), keep_dims)?;
        let original_shape = self.inner.shape().clone();
        let mut lhs = self.clone();
        let numel: <T as FloatOutBinary>::Output = axes
            .axes
            .iter()
            .map(|&ax| self.inner.shape()[ax as usize])
            .product::<i64>()
            .into_scalar();
        Ok(DiffTensor {
            inner: ret,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(
                move |mut grad: Tensor<FloatBinaryType<T>, Cpu, DEVICE>| {
                    let grad = if keep_dims {
                        grad.expand(original_shape.clone())?
                    } else {
                        for &axis in axes.axes.iter().rev() {
                            grad = grad.unsqueeze(axis)?;
                        }
                        grad.expand(&original_shape)?
                    };
                    grad.inner.par_iter_mut().for_each(|g| {
                        *g = g._div(numel).into_scalar();
                    });
                    handle_grad(&mut lhs, grad.try_astype::<T>()?, &Vec::<usize>::new())?;
                    Ok(false)
                },
            )),
        })
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
    fn reducel2<S: Into<Axis>>(
        &self,
        axes: S,
        keep_dims: bool,
    ) -> Result<DiffTensor<FloatBinaryType<T>, Cpu, DEVICE>, TensorError> {
        let axes: Axis = axes.into();
        let ret = self.inner.reducel2(axes.clone(), keep_dims)?;
        let original_shape = self.inner.shape().clone();
        let mut lhs = self.clone();

        Ok(DiffTensor {
            inner: ret.clone(),
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(
                move |mut grad: Tensor<FloatBinaryType<T>, Cpu, DEVICE>| {
                    let grad = if keep_dims {
                        grad.expand(original_shape.clone())?
                    } else {
                        for &axis in axes.axes.iter().rev() {
                            grad = grad.unsqueeze(axis)?;
                        }
                        grad.expand(&original_shape)?
                    };

                    grad.inner
                        .par_iter_mut()
                        .zip(lhs.inner.inner.par_iter())
                        .zip(ret.inner.par_iter())
                        .for_each(|((g, x), y)| {
                            *g = g._mul(x)._div(y).into_scalar();
                        });

                    handle_grad(&mut lhs, grad.try_astype::<T>()?, &Vec::<usize>::new())?;
                    Ok(false)
                },
            )),
        })
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
    fn reducel3<S: Into<Axis>>(
        &self,
        axes: S,
        keep_dims: bool,
    ) -> Result<DiffTensor<FloatBinaryType<T>, Cpu, DEVICE>, TensorError> {
        let axes: Axis = axes.into();
        let ret = self.inner.reducel3(axes.clone(), keep_dims)?;
        let original_shape = self.inner.shape().clone();
        let mut lhs = self.clone();
        Ok(DiffTensor {
            inner: ret.clone(),
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(
                move |mut grad: Tensor<FloatBinaryType<T>, Cpu, DEVICE>| {
                    let grad = if keep_dims {
                        grad.expand(original_shape.clone())?
                    } else {
                        for &axis in axes.axes.iter().rev() {
                            grad = grad.unsqueeze(axis)?;
                        }
                        grad.expand(&original_shape)?
                    };

                    grad.inner
                        .par_iter_mut()
                        .zip(lhs.inner.inner.par_iter())
                        .zip(ret.inner.par_iter())
                        .for_each(|((g, x), y)| {
                            let sign: <T as FloatOutBinary>::Output = x._signum().into_scalar();
                            *g = g._mul(sign)._mul(x._square())._div(y._square());
                        });

                    handle_grad(&mut lhs, grad.try_astype::<T>()?, &Vec::<usize>::new())?;
                    Ok(false)
                },
            )),
        })
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
    fn logsumexp<S: Into<Axis>>(
        &self,
        _: S,
        _: bool,
    ) -> Result<DiffTensor<FloatBinaryType<T>, Cpu, DEVICE>, TensorError> {
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
