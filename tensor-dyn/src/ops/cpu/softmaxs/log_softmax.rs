use crate::{
    ops::cpu::kernels::logsoftmax::{
        contiguous_dim_include,
        logsoftmax_dim_not_include,
        uncontiguous_logsoftmax_dim_include,
        uncontiguous_logsoftmax_dim_not_include,
    },
    tensor::Tensor,
    tensor_base::_Tensor,
};
use rayon::iter::{
    IndexedParallelIterator,
    IntoParallelIterator,
    IntoParallelRefIterator,
    IntoParallelRefMutIterator,
    ParallelIterator,
};
use tensor_common::shape::Shape;
use tensor_traits::{ CommonBounds, TensorInfo, TensorLike };
use tensor_types::{
    convertion::Convertor,
    dtype::TypeCommon,
    into_scalar::IntoScalar,
    into_vec::IntoVec,
    type_promote::{ FloatOutBinary, FloatOutUnary, NormalOut },
};

use super::softmax_utils::{
    contiguous_softmax_template,
    uncontiguous_softmax_template,
    SoftmaxPreprocessor,
    UCSoftmaxPreprocessor,
};

impl<T> _Tensor<T> {
    /// Applies the logsoftmax function along a specified axis.
    ///
    /// The logsoftmax function normalizes the elements along the specified axis such that they sum to 1.
    /// It is commonly used in machine learning models, particularly for multi-class classification tasks.
    /// The logsoftmax function transforms each element `x_i` in the input tensor into a probability by computing:
    ///
    /// ```text
    /// logsoftmax(x_i) = log(exp(x_i) / sum(exp(x_j))) for all j along the specified axis
    /// ```
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to apply the softmax function. The elements along this axis
    ///   will be transformed into probabilities that sum to 1.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor with the softmax values computed along
    /// the specified axis.
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn log_softmax(&self, axis: i64) -> anyhow::Result<_Tensor<<T as FloatOutUnary>::Output>>
        where
            T: CommonBounds +
                IntoScalar<<T as FloatOutUnary>::Output> +
                Convertor +
                FloatOutUnary +
                IntoScalar<<T as FloatOutUnary>::Output>,
            <T as FloatOutUnary>::Output: CommonBounds +
                NormalOut<T, Output = <T as FloatOutUnary>::Output> +
                FloatOutUnary<Output = <T as FloatOutUnary>::Output>,
            T::Vec: FloatOutUnary<Output = <<T as FloatOutUnary>::Output as TypeCommon>::Vec> +
                IntoVec<<<T as FloatOutUnary>::Output as TypeCommon>::Vec>,
            <<T as FloatOutUnary>::Output as TypeCommon>::Vec: FloatOutBinary<Output = <<T as FloatOutUnary>::Output as TypeCommon>::Vec> +
                FloatOutUnary<Output = <<T as FloatOutUnary>::Output as TypeCommon>::Vec>
    {
        let res = if self.is_contiguous() && self.parent().is_none() {
            contiguous_log_softmax(self, axis, None::<_Tensor<<T as FloatOutUnary>::Output>>)?
        } else {
            uncontiguous_log_softmax(self, axis, None::<_Tensor<<T as FloatOutUnary>::Output>>)?
        };
        Ok(res)
    }
}

impl<T> Tensor<T> {
    /// Applies the logsoftmax function along a specified axis.
    ///
    /// The logsoftmax function normalizes the elements along the specified axis such that they sum to 1.
    /// It is commonly used in machine learning models, particularly for multi-class classification tasks.
    /// The logsoftmax function transforms each element `x_i` in the input tensor into a probability by computing:
    ///
    /// ```text
    /// logsoftmax(x_i) = log(exp(x_i) / sum(exp(x_j))) for all j along the specified axis
    /// ```
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to apply the softmax function. The elements along this axis
    ///   will be transformed into probabilities that sum to 1.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor with the softmax values computed along
    /// the specified axis.
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn log_softmax(&self, axis: i64) -> anyhow::Result<Tensor<<T as FloatOutUnary>::Output>>
        where
            T: CommonBounds +
                IntoScalar<<T as FloatOutUnary>::Output> +
                Convertor +
                FloatOutUnary +
                IntoScalar<<T as FloatOutUnary>::Output>,
            <T as FloatOutUnary>::Output: CommonBounds +
                NormalOut<T, Output = <T as FloatOutUnary>::Output> +
                FloatOutUnary<Output = <T as FloatOutUnary>::Output>,
            T::Vec: FloatOutUnary<Output = <<T as FloatOutUnary>::Output as TypeCommon>::Vec> +
                IntoVec<<<T as FloatOutUnary>::Output as TypeCommon>::Vec>,
            <<T as FloatOutUnary>::Output as TypeCommon>::Vec: FloatOutBinary<Output = <<T as FloatOutUnary>::Output as TypeCommon>::Vec> +
                FloatOutUnary<Output = <<T as FloatOutUnary>::Output as TypeCommon>::Vec>
    {
        Ok(Tensor::from(_Tensor::log_softmax(self.inner.as_ref(), axis)?.into()))
    }
}

#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn contiguous_log_softmax<T, O>(
    a: &_Tensor<T>,
    axis: i64,
    c: Option<_Tensor<O>>
)
    -> anyhow::Result<_Tensor<O>>
    where
        T: CommonBounds + IntoScalar<O> + Convertor + FloatOutUnary<Output = O> + IntoScalar<O>,
        O: CommonBounds + NormalOut<T, Output = O> + FloatOutUnary<Output = O>,
        T::Vec: FloatOutUnary<Output = O::Vec> + IntoVec<O::Vec>,
        O::Vec: FloatOutBinary<Output = O::Vec> + FloatOutUnary<Output = O::Vec>
{
    let axis = (if axis < 0 { axis + (a.ndim() as i64) } else { axis }) as usize;
    contiguous_softmax_template(
        a,
        axis,
        c,
        move |res| {
            let ptr = a.ptr();
            let raw = unsafe { std::slice::from_raw_parts_mut(ptr.ptr, a.size() as usize) };
            let max = raw
                .par_iter()
                .fold(
                    || O::NEG_INF,
                    |acc, &x| acc._max(x)
                )
                .reduce(
                    || O::NEG_INF,
                    |a, b| a._max(b)
                );
            let res_raw = unsafe { std::slice::from_raw_parts_mut(res, a.size() as usize) };
            res_raw
                .par_iter_mut()
                .zip(raw.par_iter())
                .for_each(|(res, &x)| {
                    *res = x._sub(max)._exp();
                });
            let sum = res_raw
                .par_iter()
                .fold(
                    || O::ZERO,
                    |acc, &x| acc._add(x)
                )
                .reduce(
                    || O::ZERO,
                    |a, b| a._add(b)
                );
            res_raw.par_iter_mut().for_each(|x| {
                *x = x._div(sum);
            });
        },
        move |num_threads, inner_loop_size, result, transposed_tensor| {
            let reduce_shape: Shape = transposed_tensor
                .shape()
                [..transposed_tensor.ndim() - 1].to_vec()
                .into();
            let mut axes = (0..result.ndim()).collect::<Vec<_>>();
            axes.remove(axis);
            axes.push(axis);
            let transposed_res_layout = result.layout.permute(axes).unwrap();
            let transposed_res_strides = transposed_res_layout.strides();
            let iterators = SoftmaxPreprocessor::new(
                num_threads,
                reduce_shape.inner().iter().product::<i64>() as usize,
                a.ptr(),
                result.ptr(),
                transposed_tensor.strides().clone(),
                transposed_res_strides.clone(),
                transposed_tensor.shape().sub_one(),
                transposed_tensor.shape().clone(),
                reduce_shape
            );
            iterators.into_par_iter().for_each(|mut iterator| {
                let result_ptr_c = iterator.res_ptrs.clone();
                let a_data_ptr = iterator.ptrs.clone();
                let current_size = iterator.end - iterator.start;
                let shape_len = iterator.a_shape.len() as i64;
                contiguous_dim_include(
                    inner_loop_size as isize,
                    current_size as isize,
                    a_data_ptr,
                    result_ptr_c,
                    &iterator.strides,
                    &transposed_res_strides,
                    &iterator.a_shape,
                    &mut iterator.prg,
                    shape_len
                );
            });
        },
        move |num_threads, inner_loop_size, inner_loop_size_2, result, transposed_tensor| {
            let reduce_shape: Shape = transposed_tensor
                .shape()
                [..transposed_tensor.ndim() - 1].to_vec()
                .into();
            let outer_loop_size =
                (reduce_shape.inner().iter().product::<i64>() as usize) / inner_loop_size;
            let mut axes = (0..result.ndim()).collect::<Vec<_>>();
            axes.remove(axis);
            axes.push(axis);
            let transposed_res_layout = result.layout.permute(axes).unwrap();
            let transposed_res_strides = transposed_res_layout.strides();
            let iterators = SoftmaxPreprocessor::new2(
                num_threads,
                outer_loop_size,
                a.ptr(),
                result.ptr(),
                transposed_tensor.strides().clone(),
                transposed_res_strides.clone(),
                transposed_tensor.shape().sub_one(),
                reduce_shape
            );
            iterators.into_par_iter().for_each(|iterator| {
                let result_ptr_c = iterator.res_ptrs.clone();
                let a_data_ptr = iterator.ptrs.clone();
                let current_size = iterator.end - iterator.start;
                let shape_len = iterator.shape.len() as i64;
                let inp_strides = &iterator.strides;

                let inp_shape = &iterator.a_shape;
                let mut prg1 = iterator.prg.clone();
                let mut prg2 = iterator.a_prg.clone();
                logsoftmax_dim_not_include(
                    inner_loop_size as isize,
                    current_size as isize,
                    inner_loop_size_2 as isize,
                    a_data_ptr,
                    result_ptr_c,
                    &inp_strides,
                    &transposed_res_strides,
                    &inp_shape,
                    &mut prg1,
                    &mut prg2,
                    shape_len
                );
            });
        }
    )
}

#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn uncontiguous_log_softmax<T, O>(
    a: &_Tensor<T>,
    axis: i64,
    c: Option<_Tensor<O>>
)
    -> anyhow::Result<_Tensor<O>>
    where
        T: CommonBounds + IntoScalar<O> + Convertor + FloatOutUnary<Output = O>,
        O: CommonBounds + NormalOut<T, Output = O> + FloatOutUnary<Output = O>,
        T::Vec: FloatOutUnary<Output = O::Vec>,
        O::Vec: FloatOutBinary<Output = O::Vec>
{
    let axis = (if axis < 0 { axis + (a.ndim() as i64) } else { axis }) as usize;
    uncontiguous_softmax_template(
        a,
        axis,
        c,
        move |res| {
            let a = a.contiguous().unwrap();
            let ptr = a.ptr();
            let raw = unsafe { std::slice::from_raw_parts_mut(ptr.ptr, a.size() as usize) };
            let max = raw
                .par_iter()
                .fold(
                    || O::NEG_INF,
                    |acc, &x| acc._max(x)
                )
                .reduce(
                    || O::NEG_INF,
                    |a, b| a._max(b)
                );
            let res_raw = unsafe { std::slice::from_raw_parts_mut(res, a.size() as usize) };
            res_raw
                .par_iter_mut()
                .zip(raw.par_iter())
                .for_each(|(res, &x)| {
                    *res = x._sub(max)._exp();
                });
            let sum = res_raw
                .par_iter()
                .fold(
                    || O::ZERO,
                    |acc, &x| acc._add(x)
                )
                .reduce(
                    || O::ZERO,
                    |a, b| a._add(b)
                );
            res_raw.par_iter_mut().for_each(|x| {
                *x = x._div(sum);
            });
        },
        move |num_threads, inner_loop_size, result, transposed_tensor| {
            let reduce_shape: Shape = transposed_tensor
                .shape()
                [..transposed_tensor.ndim() - 1].to_vec()
                .into();
            let mut axes = (0..result.ndim()).collect::<Vec<_>>();
            axes.remove(axis);
            axes.push(axis);
            let transposed_res_layout = result.layout.permute(axes).unwrap();
            let transposed_res_strides = transposed_res_layout.strides();
            let iterators = SoftmaxPreprocessor::new(
                num_threads,
                reduce_shape.inner().iter().product::<i64>() as usize,
                a.ptr(),
                result.ptr(),
                transposed_tensor.strides().clone(),
                transposed_res_strides.clone(),
                transposed_tensor.shape().sub_one(),
                transposed_tensor.shape().clone(),
                reduce_shape
            );
            let a_last_stride = transposed_tensor.strides()[a.ndim() - 1];
            iterators.into_par_iter().for_each(|mut iterator| {
                let result_ptr_c = iterator.res_ptrs.clone();
                let a_data_ptr = iterator.ptrs.clone();
                let current_size = iterator.end - iterator.start;
                let shape_len = iterator.a_shape.len() as i64;
                uncontiguous_logsoftmax_dim_include(
                    inner_loop_size as isize,
                    current_size as isize,
                    a_data_ptr,
                    result_ptr_c,
                    &iterator.strides,
                    &iterator.a_shape,
                    &mut iterator.prg,
                    &transposed_res_strides,
                    shape_len,
                    a_last_stride as isize
                );
            });
        },
        move |num_threads, inner_loop_size, inner_loop_size_2, result, transposed_tensor| {
            let reduce_shape: Shape = transposed_tensor
                .shape()
                [..transposed_tensor.ndim() - 1].to_vec()
                .into();
            let outer_loop_size =
                (reduce_shape.inner().iter().product::<i64>() as usize) / inner_loop_size;
            let mut axes = (0..result.ndim()).collect::<Vec<_>>();
            axes.remove(axis);
            axes.push(axis);
            let transposed_res_layout = result.layout.permute(axes).unwrap();
            let transposed_res_strides = transposed_res_layout.strides();
            let iterators = UCSoftmaxPreprocessor::new2(
                num_threads,
                outer_loop_size,
                a.ptr(),
                result.ptr(),
                transposed_tensor.strides().clone(),
                transposed_tensor.shape().sub_one(),
                reduce_shape,
                transposed_res_strides.clone()
            );
            let a_last_stride = transposed_tensor.strides()[a.ndim() - 2];
            let res_last_strides = transposed_res_strides[result.ndim() - 2];
            iterators.into_par_iter().for_each(|mut iterator| {
                let result_ptr_c = iterator.res_ptrs.clone();
                let a_data_ptr = iterator.ptrs.clone();
                let current_size = iterator.end - iterator.start;
                let shape_len = iterator.shape.len() as i64;
                uncontiguous_logsoftmax_dim_not_include(
                    inner_loop_size as isize,
                    current_size as isize,
                    inner_loop_size_2 as isize,
                    a_data_ptr,
                    result_ptr_c,
                    &iterator.strides,
                    &iterator.a_shape,
                    &mut iterator.prg,
                    &mut iterator.a_prg,
                    &transposed_res_strides,
                    shape_len,
                    a_last_stride as isize,
                    res_last_strides as isize
                );
            });
        }
    )
}
