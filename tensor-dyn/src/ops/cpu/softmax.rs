use std::borrow::BorrowMut;

use crate::{
    backend::Cpu,
    ops::cpu::kernels::softmax_kernel::uncontiguous_softmax_dim_not_include,
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
use tensor_common::{ pointer::Pointer, shape::Shape, shape_utils::mt_intervals, strides::Strides };
use tensor_iterator::{ iterator_traits::ParStridedIteratorZip, TensorIterator };
use tensor_traits::{ CommonBounds, NormalReduce, ShapeManipulate, TensorCreator, TensorInfo };
use tensor_types::{
    convertion::Convertor,
    dtype::TypeCommon,
    into_scalar::IntoScalar,
    type_promote::{ Cmp, FloatOutBinary, FloatOutUnary, NormalOut },
};

use super::{
    kernels::softmax_kernel::{ contiguous_dim_include, uncontiguous_softmax_dim_include },
    reduce_utils::rearrange_array,
};

impl<T> _Tensor<T> {
    /// Applies the softmax function along a specified axis.
    ///
    /// The softmax function normalizes the elements along the specified axis such that they sum to 1.
    /// It is commonly used in machine learning models, particularly for multi-class classification tasks.
    /// The softmax function transforms each element `x_i` in the input tensor into a probability by computing:
    ///
    /// ```text
    /// softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j along the specified axis
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
    pub fn softmax(&self, axis: i64) -> anyhow::Result<_Tensor<<T as FloatOutUnary>::Output>>
        where
            T: CommonBounds + IntoScalar<<T as FloatOutUnary>::Output> + Convertor + FloatOutUnary,
            <T as FloatOutUnary>::Output: CommonBounds +
                NormalOut<T, Output = <T as FloatOutUnary>::Output> +
                FloatOutUnary<Output = <T as FloatOutUnary>::Output>,
            T::Vec: FloatOutUnary<Output = <<T as FloatOutUnary>::Output as TypeCommon>::Vec>,
            <<T as FloatOutUnary>::Output as TypeCommon>::Vec: FloatOutBinary<Output = <<T as FloatOutUnary>::Output as TypeCommon>::Vec>
    {
        let res = if self.is_contiguous() && self.parent().is_none() {
            contiguous_softmax(self, axis, None::<_Tensor<<T as FloatOutUnary>::Output>>)?
        } else {
            uncontiguous_softmax(self, axis, None::<_Tensor<<T as FloatOutUnary>::Output>>)?
        };
        Ok(res)
    }
}

impl<T> Tensor<T> {
    /// Applies the softmax function along a specified axis.
    ///
    /// The softmax function normalizes the elements along the specified axis such that they sum to 1.
    /// It is commonly used in machine learning models, particularly for multi-class classification tasks.
    /// The softmax function transforms each element `x_i` in the input tensor into a probability by computing:
    ///
    /// ```text
    /// softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j along the specified axis
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
    pub fn softmax(&self, axis: i64) -> anyhow::Result<Tensor<<T as FloatOutUnary>::Output>>
        where
            T: CommonBounds + IntoScalar<<T as FloatOutUnary>::Output> + Convertor + FloatOutUnary,
            <T as FloatOutUnary>::Output: CommonBounds +
                NormalOut<T, Output = <T as FloatOutUnary>::Output> +
                FloatOutUnary<Output = <T as FloatOutUnary>::Output>,
            T::Vec: FloatOutUnary<Output = <<T as FloatOutUnary>::Output as TypeCommon>::Vec>,
            <<T as FloatOutUnary>::Output as TypeCommon>::Vec: FloatOutBinary<Output = <<T as FloatOutUnary>::Output as TypeCommon>::Vec>
    {
        Ok(Tensor::from(_Tensor::softmax(self, axis)?.into()))
    }
}

impl<T> _Tensor<T>
    where
        T: CommonBounds + NormalOut<T, Output = T> + Cmp + FloatOutUnary + Convertor,
        <T as FloatOutUnary>::Output: CommonBounds + TypeCommon,
        <T as FloatOutUnary>::Output: NormalOut +
            NormalOut<<T as FloatOutUnary>::Output, Output = <T as FloatOutUnary>::Output> +
            FloatOutUnary,
        <<T as FloatOutUnary>::Output as FloatOutUnary>::Output: IntoScalar<<<T as FloatOutUnary>::Output as FloatOutUnary>::Output> +
            CommonBounds +
            FloatOutUnary,
        <<<T as FloatOutUnary>::Output as FloatOutUnary>::Output as FloatOutUnary>::Output: CommonBounds
{
    /// Applies the log-softmax function along a specified axis.
    ///
    /// The log-softmax function is the logarithm of the softmax function. It is useful in numerical stability,
    /// particularly in scenarios involving large exponents or probabilities, such as in classification tasks.
    /// The log-softmax function computes the logarithm of the softmax of each element `x_i` as:
    ///
    /// ```text
    /// log_softmax(x_i) = log(exp(x_i) / sum(exp(x_j))) for all j along the specified axis
    /// ```
    ///
    /// This function prevents numerical overflow by directly computing the log of the softmax,
    /// rather than computing the softmax first and then taking the log.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to apply the log-softmax function. The elements along this axis
    ///   will be transformed into log-probabilities that correspond to the softmax values.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor with the log-softmax values computed along
    /// the specified axis.
    pub fn logsoftmax(
        &self,
        axis: i64
    )
        -> anyhow::Result<_Tensor<<<T as FloatOutUnary>::Output as FloatOutUnary>::Output>>
        where
            T::Vec: NormalOut<Output = T::Vec>,
            <<T as FloatOutUnary>::Output as TypeCommon>::Vec: NormalOut<Output = <<T as FloatOutUnary>::Output as TypeCommon>::Vec>,
            <T as FloatOutUnary>::Output: FloatOutBinary<Output = <T as FloatOutUnary>::Output> +
                Convertor
    {
        let axis = (if axis < 0 { (self.layout.ndim() as i64) + axis } else { axis }) as usize;
        let max = self.max(axis as i64, true)?;
        let exp = self
            .par_iter()
            .zip(max.par_iter())
            .strided_map(|(x, y)| x._sub(y)._exp())
            .collect::<_Tensor<<T as FloatOutUnary>::Output>>();
        let sum = exp.sum(axis as i64, false)?;
        let ret = exp
            .par_iter()
            .zip(sum.par_iter())
            .strided_map(|(x, y)| x._div(y)._ln())
            .collect::<_Tensor<<<T as FloatOutUnary>::Output as FloatOutUnary>::Output>>();
        Ok(ret)
    }
}

impl<T> Tensor<T>
    where
        T: CommonBounds + NormalOut<T, Output = T> + Cmp + FloatOutUnary + Convertor,
        <T as FloatOutUnary>::Output: CommonBounds + TypeCommon,
        <T as FloatOutUnary>::Output: NormalOut +
            NormalOut<<T as FloatOutUnary>::Output, Output = <T as FloatOutUnary>::Output> +
            FloatOutUnary,
        <<T as FloatOutUnary>::Output as FloatOutUnary>::Output: IntoScalar<<<T as FloatOutUnary>::Output as FloatOutUnary>::Output> +
            CommonBounds +
            FloatOutUnary,
        <<<T as FloatOutUnary>::Output as FloatOutUnary>::Output as FloatOutUnary>::Output: CommonBounds
{
    /// Applies the log-softmax function along a specified axis.
    ///
    /// The log-softmax function is the logarithm of the softmax function. It is useful in numerical stability,
    /// particularly in scenarios involving large exponents or probabilities, such as in classification tasks.
    /// The log-softmax function computes the logarithm of the softmax of each element `x_i` as:
    ///
    /// ```text
    /// log_softmax(x_i) = log(exp(x_i) / sum(exp(x_j))) for all j along the specified axis
    /// ```
    ///
    /// This function prevents numerical overflow by directly computing the log of the softmax,
    /// rather than computing the softmax first and then taking the log.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to apply the log-softmax function. The elements along this axis
    ///   will be transformed into log-probabilities that correspond to the softmax values.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor with the log-softmax values computed along
    /// the specified axis.
    pub fn logsoftmax(
        &self,
        axis: i64
    )
        -> anyhow::Result<Tensor<<<T as FloatOutUnary>::Output as FloatOutUnary>::Output>>
        where
            T::Vec: NormalOut<Output = T::Vec>,
            <<T as FloatOutUnary>::Output as TypeCommon>::Vec: NormalOut<Output = <<T as FloatOutUnary>::Output as TypeCommon>::Vec>,
            <T as FloatOutUnary>::Output: FloatOutBinary<Output = <T as FloatOutUnary>::Output> +
                Convertor
    {
        Ok(Tensor::from(_Tensor::logsoftmax(self, axis)?.into()))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SoftmaxPreprocessor<T, U> {
    pub ptrs: Pointer<T>,
    pub res_ptrs: Pointer<U>,
    pub strides: Strides,
    pub start: usize,
    pub end: usize,
    pub prg: Vec<i64>,
    pub a_prg: Vec<i64>,
    pub shape: Shape,
    pub a_shape: Shape,
}

impl<T, U> SoftmaxPreprocessor<T, U> where T: Clone, U: Clone {
    pub fn new(
        num_threads: usize,
        loop_size: usize,
        ptrs: Pointer<T>,
        res_ptrs: Pointer<U>,
        strides: Strides,
        res_strides: Strides,
        a_shape: Shape,
        transposed_shape: Shape,
        reduce_shape: Shape
    ) -> Vec<SoftmaxPreprocessor<T, U>> {
        let intervals: Vec<(usize, usize)> = mt_intervals(loop_size, num_threads);
        let mut task_amout = 0;
        let mut iterators: Vec<SoftmaxPreprocessor<T, U>> = Vec::with_capacity(num_threads);
        let mut progress_init_a_data = vec![0; reduce_shape.len()];
        for id in 0..num_threads {
            let mut a_data_ptr_cpy = ptrs.clone();
            let mut res_ptrs_cpy = res_ptrs.clone();
            let a_data_ptr_cpy = a_data_ptr_cpy.borrow_mut();
            let res_ptrs_cpy = res_ptrs_cpy.borrow_mut();
            /*traverse the whole result shape and increment the input data ptr based on current thread id*/
            for i in (0..=reduce_shape.len() - 1).rev() {
                a_data_ptr_cpy.offset(progress_init_a_data[i] * strides[i]);
                res_ptrs_cpy.offset(progress_init_a_data[i] * res_strides[i]);
            }
            // calculate the total task amount so far based on current thread id,
            // we are splitting the whole tensor into two axes
            // [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]               thread 0
            // [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]     thread 1
            // [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]     thread 2
            // [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]     thread 3
            // [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]     thread 4
            // where the first axis is where we are splitting the tensor
            let mut tmp1 = task_amout as i64;
            let mut prg =
                vec![0; a_shape.len() - 1]; /* -1 because we want to escape the last axis */

            // since the axis we want to reduce include the most inner axis, we will skip the iteration of the last axis
            // so we use (0..=a_shape.len() - 2).rev()
            for i in (0..=a_shape.len() - 2).rev() {
                prg[i] = tmp1 % transposed_shape[i];
                tmp1 /= transposed_shape[i];
            }

            // increment the res ptr based on the current thread task amount for next thread (next iteration)
            task_amout += intervals[id].1 - intervals[id].0;

            let mut tmp2 = task_amout as i64;
            for j in (0..=reduce_shape.len() - 1).rev() {
                progress_init_a_data[j] = tmp2 % reduce_shape[j];
                tmp2 /= reduce_shape[j];
            }
            iterators.push(SoftmaxPreprocessor {
                ptrs: a_data_ptr_cpy.clone(),
                res_ptrs: res_ptrs_cpy.clone(),
                strides: strides.clone(),
                start: intervals[id].0,
                end: intervals[id].1,
                prg,
                a_prg: vec![],
                shape: reduce_shape.clone(),
                a_shape: a_shape.clone(),
            });
        }
        iterators
    }

    pub fn new2(
        num_threads: usize,
        loop_size: usize,
        ptrs: Pointer<T>,
        res_ptrs: Pointer<U>,
        transposed_strides: Strides,
        res_transposed_strides: Strides,
        transposed_shape: Shape,
        reduce_shape: Shape
    ) -> Vec<SoftmaxPreprocessor<T, U>> {
        let intervals: Vec<(usize, usize)> = mt_intervals(loop_size, num_threads);
        let mut task_amout = 0;
        let mut iterators = Vec::with_capacity(num_threads);
        let mut progress_init_a_data = vec![0; reduce_shape.len()];
        let ndim = reduce_shape.len() as i64;

        // [0, 6, 12, 18, 24, 30] res0    thread 0
        // [1, 7, 13, 19, 25, 31] res0    thread 0
        // [2, 8, 14, 20, 26, 32] res1    thread 1
        // [3, 9, 15, 21, 27, 33] res1    thread 1
        // [4, 10, 16, 22, 28, 34] res2   thread 2
        // [5, 11, 17, 23, 29, 35] res2   thread 2
        for id in 0..num_threads {
            let mut a_data_ptr_cpy = ptrs.clone();
            let mut res_ptr_cpy = res_ptrs.clone();
            let a_data_ptr_cpy = a_data_ptr_cpy.borrow_mut();
            let res_ptr_cpy = res_ptr_cpy.borrow_mut();

            for i in (0..ndim - 1).rev() {
                a_data_ptr_cpy.offset(
                    progress_init_a_data[i as usize] * transposed_strides[i as usize]
                );
                res_ptr_cpy.offset(
                    progress_init_a_data[i as usize] * res_transposed_strides[i as usize]
                );
            }

            let progress_init_a_data_cpy = progress_init_a_data.clone();

            task_amout += intervals[id].1 - intervals[id].0;

            let prg = vec![0; transposed_shape.len()];

            let mut tmp = task_amout as i64;
            for j in (0..ndim - 1).rev() {
                progress_init_a_data[j as usize] = tmp % reduce_shape[j as usize];
                tmp /= reduce_shape[j as usize];
            }

            iterators.push(SoftmaxPreprocessor {
                ptrs: a_data_ptr_cpy.clone(),
                res_ptrs: res_ptr_cpy.clone(),
                strides: transposed_strides.clone(),
                start: intervals[id].0,
                end: intervals[id].1,
                prg,
                a_prg: progress_init_a_data_cpy,
                shape: reduce_shape.clone(),
                a_shape: transposed_shape.clone(),
            });
        }
        iterators
    }
}

pub(crate) fn softmax_prepare<T: CommonBounds, O: CommonBounds>(
    a: &_Tensor<T>,
    axis: usize,
    c: Option<_Tensor<O>>
) -> anyhow::Result<(bool, _Tensor<T>, _Tensor<O>)> {
    let mut keep_fast_dim = true;
    if a.strides()[axis] == 1 {
        keep_fast_dim = false;
    }
    // get permute order, we move to_reduce axes to the end
    let mut transposed_axis = rearrange_array(a.ndim(), &[axis]);

    // sort the transposed axis based on the stride, ordering the axis can increase the cpu cache hitting rate when we do iteration
    transposed_axis[..a.ndim() - 1].sort_by(|x, y| a.strides()[*y].cmp(&a.strides()[*x]));
    transposed_axis[a.ndim() - 1..].sort_by(|x, y| a.strides()[*y].cmp(&a.strides()[*x]));

    let res = if let Some(out) = c {
        // we need a better logic to verify the out is valid.
        // we need to get the real size and compare the real size with the res_shape
        if a.shape().inner() != out.shape().inner() {
            return Err(anyhow::Error::msg("Output array has incorrect shape".to_string()));
        } else if !out.is_contiguous() {
            return Err(anyhow::Error::msg("Output array is not contiguous".to_string()));
        }
        Ok(out)
    } else {
        _Tensor::<O, Cpu>::empty(a.shape())
    };
    Ok((keep_fast_dim, a.permute(transposed_axis)?, res?))
}

#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn contiguous_softmax_template<T, F1, F2, F3, O>(
    a: &_Tensor<T>,
    axis: usize,
    c: Option<_Tensor<O>>,
    full_reduce: F1,
    nkd: F2,
    kd: F3
)
    -> anyhow::Result<_Tensor<O>>
    where
        T: CommonBounds + IntoScalar<O>,
        O: CommonBounds,
        F1: Fn(&mut O),
        F2: Fn(usize, usize, &_Tensor<O>, &_Tensor<T>),
        F3: Fn(usize, usize, usize, &_Tensor<O>, &_Tensor<T>)
{
    let (keep_fast_dim, transposed_tensor, result) = softmax_prepare(a, axis, c)?;

    let a_last_stride = if keep_fast_dim {
        transposed_tensor.strides()[a.ndim() - 2]
    } else {
        transposed_tensor.strides()[a.ndim() - 1]
    };
    assert_eq!(a_last_stride, 1);
    let result_data = result.ptr();
    if a.ndim() == 1 {
        full_reduce(unsafe { result_data.get_ptr().as_mut().unwrap() });
    } else {
        let inner_loop_size = *a.shape().last().unwrap() as usize;
        if !keep_fast_dim {
            let num_threads = if result.size() < rayon::current_num_threads() {
                result.size()
            } else {
                rayon::current_num_threads()
            };
            nkd(num_threads, inner_loop_size, &result, &transposed_tensor);
        } else {
            let a_reduce_size = a.size() / (a.shape()[axis] as usize);
            let outer_loop_size = a_reduce_size / inner_loop_size;
            let inner_loop_size_2 = a.shape()[axis] as usize;
            let num_threads = if outer_loop_size < rayon::current_num_threads() {
                outer_loop_size
            } else {
                rayon::current_num_threads()
            };
            assert!(inner_loop_size > 1);
            kd(num_threads, inner_loop_size, inner_loop_size_2, &result, &transposed_tensor);
        }
    }
    Ok(result)
}

#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn uncontiguous_softmax_template<T, F1, F2, F3, O>(
    a: &_Tensor<T>,
    axis: usize,
    c: Option<_Tensor<O>>,
    full_reduce: F1,
    nkd: F2,
    kd: F3
)
    -> anyhow::Result<_Tensor<O>>
    where
        T: CommonBounds + IntoScalar<O>,
        O: CommonBounds,
        F1: Fn(&mut O),
        F2: Fn(usize, usize, &_Tensor<O>, &_Tensor<T>),
        F3: Fn(usize, usize, usize, &_Tensor<O>, &_Tensor<T>)
{
    let (keep_fast_dim, transposed_tensor, result) = softmax_prepare(a, axis, c)?;

    let a_last_stride = if keep_fast_dim {
        transposed_tensor.strides()[a.ndim() - 2]
    } else {
        transposed_tensor.strides()[a.ndim() - 1]
    };
    assert_eq!(a_last_stride, 1);
    let result_data = result.ptr();
    if a.ndim() == 1 {
        full_reduce(unsafe { result_data.get_ptr().as_mut().unwrap() });
    } else {
        let inner_loop_size = *a.shape().last().unwrap() as usize;
        if !keep_fast_dim {
            let num_threads = if result.size() < rayon::current_num_threads() {
                result.size()
            } else {
                rayon::current_num_threads()
            };
            nkd(num_threads, inner_loop_size, &result, &transposed_tensor);
        } else {
            let a_reduce_size = a.size() / (a.shape()[axis] as usize);
            let outer_loop_size = a_reduce_size / inner_loop_size;
            let inner_loop_size_2 = a.shape()[axis] as usize;
            let num_threads = if outer_loop_size < rayon::current_num_threads() {
                outer_loop_size
            } else {
                rayon::current_num_threads()
            };
            assert!(inner_loop_size > 1);
            kd(num_threads, inner_loop_size, inner_loop_size_2, &result, &transposed_tensor);
        }
    }
    Ok(result)
}

#[cfg_attr(feature = "track_caller", track_caller)]
pub(crate) fn contiguous_softmax<T, O>(
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
                use crate::ops::cpu::kernels::softmax_kernel::softmax_dim_not_include;
                softmax_dim_not_include(
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
pub(crate) fn uncontiguous_softmax<T, O>(
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
                uncontiguous_softmax_dim_include(
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
            let a_last_stride = transposed_tensor.strides()[a.ndim() - 1];
            iterators.into_par_iter().for_each(|mut iterator| {
                let result_ptr_c = iterator.res_ptrs.clone();
                let a_data_ptr = iterator.ptrs.clone();
                let current_size = iterator.end - iterator.start;
                let shape_len = iterator.shape.len() as i64;
                uncontiguous_softmax_dim_not_include(
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
                    a_last_stride as isize
                );
            });
        }
    )
}
