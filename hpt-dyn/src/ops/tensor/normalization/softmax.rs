use hpt_common::{error::base::TensorError, shape::shape::Shape};
use hpt_traits::tensor::{CommonBounds, TensorInfo};
use hpt_types::{
    dtype::ToDType,
    into_scalar::Cast,
    type_promote::{FloatOutBinary, FloatOutUnary, NormalOut},
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};

use crate::Tensor;

use super::{
    kernels::{
        contiguous_softmax_dim_include, softmax_dim_not_include, uncontiguous_softmax_dim_include,
        uncontiguous_softmax_dim_not_include,
    },
    normalize_utils::{
        NormalizePreprocessor, UCNormalizePreprocessor, contiguous_normalize_template,
        uncontiguous_normalize_template,
    },
};

#[track_caller]
pub(crate) fn contiguous_softmax<T, O: ToDType>(
    a: &Tensor,
    axis: i64,
    c: Option<Tensor>,
) -> Result<Tensor, TensorError>
where
    T: CommonBounds + Cast<O> + FloatOutUnary<Output = O>,
    O: CommonBounds + NormalOut<T, Output = O> + FloatOutUnary<Output = O>,
    T::Vec: FloatOutUnary<Output = O::Vec>,
    O::Vec: FloatOutBinary<Output = O::Vec>,
{
    let axis = (if axis < 0 {
        axis + (a.ndim() as i64)
    } else {
        axis
    }) as usize;
    contiguous_normalize_template::<T, O, _, _, _>(
        a,
        axis,
        O::to_dtype(),
        c,
        move |res| {
            let ptr = a.ptr::<T>();
            let raw = unsafe { std::slice::from_raw_parts_mut(ptr.ptr, a.size() as usize) };
            let max = raw
                .par_iter()
                .fold(|| O::NEG_INF, |acc, &x| acc._max(x))
                .reduce(|| O::NEG_INF, |a, b| a._max(b));
            let res_raw = unsafe { std::slice::from_raw_parts_mut(res, a.size() as usize) };
            res_raw
                .par_iter_mut()
                .zip(raw.par_iter())
                .for_each(|(res, &x)| {
                    *res = x._sub(max)._exp();
                });
            let sum = res_raw
                .par_iter()
                .fold(|| O::ZERO, |acc, &x| acc._add(x))
                .reduce(|| O::ZERO, |a, b| a._add(b));
            res_raw.par_iter_mut().for_each(|x| {
                *x = x._div(sum);
            });
        },
        move |num_threads, inner_loop_size, result, transposed_tensor| {
            let reduce_shape: Shape = transposed_tensor.shape()[..transposed_tensor.ndim() - 1]
                .to_vec()
                .into();
            let mut axes = (0..result.ndim()).collect::<Vec<_>>();
            axes.remove(axis);
            axes.push(axis);
            let transposed_res_layout = result.layout.permute(axes).unwrap();
            let transposed_res_strides = transposed_res_layout.strides();
            let iterators = NormalizePreprocessor::new(
                num_threads,
                reduce_shape.inner().iter().product::<i64>() as usize,
                a.ptr::<T>(),
                result.ptr::<O>(),
                transposed_tensor.strides().clone(),
                transposed_res_strides.clone(),
                transposed_tensor.shape().sub_one(),
                transposed_tensor.shape().clone(),
                reduce_shape,
            );
            iterators.into_par_iter().for_each(|mut iterator| {
                let result_ptr_c = iterator.res_ptrs.clone();
                let a_data_ptr = iterator.ptrs.clone();
                let current_size = iterator.end - iterator.start;
                let shape_len = iterator.a_shape.len() as i64;
                contiguous_softmax_dim_include(
                    inner_loop_size as isize,
                    current_size as isize,
                    a_data_ptr,
                    result_ptr_c,
                    &iterator.strides,
                    &transposed_res_strides,
                    &iterator.a_shape,
                    &mut iterator.prg,
                    shape_len,
                );
            });
        },
        move |num_threads, inner_loop_size, inner_loop_size_2, result, transposed_tensor| {
            let reduce_shape: Shape = transposed_tensor.shape()[..transposed_tensor.ndim() - 1]
                .to_vec()
                .into();
            let outer_loop_size =
                (reduce_shape.inner().iter().product::<i64>() as usize) / inner_loop_size;
            let mut axes = (0..result.ndim()).collect::<Vec<_>>();
            axes.remove(axis);
            axes.push(axis);
            let transposed_res_layout = result.layout.permute(axes).unwrap();
            let transposed_res_strides = transposed_res_layout.strides();
            let iterators = NormalizePreprocessor::new2(
                num_threads,
                outer_loop_size,
                a.ptr::<T>(),
                result.ptr::<O>(),
                transposed_tensor.strides().clone(),
                transposed_res_strides.clone(),
                transposed_tensor.shape().sub_one(),
                reduce_shape,
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
                    shape_len,
                );
            });
        },
    )
}

#[track_caller]
pub(crate) fn uncontiguous_softmax<T, O: ToDType>(
    a: &Tensor,
    axis: i64,
    c: Option<Tensor>,
) -> Result<Tensor, TensorError>
where
    T: CommonBounds + Cast<O> + FloatOutUnary<Output = O>,
    O: CommonBounds + NormalOut<T, Output = O> + FloatOutUnary<Output = O>,
    T::Vec: FloatOutUnary<Output = O::Vec>,
    O::Vec: FloatOutBinary<Output = O::Vec>,
{
    let axis = (if axis < 0 {
        axis + (a.ndim() as i64)
    } else {
        axis
    }) as usize;
    uncontiguous_normalize_template::<T, O, _, _, _>(
        a,
        axis,
        O::to_dtype(),
        c,
        move |res| {
            let a = a.contiguous().expect("contiguous failed");
            let ptr = a.ptr::<T>();
            let raw = unsafe { std::slice::from_raw_parts_mut(ptr.ptr, a.size() as usize) };
            let max = raw
                .par_iter()
                .fold(|| O::NEG_INF, |acc, &x| acc._max(x))
                .reduce(|| O::NEG_INF, |a, b| a._max(b));
            let res_raw = unsafe { std::slice::from_raw_parts_mut(res, a.size() as usize) };
            res_raw
                .par_iter_mut()
                .zip(raw.par_iter())
                .for_each(|(res, &x)| {
                    *res = x._sub(max)._exp();
                });
            let sum = res_raw
                .par_iter()
                .fold(|| O::ZERO, |acc, &x| acc._add(x))
                .reduce(|| O::ZERO, |a, b| a._add(b));
            res_raw.par_iter_mut().for_each(|x| {
                *x = x._div(sum);
            });
        },
        move |num_threads, inner_loop_size, result, transposed_tensor| {
            let reduce_shape: Shape = transposed_tensor.shape()[..transposed_tensor.ndim() - 1]
                .to_vec()
                .into();
            let mut axes = (0..result.ndim()).collect::<Vec<_>>();
            axes.remove(axis);
            axes.push(axis);
            let transposed_res_layout = result.layout.permute(axes).unwrap();
            let transposed_res_strides = transposed_res_layout.strides();
            let iterators = NormalizePreprocessor::new(
                num_threads,
                reduce_shape.inner().iter().product::<i64>() as usize,
                a.ptr::<T>(),
                result.ptr::<O>(),
                transposed_tensor.strides().clone(),
                transposed_res_strides.clone(),
                transposed_tensor.shape().sub_one(),
                transposed_tensor.shape().clone(),
                reduce_shape,
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
                    a_last_stride as isize,
                );
            });
        },
        move |num_threads, inner_loop_size, inner_loop_size_2, result, transposed_tensor| {
            let reduce_shape: Shape = transposed_tensor.shape()[..transposed_tensor.ndim() - 1]
                .to_vec()
                .into();
            let outer_loop_size =
                (reduce_shape.inner().iter().product::<i64>() as usize) / inner_loop_size;
            let mut axes = (0..result.ndim()).collect::<Vec<_>>();
            axes.remove(axis);
            axes.push(axis);
            let transposed_res_layout = result.layout.permute(axes).unwrap();
            let transposed_res_strides = transposed_res_layout.strides();
            let iterators = UCNormalizePreprocessor::new2(
                num_threads,
                outer_loop_size,
                a.ptr::<T>(),
                result.ptr::<O>(),
                transposed_tensor.strides().clone(),
                transposed_tensor.shape().sub_one(),
                reduce_shape,
                transposed_res_strides.clone(),
            );
            let a_last_stride = transposed_tensor.strides()[a.ndim() - 2];
            let res_last_strides = transposed_res_strides[result.ndim() - 2];
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
                    a_last_stride as isize,
                    res_last_strides as isize,
                );
            });
        },
    )
}

impl Tensor {
    pub fn softmax(&self, axis: i64) -> Result<Tensor, TensorError> {
        if self.is_contiguous() {
            macro_rules! softmax {
                ($dtype:ty) => {{
                    type T = $dtype;
                    type O = <$dtype as FloatOutUnary>::Output;
                    contiguous_softmax::<T, O>(self, axis, None)
                }};
            }
            match self.dtype {
                #[cfg(feature = "i8")]
                hpt_types::dtype::DType::I8 => softmax!(i8),
                #[cfg(feature = "u8")]
                hpt_types::dtype::DType::U8 => softmax!(u8),
                #[cfg(feature = "f32")]
                hpt_types::dtype::DType::F32 => softmax!(f32),
                #[cfg(feature = "f16")]
                hpt_types::dtype::DType::F16 => softmax!(half::f16),
                #[cfg(feature = "bf16")]
                hpt_types::dtype::DType::BF16 => softmax!(half::bf16),
                #[cfg(feature = "bool")]
                hpt_types::dtype::DType::Bool => softmax!(bool),
                #[cfg(feature = "i16")]
                hpt_types::dtype::DType::I16 => softmax!(i16),
                #[cfg(feature = "u16")]
                hpt_types::dtype::DType::U16 => softmax!(u16),
                #[cfg(feature = "i32")]
                hpt_types::dtype::DType::I32 => softmax!(i32),
                #[cfg(feature = "u32")]
                hpt_types::dtype::DType::U32 => softmax!(u32),
                #[cfg(feature = "i64")]
                hpt_types::dtype::DType::I64 => softmax!(i64),
                #[cfg(feature = "u64")]
                hpt_types::dtype::DType::U64 => softmax!(u64),
                #[cfg(feature = "f64")]
                hpt_types::dtype::DType::F64 => softmax!(f64),
                _ => unreachable!("unsupported dtype: {:?}", self.dtype),
            }
        } else {
            macro_rules! softmax {
                ($dtype:ty) => {{
                    type T = $dtype;
                    type O = <$dtype as FloatOutUnary>::Output;
                    uncontiguous_softmax::<T, O>(self, axis, None)
                }};
            }
            match self.dtype {
                #[cfg(feature = "i8")]
                hpt_types::dtype::DType::I8 => softmax!(i8),
                #[cfg(feature = "u8")]
                hpt_types::dtype::DType::U8 => softmax!(u8),
                #[cfg(feature = "f32")]
                hpt_types::dtype::DType::F32 => softmax!(f32),
                #[cfg(feature = "f16")]
                hpt_types::dtype::DType::F16 => softmax!(half::f16),
                #[cfg(feature = "bf16")]
                hpt_types::dtype::DType::BF16 => softmax!(half::bf16),
                #[cfg(feature = "bool")]
                hpt_types::dtype::DType::Bool => softmax!(bool),
                #[cfg(feature = "i16")]
                hpt_types::dtype::DType::I16 => softmax!(i16),
                #[cfg(feature = "u16")]
                hpt_types::dtype::DType::U16 => softmax!(u16),
                #[cfg(feature = "i32")]
                hpt_types::dtype::DType::I32 => softmax!(i32),
                #[cfg(feature = "u32")]
                hpt_types::dtype::DType::U32 => softmax!(u32),
                #[cfg(feature = "i64")]
                hpt_types::dtype::DType::I64 => softmax!(i64),
                #[cfg(feature = "u64")]
                hpt_types::dtype::DType::U64 => softmax!(u64),
                #[cfg(feature = "f64")]
                hpt_types::dtype::DType::F64 => softmax!(f64),
                _ => unreachable!("unsupported dtype: {:?}", self.dtype),
            }
        }
    }
}
