use super::{
    reduce_template::{contiguous_reduce_template, uncontiguos_reduce_template},
    reduce_utils::{ReductionPreprocessor, UCReductionPreprocessor},
};
use crate::{
    Tensor,
    ops::tensor::reduce::kernels::{
        contiguous_reduce_dim_include, contiguous_reduce_dim_include_simd, fast_reduce_no_simd,
        fast_reduce_simd, reduce_dim_not_include, reduce_dim_not_include_simd,
        uncontiguous_reduce_dim_include, uncontiguous_reduce_dim_not_include,
    },
};
#[cfg(feature = "f16")]
use half::f16;
#[cfg(feature = "bf16")]
use half::bf16;
use hpt_common::axis::axis::process_axes;
use hpt_common::{
    error::base::TensorError,
    shape::shape_utils::{mt_intervals, mt_intervals_simd},
};
use hpt_iterator::TensorIterator;
use hpt_iterator::iterator_traits::StridedIterator;
use hpt_traits::tensor::{CommonBounds, TensorInfo};
use hpt_types::dtype::TypeCommon;
use hpt_types::traits::VecTrait;
use hpt_types::type_promote::FloatOutBinary;
use hpt_types::type_promote::FloatOutUnary;
use hpt_types::type_promote::NormalOutUnary;
use hpt_types::{
    dtype::DType,
    dtype::ToDType,
    into_scalar::Cast,
    traits::SimdSelect,
    type_promote::{Eval, NormalOut},
};
use hpt_types::{into_vec::IntoVec, promote_normal_unary};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

pub(crate) fn contiguous_reduce<
    T: CommonBounds,
    O: CommonBounds,
    F: Fn(O) -> O + Send + Sync + Copy,
    VF: Fn(<O as TypeCommon>::Vec) -> <O as TypeCommon>::Vec + Send + Sync + Copy,
>(
    a: &Tensor,
    axes: &[usize],
    init_val: f64,
    keepdims: bool,
    init_out: bool,
    res_dtype: DType,
    c: Option<Tensor>,
    preop: fn(T) -> O,
    preop_no_cast: fn(O) -> O,
    cumulate: fn(O, O) -> O,
    postop: Option<F>,
    vec_preop: fn(<T as TypeCommon>::Vec) -> <O as TypeCommon>::Vec,
    vec_preop_no_cast: fn(<O as TypeCommon>::Vec) -> <O as TypeCommon>::Vec,
    vec_cumulate: fn(<O as TypeCommon>::Vec, <O as TypeCommon>::Vec) -> <O as TypeCommon>::Vec,
    vec_postop: Option<VF>,
) -> std::result::Result<Tensor, TensorError>
where
    f64: Cast<O>,
    O: ToDType,
{
    assert_eq!(res_dtype, O::to_dtype());
    let res_shape = a.layout.reduce(axes, keepdims)?.shape().clone();
    let res = contiguous_reduce_template(
        &a,
        axes,
        init_val,
        keepdims,
        init_out,
        res_dtype,
        c,
        |res| {
            let res = res as *mut O;
            let ptr = a.ptr();
            let raw =
                unsafe { std::slice::from_raw_parts_mut(ptr.ptr as *mut T, a.size() as usize) };
            let init_val: O = init_val.cast();
            let val = raw
                .par_iter()
                .fold(|| init_val, |acc, &x| cumulate(acc, preop(x)))
                .reduce(|| init_val, |a, b| cumulate(a, b));
            if let Some(postop) = postop {
                unsafe { *res = postop(val) };
            } else {
                unsafe { *res = val };
            }
        },
        move |num_threads, inner_loop_size, inner_loop_size_2, result, transposed_tensor| {
            let iterators = ReductionPreprocessor::new(
                num_threads,
                result.size(),
                inner_loop_size_2,
                a.ptr::<T>(),
                result.ptr::<O>(),
                transposed_tensor.strides().clone(),
                transposed_tensor.shape().sub_one(),
                transposed_tensor.shape().clone(),
                result.shape().clone(),
            );
            iterators.into_iter().for_each(|mut iterator| {
                let result_ptr_c = iterator.res_ptrs.clone();
                let a_data_ptr = iterator.ptrs.clone();
                let current_size = iterator.end - iterator.start;
                let shape_len = iterator.a_shape.len() as i64;
                if T::STR == O::STR {
                    contiguous_reduce_dim_include_simd(
                        init_val.cast(),
                        inner_loop_size as isize,
                        current_size as isize,
                        inner_loop_size_2 as isize,
                        a_data_ptr.cast::<O>(),
                        result_ptr_c.cast::<O>(),
                        &iterator.strides,
                        &iterator.a_shape,
                        &mut iterator.prg,
                        shape_len,
                        preop_no_cast,
                        cumulate,
                        vec_preop_no_cast,
                        vec_cumulate,
                        postop,
                    );
                } else {
                    contiguous_reduce_dim_include(
                        inner_loop_size as isize,
                        current_size as isize,
                        inner_loop_size_2 as isize,
                        a_data_ptr,
                        result_ptr_c,
                        &iterator.strides,
                        &iterator.a_shape,
                        &mut iterator.prg,
                        shape_len,
                        preop,
                        cumulate,
                        postop,
                    );
                }
            });
        },
        move |num_threads, inner_loop_size, result, a| {
            let intervals =
                mt_intervals_simd(inner_loop_size, num_threads, <O as TypeCommon>::Vec::SIZE);
            let mut slices = vec![(0, 0x7FFFFFFFFFFFFFFF, 1); a.ndim()];
            let mut slices_res = vec![(0, 0x7FFFFFFFFFFFFFFF, 1); result.ndim()];
            let mut sliced_tensors = Vec::with_capacity(num_threads);
            let mut sliced_res = Vec::with_capacity(num_threads);
            assert_eq!(inner_loop_size, result.size());
            for (start, end) in intervals.into_iter() {
                if end - start == 0 {
                    continue;
                }
                slices[a.ndim() - 1] = (start as i64, end as i64, 1);
                slices_res[result.ndim() - 1] = (start as i64, end as i64, 1);
                sliced_tensors.push(a.slice(&slices).expect("Slice failed"));
                sliced_res.push(result.slice(&slices_res).expect("Slice failed"));
            }
            sliced_tensors
                .into_par_iter()
                .zip(sliced_res.into_par_iter())
                .for_each(move |(inp, res)| {
                    let inp_ptr = inp.ptr::<T>();
                    let res_ptr = res.ptr::<O>();

                    let inner_loop_size = *res.shape().last().unwrap() as isize;
                    let outer_loop_size = (inp.size() as isize) / inner_loop_size;
                    if O::BYTE_SIZE == T::BYTE_SIZE {
                        fast_reduce_simd(
                            inner_loop_size,
                            outer_loop_size,
                            inp_ptr,
                            res_ptr,
                            inp.strides().inner(),
                            inp.shape().inner(),
                            <O as TypeCommon>::Vec::SIZE as isize,
                            preop,
                            cumulate,
                            postop,
                            vec_preop,
                            vec_cumulate,
                            vec_postop,
                        );
                    } else {
                        fast_reduce_no_simd(
                            inner_loop_size,
                            outer_loop_size,
                            inp_ptr,
                            res_ptr,
                            inp.strides().inner(),
                            inp.shape().inner(),
                            preop,
                            cumulate,
                            postop,
                        );
                    }
                });
        },
        move |num_threads,
              outer_loop_size,
              inner_loop_size,
              inner_loop_size_2,
              result,
              transposed_tensor| {
            let iterators = ReductionPreprocessor::new2(
                num_threads,
                outer_loop_size,
                inner_loop_size,
                a.ptr::<T>(),
                result.ptr::<O>(),
                transposed_tensor.strides().clone(),
                transposed_tensor.shape().sub_one(),
                result.shape().clone(),
            );
            iterators.into_par_iter().for_each(|iterator| {
                let result_ptr_c = iterator.res_ptrs.cast::<O>();
                let a_data_ptr = iterator.ptrs.cast::<T>();
                let current_size = iterator.end - iterator.start;
                let shape_len = iterator.shape.len() as i64;
                let inp_strides = &iterator.strides;
                let inp_shape = &iterator.a_shape;
                let mut prg1 = iterator.prg.clone();
                let mut prg2 = iterator.a_prg.clone();
                if O::BYTE_SIZE == T::BYTE_SIZE {
                    reduce_dim_not_include_simd(
                        inner_loop_size as isize,
                        current_size as isize,
                        inner_loop_size_2 as isize,
                        a_data_ptr,
                        result_ptr_c,
                        &inp_strides,
                        &inp_shape,
                        &mut prg1,
                        &mut prg2,
                        shape_len,
                        preop,
                        cumulate,
                        postop,
                        vec_preop,
                        vec_cumulate,
                        vec_postop,
                    );
                } else {
                    reduce_dim_not_include(
                        inner_loop_size as isize,
                        current_size as isize,
                        inner_loop_size_2 as isize,
                        a_data_ptr,
                        result_ptr_c,
                        &inp_strides,
                        &inp_shape,
                        &mut prg1,
                        &mut prg2,
                        shape_len,
                        preop,
                        cumulate,
                        postop,
                    );
                }
            });
        },
    )?;
    res.reshape(&res_shape)
}

pub(crate) fn uncontiguous_reduce<
    T: CommonBounds,
    O: CommonBounds,
    F: Fn(O) -> O + Send + Sync + Copy,
>(
    a: &Tensor,
    axes: &[usize],
    init_val: f64,
    keepdims: bool,
    init_out: bool,
    res_dtype: DType,
    c: Option<Tensor>,
    preop: fn(T) -> O,
    cumulate: fn(O, O) -> O,
    postop: Option<F>,
) -> std::result::Result<Tensor, TensorError>
where
    f64: Cast<O>,
    O: ToDType,
{
    assert_eq!(res_dtype, O::to_dtype());
    uncontiguos_reduce_template(
        a,
        axes,
        init_val,
        keepdims,
        init_out,
        res_dtype,
        c,
        move |res| {
            let res = res as *mut O;
            let init_val: O = init_val.cast();
            let val = a
                .par_iter()
                .par_strided_fold(init_val, |acc, x| cumulate(acc, preop(x)))
                .reduce(|| init_val, |a, b| cumulate(a, b));
            if let Some(postop) = postop {
                unsafe {
                    *res = postop(val);
                }
            } else {
                unsafe {
                    *res = val;
                }
            }
        },
        move |num_threads, inner_loop_size, inner_loop_size_2, result, transposed_tensor| {
            let a_last_stride = transposed_tensor.strides()[a.ndim() - 1];
            let iterators = UCReductionPreprocessor::new(
                num_threads,
                result.size(),
                inner_loop_size_2,
                a.ptr::<T>(),
                result.ptr::<O>(),
                transposed_tensor.strides().clone(),
                transposed_tensor.shape().sub_one(),
                transposed_tensor.shape().clone(),
                result.shape().clone(),
                result.strides().inner(),
            );
            let res_shape = result.shape().clone();
            iterators.into_par_iter().for_each(|mut iterator| {
                let result_ptr_c = iterator.res_ptrs;
                let a_data_ptr = iterator.ptrs;
                let current_size = iterator.end - iterator.start;
                let res_shape = res_shape.clone();
                let res_strides = result.strides().clone();
                let shape_len = iterator.a_shape.len() as i64;

                uncontiguous_reduce_dim_include(
                    inner_loop_size as isize,
                    current_size as isize,
                    inner_loop_size_2 as isize,
                    a_data_ptr,
                    result_ptr_c,
                    &iterator.strides,
                    &iterator.a_shape,
                    &mut iterator.prg,
                    &mut iterator.res_prg,
                    &res_strides,
                    &res_shape,
                    shape_len,
                    a_last_stride as isize,
                    preop,
                    cumulate,
                    postop,
                );
            });
        },
        move |num_threads, inner_loop_size, ap, result| {
            let intervals = mt_intervals(inner_loop_size, num_threads);
            let mut slices = vec![(0, 0x7FFFFFFFFFFFFFFF, 1); ap.ndim()];
            let mut slices_res = vec![(0, 0x7FFFFFFFFFFFFFFF, 1); result.ndim()];
            let mut sliced_tensors = Vec::with_capacity(num_threads);
            let mut sliced_res = Vec::with_capacity(num_threads);
            assert_eq!(inner_loop_size, result.size());
            for (start, end) in intervals.into_iter() {
                if end - start == 0 {
                    continue;
                }
                slices[ap.ndim() - 1] = (start as i64, end as i64, 1);
                slices_res[result.ndim() - 1] = (start as i64, end as i64, 1);
                sliced_tensors.push(ap.slice(&slices).expect("Slice failed"));
                sliced_res.push(result.slice(&slices_res).expect("Slice failed"));
            }
            sliced_tensors
                .into_par_iter()
                .zip(sliced_res.into_par_iter())
                .for_each(move |(inp, mut res)| {
                    res.iter_mut().zip(inp.iter()).for_each(|(x, y)| {
                        *x = cumulate(*x, preop(y));
                    });
                    if let Some(postop) = postop {
                        res.iter_mut().for_each(|x| {
                            *x = postop(*x);
                        });
                    }
                });
        },
        move |num_threads, inner_loop_size, inner_loop_size_2, result, transposed_tensor| {
            let outer_loop_size = result.size() / inner_loop_size;
            let iterators = UCReductionPreprocessor::new2(
                num_threads,
                outer_loop_size,
                inner_loop_size,
                a.ptr::<T>(),
                result.ptr::<O>(),
                transposed_tensor.strides().clone(),
                transposed_tensor.shape().sub_one(),
                result.shape().clone(),
                result.strides().inner(),
            );
            let res_shape = result.shape().clone();
            iterators.into_par_iter().for_each(|mut iterator| {
                let a_last_stride = transposed_tensor.strides()[a.ndim() - axes.len() - 1];
                let result_ptr_c = iterator.res_ptrs.clone();
                let a_data_ptr = iterator.ptrs.clone();
                let current_size = iterator.end - iterator.start;
                let res_last_strides = *result.strides().inner().last().unwrap();
                let res_strides = result.strides().clone();
                let res_shape = res_shape.clone();
                let shape_len = iterator.shape.len() as i64;
                uncontiguous_reduce_dim_not_include(
                    inner_loop_size as isize,
                    current_size as isize,
                    inner_loop_size_2 as isize,
                    a_data_ptr,
                    result_ptr_c,
                    &iterator.strides,
                    &iterator.a_shape,
                    &mut iterator.prg,
                    &mut iterator.a_prg,
                    &mut iterator.res_prg,
                    &res_strides,
                    &res_shape,
                    shape_len,
                    a_last_stride as isize,
                    res_last_strides as isize,
                    preop,
                    cumulate,
                    postop,
                );
            });
        },
    )
}

pub(crate) fn sum<T: CommonBounds + ToDType>(
    inp: &Tensor,
    axes: &[i64],
    keepdims: bool,
    out: Option<Tensor>,
) -> Result<Tensor, TensorError>
where
    f64: Cast<T>,
{
    let axes = process_axes(axes, inp.ndim())?;
    let res_dtype = promote_normal_unary(inp.dtype);
    if inp.is_contiguous() && inp.parent.is_none() {
        contiguous_reduce(
            inp,
            &axes,
            0.0,
            keepdims,
            true,
            res_dtype,
            out,
            |x: T| x,
            |x| x,
            |x: T, y: T| x._add(y),
            None::<fn(T) -> T>,
            |x| x.into_vec(),
            |x| x,
            |x, y| x._add(y),
            None::<fn(<T as TypeCommon>::Vec) -> <T as TypeCommon>::Vec>,
        )
    } else {
        uncontiguous_reduce(
            inp,
            &axes,
            0.0,
            keepdims,
            true,
            res_dtype,
            out,
            |x: T| x,
            |x: T, y: T| x._add(y),
            None::<fn(T) -> T>,
        )
    }
}

pub(crate) fn nansum<T: CommonBounds + ToDType>(
    inp: &Tensor,
    axes: &[i64],
    keepdims: bool,
    out: Option<Tensor>,
) -> Result<Tensor, TensorError>
where
    T: Eval<Output = bool>,
    T::Vec: Eval,
    <T::Vec as Eval>::Output: SimdSelect<T::Vec>,
    f64: Cast<T>,
{
    let axes = process_axes(axes, inp.ndim())?;
    let res_dtype = promote_normal_unary(inp.dtype);
    if inp.is_contiguous() && inp.parent.is_none() {
        contiguous_reduce(
            inp,
            &axes,
            0.0,
            keepdims,
            true,
            res_dtype,
            out,
            |x: T| if x._is_nan() { T::ZERO } else { x.cast() },
            |x| x,
            |x: T, y: T| x._add(y),
            None::<fn(T) -> T>,
            |x| x._is_nan().select(T::Vec::splat(T::ZERO), x),
            |x| x._is_nan().select(T::Vec::splat(T::ZERO), x),
            |x, y| x._add(y),
            None::<fn(<T as TypeCommon>::Vec) -> <T as TypeCommon>::Vec>,
        )
    } else {
        uncontiguous_reduce(
            inp,
            &axes,
            0.0,
            keepdims,
            true,
            res_dtype,
            out,
            |x: T| if x._is_nan() { T::ZERO } else { x.cast() },
            |x: T, y: T| x._add(y),
            None::<fn(T) -> T>,
        )
    }
}

pub(crate) fn prod<T: CommonBounds + ToDType>(
    inp: &Tensor,
    axes: &[i64],
    keepdims: bool,
    out: Option<Tensor>,
) -> Result<Tensor, TensorError>
where
    f64: Cast<T>,
{
    let axes = process_axes(axes, inp.ndim())?;
    let res_dtype = promote_normal_unary(inp.dtype);
    if inp.is_contiguous() && inp.parent.is_none() {
        contiguous_reduce(
            inp,
            &axes,
            1.0,
            keepdims,
            true,
            res_dtype,
            out,
            |x: T| x,
            |x| x,
            |x: T, y: T| x._mul(y),
            None::<fn(T) -> T>,
            |x| x,
            |x| x,
            |x, y| x._mul(y),
            None::<fn(<T as TypeCommon>::Vec) -> <T as TypeCommon>::Vec>,
        )
    } else {
        uncontiguous_reduce(
            inp,
            &axes,
            1.0,
            keepdims,
            true,
            res_dtype,
            out,
            |x: T| x,
            |x: T, y: T| x._mul(y),
            None::<fn(T) -> T>,
        )
    }
}

pub(crate) fn nanprod<T: CommonBounds + ToDType>(
    inp: &Tensor,
    axes: &[i64],
    keepdims: bool,
    out: Option<Tensor>,
) -> Result<Tensor, TensorError>
where
    T: Eval<Output = bool>,
    T::Vec: Eval,
    <T::Vec as Eval>::Output: SimdSelect<T::Vec>,
    f64: Cast<T>,
{
    let axes = process_axes(axes, inp.ndim())?;
    let res_dtype = promote_normal_unary(inp.dtype);
    if inp.is_contiguous() && inp.parent.is_none() {
        contiguous_reduce(
            inp,
            &axes,
            1.0,
            keepdims,
            true,
            res_dtype,
            out,
            |x: T| if x._is_nan() { T::ONE } else { x },
            |x| x,
            |x: T, y: T| x._mul(y),
            None::<fn(T) -> T>,
            |x| x._is_nan().select(T::Vec::splat(T::ONE), x),
            |x| x._is_nan().select(T::Vec::splat(T::ONE), x),
            |x, y| x._mul(y),
            None::<fn(<T as TypeCommon>::Vec) -> <T as TypeCommon>::Vec>,
        )
    } else {
        uncontiguous_reduce(
            inp,
            &axes,
            1.0,
            keepdims,
            true,
            res_dtype,
            out,
            |x: T| if x._is_nan() { T::ONE } else { x.cast() },
            |x: T, y: T| x._mul(y),
            None::<fn(T) -> T>,
        )
    }
}

pub(crate) fn min<T: CommonBounds + ToDType>(
    inp: &Tensor,
    axes: &[i64],
    keepdims: bool,
    out: Option<Tensor>,
) -> Result<Tensor, TensorError>
where
    f64: Cast<T>,
    T: Cast<f64>,
{
    let axes = process_axes(axes, inp.ndim())?;
    let res_dtype = promote_normal_unary(inp.dtype);
    if inp.is_contiguous() && inp.parent.is_none() {
        contiguous_reduce(
            inp,
            &axes,
            T::MAX.cast(),
            keepdims,
            true,
            res_dtype,
            out,
            |x: T| x,
            |x| x,
            |x: T, y: T| x._min(y),
            None::<fn(T) -> T>,
            |x| x,
            |x| x,
            |x, y| x._min(y),
            None::<fn(<T as TypeCommon>::Vec) -> <T as TypeCommon>::Vec>,
        )
    } else {
        uncontiguous_reduce(
            inp,
            &axes,
            T::MAX.cast(),
            keepdims,
            true,
            res_dtype,
            out,
            |x: T| x,
            |x: T, y: T| x._min(y),
            None::<fn(T) -> T>,
        )
    }
}

pub(crate) fn max<T: CommonBounds + ToDType>(
    inp: &Tensor,
    axes: &[i64],
    keepdims: bool,
    out: Option<Tensor>,
) -> Result<Tensor, TensorError>
where
    f64: Cast<T>,
    T: Cast<f64>,
{
    let axes = process_axes(axes, inp.ndim())?;
    let res_dtype = promote_normal_unary(inp.dtype);
    if inp.is_contiguous() && inp.parent.is_none() {
        contiguous_reduce(
            inp,
            &axes,
            T::MIN.cast(),
            keepdims,
            true,
            res_dtype,
            out,
            |x: T| x,
            |x| x,
            |x: T, y: T| x._max(y),
            None::<fn(T) -> T>,
            |x| x,
            |x| x,
            |x, y| x._max(y),
            None::<fn(<T as TypeCommon>::Vec) -> <T as TypeCommon>::Vec>,
        )
    } else {
        uncontiguous_reduce(
            inp,
            &axes,
            T::MIN.cast(),
            keepdims,
            true,
            res_dtype,
            out,
            |x: T| x,
            |x: T, y: T| x._max(y),
            None::<fn(T) -> T>,
        )
    }
}

pub(crate) fn sum_square<T: CommonBounds + ToDType>(
    inp: &Tensor,
    axes: &[i64],
    keepdims: bool,
    out: Option<Tensor>,
) -> Result<Tensor, TensorError>
where
    f64: Cast<T>,
{
    let axes = process_axes(axes, inp.ndim())?;
    let res_dtype = promote_normal_unary(inp.dtype);
    if inp.is_contiguous() && inp.parent.is_none() {
        contiguous_reduce(
            inp,
            &axes,
            0.0,
            keepdims,
            true,
            res_dtype,
            out,
            |x: T| x._square(),
            |x| x._square(),
            |x: T, y: T| x._add(y),
            None::<fn(T) -> T>,
            |x| x._square(),
            |x| x._square(),
            |x, y| x._add(y),
            None::<fn(<T as TypeCommon>::Vec) -> <T as TypeCommon>::Vec>,
        )
    } else {
        uncontiguous_reduce(
            inp,
            &axes,
            0.0,
            keepdims,
            true,
            res_dtype,
            out,
            |x: T| x._square(),
            |x: T, y: T| x._add(y),
            None::<fn(T) -> T>,
        )
    }
}

pub(crate) fn reducel1<T: CommonBounds + ToDType>(
    inp: &Tensor,
    axes: &[i64],
    keepdims: bool,
    out: Option<Tensor>,
) -> Result<Tensor, TensorError>
where
    f64: Cast<T>,
{
    let axes = process_axes(axes, inp.ndim())?;
    let res_dtype = promote_normal_unary(inp.dtype);
    if inp.is_contiguous() && inp.parent.is_none() {
        contiguous_reduce(
            inp,
            &axes,
            0.0,
            keepdims,
            true,
            res_dtype,
            out,
            |x: T| x._abs(),
            |x| x._abs(),
            |x: T, y: T| x._add(y),
            None::<fn(T) -> T>,
            |x| x._abs(),
            |x| x._abs(),
            |x, y| x._add(y),
            None::<fn(<T as TypeCommon>::Vec) -> <T as TypeCommon>::Vec>,
        )
    } else {
        uncontiguous_reduce(
            inp,
            &axes,
            0.0,
            keepdims,
            true,
            res_dtype,
            out,
            |x: T| x._abs(),
            |x: T, y: T| x._add(y),
            None::<fn(T) -> T>,
        )
    }
}

pub(crate) fn all<T: CommonBounds + ToDType>(
    inp: &Tensor,
    axes: &[i64],
    keepdims: bool,
    out: Option<Tensor>,
) -> Result<Tensor, TensorError>
where
    f64: Cast<T>,
    T: Eval<Output = bool>,
    T::Vec: IntoVec<<bool as TypeCommon>::Vec>,
{
    let axes = process_axes(axes, inp.ndim())?;
    let res_dtype = promote_normal_unary(inp.dtype);
    if inp.is_contiguous() && inp.parent.is_none() {
        contiguous_reduce::<T, bool, _, _>(
            inp,
            &axes,
            1.0,
            keepdims,
            true,
            res_dtype,
            out,
            |x: T| x._is_true(),
            |x| x,
            |x, y| x & y,
            None::<fn(bool) -> bool>,
            |x| x.into_vec(),
            |x| x,
            |x, y| x & y,
            None::<fn(<bool as TypeCommon>::Vec) -> <bool as TypeCommon>::Vec>,
        )
    } else {
        uncontiguous_reduce::<T, bool, _>(
            inp,
            &axes,
            1.0,
            keepdims,
            true,
            res_dtype,
            out,
            |x: T| x._is_true(),
            |x, y| x & y,
            None::<fn(bool) -> bool>,
        )
    }
}

pub(crate) fn any<T: CommonBounds + ToDType>(
    inp: &Tensor,
    axes: &[i64],
    keepdims: bool,
    out: Option<Tensor>,
) -> Result<Tensor, TensorError>
where
    f64: Cast<T>,
    T: Eval<Output = bool>,
    T::Vec: IntoVec<<bool as TypeCommon>::Vec>,
{
    let axes = process_axes(axes, inp.ndim())?;
    let res_dtype = promote_normal_unary(inp.dtype);
    if inp.is_contiguous() && inp.parent.is_none() {
        contiguous_reduce::<T, bool, _, _>(
            inp,
            &axes,
            0.0,
            keepdims,
            true,
            res_dtype,
            out,
            |x: T| x._is_true(),
            |x| x,
            |x, y| x | y,
            None::<fn(bool) -> bool>,
            |x| x.into_vec(),
            |x| x,
            |x, y| x | y,
            None::<fn(<bool as TypeCommon>::Vec) -> <bool as TypeCommon>::Vec>,
        )
    } else {
        uncontiguous_reduce::<T, bool, _>(
            inp,
            &axes,
            0.0,
            keepdims,
            true,
            res_dtype,
            out,
            |x: T| x._is_true(),
            |x, y| x | y,
            None::<fn(bool) -> bool>,
        )
    }
}

pub(crate) fn mean<T: CommonBounds, O: CommonBounds + ToDType>(
    inp: &Tensor,
    axes: &[i64],
    keepdims: bool,
    out: Option<Tensor>,
) -> Result<Tensor, TensorError>
where
    f64: Cast<O>,
    T: Cast<O>,
    <T as TypeCommon>::Vec: IntoVec<<O as TypeCommon>::Vec>,
    O: FloatOutBinary<Output = O>,
    <O as TypeCommon>::Vec: FloatOutBinary<Output = <O as TypeCommon>::Vec>,
{
    let axes = process_axes(axes, inp.ndim())?;
    let res_dtype = promote_normal_unary(inp.dtype);
    let reduce_size: O = (axes
        .iter()
        .fold(1, |acc, &x| acc * (inp.shape()[x] as usize)) as f64)
        .cast();
    let reduce_vec = <O as TypeCommon>::Vec::splat(reduce_size);
    if inp.is_contiguous() && inp.parent.is_none() {
        contiguous_reduce::<T, O, _, _>(
            inp,
            &axes,
            0.0,
            keepdims,
            true,
            res_dtype,
            out,
            |x: T| x.cast(),
            |x| x,
            |x, y| x._add(y),
            Some(|x: O| x._div(reduce_size)),
            |x| x.into_vec(),
            |x| x,
            |x, y| x._add(y),
            Some(|x: <O as TypeCommon>::Vec| x._div(reduce_vec)),
        )
    } else {
        uncontiguous_reduce::<T, O, _>(
            inp,
            &axes,
            0.0,
            keepdims,
            true,
            res_dtype,
            out,
            |x: T| x.cast(),
            |x, y| x._add(y),
            Some(|x: O| x._div(reduce_size)),
        )
    }
}

pub(crate) fn reducel2<T: CommonBounds, O: CommonBounds + ToDType>(
    inp: &Tensor,
    axes: &[i64],
    keepdims: bool,
    out: Option<Tensor>,
) -> Result<Tensor, TensorError>
where
    f64: Cast<O>,
    T: Cast<O>,
    <T as TypeCommon>::Vec: IntoVec<<O as TypeCommon>::Vec>,
    O: FloatOutBinary<Output = O> + FloatOutUnary<Output = O>,
    <O as TypeCommon>::Vec: FloatOutBinary<Output = <O as TypeCommon>::Vec>,
    <O as TypeCommon>::Vec: FloatOutUnary<Output = <O as TypeCommon>::Vec>,
{
    let axes = process_axes(axes, inp.ndim())?;
    let res_dtype = promote_normal_unary(inp.dtype);
    if inp.is_contiguous() && inp.parent.is_none() {
        contiguous_reduce::<T, O, _, _>(
            inp,
            &axes,
            0.0,
            keepdims,
            true,
            res_dtype,
            out,
            |x: T| {
                let c: O = x.cast();
                c._square()
            },
            |x| x._square(),
            |x, y| x._add(y),
            Some(|x: O| x._sqrt()),
            |x| {
                let c: <O as TypeCommon>::Vec = x.into_vec();
                c._square()
            },
            |x| x._square(),
            |x, y| x._add(y),
            Some(|x: <O as TypeCommon>::Vec| x._sqrt()),
        )
    } else {
        uncontiguous_reduce::<T, O, _>(
            inp,
            &axes,
            0.0,
            keepdims,
            true,
            res_dtype,
            out,
            |x: T| {
                let c: O = x.cast();
                c._square()
            },
            |x, y| x._add(y),
            Some(|x: O| x._sqrt()),
        )
    }
}

pub(crate) fn reducel3<T: CommonBounds, O: CommonBounds + ToDType>(
    inp: &Tensor,
    axes: &[i64],
    keepdims: bool,
    out: Option<Tensor>,
) -> Result<Tensor, TensorError>
where
    f64: Cast<O>,
    T: Cast<O>,
    <T as TypeCommon>::Vec: IntoVec<<O as TypeCommon>::Vec>,
    O: FloatOutBinary<Output = O> + FloatOutUnary<Output = O>,
    <O as TypeCommon>::Vec: FloatOutBinary<Output = <O as TypeCommon>::Vec>,
    <O as TypeCommon>::Vec: FloatOutUnary<Output = <O as TypeCommon>::Vec>,
{
    let axes = process_axes(axes, inp.ndim())?;
    let res_dtype = promote_normal_unary(inp.dtype);

    if inp.is_contiguous() && inp.parent.is_none() {
        contiguous_reduce::<T, O, _, _>(
            inp,
            &axes,
            0.0,
            keepdims,
            true,
            res_dtype,
            out,
            |x: T| {
                let c: O = x._abs().cast();
                c._pow((3.0).cast())
            },
            |x| x._abs()._pow((3.0).cast()),
            |x, y| x._add(y),
            Some(|x: O| x._pow((1.0 / 3.0).cast())),
            |x| {
                let c: <O as TypeCommon>::Vec = x.into_vec();
                c._pow(<O as TypeCommon>::Vec::splat((3.0).cast()))
            },
            |x| x._abs()._pow(<O as TypeCommon>::Vec::splat((3.0).cast())),
            |x, y| x._add(y),
            Some(|x: <O as TypeCommon>::Vec| {
                x._pow(<O as TypeCommon>::Vec::splat((1.0 / 3.0).cast()))
            }),
        )
    } else {
        uncontiguous_reduce::<T, O, _>(
            inp,
            &axes,
            0.0,
            keepdims,
            true,
            res_dtype,
            out,
            |x: T| {
                let c: O = x._abs().cast();
                c._pow((3.0).cast())
            },
            |x, y| x._add(y),
            Some(|x: O| x._pow((1.0 / 3.0).cast())),
        )
    }
}

pub(crate) fn logsumexp<T: CommonBounds, O: CommonBounds + ToDType>(
    inp: &Tensor,
    axes: &[i64],
    keepdims: bool,
    out: Option<Tensor>,
) -> Result<Tensor, TensorError>
where
    f64: Cast<O>,
    T: Cast<O>,
    <T as TypeCommon>::Vec: IntoVec<<O as TypeCommon>::Vec>,
    O: FloatOutBinary<Output = O> + FloatOutUnary<Output = O>,
    <O as TypeCommon>::Vec: FloatOutBinary<Output = <O as TypeCommon>::Vec>,
    <O as TypeCommon>::Vec: FloatOutUnary<Output = <O as TypeCommon>::Vec>,
{
    let axes = process_axes(axes, inp.ndim())?;
    let res_dtype = promote_normal_unary(inp.dtype);

    if inp.is_contiguous() && inp.parent.is_none() {
        contiguous_reduce::<T, O, _, _>(
            inp,
            &axes,
            0.0,
            keepdims,
            true,
            res_dtype,
            out,
            |x: T| {
                let c: O = x._abs().cast();
                c._exp()
            },
            |x| x._exp(),
            |x, y| x._add(y),
            Some(|x: O| x._ln()),
            |x| {
                let c: <O as TypeCommon>::Vec = x.into_vec();
                c._exp()
            },
            |x| x._exp(),
            |x, y| x._add(y),
            Some(|x: <O as TypeCommon>::Vec| x._ln()),
        )
    } else {
        uncontiguous_reduce::<T, O, _>(
            inp,
            &axes,
            0.0,
            keepdims,
            true,
            res_dtype,
            out,
            |x: T| {
                let c: O = x.cast();
                c._exp()
            },
            |x, y| x._add(y),
            Some(|x: O| x._ln()),
        )
    }
}

impl Tensor {
    pub fn sum(&self, axes: &[i64], keepdims: bool) -> Result<Tensor, TensorError> {
        match self.dtype {
            #[cfg(feature = "f32")]
            DType::F32 => sum::<f32>(self, axes, keepdims, None),
            #[cfg(feature = "f16")]
            DType::F16 => sum::<f16>(self, axes, keepdims, None),
            #[cfg(feature = "bf16")]
            DType::BF16 => sum::<bf16>(self, axes, keepdims, None),
            #[cfg(feature = "i8")]
            DType::I8 => sum::<i8>(self, axes, keepdims, None),
            #[cfg(feature = "u8")]
            DType::U8 => sum::<u8>(self, axes, keepdims, None),
            #[cfg(feature = "bool")]
            DType::Bool => sum::<bool>(self, axes, keepdims, None),
            #[cfg(feature = "i16")]
            DType::I16 => sum::<i16>(self, axes, keepdims, None),
            #[cfg(feature = "u16")]
            DType::U16 => sum::<u16>(self, axes, keepdims, None),
            #[cfg(feature = "i32")]
            DType::I32 => sum::<i32>(self, axes, keepdims, None),
            #[cfg(feature = "u32")]
            DType::U32 => sum::<u32>(self, axes, keepdims, None),
            #[cfg(feature = "i64")]
            DType::I64 => sum::<i64>(self, axes, keepdims, None),
            #[cfg(feature = "u64")]
            DType::U64 => sum::<u64>(self, axes, keepdims, None),
            #[cfg(feature = "f64")]
            DType::F64 => sum::<f64>(self, axes, keepdims, None),
            _ => unreachable!("unsupported dtype: {:?}", self.dtype),
        }
    }

    pub fn prod(&self, axes: &[i64], keepdims: bool) -> Result<Tensor, TensorError> {
        match self.dtype {
            #[cfg(feature = "f32")]
            DType::F32 => prod::<f32>(self, axes, keepdims, None),
            #[cfg(feature = "f16")]
            DType::F16 => prod::<f16>(self, axes, keepdims, None),
            #[cfg(feature = "bf16")]
            DType::BF16 => prod::<bf16>(self, axes, keepdims, None),
            #[cfg(feature = "i8")]
            DType::I8 => prod::<i8>(self, axes, keepdims, None),
            #[cfg(feature = "u8")]
            DType::U8 => prod::<u8>(self, axes, keepdims, None),
            #[cfg(feature = "bool")]
            DType::Bool => prod::<bool>(self, axes, keepdims, None),
            #[cfg(feature = "i16")]
            DType::I16 => prod::<i16>(self, axes, keepdims, None),
            #[cfg(feature = "u16")]
            DType::U16 => prod::<u16>(self, axes, keepdims, None),
            #[cfg(feature = "i32")]
            DType::I32 => prod::<i32>(self, axes, keepdims, None),
            #[cfg(feature = "u32")]
            DType::U32 => prod::<u32>(self, axes, keepdims, None),
            #[cfg(feature = "i64")]
            DType::I64 => prod::<i64>(self, axes, keepdims, None),
            #[cfg(feature = "u64")]
            DType::U64 => prod::<u64>(self, axes, keepdims, None),
            #[cfg(feature = "f64")]
            DType::F64 => prod::<f64>(self, axes, keepdims, None),
            _ => unreachable!("unsupported dtype: {:?}", self.dtype),
        }
    }

    pub fn nansum(&self, axes: &[i64], keepdims: bool) -> Result<Tensor, TensorError> {
        match self.dtype {
            #[cfg(feature = "f32")]
            DType::F32 => nansum::<f32>(self, axes, keepdims, None),
            #[cfg(feature = "f16")]
            DType::F16 => nansum::<f16>(self, axes, keepdims, None),
            #[cfg(feature = "bf16")]
            DType::BF16 => nansum::<bf16>(self, axes, keepdims, None),
            #[cfg(feature = "i8")]
            DType::I8 => nansum::<i8>(self, axes, keepdims, None),
            #[cfg(feature = "u8")]
            DType::U8 => nansum::<u8>(self, axes, keepdims, None),
            #[cfg(feature = "bool")]
            DType::Bool => nansum::<bool>(self, axes, keepdims, None),
            #[cfg(feature = "i16")]
            DType::I16 => nansum::<i16>(self, axes, keepdims, None),
            #[cfg(feature = "u16")]
            DType::U16 => nansum::<u16>(self, axes, keepdims, None),
            #[cfg(feature = "i32")]
            DType::I32 => nansum::<i32>(self, axes, keepdims, None),
            #[cfg(feature = "u32")]
            DType::U32 => nansum::<u32>(self, axes, keepdims, None),
            #[cfg(feature = "i64")]
            DType::I64 => nansum::<i64>(self, axes, keepdims, None),
            #[cfg(feature = "u64")]
            DType::U64 => nansum::<u64>(self, axes, keepdims, None),
            #[cfg(feature = "f64")]
            DType::F64 => nansum::<f64>(self, axes, keepdims, None),
            _ => unreachable!("unsupported dtype: {:?}", self.dtype),
        }
    }

    pub fn nanprod(&self, axes: &[i64], keepdims: bool) -> Result<Tensor, TensorError> {
        match self.dtype {
            #[cfg(feature = "f32")]
            DType::F32 => nanprod::<f32>(self, axes, keepdims, None),
            #[cfg(feature = "f16")]
            DType::F16 => nanprod::<f16>(self, axes, keepdims, None),
            #[cfg(feature = "bf16")]
            DType::BF16 => nanprod::<bf16>(self, axes, keepdims, None),
            #[cfg(feature = "i8")]
            DType::I8 => nanprod::<i8>(self, axes, keepdims, None),
            #[cfg(feature = "u8")]
            DType::U8 => nanprod::<u8>(self, axes, keepdims, None),
            #[cfg(feature = "bool")]
            DType::Bool => nanprod::<bool>(self, axes, keepdims, None),
            #[cfg(feature = "i16")]
            DType::I16 => nanprod::<i16>(self, axes, keepdims, None),
            #[cfg(feature = "u16")]
            DType::U16 => nanprod::<u16>(self, axes, keepdims, None),
            #[cfg(feature = "i32")]
            DType::I32 => nanprod::<i32>(self, axes, keepdims, None),
            #[cfg(feature = "u32")]
            DType::U32 => nanprod::<u32>(self, axes, keepdims, None),
            #[cfg(feature = "i64")]
            DType::I64 => nanprod::<i64>(self, axes, keepdims, None),
            #[cfg(feature = "u64")]
            DType::U64 => nanprod::<u64>(self, axes, keepdims, None),
            #[cfg(feature = "f64")]
            DType::F64 => nanprod::<f64>(self, axes, keepdims, None),
            _ => unreachable!("unsupported dtype: {:?}", self.dtype),
        }
    }

    pub fn min(&self, axes: &[i64], keepdims: bool) -> Result<Tensor, TensorError> {
        match self.dtype {
            #[cfg(feature = "f32")]
            DType::F32 => min::<f32>(self, axes, keepdims, None),
            #[cfg(feature = "f16")]
            DType::F16 => min::<f16>(self, axes, keepdims, None),
            #[cfg(feature = "bf16")]
            DType::BF16 => min::<bf16>(self, axes, keepdims, None),
            #[cfg(feature = "i8")]
            DType::I8 => min::<i8>(self, axes, keepdims, None),
            #[cfg(feature = "u8")]
            DType::U8 => min::<u8>(self, axes, keepdims, None),
            #[cfg(feature = "bool")]
            DType::Bool => min::<bool>(self, axes, keepdims, None),
            #[cfg(feature = "i16")]
            DType::I16 => min::<i16>(self, axes, keepdims, None),
            #[cfg(feature = "u16")]
            DType::U16 => min::<u16>(self, axes, keepdims, None),
            #[cfg(feature = "i32")]
            DType::I32 => min::<i32>(self, axes, keepdims, None),
            #[cfg(feature = "u32")]
            DType::U32 => min::<u32>(self, axes, keepdims, None),
            #[cfg(feature = "i64")]
            DType::I64 => min::<i64>(self, axes, keepdims, None),
            #[cfg(feature = "u64")]
            DType::U64 => min::<u64>(self, axes, keepdims, None),
            #[cfg(feature = "f64")]
            DType::F64 => min::<f64>(self, axes, keepdims, None),
            _ => unreachable!("unsupported dtype: {:?}", self.dtype),
        }
    }

    pub fn max(&self, axes: &[i64], keepdims: bool) -> Result<Tensor, TensorError> {
        match self.dtype {
            #[cfg(feature = "f32")]
            DType::F32 => max::<f32>(self, axes, keepdims, None),
            #[cfg(feature = "f16")]
            DType::F16 => max::<f16>(self, axes, keepdims, None),
            #[cfg(feature = "bf16")]
            DType::BF16 => max::<bf16>(self, axes, keepdims, None),
            #[cfg(feature = "i8")]
            DType::I8 => max::<i8>(self, axes, keepdims, None),
            #[cfg(feature = "u8")]
            DType::U8 => max::<u8>(self, axes, keepdims, None),
            #[cfg(feature = "bool")]
            DType::Bool => max::<bool>(self, axes, keepdims, None),
            #[cfg(feature = "i16")]
            DType::I16 => max::<i16>(self, axes, keepdims, None),
            #[cfg(feature = "u16")]
            DType::U16 => max::<u16>(self, axes, keepdims, None),
            #[cfg(feature = "i32")]
            DType::I32 => max::<i32>(self, axes, keepdims, None),
            #[cfg(feature = "u32")]
            DType::U32 => max::<u32>(self, axes, keepdims, None),
            #[cfg(feature = "i64")]
            DType::I64 => max::<i64>(self, axes, keepdims, None),
            #[cfg(feature = "u64")]
            DType::U64 => max::<u64>(self, axes, keepdims, None),
            #[cfg(feature = "f64")]
            DType::F64 => max::<f64>(self, axes, keepdims, None),
            _ => unreachable!("unsupported dtype: {:?}", self.dtype),
        }
    }

    pub fn sum_square(&self, axes: &[i64], keepdims: bool) -> Result<Tensor, TensorError> {
        match self.dtype {
            #[cfg(feature = "f32")]
            DType::F32 => sum_square::<f32>(self, axes, keepdims, None),
            #[cfg(feature = "f16")]
            DType::F16 => sum_square::<f16>(self, axes, keepdims, None),
            #[cfg(feature = "bf16")]
            DType::BF16 => sum_square::<bf16>(self, axes, keepdims, None),
            #[cfg(feature = "i8")]
            DType::I8 => sum_square::<i8>(self, axes, keepdims, None),
            #[cfg(feature = "u8")]
            DType::U8 => sum_square::<u8>(self, axes, keepdims, None),
            #[cfg(feature = "bool")]
            DType::Bool => sum_square::<bool>(self, axes, keepdims, None),
            #[cfg(feature = "i16")]
            DType::I16 => sum_square::<i16>(self, axes, keepdims, None),
            #[cfg(feature = "u16")]
            DType::U16 => sum_square::<u16>(self, axes, keepdims, None),
            #[cfg(feature = "i32")]
            DType::I32 => sum_square::<i32>(self, axes, keepdims, None),
            #[cfg(feature = "u32")]
            DType::U32 => sum_square::<u32>(self, axes, keepdims, None),
            #[cfg(feature = "i64")]
            DType::I64 => sum_square::<i64>(self, axes, keepdims, None),
            #[cfg(feature = "u64")]
            DType::U64 => sum_square::<u64>(self, axes, keepdims, None),
            #[cfg(feature = "f64")]
            DType::F64 => sum_square::<f64>(self, axes, keepdims, None),
            _ => unreachable!("unsupported dtype: {:?}", self.dtype),
        }
    }

    pub fn reducel1(&self, axes: &[i64], keepdims: bool) -> Result<Tensor, TensorError> {
        match self.dtype {
            #[cfg(feature = "f32")]
            DType::F32 => reducel1::<f32>(self, axes, keepdims, None),
            #[cfg(feature = "f16")]
            DType::F16 => reducel1::<f16>(self, axes, keepdims, None),
            #[cfg(feature = "bf16")]
            DType::BF16 => reducel1::<bf16>(self, axes, keepdims, None),
            #[cfg(feature = "i8")]
            DType::I8 => reducel1::<i8>(self, axes, keepdims, None),
            #[cfg(feature = "u8")]
            DType::U8 => reducel1::<u8>(self, axes, keepdims, None),
            #[cfg(feature = "bool")]
            DType::Bool => reducel1::<bool>(self, axes, keepdims, None),
            #[cfg(feature = "i16")]
            DType::I16 => reducel1::<i16>(self, axes, keepdims, None),
            #[cfg(feature = "u16")]
            DType::U16 => reducel1::<u16>(self, axes, keepdims, None),
            #[cfg(feature = "i32")]
            DType::I32 => reducel1::<i32>(self, axes, keepdims, None),
            #[cfg(feature = "u32")]
            DType::U32 => reducel1::<u32>(self, axes, keepdims, None),
            #[cfg(feature = "i64")]
            DType::I64 => reducel1::<i64>(self, axes, keepdims, None),
            #[cfg(feature = "u64")]
            DType::U64 => reducel1::<u64>(self, axes, keepdims, None),
            #[cfg(feature = "f64")]
            DType::F64 => reducel1::<f64>(self, axes, keepdims, None),
            _ => unreachable!("unsupported dtype: {:?}", self.dtype),
        }
    }
    pub fn all(&self, axes: &[i64], keepdims: bool) -> Result<Tensor, TensorError> {
        match self.dtype {
            #[cfg(feature = "f32")]
            DType::F32 => all::<f32>(self, axes, keepdims, None),
            #[cfg(feature = "f16")]
            DType::F16 => all::<f16>(self, axes, keepdims, None),
            #[cfg(feature = "bf16")]
            DType::BF16 => all::<bf16>(self, axes, keepdims, None),
            #[cfg(feature = "i8")]
            DType::I8 => all::<i8>(self, axes, keepdims, None),
            #[cfg(feature = "u8")]
            DType::U8 => all::<u8>(self, axes, keepdims, None),
            #[cfg(feature = "bool")]
            DType::Bool => all::<bool>(self, axes, keepdims, None),
            #[cfg(feature = "i16")]
            DType::I16 => all::<i16>(self, axes, keepdims, None),
            #[cfg(feature = "u16")]
            DType::U16 => all::<u16>(self, axes, keepdims, None),
            #[cfg(feature = "i32")]
            DType::I32 => all::<i32>(self, axes, keepdims, None),
            #[cfg(feature = "u32")]
            DType::U32 => all::<u32>(self, axes, keepdims, None),
            #[cfg(feature = "i64")]
            DType::I64 => all::<i64>(self, axes, keepdims, None),
            #[cfg(feature = "u64")]
            DType::U64 => all::<u64>(self, axes, keepdims, None),
            #[cfg(feature = "f64")]
            DType::F64 => all::<f64>(self, axes, keepdims, None),
            _ => unreachable!("unsupported dtype: {:?}", self.dtype),
        }
    }

    pub fn any(&self, axes: &[i64], keepdims: bool) -> Result<Tensor, TensorError> {
        match self.dtype {
            #[cfg(feature = "f32")]
            DType::F32 => any::<f32>(self, axes, keepdims, None),
            #[cfg(feature = "f16")]
            DType::F16 => any::<f16>(self, axes, keepdims, None),
            #[cfg(feature = "bf16")]
            DType::BF16 => any::<bf16>(self, axes, keepdims, None),
            #[cfg(feature = "i8")]
            DType::I8 => any::<i8>(self, axes, keepdims, None),
            #[cfg(feature = "u8")]
            DType::U8 => any::<u8>(self, axes, keepdims, None),
            #[cfg(feature = "bool")]
            DType::Bool => any::<bool>(self, axes, keepdims, None),
            #[cfg(feature = "i16")]
            DType::I16 => any::<i16>(self, axes, keepdims, None),
            #[cfg(feature = "u16")]
            DType::U16 => any::<u16>(self, axes, keepdims, None),
            #[cfg(feature = "i32")]
            DType::I32 => any::<i32>(self, axes, keepdims, None),
            #[cfg(feature = "u32")]
            DType::U32 => any::<u32>(self, axes, keepdims, None),
            #[cfg(feature = "i64")]
            DType::I64 => any::<i64>(self, axes, keepdims, None),
            #[cfg(feature = "u64")]
            DType::U64 => any::<u64>(self, axes, keepdims, None),
            #[cfg(feature = "f64")]
            DType::F64 => any::<f64>(self, axes, keepdims, None),
            _ => unreachable!("unsupported dtype: {:?}", self.dtype),
        }
    }
    pub fn logsumexp(&self, axes: &[i64], keepdims: bool) -> Result<Tensor, TensorError> {
        match self.dtype {
            #[cfg(feature = "f32")]
            DType::F32 => logsumexp::<f32, f32>(self, axes, keepdims, None),
            #[cfg(feature = "f16")]
            DType::F16 => logsumexp::<f16, f16>(self, axes, keepdims, None),
            #[cfg(feature = "bf16")]
            DType::BF16 => logsumexp::<bf16, bf16>(self, axes, keepdims, None),
            #[cfg(feature = "i8")]
            DType::I8 => logsumexp::<i8, <i8 as FloatOutUnary>::Output>(self, axes, keepdims, None),
            #[cfg(feature = "u8")]
            DType::U8 => logsumexp::<u8, <u8 as FloatOutUnary>::Output>(self, axes, keepdims, None),
            #[cfg(feature = "bool")]
            DType::Bool => {
                logsumexp::<bool, <bool as FloatOutUnary>::Output>(self, axes, keepdims, None)
            }
            #[cfg(feature = "i16")]
            DType::I16 => {
                logsumexp::<i16, <i16 as FloatOutUnary>::Output>(self, axes, keepdims, None)
            }
            #[cfg(feature = "u16")]
            DType::U16 => {
                logsumexp::<u16, <u16 as FloatOutUnary>::Output>(self, axes, keepdims, None)
            }
            #[cfg(feature = "i32")]
            DType::I32 => {
                logsumexp::<i32, <i32 as FloatOutUnary>::Output>(self, axes, keepdims, None)
            }
            #[cfg(feature = "u32")]
            DType::U32 => {
                logsumexp::<u32, <u32 as FloatOutUnary>::Output>(self, axes, keepdims, None)
            }
            #[cfg(feature = "i64")]
            DType::I64 => {
                logsumexp::<i64, <i64 as FloatOutUnary>::Output>(self, axes, keepdims, None)
            }
            #[cfg(feature = "u64")]
            DType::U64 => {
                logsumexp::<u64, <u64 as FloatOutUnary>::Output>(self, axes, keepdims, None)
            }
            #[cfg(feature = "f64")]
            DType::F64 => logsumexp::<f64, f64>(self, axes, keepdims, None),
            _ => unreachable!("unsupported dtype: {:?}", self.dtype),
        }
    }
    pub fn reducel2(&self, axes: &[i64], keepdims: bool) -> Result<Tensor, TensorError> {
        match self.dtype {
            #[cfg(feature = "f32")]
            DType::F32 => reducel2::<f32, f32>(self, axes, keepdims, None),
            #[cfg(feature = "f16")]
            DType::F16 => reducel2::<f16, f16>(self, axes, keepdims, None),
            #[cfg(feature = "bf16")]
            DType::BF16 => reducel2::<bf16, bf16>(self, axes, keepdims, None),
            #[cfg(feature = "i8")]
            DType::I8 => reducel2::<i8, <i8 as FloatOutUnary>::Output>(self, axes, keepdims, None),
            #[cfg(feature = "u8")]
            DType::U8 => reducel2::<u8, <u8 as FloatOutUnary>::Output>(self, axes, keepdims, None),
            #[cfg(feature = "bool")]
            DType::Bool => {
                reducel2::<bool, <bool as FloatOutUnary>::Output>(self, axes, keepdims, None)
            }
            #[cfg(feature = "i16")]
            DType::I16 => {
                reducel2::<i16, <i16 as FloatOutUnary>::Output>(self, axes, keepdims, None)
            }
            #[cfg(feature = "u16")]
            DType::U16 => {
                reducel2::<u16, <u16 as FloatOutUnary>::Output>(self, axes, keepdims, None)
            }
            #[cfg(feature = "i32")]
            DType::I32 => {
                reducel2::<i32, <i32 as FloatOutUnary>::Output>(self, axes, keepdims, None)
            }
            #[cfg(feature = "u32")]
            DType::U32 => {
                reducel2::<u32, <u32 as FloatOutUnary>::Output>(self, axes, keepdims, None)
            }
            #[cfg(feature = "i64")]
            DType::I64 => {
                reducel2::<i64, <i64 as FloatOutUnary>::Output>(self, axes, keepdims, None)
            }
            #[cfg(feature = "u64")]
            DType::U64 => {
                reducel2::<u64, <u64 as FloatOutUnary>::Output>(self, axes, keepdims, None)
            }
            #[cfg(feature = "f64")]
            DType::F64 => reducel2::<f64, f64>(self, axes, keepdims, None),
            _ => unreachable!("unsupported dtype: {:?}", self.dtype),
        }
    }
    pub fn reducel3(&self, axes: &[i64], keepdims: bool) -> Result<Tensor, TensorError> {
        match self.dtype {
            #[cfg(feature = "f32")]
            DType::F32 => reducel3::<f32, f32>(self, axes, keepdims, None),
            #[cfg(feature = "f16")]
            DType::F16 => reducel3::<f16, f16>(self, axes, keepdims, None),
            #[cfg(feature = "bf16")]
            DType::BF16 => reducel3::<bf16, bf16>(self, axes, keepdims, None),
            #[cfg(feature = "i8")]
            DType::I8 => reducel3::<i8, <i8 as FloatOutUnary>::Output>(self, axes, keepdims, None),
            #[cfg(feature = "u8")]
            DType::U8 => reducel3::<u8, <u8 as FloatOutUnary>::Output>(self, axes, keepdims, None),
            #[cfg(feature = "bool")]
            DType::Bool => {
                reducel3::<bool, <bool as FloatOutUnary>::Output>(self, axes, keepdims, None)
            }
            #[cfg(feature = "i16")]
            DType::I16 => {
                reducel3::<i16, <i16 as FloatOutUnary>::Output>(self, axes, keepdims, None)
            }
            #[cfg(feature = "u16")]
            DType::U16 => {
                reducel3::<u16, <u16 as FloatOutUnary>::Output>(self, axes, keepdims, None)
            }
            #[cfg(feature = "i32")]
            DType::I32 => {
                reducel3::<i32, <i32 as FloatOutUnary>::Output>(self, axes, keepdims, None)
            }
            #[cfg(feature = "u32")]
            DType::U32 => {
                reducel3::<u32, <u32 as FloatOutUnary>::Output>(self, axes, keepdims, None)
            }
            #[cfg(feature = "i64")]
            DType::I64 => {
                reducel3::<i64, <i64 as FloatOutUnary>::Output>(self, axes, keepdims, None)
            }
            #[cfg(feature = "u64")]
            DType::U64 => {
                reducel3::<u64, <u64 as FloatOutUnary>::Output>(self, axes, keepdims, None)
            }
            #[cfg(feature = "f64")]
            DType::F64 => reducel3::<f64, f64>(self, axes, keepdims, None),
            _ => unreachable!("unsupported dtype: {:?}", self.dtype),
        }
    }
    pub fn mean(&self, axes: &[i64], keepdims: bool) -> Result<Tensor, TensorError> {
        match self.dtype {
            #[cfg(feature = "f32")]
            DType::F32 => mean::<f32, f32>(self, axes, keepdims, None),
            #[cfg(feature = "f16")]
            DType::F16 => mean::<f16, f16>(self, axes, keepdims, None),
            #[cfg(feature = "bf16")]
            DType::BF16 => mean::<bf16, bf16>(self, axes, keepdims, None),
            #[cfg(feature = "i8")]
            DType::I8 => mean::<i8, <i8 as FloatOutUnary>::Output>(self, axes, keepdims, None),
            #[cfg(feature = "u8")]
            DType::U8 => mean::<u8, <u8 as FloatOutUnary>::Output>(self, axes, keepdims, None),
            #[cfg(feature = "bool")]
            DType::Bool => {
                mean::<bool, <bool as FloatOutUnary>::Output>(self, axes, keepdims, None)
            }
            #[cfg(feature = "i16")]
            DType::I16 => mean::<i16, <i16 as FloatOutUnary>::Output>(self, axes, keepdims, None),
            #[cfg(feature = "u16")]
            DType::U16 => mean::<u16, <u16 as FloatOutUnary>::Output>(self, axes, keepdims, None),
            #[cfg(feature = "i32")]
            DType::I32 => mean::<i32, <i32 as FloatOutUnary>::Output>(self, axes, keepdims, None),
            #[cfg(feature = "u32")]
            DType::U32 => mean::<u32, <u32 as FloatOutUnary>::Output>(self, axes, keepdims, None),
            #[cfg(feature = "i64")]
            DType::I64 => mean::<i64, <i64 as FloatOutUnary>::Output>(self, axes, keepdims, None),
            #[cfg(feature = "u64")]
            DType::U64 => mean::<u64, <u64 as FloatOutUnary>::Output>(self, axes, keepdims, None),
            #[cfg(feature = "f64")]
            DType::F64 => mean::<f64, f64>(self, axes, keepdims, None),
            _ => unreachable!("unsupported dtype: {:?}", self.dtype),
        }
    }
    pub fn sum_(
        &self,
        axes: &[i64],
        keepdims: bool,
        out: &mut Tensor,
    ) -> Result<Tensor, TensorError> {
        match self.dtype {
            #[cfg(feature = "f32")]
            DType::F32 => sum::<f32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "f16")]
            DType::F16 => sum::<f16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "bf16")]
            DType::BF16 => sum::<bf16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i8")]
            DType::I8 => sum::<i8>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u8")]
            DType::U8 => sum::<u8>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "bool")]
            DType::Bool => sum::<bool>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i16")]
            DType::I16 => sum::<i16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u16")]
            DType::U16 => sum::<u16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i32")]
            DType::I32 => sum::<i32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u32")]
            DType::U32 => sum::<u32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i64")]
            DType::I64 => sum::<i64>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u64")]
            DType::U64 => sum::<u64>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "f64")]
            DType::F64 => sum::<f64>(self, axes, keepdims, Some(out.clone())),
            _ => unreachable!("unsupported dtype: {:?}", self.dtype),
        }
    }

    pub fn prod_(
        &self,
        axes: &[i64],
        keepdims: bool,
        out: &mut Tensor,
    ) -> Result<Tensor, TensorError> {
        match self.dtype {
            #[cfg(feature = "f32")]
            DType::F32 => prod::<f32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "f16")]
            DType::F16 => prod::<f16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "bf16")]
            DType::BF16 => prod::<bf16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i8")]
            DType::I8 => prod::<i8>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u8")]
            DType::U8 => prod::<u8>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "bool")]
            DType::Bool => prod::<bool>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i16")]
            DType::I16 => prod::<i16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u16")]
            DType::U16 => prod::<u16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i32")]
            DType::I32 => prod::<i32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u32")]
            DType::U32 => prod::<u32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i64")]
            DType::I64 => prod::<i64>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u64")]
            DType::U64 => prod::<u64>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "f64")]
            DType::F64 => prod::<f64>(self, axes, keepdims, Some(out.clone())),
            _ => unreachable!("unsupported dtype: {:?}", self.dtype),
        }
    }

    pub fn nansum_(
        &self,
        axes: &[i64],
        keepdims: bool,
        out: &mut Tensor,
    ) -> Result<Tensor, TensorError> {
        match self.dtype {
            #[cfg(feature = "f32")]
            DType::F32 => nansum::<f32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "f16")]
            DType::F16 => nansum::<f16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "bf16")]
            DType::BF16 => nansum::<bf16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i8")]
            DType::I8 => nansum::<i8>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u8")]
            DType::U8 => nansum::<u8>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "bool")]
            DType::Bool => nansum::<bool>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i16")]
            DType::I16 => nansum::<i16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u16")]
            DType::U16 => nansum::<u16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i32")]
            DType::I32 => nansum::<i32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u32")]
            DType::U32 => nansum::<u32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i64")]
            DType::I64 => nansum::<i64>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u64")]
            DType::U64 => nansum::<u64>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "f64")]
            DType::F64 => nansum::<f64>(self, axes, keepdims, Some(out.clone())),
            _ => unreachable!("unsupported dtype: {:?}", self.dtype),
        }
    }

    pub fn nanprod_(
        &self,
        axes: &[i64],
        keepdims: bool,
        out: &mut Tensor,
    ) -> Result<Tensor, TensorError> {
        match self.dtype {
            #[cfg(feature = "f32")]
            DType::F32 => nanprod::<f32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "f16")]
            DType::F16 => nanprod::<f16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "bf16")]
            DType::BF16 => nanprod::<bf16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i8")]
            DType::I8 => nanprod::<i8>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u8")]
            DType::U8 => nanprod::<u8>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "bool")]
            DType::Bool => nanprod::<bool>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i16")]
            DType::I16 => nanprod::<i16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u16")]
            DType::U16 => nanprod::<u16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i32")]
            DType::I32 => nanprod::<i32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u32")]
            DType::U32 => nanprod::<u32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i64")]
            DType::I64 => nanprod::<i64>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u64")]
            DType::U64 => nanprod::<u64>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "f64")]
            DType::F64 => nanprod::<f64>(self, axes, keepdims, Some(out.clone())),
            _ => unreachable!("unsupported dtype: {:?}", self.dtype),
        }
    }

    pub fn min_(
        &self,
        axes: &[i64],
        keepdims: bool,
        out: &mut Tensor,
    ) -> Result<Tensor, TensorError> {
        match self.dtype {
            #[cfg(feature = "f32")]
            DType::F32 => min::<f32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "f16")]
            DType::F16 => min::<f16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "bf16")]
            DType::BF16 => min::<bf16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i8")]
            DType::I8 => min::<i8>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u8")]
            DType::U8 => min::<u8>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "bool")]
            DType::Bool => min::<bool>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i16")]
            DType::I16 => min::<i16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u16")]
            DType::U16 => min::<u16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i32")]
            DType::I32 => min::<i32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u32")]
            DType::U32 => min::<u32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i64")]
            DType::I64 => min::<i64>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u64")]
            DType::U64 => min::<u64>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "f64")]
            DType::F64 => min::<f64>(self, axes, keepdims, Some(out.clone())),
            _ => unreachable!("unsupported dtype: {:?}", self.dtype),
        }
    }

    pub fn max_(
        &self,
        axes: &[i64],
        keepdims: bool,
        out: &mut Tensor,
    ) -> Result<Tensor, TensorError> {
        match self.dtype {
            #[cfg(feature = "f32")]
            DType::F32 => max::<f32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "f16")]
            DType::F16 => max::<f16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "bf16")]
            DType::BF16 => max::<bf16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i8")]
            DType::I8 => max::<i8>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u8")]
            DType::U8 => max::<u8>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "bool")]
            DType::Bool => max::<bool>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i16")]
            DType::I16 => max::<i16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u16")]
            DType::U16 => max::<u16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i32")]
            DType::I32 => max::<i32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u32")]
            DType::U32 => max::<u32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i64")]
            DType::I64 => max::<i64>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u64")]
            DType::U64 => max::<u64>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "f64")]
            DType::F64 => max::<f64>(self, axes, keepdims, Some(out.clone())),
            _ => unreachable!("unsupported dtype: {:?}", self.dtype),
        }
    }

    pub fn sum_square_(
        &self,
        axes: &[i64],
        keepdims: bool,
        out: &mut Tensor,
    ) -> Result<Tensor, TensorError> {
        match self.dtype {
            #[cfg(feature = "f32")]
            DType::F32 => sum_square::<f32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "f16")]
            DType::F16 => sum_square::<f16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "bf16")]
            DType::BF16 => sum_square::<bf16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i8")]
            DType::I8 => sum_square::<i8>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u8")]
            DType::U8 => sum_square::<u8>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "bool")]
            DType::Bool => sum_square::<bool>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i16")]
            DType::I16 => sum_square::<i16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u16")]
            DType::U16 => sum_square::<u16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i32")]
            DType::I32 => sum_square::<i32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u32")]
            DType::U32 => sum_square::<u32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i64")]
            DType::I64 => sum_square::<i64>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u64")]
            DType::U64 => sum_square::<u64>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "f64")]
            DType::F64 => sum_square::<f64>(self, axes, keepdims, Some(out.clone())),
            _ => unreachable!("unsupported dtype: {:?}", self.dtype),
        }
    }

    pub fn reducel1_(
        &self,
        axes: &[i64],
        keepdims: bool,
        out: &mut Tensor,
    ) -> Result<Tensor, TensorError> {
        match self.dtype {
            #[cfg(feature = "f32")]
            DType::F32 => reducel1::<f32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "f16")]
            DType::F16 => reducel1::<f16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "bf16")]
            DType::BF16 => reducel1::<bf16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i8")]
            DType::I8 => reducel1::<i8>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u8")]
            DType::U8 => reducel1::<u8>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "bool")]
            DType::Bool => reducel1::<bool>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i16")]
            DType::I16 => reducel1::<i16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u16")]
            DType::U16 => reducel1::<u16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i32")]
            DType::I32 => reducel1::<i32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u32")]
            DType::U32 => reducel1::<u32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i64")]
            DType::I64 => reducel1::<i64>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u64")]
            DType::U64 => reducel1::<u64>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "f64")]
            DType::F64 => reducel1::<f64>(self, axes, keepdims, Some(out.clone())),
            _ => unreachable!("unsupported dtype: {:?}", self.dtype),
        }
    }
    pub fn all_(
        &self,
        axes: &[i64],
        keepdims: bool,
        out: &mut Tensor,
    ) -> Result<Tensor, TensorError> {
        match self.dtype {
            #[cfg(feature = "f32")]
            DType::F32 => all::<f32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "f16")]
            DType::F16 => all::<f16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "bf16")]
            DType::BF16 => all::<bf16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i8")]
            DType::I8 => all::<i8>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u8")]
            DType::U8 => all::<u8>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "bool")]
            DType::Bool => all::<bool>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i16")]
            DType::I16 => all::<i16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u16")]
            DType::U16 => all::<u16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i32")]
            DType::I32 => all::<i32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u32")]
            DType::U32 => all::<u32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i64")]
            DType::I64 => all::<i64>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u64")]
            DType::U64 => all::<u64>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "f64")]
            DType::F64 => all::<f64>(self, axes, keepdims, Some(out.clone())),
            _ => unreachable!("unsupported dtype: {:?}", self.dtype),
        }
    }

    pub fn any_(
        &self,
        axes: &[i64],
        keepdims: bool,
        out: &mut Tensor,
    ) -> Result<Tensor, TensorError> {
        match self.dtype {
            #[cfg(feature = "f32")]
            DType::F32 => any::<f32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "f16")]
            DType::F16 => any::<f16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "bf16")]
            DType::BF16 => any::<bf16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i8")]
            DType::I8 => any::<i8>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u8")]
            DType::U8 => any::<u8>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "bool")]
            DType::Bool => any::<bool>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i16")]
            DType::I16 => any::<i16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u16")]
            DType::U16 => any::<u16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i32")]
            DType::I32 => any::<i32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u32")]
            DType::U32 => any::<u32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i64")]
            DType::I64 => any::<i64>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "u64")]
            DType::U64 => any::<u64>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "f64")]
            DType::F64 => any::<f64>(self, axes, keepdims, Some(out.clone())),
            _ => unreachable!("unsupported dtype: {:?}", self.dtype),
        }
    }
    pub fn logsumexp_(
        &self,
        axes: &[i64],
        keepdims: bool,
        out: &mut Tensor,
    ) -> Result<Tensor, TensorError> {
        match self.dtype {
            #[cfg(feature = "f32")]
            DType::F32 => logsumexp::<f32, f32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "f16")]
            DType::F16 => logsumexp::<f16, f16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "bf16")]
            DType::BF16 => logsumexp::<bf16, bf16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i8")]
            DType::I8 => logsumexp::<i8, <i8 as FloatOutUnary>::Output>(
                self,
                axes,
                keepdims,
                Some(out.clone()),
            ),
            #[cfg(feature = "u8")]
            DType::U8 => logsumexp::<u8, <u8 as FloatOutUnary>::Output>(
                self,
                axes,
                keepdims,
                Some(out.clone()),
            ),
            #[cfg(feature = "bool")]
            DType::Bool => logsumexp::<bool, <bool as FloatOutUnary>::Output>(
                self,
                axes,
                keepdims,
                Some(out.clone()),
            ),
            #[cfg(feature = "i16")]
            DType::I16 => logsumexp::<i16, <i16 as FloatOutUnary>::Output>(
                self,
                axes,
                keepdims,
                Some(out.clone()),
            ),
            #[cfg(feature = "u16")]
            DType::U16 => logsumexp::<u16, <u16 as FloatOutUnary>::Output>(
                self,
                axes,
                keepdims,
                Some(out.clone()),
            ),
            #[cfg(feature = "i32")]
            DType::I32 => logsumexp::<i32, <i32 as FloatOutUnary>::Output>(
                self,
                axes,
                keepdims,
                Some(out.clone()),
            ),
            #[cfg(feature = "u32")]
            DType::U32 => logsumexp::<u32, <u32 as FloatOutUnary>::Output>(
                self,
                axes,
                keepdims,
                Some(out.clone()),
            ),
            #[cfg(feature = "i64")]
            DType::I64 => logsumexp::<i64, <i64 as FloatOutUnary>::Output>(
                self,
                axes,
                keepdims,
                Some(out.clone()),
            ),
            #[cfg(feature = "u64")]
            DType::U64 => logsumexp::<u64, <u64 as FloatOutUnary>::Output>(
                self,
                axes,
                keepdims,
                Some(out.clone()),
            ),
            #[cfg(feature = "f64")]
            DType::F64 => logsumexp::<f64, f64>(self, axes, keepdims, Some(out.clone())),
            _ => unreachable!("unsupported dtype: {:?}", self.dtype),
        }
    }
    pub fn reducel2_(
        &self,
        axes: &[i64],
        keepdims: bool,
        out: &mut Tensor,
    ) -> Result<Tensor, TensorError> {
        match self.dtype {
            #[cfg(feature = "f32")]
            DType::F32 => reducel2::<f32, f32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "f16")]
            DType::F16 => reducel2::<f16, f16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "bf16")]
            DType::BF16 => reducel2::<bf16, bf16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i8")]
            DType::I8 => reducel2::<i8, <i8 as FloatOutUnary>::Output>(
                self,
                axes,
                keepdims,
                Some(out.clone()),
            ),
            #[cfg(feature = "u8")]
            DType::U8 => reducel2::<u8, <u8 as FloatOutUnary>::Output>(
                self,
                axes,
                keepdims,
                Some(out.clone()),
            ),
            #[cfg(feature = "bool")]
            DType::Bool => reducel2::<bool, <bool as FloatOutUnary>::Output>(
                self,
                axes,
                keepdims,
                Some(out.clone()),
            ),
            #[cfg(feature = "i16")]
            DType::I16 => reducel2::<i16, <i16 as FloatOutUnary>::Output>(
                self,
                axes,
                keepdims,
                Some(out.clone()),
            ),
            #[cfg(feature = "u16")]
            DType::U16 => reducel2::<u16, <u16 as FloatOutUnary>::Output>(
                self,
                axes,
                keepdims,
                Some(out.clone()),
            ),
            #[cfg(feature = "f64")]
            DType::F64 => reducel2::<f64, f64>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i32")]
            DType::I32 => reducel2::<i32, <i32 as FloatOutUnary>::Output>(
                self,
                axes,
                keepdims,
                Some(out.clone()),
            ),
            #[cfg(feature = "u32")]
            DType::U32 => reducel2::<u32, <u32 as FloatOutUnary>::Output>(
                self,
                axes,
                keepdims,
                Some(out.clone()),
            ),
            #[cfg(feature = "i64")]
            DType::I64 => reducel2::<i64, <i64 as FloatOutUnary>::Output>(
                self,
                axes,
                keepdims,
                Some(out.clone()),
            ),
            #[cfg(feature = "u64")]
            DType::U64 => reducel2::<u64, <u64 as FloatOutUnary>::Output>(
                self,
                axes,
                keepdims,
                Some(out.clone()),
            ),
            _ => unreachable!("unsupported dtype: {:?}", self.dtype),
        }
    }
    pub fn reducel3_(
        &self,
        axes: &[i64],
        keepdims: bool,
        out: &mut Tensor,
    ) -> Result<Tensor, TensorError> {
        match self.dtype {
            #[cfg(feature = "f32")]
            DType::F32 => reducel3::<f32, f32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "f16")]
            DType::F16 => reducel3::<f16, f16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "bf16")]
            DType::BF16 => reducel3::<bf16, bf16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i8")]
            DType::I8 => reducel3::<i8, <i8 as FloatOutUnary>::Output>(
                self,
                axes,
                keepdims,
                Some(out.clone()),
            ),
            #[cfg(feature = "u8")]
            DType::U8 => reducel3::<u8, <u8 as FloatOutUnary>::Output>(
                self,
                axes,
                keepdims,
                Some(out.clone()),
            ),
            #[cfg(feature = "bool")]
            DType::Bool => reducel3::<bool, <bool as FloatOutUnary>::Output>(
                self,
                axes,
                keepdims,
                Some(out.clone()),
            ),
            #[cfg(feature = "i16")]
            DType::I16 => reducel3::<i16, <i16 as FloatOutUnary>::Output>(
                self,
                axes,
                keepdims,
                Some(out.clone()),
            ),
            #[cfg(feature = "u16")]
            DType::U16 => reducel3::<u16, <u16 as FloatOutUnary>::Output>(
                self,
                axes,
                keepdims,
                Some(out.clone()),
            ),
            #[cfg(feature = "f64")]
            DType::F64 => reducel3::<f64, f64>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i32")]
            DType::I32 => reducel3::<i32, <i32 as FloatOutUnary>::Output>(
                self,
                axes,
                keepdims,
                Some(out.clone()),
            ),
            #[cfg(feature = "u32")]
            DType::U32 => reducel3::<u32, <u32 as FloatOutUnary>::Output>(
                self,
                axes,
                keepdims,
                Some(out.clone()),
            ),
            #[cfg(feature = "i64")]
            DType::I64 => reducel3::<i64, <i64 as FloatOutUnary>::Output>(
                self,
                axes,
                keepdims,
                Some(out.clone()),
            ),
            #[cfg(feature = "u64")]
            DType::U64 => reducel3::<u64, <u64 as FloatOutUnary>::Output>(
                self,
                axes,
                keepdims,
                Some(out.clone()),
            ),
            _ => unreachable!("unsupported dtype: {:?}", self.dtype),
        }
    }
    pub fn mean_(
        &self,
        axes: &[i64],
        keepdims: bool,
        out: &mut Tensor,
    ) -> Result<Tensor, TensorError> {
        match self.dtype {
            #[cfg(feature = "f32")]
            DType::F32 => mean::<f32, f32>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "f16")]
            DType::F16 => mean::<f16, f16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "bf16")]
            DType::BF16 => mean::<bf16, bf16>(self, axes, keepdims, Some(out.clone())),
            #[cfg(feature = "i8")]
            DType::I8 => {
                mean::<i8, <i8 as FloatOutUnary>::Output>(self, axes, keepdims, Some(out.clone()))
            }
            #[cfg(feature = "u8")]
            DType::U8 => {
                mean::<u8, <u8 as FloatOutUnary>::Output>(self, axes, keepdims, Some(out.clone()))
            }
            #[cfg(feature = "bool")]
            DType::Bool => mean::<bool, <bool as FloatOutUnary>::Output>(
                self,
                axes,
                keepdims,
                Some(out.clone()),
            ),
            #[cfg(feature = "i16")]
            DType::I16 => {
                mean::<i16, <i16 as FloatOutUnary>::Output>(self, axes, keepdims, Some(out.clone()))
            }
            #[cfg(feature = "u16")]
            DType::U16 => {
                mean::<u16, <u16 as FloatOutUnary>::Output>(self, axes, keepdims, Some(out.clone()))
            }
            #[cfg(feature = "i32")]
            DType::I32 => {
                mean::<i32, <i32 as FloatOutUnary>::Output>(self, axes, keepdims, Some(out.clone()))
            }
            #[cfg(feature = "u32")]
            DType::U32 => {
                mean::<u32, <u32 as FloatOutUnary>::Output>(self, axes, keepdims, Some(out.clone()))
            }
            #[cfg(feature = "i64")]
            DType::I64 => {
                mean::<i64, <i64 as FloatOutUnary>::Output>(self, axes, keepdims, Some(out.clone()))
            }
            #[cfg(feature = "u64")]
            DType::U64 => {
                mean::<u64, <u64 as FloatOutUnary>::Output>(self, axes, keepdims, Some(out.clone()))
            }
            #[cfg(feature = "f64")]
            DType::F64 => mean::<f64, f64>(self, axes, keepdims, Some(out.clone())),
            _ => unreachable!("unsupported dtype: {:?}", self.dtype),
        }
    }
}
