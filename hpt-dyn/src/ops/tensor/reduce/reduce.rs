use hpt_common::{error::base::TensorError, shape::shape_utils::mt_intervals_simd};
use hpt_types::{dtype::DType, into_scalar::Cast, type_promote::NormalOut};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

use crate::{
    Tensor,
    ops::tensor::reduce::kernels::{
        contiguous_reduce_dim_include, contiguous_reduce_dim_include_simd, fast_reduce_no_simd,
        fast_reduce_simd, reduce_dim_not_include, reduce_dim_not_include_simd,
    },
};

use super::{reduce_template::contiguous_reduce_template, reduce_utils::ReductionPreprocessor};

use hpt_types::dtype::TypeCommon;
use hpt_types::traits::VecTrait;
use half::{f16, bf16};

#[duplicate::duplicate_item(
    func_name               in_type   trait_name   pre_op       pre_op_no_cast      cumulate_op             post_op                 vec_pre_op          vec_preop_no_cast   vec_cumulate                        vec_post_op;
    [contiguous_sum_f32]    [f32]     [NormalOut]  [|x: T| x]   [preop]             [|a: O, b: O| a + b]    [None::<fn(O) -> O>]    [|x: InVec| x]      [|x: InVec| x]      [|x: OutVec, y: OutVec| x + y]      [None::<fn(OutVec) -> OutVec>];
    [contiguous_sum_f16]    [f16]     [NormalOut]  [|x: T| x]   [preop]             [|a: O, b: O| a + b]    [None::<fn(O) -> O>]    [|x: InVec| x]      [|x: InVec| x]      [|x: OutVec, y: OutVec| x + y]      [None::<fn(OutVec) -> OutVec>];
    [contiguous_sum_bf16]   [bf16]    [NormalOut]  [|x: T| x]   [preop]             [|a: O, b: O| a + b]    [None::<fn(O) -> O>]    [|x: InVec| x]      [|x: InVec| x]      [|x: OutVec, y: OutVec| x + y]      [None::<fn(OutVec) -> OutVec>];
    [contiguous_sum_i8]     [i8]      [NormalOut]  [|x: T| x]   [preop]             [|a: O, b: O| a + b]    [None::<fn(O) -> O>]    [|x: InVec| x]      [|x: InVec| x]      [|x: OutVec, y: OutVec| x + y]      [None::<fn(OutVec) -> OutVec>];
    [contiguous_sum_u8]     [u8]      [NormalOut]  [|x: T| x]   [preop]             [|a: O, b: O| a + b]    [None::<fn(O) -> O>]    [|x: InVec| x]      [|x: InVec| x]      [|x: OutVec, y: OutVec| x + y]      [None::<fn(OutVec) -> OutVec>];
    
)]
pub(crate) fn func_name(
    a: &Tensor,
    axes: &[usize],
    init_val: f64,
    keepdims: bool,
    init_out: bool,
    res_dtype: DType,
    c: Option<Tensor>,
) -> std::result::Result<Tensor, TensorError> {
    type T = in_type;
    type O = <T as trait_name>::Output;
    type InVec = <T as TypeCommon>::Vec;
    type OutVec = <O as TypeCommon>::Vec;
    let preop = pre_op;
    let cumulate = cumulate_op;
    let postop = post_op;
    let preop_no_cast = pre_op_no_cast;
    let vec_preop = vec_pre_op;
    let vec_postop = vec_post_op;
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
        |num_threads, inner_loop_size, inner_loop_size_2, result, transposed_tensor| {
            let iterators = ReductionPreprocessor::new(
                num_threads,
                result.size(),
                inner_loop_size_2,
                a.ptr().cast::<T>(),
                result.ptr().cast::<O>(),
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
        |num_threads, inner_loop_size, result, a| {
            let intervals = mt_intervals_simd(inner_loop_size, num_threads, OutVec::SIZE);
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
                    let inp_ptr = inp.ptr().cast::<T>();
                    let res_ptr = res.ptr().cast::<O>();

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
        |num_threads,
         outer_loop_size,
         inner_loop_size,
         inner_loop_size_2,
         result,
         transposed_tensor| {
            let iterators = ReductionPreprocessor::new2(
                num_threads,
                outer_loop_size,
                inner_loop_size,
                a.ptr(),
                result.ptr(),
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
