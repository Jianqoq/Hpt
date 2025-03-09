use crate::ALIGN;
use hpt_common::utils::pointer::Pointer;
use hpt_traits::tensor::CommonBounds;
use hpt_types::type_promote::{FloatOutBinary, FloatOutUnary, NormalOut};
use hpt_types::utils::{array_vec_reduce, vec_sum};
use hpt_types::vectors::traits::*;

/// used for updating prg and inp_ptr for case2, first next
#[inline]
fn update_prg2<T>(
    prg: &mut [i64],
    shape_len: i64,
    inp_ptr: &mut hpt_common::utils::pointer::Pointer<T>,
    strides: &[i64],
    shape: &[i64],
) {
    for j in (shape_len..shape.len() as i64).rev() {
        let j = j as usize;
        if prg[j] < shape[j] {
            prg[j] += 1;
            inp_ptr.offset(strides[j]);
            break;
        } else {
            prg[j] = 0;
            inp_ptr.offset(-strides[j] * shape[j]);
        }
    }
}

#[inline]
fn update_prg2_softmax<T, O>(
    prg: &mut [i64],
    shape_len: i64,
    inp_ptr: &mut hpt_common::utils::pointer::Pointer<T>,
    res_ptr: &mut hpt_common::utils::pointer::Pointer<O>,
    strides: &[i64],
    res_strides: &[i64],
    shape: &[i64],
) {
    for j in (shape_len..shape.len() as i64).rev() {
        let j = j as usize;
        if prg[j] < shape[j] {
            prg[j] += 1;
            inp_ptr.offset(strides[j]);
            res_ptr.offset(res_strides[j]);
            break;
        } else {
            prg[j] = 0;
            inp_ptr.offset(-strides[j] * shape[j]);
            res_ptr.offset(-res_strides[j] * shape[j]);
        }
    }
}

#[inline]
fn update_prg3_softmax<T, O>(
    prg: &mut [i64],
    shape_len: i64,
    inp_ptr: &mut Pointer<T>,
    res_ptr: &mut Pointer<O>,
    strides: &[i64],
    res_strides: &[i64],
    shape: &[i64],
) {
    for j in (0..shape_len - 1).rev() {
        let j = j as usize;
        if prg[j] < shape[j] {
            prg[j] += 1;
            inp_ptr.offset(strides[j]);
            res_ptr.offset(res_strides[j]);
            break;
        } else {
            prg[j] = 0;
            inp_ptr.offset(-strides[j] * shape[j]);
            res_ptr.offset(-res_strides[j] * shape[j]);
        }
    }
}

#[inline]
pub(crate) fn softmax_dim_not_include<T, O>(
    inner_loop_size: isize,
    outer_loop_size: isize,
    intermediate_size: isize,
    mut inp_ptr: hpt_common::utils::pointer::Pointer<T>,
    mut res_ptr: hpt_common::utils::pointer::Pointer<O>,
    inp_strides: &[i64],
    res_strides: &[i64],
    inp_shape: &[i64],
    prg1: &mut [i64],
    prg2: &mut [i64],
    shape_len: i64,
) where
    T: CommonBounds + FloatOutUnary<Output = O>,
    O: CommonBounds + NormalOut<T, Output = O> + FloatOutUnary<Output = O>,
{
    let buffer = unsafe {
        std::alloc::alloc(
            std::alloc::Layout::from_size_align((inner_loop_size as usize) * T::BIT_SIZE, ALIGN)
                .unwrap(),
        )
    };
    let max_buffer = buffer as *mut T;
    let buffer2 = unsafe {
        std::alloc::alloc_zeroed(
            std::alloc::Layout::from_size_align((inner_loop_size as usize) * O::BIT_SIZE, ALIGN)
                .unwrap(),
        )
    };
    let sum_buffer = buffer2 as *mut O;
    for _ in 0..outer_loop_size {
        for i in 0..inner_loop_size as usize {
            unsafe {
                *max_buffer.offset(i as isize) = T::NEG_INF;
                *sum_buffer.offset(i as isize) = O::ZERO;
            }
        }
        let inp_ptr_origin = inp_ptr.clone(); // save original inp_ptr'
                                              // first loop to get max value
        for _ in 0..intermediate_size {
            for i in 0..inner_loop_size as i64 {
                let buffer_val = unsafe { max_buffer.offset(i as isize).read() };
                unsafe {
                    max_buffer
                        .offset(i as isize)
                        .write(inp_ptr[i]._max(buffer_val));
                }
            }
            update_prg2(prg1, shape_len, &mut inp_ptr, inp_strides, inp_shape);
        }
        prg1.iter_mut().for_each(|x| {
            *x = 0;
        });
        inp_ptr = inp_ptr_origin.clone(); // reset inp_ptr
        let res_ptr_origin = res_ptr.clone(); // save original res_ptr
                                              // second loop to get exp value
        for _ in 0..intermediate_size {
            for i in 0..inner_loop_size as i64 {
                let max = unsafe { max_buffer.offset(i as isize).read() };
                res_ptr[i] = inp_ptr[i]._sub(max)._exp();
            }
            update_prg2_softmax(
                prg1,
                shape_len,
                &mut inp_ptr,
                &mut res_ptr,
                inp_strides,
                res_strides,
                inp_shape,
            );
        }
        prg1.iter_mut().for_each(|x| {
            *x = 0;
        });
        res_ptr = res_ptr_origin.clone(); // reset res_ptr

        for _ in 0..intermediate_size {
            for i in 0..inner_loop_size as i64 {
                unsafe {
                    let buffer_val = sum_buffer.offset(i as isize).read();
                    sum_buffer
                        .offset(i as isize)
                        .write(res_ptr[i]._add(buffer_val));
                }
            }
            update_prg2(prg1, shape_len, &mut res_ptr, res_strides, inp_shape);
        }
        prg1.iter_mut().for_each(|x| {
            *x = 0;
        });

        res_ptr = res_ptr_origin.clone(); // reset res_ptr
        for _ in 0..intermediate_size {
            for i in 0..inner_loop_size as i64 {
                unsafe {
                    let sum = sum_buffer.offset(i as isize).read();
                    res_ptr[i] = res_ptr[i]._div(sum);
                }
            }
            update_prg2(prg1, shape_len, &mut res_ptr, res_strides, inp_shape);
        }
        update_prg3_softmax(
            prg2,
            shape_len,
            &mut inp_ptr,
            &mut res_ptr,
            inp_strides,
            res_strides,
            inp_shape,
        );
        prg1.iter_mut().for_each(|x| {
            *x = 0;
        });
    }
    unsafe {
        std::alloc::dealloc(
            buffer as *mut _,
            std::alloc::Layout::from_size_align(inner_loop_size as usize, ALIGN).unwrap(),
        )
    }
}

#[inline]
pub(crate) fn contiguous_dim_include<T, O>(
    inner_loop_size: isize,
    outer_loop_size: isize,
    mut inp_ptr: hpt_common::utils::pointer::Pointer<T>,
    mut res_ptr: hpt_common::utils::pointer::Pointer<O>,
    inp_strides: &[i64],
    res_strides: &[i64],
    inp_shape: &[i64],
    prg1: &mut [i64],
    shape_len: i64,
) where
    T: CommonBounds + FloatOutUnary<Output = O>,
    O: CommonBounds + NormalOut<T, Output = O> + FloatOutUnary<Output = O>,
    T::Vec: FloatOutUnary<Output = O::Vec>,
    O::Vec: FloatOutBinary<Output = O::Vec>,
{
    let remain = (inner_loop_size as usize) % T::Vec::SIZE;
    if T::Vec::SIZE == O::Vec::SIZE {
        for _ in 0..outer_loop_size {
            let inp_arr = unsafe {
                std::slice::from_raw_parts(inp_ptr.ptr as *const T, inner_loop_size as usize)
            };
            let inp_vecs = inp_ptr.ptr as *const T::Vec;
            let res_vecs = res_ptr.ptr as *mut O::Vec;
            let max = array_vec_reduce(
                inp_arr,
                T::NEG_INF,
                |x| x,
                |x, y| x._max(y),
                |x| x,
                |x, y| x._max(y),
            );
            let max_vec = T::Vec::splat(max);

            let mut sum_vec = O::Vec::splat(O::ZERO);

            for i in 0..(inner_loop_size as usize) / T::Vec::SIZE {
                let inp_vec = unsafe { inp_vecs.offset(i as isize).read_unaligned() };
                let val = inp_vec._sub(max_vec)._exp();
                unsafe { res_vecs.offset(i as isize).write_unaligned(val) };
                sum_vec = sum_vec._add(val);
            }

            let mut sum = O::ZERO;
            for i in (inner_loop_size as usize) - remain..inner_loop_size as usize {
                let val = inp_ptr[i]._sub(max)._exp();
                res_ptr[i] = val;
                sum = sum._add(val);
            }
            sum = sum._add(vec_sum::<O>(sum_vec));
            let sum_vec = O::Vec::splat(sum);

            for i in 0..(inner_loop_size as usize) / T::Vec::SIZE {
                unsafe {
                    res_vecs.offset(i as isize).write_unaligned(
                        res_vecs.offset(i as isize).read_unaligned()._div(sum_vec),
                    );
                }
            }
            for i in (inner_loop_size as usize) - remain..inner_loop_size as usize {
                res_ptr[i] = res_ptr[i]._div(sum);
            }
            update_prg3_softmax(
                prg1,
                shape_len,
                &mut inp_ptr,
                &mut res_ptr,
                inp_strides,
                res_strides,
                inp_shape,
            );
        }
    } else {
        for _ in 0..outer_loop_size {
            let inp_arr = unsafe {
                std::slice::from_raw_parts(inp_ptr.ptr as *const T, inner_loop_size as usize)
            };
            let res_vecs = res_ptr.ptr as *mut O::Vec;
            let max = array_vec_reduce(
                inp_arr,
                T::NEG_INF,
                |x| x,
                |x, y| x._max(y),
                |x| x,
                |x, y| x._max(y),
            );

            let mut sum = O::ZERO;
            for i in 0..inner_loop_size {
                let val = inp_ptr[i]._sub(max)._exp();
                res_ptr[i] = val;
                sum = sum._add(val);
            }
            let sum_vec = O::Vec::splat(sum);
            for i in 0..(inner_loop_size as usize) / T::Vec::SIZE {
                unsafe {
                    res_vecs.offset(i as isize).write_unaligned(
                        res_vecs.offset(i as isize).read_unaligned()._div(sum_vec),
                    );
                }
            }
            for i in (inner_loop_size as usize) - remain..inner_loop_size as usize {
                res_ptr[i] = res_ptr[i]._div(sum);
            }
            update_prg3_softmax(
                prg1,
                shape_len,
                &mut inp_ptr,
                &mut res_ptr,
                inp_strides,
                res_strides,
                inp_shape,
            );
        }
    }
}

#[inline]
pub(crate) fn uncontiguous_softmax_dim_include<T, O>(
    inner_loop_size: isize,
    outer_loop_size: isize,
    mut inp_ptr: hpt_common::utils::pointer::Pointer<T>,
    mut res_ptr: hpt_common::utils::pointer::Pointer<O>,
    inp_strides: &[i64],
    inp_shape: &[i64],
    prg1: &mut [i64],
    res_strides: &[i64],
    shape_len: i64,
    inp_last_stride: isize,
) where
    T: CommonBounds + FloatOutUnary<Output = O>,
    O: CommonBounds + NormalOut<T, Output = O> + FloatOutUnary<Output = O>,
{
    for _ in 0..outer_loop_size {
        let mut max = T::NEG_INF;
        for i in 0..inner_loop_size {
            let a_val = inp_ptr[i * inp_last_stride];
            max = max._max(a_val);
        }
        let mut sum = O::ZERO;
        for i in 0..inner_loop_size {
            let val = inp_ptr[i * inp_last_stride]._sub(max)._exp();
            res_ptr[i] = val;
            sum = sum._add(val);
        }
        for i in 0..inner_loop_size {
            res_ptr[i] = res_ptr[i]._div(sum);
        }
        update_prg3_softmax(
            prg1,
            shape_len,
            &mut inp_ptr,
            &mut res_ptr,
            inp_strides,
            res_strides,
            inp_shape,
        );
    }
}

#[inline]
pub(crate) fn uncontiguous_softmax_dim_not_include<T, O>(
    inner_loop_size: isize,
    outer_loop_size: isize,
    intermediate_size: isize,
    mut inp_ptr: hpt_common::utils::pointer::Pointer<T>,
    mut res_ptr: hpt_common::utils::pointer::Pointer<O>,
    inp_strides: &[i64],
    inp_shape: &[i64],
    prg1: &mut [i64],
    prg2: &mut [i64],
    res_strides: &[i64],
    shape_len: i64,
    inp_last_stride: isize,
    res_last_strides: isize,
) where
    T: CommonBounds + FloatOutUnary<Output = O>,
    O: CommonBounds + NormalOut<T, Output = O> + FloatOutUnary<Output = O>,
{
    let buffer = unsafe {
        std::alloc::alloc(
            std::alloc::Layout::from_size_align((inner_loop_size as usize) * T::BIT_SIZE, ALIGN)
                .unwrap(),
        )
    };
    let max_buffer = buffer as *mut T;
    let buffer2 = unsafe {
        std::alloc::alloc_zeroed(
            std::alloc::Layout::from_size_align((inner_loop_size as usize) * O::BIT_SIZE, ALIGN)
                .unwrap(),
        )
    };
    let sum_buffer = buffer2 as *mut O;
    for _ in 0..outer_loop_size {
        for i in 0..inner_loop_size as usize {
            unsafe {
                *max_buffer.offset(i as isize) = T::NEG_INF;
                *sum_buffer.offset(i as isize) = O::ZERO;
            }
        }
        let inp_ptr_origin = inp_ptr.clone(); // save original inp_ptr'
                                              // first loop to get max value
        for _ in 0..intermediate_size {
            for i in 0..inner_loop_size as i64 {
                let buffer_val = unsafe { max_buffer.offset(i as isize).read() };
                unsafe {
                    max_buffer
                        .offset(i as isize)
                        .write(inp_ptr[i * (inp_last_stride as i64)]._max(buffer_val));
                }
            }
            update_prg2(prg1, shape_len, &mut inp_ptr, inp_strides, inp_shape);
        }
        prg1.iter_mut().for_each(|x| {
            *x = 0;
        });
        inp_ptr = inp_ptr_origin.clone(); // reset inp_ptr
        let res_ptr_origin = res_ptr.clone(); // save original res_ptr
                                              // second loop to get exp value
        for _ in 0..intermediate_size {
            for i in 0..inner_loop_size as i64 {
                let max = unsafe { max_buffer.offset(i as isize).read() };
                res_ptr[i * (res_last_strides as i64)] =
                    inp_ptr[i * (inp_last_stride as i64)]._sub(max)._exp();
            }
            update_prg2_softmax(
                prg1,
                shape_len,
                &mut inp_ptr,
                &mut res_ptr,
                inp_strides,
                res_strides,
                inp_shape,
            );
        }
        prg1.iter_mut().for_each(|x| {
            *x = 0;
        });
        res_ptr = res_ptr_origin.clone(); // reset res_ptr

        for _ in 0..intermediate_size {
            for i in 0..inner_loop_size as i64 {
                unsafe {
                    let buffer_val = sum_buffer.offset(i as isize).read();
                    sum_buffer
                        .offset(i as isize)
                        .write(res_ptr[i * (res_last_strides as i64)]._add(buffer_val));
                }
            }
            update_prg2(prg1, shape_len, &mut res_ptr, res_strides, inp_shape);
        }
        prg1.iter_mut().for_each(|x| {
            *x = 0;
        });

        res_ptr = res_ptr_origin.clone(); // reset res_ptr
        for _ in 0..intermediate_size {
            for i in 0..inner_loop_size as i64 {
                unsafe {
                    let sum = sum_buffer.offset(i as isize).read();
                    res_ptr[i * (res_last_strides as i64)] =
                        res_ptr[i * (res_last_strides as i64)]._div(sum);
                }
            }
            update_prg2(prg1, shape_len, &mut res_ptr, res_strides, inp_shape);
        }
        update_prg3_softmax(
            prg2,
            shape_len,
            &mut inp_ptr,
            &mut res_ptr,
            inp_strides,
            res_strides,
            inp_shape,
        );
        prg1.iter_mut().for_each(|x| {
            *x = 0;
        });
    }
    unsafe {
        std::alloc::dealloc(
            buffer as *mut _,
            std::alloc::Layout::from_size_align(inner_loop_size as usize, ALIGN).unwrap(),
        )
    }
}
