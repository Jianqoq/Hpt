use tensor_common::pointer::Pointer;
use tensor_traits::CommonBounds;
use tensor_types::into_scalar::IntoScalar;
use tensor_types::type_promote::{ FloatOutUnary, NormalOut, FloatOutBinary };
use tensor_types::utils::{ array_vec_reduce, vec_sum };
use tensor_types::vectors::traits::*;
use crate::ALIGN;
use tensor_types::into_vec::IntoVec;
/// used for updating prg and inp_ptr for case2, first next
#[inline]
fn update_prg2<T>(
    prg: &mut [i64],
    shape_len: i64,
    inp_ptr: &mut tensor_common::pointer::Pointer<T>,
    strides: &[i64],
    shape: &[i64]
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
    inp_ptr: &mut tensor_common::pointer::Pointer<T>,
    res_ptr: &mut tensor_common::pointer::Pointer<O>,
    strides: &[i64],
    res_strides: &[i64],
    shape: &[i64]
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
    shape: &[i64]
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
pub(crate) fn logsoftmax_dim_not_include<T, O>(
    inner_loop_size: isize,
    outer_loop_size: isize,
    intermediate_size: isize,
    mut inp_ptr: tensor_common::pointer::Pointer<T>,
    mut res_ptr: tensor_common::pointer::Pointer<O>,
    inp_strides: &[i64],
    res_strides: &[i64],
    inp_shape: &[i64],
    prg1: &mut [i64],
    prg2: &mut [i64],
    shape_len: i64
)
    where
        T: CommonBounds + FloatOutUnary<Output = O> + IntoScalar<O>,
        O: CommonBounds + NormalOut<T, Output = O> + FloatOutUnary<Output = O>
{
    let buffer = unsafe {
        std::alloc::alloc(
            std::alloc::Layout
                ::from_size_align((inner_loop_size as usize) * T::BIT_SIZE, ALIGN)
                .unwrap()
        )
    };
    let max_buffer = buffer as *mut T;
    let buffer2 = unsafe {
        std::alloc::alloc_zeroed(
            std::alloc::Layout
                ::from_size_align((inner_loop_size as usize) * O::BIT_SIZE, ALIGN)
                .unwrap()
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
                    max_buffer.offset(i as isize).write(inp_ptr[i]._max(buffer_val));
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
                let val = inp_ptr[i]._sub(max);
                res_ptr[i] = val.into_scalar();
                unsafe {
                    let buffer_val = sum_buffer.offset(i as isize).read();
                    sum_buffer.offset(i as isize).write(val._exp()._add(buffer_val));
                }
            }
            update_prg2_softmax(
                prg1,
                shape_len,
                &mut inp_ptr,
                &mut res_ptr,
                inp_strides,
                res_strides,
                inp_shape
            );
        }
        prg1.iter_mut().for_each(|x| {
            *x = 0;
        });

        for i in 0..inner_loop_size as i64 {
            unsafe {
                sum_buffer.offset(i as isize).write(
                    sum_buffer
                        .offset(i as isize)
                        .read()
                        ._ln()
                );
            }
        }

        res_ptr = res_ptr_origin.clone(); // reset res_ptr
        for _ in 0..intermediate_size {
            for i in 0..inner_loop_size as i64 {
                unsafe {
                    let sum = sum_buffer.offset(i as isize).read();
                    res_ptr[i] = res_ptr[i]._sub(sum);
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
            inp_shape
        );
        prg1.iter_mut().for_each(|x| {
            *x = 0;
        });
    }
    unsafe {
        std::alloc::dealloc(
            buffer as *mut _,
            std::alloc::Layout::from_size_align(inner_loop_size as usize, ALIGN).unwrap()
        )
    }
}

#[inline]
pub(crate) fn contiguous_dim_include<T, O>(
    inner_loop_size: isize,
    outer_loop_size: isize,
    mut inp_ptr: tensor_common::pointer::Pointer<T>,
    mut res_ptr: tensor_common::pointer::Pointer<O>,
    inp_strides: &[i64],
    res_strides: &[i64],
    inp_shape: &[i64],
    prg1: &mut [i64],
    shape_len: i64
)
    where
        T: CommonBounds + FloatOutUnary<Output = O> + IntoScalar<O>,
        O: CommonBounds + NormalOut<T, Output = O> + FloatOutUnary<Output = O>,
        T::Vec: FloatOutUnary<Output = O::Vec> + IntoVec<O::Vec>,
        O::Vec: FloatOutBinary<Output = O::Vec> + FloatOutUnary<Output = O::Vec>
{
    let remain = (inner_loop_size as usize) % T::Vec::SIZE;
    if T::Vec::SIZE == O::Vec::SIZE {
        for _ in 0..outer_loop_size {
            let inp_arr = unsafe {
                std::slice::from_raw_parts(inp_ptr.ptr as *const T, inner_loop_size as usize)
            };
            let inp_vecs = unsafe {
                std::slice::from_raw_parts(
                    inp_ptr.ptr as *const T::Vec,
                    (inner_loop_size as usize) / T::Vec::SIZE
                )
            };
            let res_vecs = unsafe {
                std::slice::from_raw_parts_mut(
                    res_ptr.ptr as *mut O::Vec,
                    (inner_loop_size as usize) / O::Vec::SIZE
                )
            };
            let max = array_vec_reduce(
                inp_arr,
                T::NEG_INF,
                |x, y| x._max(y),
                |x, y| x._max(y)
            );
            let max_vec = T::Vec::splat(max);

            let mut sum_vec = O::Vec::splat(O::ZERO);
            inp_vecs
                .iter()
                .zip(res_vecs.iter_mut())
                .for_each(|(inp_vec, res_vec)| {
                    let val: T::Vec = inp_vec.read_unaligned()._sub(max_vec);
                    let val: O::Vec = val.into_vec();
                    res_vec.write_unaligned(val);
                    sum_vec = sum_vec._add(val._exp());
                });

            let mut sum = O::ZERO;
            for i in (inner_loop_size as usize) - remain..inner_loop_size as usize {
                let val = inp_ptr[i]._sub(max);
                res_ptr[i] = val.into_scalar();
                sum = sum._add(val._exp());
            }
            sum = sum._add(vec_sum::<O>(sum_vec));
            let log_sum = sum._ln();
            let sum_vec = O::Vec::splat(log_sum);
            res_vecs.iter_mut().for_each(|res_vec| {
                res_vec.write_unaligned(res_vec.read_unaligned()._sub(sum_vec));
            });
            for i in (inner_loop_size as usize) - remain..inner_loop_size as usize {
                res_ptr[i] = res_ptr[i]._sub(log_sum);
            }
            update_prg3_softmax(
                prg1,
                shape_len,
                &mut inp_ptr,
                &mut res_ptr,
                inp_strides,
                res_strides,
                inp_shape
            );
        }
    } else {
        for _ in 0..outer_loop_size {
            let inp_arr = unsafe {
                std::slice::from_raw_parts(inp_ptr.ptr as *const T, inner_loop_size as usize)
            };
            let res_vecs = unsafe {
                std::slice::from_raw_parts_mut(
                    res_ptr.ptr as *mut O::Vec,
                    (inner_loop_size as usize) / O::Vec::SIZE
                )
            };
            let max = array_vec_reduce(
                inp_arr,
                T::NEG_INF,
                |x, y| x._max(y),
                |x, y| x._max(y)
            );

            let mut sum = O::ZERO;
            for i in 0..inner_loop_size {
                let val = inp_ptr[i]._sub(max);
                res_ptr[i] = val.into_scalar();
                sum = sum._add(val._exp());
            }
            let log_sum = sum._ln();
            let sum_vec = O::Vec::splat(log_sum);
            res_vecs.iter_mut().for_each(|res_vec| {
                res_vec.write_unaligned(res_vec.read_unaligned()._sub(sum_vec));
            });
            for i in (inner_loop_size as usize) - remain..inner_loop_size as usize {
                res_ptr[i] = res_ptr[i]._sub(log_sum);
            }
            update_prg3_softmax(
                prg1,
                shape_len,
                &mut inp_ptr,
                &mut res_ptr,
                inp_strides,
                res_strides,
                inp_shape
            );
        }
    }
}

#[inline]
pub(crate) fn uncontiguous_logsoftmax_dim_include<T, O>(
    inner_loop_size: isize,
    outer_loop_size: isize,
    mut inp_ptr: tensor_common::pointer::Pointer<T>,
    mut res_ptr: tensor_common::pointer::Pointer<O>,
    inp_strides: &[i64],
    inp_shape: &[i64],
    prg1: &mut [i64],
    res_strides: &[i64],
    shape_len: i64,
    inp_last_stride: isize
)
    where
        T: CommonBounds + FloatOutUnary<Output = O> + IntoScalar<O>,
        O: CommonBounds + NormalOut<T, Output = O> + FloatOutUnary<Output = O>
{
    for _ in 0..outer_loop_size {
        let mut max = T::NEG_INF;
        for i in 0..inner_loop_size {
            let a_val = inp_ptr[i * inp_last_stride];
            max = max._max(a_val);
        }
        let mut sum = O::ZERO;
        for i in 0..inner_loop_size {
            let val = inp_ptr[i * inp_last_stride]._sub(max);
            res_ptr[i] = val.into_scalar();
            sum = sum._add(val._exp());
        }
        let log_sum = sum._ln();
        for i in 0..inner_loop_size {
            res_ptr[i] = res_ptr[i]._sub(log_sum);
        }
        update_prg3_softmax(
            prg1,
            shape_len,
            &mut inp_ptr,
            &mut res_ptr,
            inp_strides,
            res_strides,
            inp_shape
        );
    }
}

#[inline]
pub(crate) fn uncontiguous_logsoftmax_dim_not_include<T, O>(
    inner_loop_size: isize,
    outer_loop_size: isize,
    intermediate_size: isize,
    mut inp_ptr: tensor_common::pointer::Pointer<T>,
    mut res_ptr: tensor_common::pointer::Pointer<O>,
    inp_strides: &[i64],
    inp_shape: &[i64],
    prg1: &mut [i64],
    prg2: &mut [i64],
    res_strides: &[i64],
    shape_len: i64,
    inp_last_stride: isize,
    res_last_strides: isize
)
    where
        T: CommonBounds + FloatOutUnary<Output = O> + IntoScalar<O>,
        O: CommonBounds + NormalOut<T, Output = O> + FloatOutUnary<Output = O>
{
    let buffer = unsafe {
        std::alloc::alloc(
            std::alloc::Layout
                ::from_size_align((inner_loop_size as usize) * T::BIT_SIZE, ALIGN)
                .unwrap()
        )
    };
    let max_buffer = buffer as *mut T;
    let buffer2 = unsafe {
        std::alloc::alloc_zeroed(
            std::alloc::Layout
                ::from_size_align((inner_loop_size as usize) * O::BIT_SIZE, ALIGN)
                .unwrap()
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
                let val = inp_ptr[i * (inp_last_stride as i64)]._sub(max);
                res_ptr[i * (res_last_strides as i64)] = val.into_scalar();
                unsafe {
                    let buffer_val = sum_buffer.offset(i as isize).read();
                    sum_buffer.offset(i as isize).write(val._exp()._add(buffer_val));
                }
            }
            update_prg2_softmax(
                prg1,
                shape_len,
                &mut inp_ptr,
                &mut res_ptr,
                inp_strides,
                res_strides,
                inp_shape
            );
        }
        prg1.iter_mut().for_each(|x| {
            *x = 0;
        });
        for i in 0..inner_loop_size as i64 {
            unsafe {
                sum_buffer.offset(i as isize).write(
                    sum_buffer
                        .offset(i as isize)
                        .read()
                        ._ln()
                );
            }
        }

        res_ptr = res_ptr_origin.clone(); // reset res_ptr
        for _ in 0..intermediate_size {
            for i in 0..inner_loop_size as i64 {
                unsafe {
                    let sum = sum_buffer.offset(i as isize).read();
                    res_ptr[i * (res_last_strides as i64)] =
                        res_ptr[i * (res_last_strides as i64)]._sub(sum);
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
            inp_shape
        );
        prg1.iter_mut().for_each(|x| {
            *x = 0;
        });
    }
    unsafe {
        std::alloc::dealloc(
            buffer as *mut _,
            std::alloc::Layout::from_size_align(inner_loop_size as usize, ALIGN).unwrap()
        )
    }
}
