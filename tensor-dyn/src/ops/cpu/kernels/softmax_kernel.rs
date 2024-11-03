use tensor_common::pointer::Pointer;
use tensor_traits::CommonBounds;

use paste::paste;
use tensor_macros::{ gen_fast_reduce_simd_helper, gen_reduce_dim_not_include_simd_helper };
use tensor_types::dtype::TypeCommon;
use tensor_types::type_promote::{ FloatOutUnary, NormalOut, FloatOutBinary };
use tensor_types::utils::{ array_vec_reduce, vec_sum };
use tensor_types::vectors::traits::*;

use crate::ALIGN;

#[inline]
fn update_prg<T>(prg: &mut [i64], inp_ptr: &mut Pointer<T>, strides: &[i64], shape: &[i64]) {
    for j in (0..strides.len() - 1).rev() {
        if
            prg[j] < shape[j] - 1
            /*we need to subtract one because we didn't subtract it before we execute the kernel*/
        {
            prg[j] += 1;
            inp_ptr.offset(strides[j]);
            break;
        } else {
            prg[j] = 0;
            inp_ptr.offset(-strides[j] * (shape[j] - 1));
        }
    }
}

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

/// used for updating prg and inp_ptr for case2, second next
#[inline]
fn update_prg3<T>(
    prg: &mut [i64],
    shape_len: i64,
    inp_ptr: &mut Pointer<T>,
    strides: &[i64],
    shape: &[i64]
) {
    for j in (0..shape_len - 1).rev() {
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
fn update_prg4<T>(prg: &mut [i64], inp_ptr: &mut Pointer<T>, strides: &[i64], shape: &[i64]) {
    for j in (0..strides.len()).rev() {
        if
            prg[j] < shape[j] - 1
            /*we need to subtract one because we didn't subtract it before we execute the kernel*/
        {
            prg[j] += 1;
            inp_ptr.offset(strides[j]);
            break;
        } else {
            prg[j] = 0;
            inp_ptr.offset(-strides[j] * (shape[j] - 1));
        }
    }
}

macro_rules! gen_kernel {
    (
        $num_largest_vecs:expr,
        $unroll_num:expr,
        $inp_ptr:ident,
        $res_ptr:ident,
        $vec_size:ident,
        $outer_loop_size:ident,
        $vec_op:ident,
        $inp_strides:ident,
        $inp_shape:ident,
        $prg:ident,
        $vec_post:ident,
        [$($idx:expr),*]
    ) => {
        let origin_ptr = $inp_ptr.clone();
        for i in 0..$num_largest_vecs {
            $inp_ptr = origin_ptr.clone();
            $inp_ptr.offset(i as i64 * ($unroll_num * $vec_size) as i64);
                paste! {
                    $(
                    let mut [<res_vec $idx>] = unsafe {
                        O::Vec::from_ptr($res_ptr.ptr.offset((i * $unroll_num + ($idx - 1)) * O::Vec::SIZE as isize))
                    };
                    )*
                }
            for _ in 0..$outer_loop_size {
                paste! {
                    $(
                        let [<inp_vec $idx>] = unsafe { T::Vec::from_ptr($inp_ptr.ptr.offset(($idx - 1) * O::Vec::SIZE as isize)) };
                        [<res_vec $idx>] = $vec_op([<res_vec $idx>], [<inp_vec $idx>]);
                    )*
                }
                update_prg(&mut $prg, &mut $inp_ptr, $inp_strides, $inp_shape);
            }
            if let Some(vec_post) = &$vec_post {
                paste! {
                    $(
                        let mut_ref = unsafe { $res_ptr.ptr.offset((i * $unroll_num + ($idx - 1)) * O::Vec::SIZE as isize) as *mut O::Vec };
                        unsafe {
                            mut_ref.write_unaligned(vec_post([<res_vec $idx>]));
                        }
                    )*
                }
            } else {
                unsafe {
                    paste! {
                        $(
                            core::ptr::copy_nonoverlapping(
                                [<res_vec $idx>].as_ptr(),
                                $res_ptr.ptr.offset((i * $unroll_num + ($idx - 1)) * O::Vec::SIZE as isize),
                                O::Vec::SIZE
                            );
                        )*
                    }
                }
            }
        }
        $prg.iter_mut().for_each(|x| {
            *x = 0;
        });
    };
}

/// case when reduce along all axes except the fastest dimension, this case, inner loop stride is always 1
#[inline]
pub(crate) fn fast_reduce_simd<T, O, F, F2, F3, F4>(
    inner_loop_size: isize,
    outer_loop_size: isize,
    mut inp_ptr: tensor_common::pointer::Pointer<T>,
    mut res_ptr: tensor_common::pointer::Pointer<O>,
    inp_strides: &[i64],
    inp_shape: &[i64],
    vec_size: isize,
    op: F,
    vec_op: F2,
    op_post: Option<F3>,
    vec_op_post: Option<F4>
)
    where
        T: CommonBounds,
        O: CommonBounds,
        F: Fn(O, T) -> O,
        F2: Fn(O::Vec, T::Vec) -> O::Vec,
        F3: Fn(O) -> O,
        F4: Fn(O::Vec) -> O::Vec
{
    use crate::REGNUM;

    let origin = inp_ptr.clone(); // save original inp_ptr
    let origin_res = res_ptr.clone(); // save original res_ptr
    let ndim = inp_strides.len();
    let mut prg = vec![0; ndim]; // intialize loop progress
    let remain = inner_loop_size % vec_size; // get inner loop size remainder
    let inner = inner_loop_size - remain; // get inner loop size that is multiple of vec_size
    let num_vecs = inner / vec_size; // get number of vectors
    let remain_vec = num_vecs % (REGNUM as isize);
    let num_largest_vecs = (num_vecs - remain_vec) / (REGNUM as isize);
    #[cfg(target_feature = "avx2")]
    gen_kernel!(
        num_largest_vecs,
        REGNUM as isize,
        inp_ptr,
        res_ptr,
        vec_size,
        outer_loop_size,
        vec_op,
        inp_strides,
        inp_shape,
        prg,
        vec_op_post,
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    );
    #[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
    gen_kernel!(
        num_largest_vecs,
        REGNUM as isize,
        inp_ptr,
        res_ptr,
        vec_size,
        outer_loop_size,
        vec_op,
        inp_strides,
        inp_shape,
        prg,
        vec_op_post,
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        ]
    );
    #[cfg(all(target_feature = "sse", not(target_feature = "avx2")))]
    gen_kernel!(
        num_largest_vecs,
        REGNUM as isize,
        inp_ptr,
        res_ptr,
        vec_size,
        outer_loop_size,
        vec_op,
        inp_strides,
        inp_shape,
        prg,
        vec_op_post,
        [1, 2, 3, 4, 5, 6, 7, 8]
    );
    let remain_vec = remain_vec as u32;
    inp_ptr = origin.clone(); // reset inp_ptr
    res_ptr = origin_res.clone(); // reset res_ptr
    inp_ptr += (num_largest_vecs as i64) * (REGNUM as i64) * (vec_size as i64);
    res_ptr += (num_largest_vecs as i64) * (REGNUM as i64) * (vec_size as i64);
    gen_fast_reduce_simd_helper!(remain_vec);
    if remain > 0 {
        inp_ptr = origin; // reset inp_ptr
        res_ptr = origin_res; // reset res_ptr
        for _ in 0..outer_loop_size {
            for idx in inner..inner_loop_size {
                let inp = inp_ptr[idx];
                res_ptr[idx] = op(res_ptr[idx], inp);
            }
            update_prg(&mut prg, &mut inp_ptr, inp_strides, inp_shape);
        }
        if let Some(op_post) = &op_post {
            for i in inner..inner_loop_size {
                res_ptr[i] = op_post(res_ptr[i]);
            }
        }
    }
}

#[inline]
pub(crate) fn fast_reduce_no_simd<T, O>(
    inner_loop_size: isize,
    outer_loop_size: isize,
    mut inp_ptr: tensor_common::pointer::Pointer<T>,
    mut res_ptr: tensor_common::pointer::Pointer<O>,
    inp_strides: &[i64],
    inp_shape: &[i64]
)
    where T: CommonBounds, O: CommonBounds
{
    todo!()
    // let ndim = inp_strides.len();
    // let mut prg = vec![0; ndim]; // intialize loop progress
    // for _ in 0..outer_loop_size {
    //     for idx in 0..inner_loop_size {
    //         let inp = inp_ptr[idx];
    //         res_ptr[idx] = op(res_ptr[idx], inp);
    //     }
    //     update_prg(&mut prg, &mut inp_ptr, inp_strides, inp_shape);
    // }
    // if let Some(op_post) = op_post {
    //     for i in 0..inner_loop_size {
    //         res_ptr[i] = op_post(res_ptr[i]);
    //     }
    // }
}

macro_rules! gen_kernel2 {
    (
        $num_largest_vecs:expr,
        $unroll_num:expr,
        $inp_ptr:ident,
        $res_ptr:ident,
        $vec_size:expr,
        $intermediate_size:ident,
        $vec_op:ident,
        $inp_strides:ident,
        $inp_shape:ident,
        $prg:ident,
        $shape_len:ident,
        $vec_post:expr,
        [$($idx:expr),*]
    ) => {
        let origin_ptr = $inp_ptr.clone();
        unsafe {
            for i in 0..$num_largest_vecs {
                $inp_ptr = origin_ptr.clone();
                $inp_ptr.offset(i as i64 * ($unroll_num * $vec_size) as i64);
                    paste! {
                        $(
                        let mut [<res_vec $idx>] = O::Vec::from_ptr($res_ptr.ptr.offset((i * $unroll_num + ($idx - 1)) * O::Vec::SIZE as isize));
                        )*
                    }
                for _ in 0..$intermediate_size {
                        paste! {
                            $(
                                let [<inp_vec $idx>] = T::Vec::from_ptr($inp_ptr.ptr.offset(($idx - 1) * O::Vec::SIZE as isize));
                                [<res_vec $idx>] = $vec_op([<res_vec $idx>], [<inp_vec $idx>]);
                            )*
                        }
                    update_prg2($prg, $shape_len, &mut $inp_ptr, $inp_strides, $inp_shape);
                }
                if let Some(vec_post) = &$vec_post {
                    paste! {
                        $(
                            let mut_ref = $res_ptr.ptr.offset((i * $unroll_num + ($idx - 1)) * O::Vec::SIZE as isize) as *mut O::Vec;
                            mut_ref.write_unaligned(vec_post([<res_vec $idx>]));
                        )*
                    }
                } else {
                    paste! {
                        $(
                            core::ptr::copy_nonoverlapping(
                                [<res_vec $idx>].as_ptr(),
                                $res_ptr.ptr.offset((i * $unroll_num + ($idx - 1)) * O::Vec::SIZE as isize),
                                O::Vec::SIZE
                            );
                        )*
                    }
                }
            }
        }
        $inp_ptr = origin_ptr; // reset inp_ptr
    };
}

macro_rules! gen_kernel3 {
    (
        $num_largest_vecs:expr,
        $unroll_num:expr,
        $outer_loop_size:expr,
        $inp_ptr:ident,
        $res_ptr:ident,
        $vec_size:expr,
        $intermediate_size:ident,
        $vec_op:ident,
        $inp_strides:ident,
        $inp_shape:ident,
        $prg1:ident,
        $prg2:ident,
        $shape_len:ident,
        $inner_loop_size:expr,
        $vec_post:expr,
        [$($idx:expr),*]
    ) => {
        for _ in 0..$outer_loop_size {
            gen_kernel2!(
                $num_largest_vecs,
                $unroll_num,
                $inp_ptr,
                $res_ptr,
                $vec_size,
                $intermediate_size,
                $vec_op,
                $inp_strides,
                $inp_shape,
                $prg1,
                $shape_len,
                $vec_post,
                [$($idx),*]
            );
            update_prg3($prg2, $shape_len, &mut $inp_ptr, $inp_strides, $inp_shape);
            $res_ptr.offset($inner_loop_size as i64);
            $prg1.iter_mut().for_each(|x| {
                *x = 0;
            });
        }
    };
}

// case when reduce doesn't contain fastest dim, inner loop stride is always 1
#[inline]
pub(crate) fn reduce_dim_not_include_simd<T, O, F, F2, F3, F4>(
    inner_loop_size: isize,
    outer_loop_size: isize,
    intermediate_size: isize,
    mut inp_ptr: tensor_common::pointer::Pointer<T>,
    mut res_ptr: tensor_common::pointer::Pointer<O>,
    inp_strides: &[i64],
    inp_shape: &[i64],
    prg1: &mut [i64],
    prg2: &mut [i64],
    shape_len: i64,
    op: F,
    op_post: Option<F3>,
    vec_op: F2,
    vec_post: Option<F4>
)
    where
        T: CommonBounds,
        O: CommonBounds,
        F: Fn(O, T) -> O,
        F2: Fn(O::Vec, T::Vec) -> O::Vec,
        F3: Fn(O) -> O,
        F4: Fn(O::Vec) -> O::Vec
{
    use std::ops::IndexMut;

    use crate::REGNUM;

    let origin = inp_ptr.clone(); // save original inp_ptr
    let origin_res = res_ptr.clone(); // save original res_ptr
    let remain = inner_loop_size % (O::Vec::SIZE as isize); // get inner loop size remainder
    let inner = inner_loop_size - remain; // get inner loop size that is multiple of vec_size
    let num_vecs = inner / (O::Vec::SIZE as isize); // get number of vectors
    let remain_vec = num_vecs % (REGNUM as isize);
    let num_largest_vecs = (num_vecs - remain_vec) / (REGNUM as isize);
    let origin_prg2 = prg2.iter().cloned().collect::<Vec<_>>();
    if num_largest_vecs > 0 {
        for _ in 0..outer_loop_size {
            #[cfg(target_feature = "avx2")]
            gen_kernel2!(
                num_largest_vecs,
                16,
                inp_ptr,
                res_ptr,
                O::Vec::SIZE as isize,
                intermediate_size,
                vec_op,
                inp_strides,
                inp_shape,
                prg1,
                shape_len,
                vec_post,
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            );
            #[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
            gen_kernel2!(
                num_largest_vecs,
                32,
                inp_ptr,
                res_ptr,
                O::Vec::SIZE as isize,
                intermediate_size,
                vec_op,
                inp_strides,
                inp_shape,
                prg1,
                shape_len,
                vec_post,
                [
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                    23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                ]
            );
            #[cfg(all(target_feature = "sse", not(target_feature = "avx2")))]
            gen_kernel2!(
                num_largest_vecs,
                8,
                inp_ptr,
                res_ptr,
                O::Vec::SIZE as isize,
                intermediate_size,
                vec_op,
                inp_strides,
                inp_shape,
                prg1,
                shape_len,
                vec_post,
                [1, 2, 3, 4, 5, 6, 7, 8]
            );
            update_prg3(prg2, shape_len, &mut inp_ptr, inp_strides, inp_shape);
            if let Some(vec_post) = &vec_post {
                for i in 0..num_largest_vecs {
                    for j in 0..REGNUM as isize {
                        let mut_ref = unsafe {
                            res_ptr.ptr.offset(
                                (i * (REGNUM as isize) + j) * (O::Vec::SIZE as isize)
                            ) as *mut O::Vec
                        };
                        unsafe {
                            mut_ref.write_unaligned(vec_post(mut_ref.read_unaligned()));
                        }
                    }
                }
            }
            res_ptr.offset(inner_loop_size as i64);
            prg1.iter_mut().for_each(|x| {
                *x = 0;
            });
        }
    }
    let remain_vec = remain_vec as u32;
    inp_ptr = origin.clone(); // reset inp_ptr
    res_ptr = origin_res.clone(); // reset res_ptr
    inp_ptr.offset((num_largest_vecs as i64) * (REGNUM as i64) * (O::Vec::SIZE as i64));
    res_ptr.offset((num_largest_vecs as i64) * (REGNUM as i64) * (O::Vec::SIZE as i64));
    origin_prg2
        .iter()
        .enumerate()
        .for_each(|(i, x)| {
            *prg2.index_mut(i) = *x;
        });
    gen_reduce_dim_not_include_simd_helper!(remain_vec);
    origin_prg2
        .iter()
        .enumerate()
        .for_each(|(i, x)| {
            *prg2.index_mut(i) = *x;
        });
    if remain > 0 {
        inp_ptr = origin; // reset inp_ptr
        res_ptr = origin_res; // reset res_ptr
        for _ in 0..outer_loop_size {
            for _ in 0..intermediate_size {
                for i in inner..inner_loop_size {
                    let a_val = inp_ptr[i];
                    let mut_ref = unsafe { &mut *res_ptr.ptr.offset(i) };
                    *mut_ref = op(*mut_ref, a_val);
                }
                update_prg2(prg1, shape_len, &mut inp_ptr, inp_strides, inp_shape);
            }
            update_prg3(prg2, shape_len, &mut inp_ptr, inp_strides, inp_shape);
            if let Some(op_post) = &op_post {
                for i in inner..inner_loop_size {
                    let mut_ref = unsafe { &mut *res_ptr.ptr.offset(i) };
                    *mut_ref = op_post(*mut_ref);
                }
            }
            res_ptr += inner_loop_size as usize;
            prg1.iter_mut().for_each(|x| {
                *x = 0;
            });
        }
    }
}

#[inline]
pub(crate) fn softmax_dim_not_include<T, O>(
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
        T: CommonBounds + FloatOutUnary<Output = O>,
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
    // println!("inner loop size: {}, outer loop size: {}, intermediate_size: {}", inner_loop_size, outer_loop_size, intermediate_size);
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
        // for i in 0..(inner_loop_size as usize) {
        //     unsafe {
        //         let val = max_buffer.offset(i as isize).read();
        //         println!("{}", val);
        //     }
        // }
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
                inp_shape
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
                    sum_buffer.offset(i as isize).write(res_ptr[i]._add(buffer_val));
                }
            }
            update_prg2(prg1, shape_len, &mut res_ptr, res_strides, inp_shape);
        }
        prg1.iter_mut().for_each(|x| {
            *x = 0;
        });
        // for i in 0..inner_loop_size as i64 {
        //     unsafe {
        //         let val = sum_buffer.offset(i as isize).read();
        //         println!("sum: {}", val);
        //     }
        // }

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
            inp_shape
        );
        // println!("next");
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
        T: CommonBounds + FloatOutUnary<Output = O>,
        O: CommonBounds + NormalOut<T, Output = O> + FloatOutUnary<Output = O>,
        T::Vec: FloatOutUnary<Output = O::Vec>,
        O::Vec: FloatOutBinary<Output = O::Vec>
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
                |x, y| x._max(y),
                |x, y| x._max(y)
            );
            let max_vec = T::Vec::splat(max);

            let mut sum_vec = O::Vec::splat(O::ZERO);
            inp_vecs
                .iter()
                .zip(res_vecs.iter_mut())
                .for_each(|(inp_vec, res_vec)| {
                    let val = inp_vec.read_unaligned()._sub(max_vec)._exp();
                    res_vec.write_unaligned(val);
                    sum_vec = sum_vec._add(val);
                });

            let mut sum = O::ZERO;
            for i in (inner_loop_size as usize) - remain..inner_loop_size as usize {
                let val = inp_ptr[i]._sub(max)._exp();
                res_ptr[i] = val;
                sum = sum._add(val);
            }
            sum = sum._add(vec_sum::<O>(sum_vec));
            let sum_vec = O::Vec::splat(sum);
            res_vecs.iter_mut().for_each(|res_vec| {
                res_vec.write_unaligned(res_vec.read_unaligned()._div(sum_vec));
            });
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
                |x, y| x._max(y),
                |x, y| x._max(y)
            );

            let mut sum = O::ZERO;
            for i in 0..inner_loop_size {
                let val = inp_ptr[i]._sub(max)._exp();
                res_ptr[i] = val;
                sum = sum._add(val);
            }
            let sum_vec = O::Vec::splat(sum);
            res_vecs.iter_mut().for_each(|res_vec| {
                res_vec.write_unaligned(res_vec.read_unaligned()._div(sum_vec));
            });
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
pub(crate) fn uncontiguous_reduce_dim_include<T, O, F, F2>(
    inner_loop_size: isize,
    outer_loop_size: isize,
    intermediate_size: isize,
    mut inp_ptr: tensor_common::pointer::Pointer<T>,
    mut res_ptr: tensor_common::pointer::Pointer<O>,
    inp_strides: &[i64],
    inp_shape: &[i64],
    prg1: &mut [i64],
    prg3: &mut [i64],
    res_strides: &[i64],
    res_shape: &[i64],
    shape_len: i64,
    inp_last_stride: isize,
    op: F,
    op_post: Option<F2>
)
    where T: CommonBounds, O: CommonBounds, F: Fn(O, T) -> O, F2: Fn(O) -> O
{
    for _ in 0..outer_loop_size {
        for _ in 0..intermediate_size {
            let mut tmp = res_ptr[0isize];
            for i in 0..inner_loop_size {
                let a_val = inp_ptr[i * inp_last_stride];
                tmp = op(tmp, a_val);
            }
            res_ptr[0isize] = tmp;
            update_prg3(prg1, shape_len, &mut inp_ptr, inp_strides, inp_shape);
        }
        if let Some(op_post) = &op_post {
            res_ptr[0isize] = op_post(res_ptr[0isize]);
        }
        update_prg4(prg3, &mut res_ptr, res_strides, res_shape);
    }
}

#[inline]
pub(crate) fn uncontiguous_reduce_dim_not_include<T, O, F, F2>(
    inner_loop_size: isize,
    outer_loop_size: isize,
    intermediate_size: isize,
    mut inp_ptr: tensor_common::pointer::Pointer<T>,
    mut res_ptr: tensor_common::pointer::Pointer<O>,
    inp_strides: &[i64],
    inp_shape: &[i64],
    prg1: &mut [i64],
    prg2: &mut [i64],
    prg3: &mut [i64],
    res_strides: &[i64],
    res_shape: &[i64],
    shape_len: i64,
    inp_last_stride: isize,
    res_last_strides: isize,
    op: F,
    op_post: Option<F2>
)
    where T: CommonBounds, O: CommonBounds, F: Fn(O, T) -> O, F2: Fn(O) -> O
{
    for _ in 0..outer_loop_size {
        for _ in 0..intermediate_size {
            for i in 0..inner_loop_size {
                res_ptr[i * res_last_strides] = op(
                    res_ptr[i * res_last_strides],
                    inp_ptr[i * inp_last_stride]
                );
            }
            update_prg2(prg1, shape_len, &mut inp_ptr, inp_strides, inp_shape);
        }
        update_prg3(prg2, shape_len, &mut inp_ptr, inp_strides, inp_shape);
        if let Some(op_post) = &op_post {
            for i in 0..inner_loop_size {
                let mut_ref = unsafe { &mut *res_ptr.ptr.offset(i * res_last_strides) };
                *mut_ref = op_post(*mut_ref);
            }
        }
        update_prg(prg3, &mut res_ptr, res_strides, res_shape);
        prg1.iter_mut().for_each(|x| {
            *x = 0;
        });
    }
}
