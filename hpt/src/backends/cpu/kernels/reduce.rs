use hpt_common::utils::pointer::Pointer;
use hpt_traits::tensor::CommonBounds;

use hpt_macros::{gen_fast_reduce_simd_helper, gen_reduce_dim_not_include_simd_helper};
use hpt_types::dtype::TypeCommon;
use hpt_types::utils::array_vec_reduce;
use hpt_types::vectors::traits::*;
use paste::paste;

#[inline]
fn update_prg<T>(prg: &mut [i64], mut inp_ptr: &mut Pointer<T>, strides: &[i64], shape: &[i64]) {
    for j in (0..strides.len() - 1).rev() {
        if prg[j] < shape[j] - 1
        /*we need to subtract one because we didn't subtract it before we execute the kernel*/
        {
            prg[j] += 1;
            inp_ptr += strides[j];
            break;
        } else {
            prg[j] = 0;
            inp_ptr += -strides[j] * (shape[j] - 1);
        }
    }
}

/// used for updating prg and inp_ptr for case2, first next
#[inline]
fn update_prg2<T>(
    prg: &mut [i64],
    shape_len: i64,
    mut inp_ptr: &mut hpt_common::utils::pointer::Pointer<T>,
    strides: &[i64],
    shape: &[i64],
) {
    for j in (shape_len..shape.len() as i64).rev() {
        let j = j as usize;
        if prg[j] < shape[j] {
            prg[j] += 1;
            inp_ptr += strides[j];
            break;
        } else {
            prg[j] = 0;
            inp_ptr += -strides[j] * shape[j];
        }
    }
}

/// used for updating prg and inp_ptr for case2, second next
#[inline]
fn update_prg3<T>(
    prg: &mut [i64],
    shape_len: i64,
    mut inp_ptr: &mut Pointer<T>,
    strides: &[i64],
    shape: &[i64],
) {
    for j in (0..shape_len - 1).rev() {
        let j = j as usize;
        if prg[j] < shape[j] {
            prg[j] += 1;
            inp_ptr += strides[j];
            break;
        } else {
            prg[j] = 0;
            inp_ptr += -strides[j] * shape[j];
        }
    }
}

#[inline]
fn update_prg4<T>(prg: &mut [i64], mut inp_ptr: &mut Pointer<T>, strides: &[i64], shape: &[i64]) {
    for j in (0..strides.len()).rev() {
        if prg[j] < shape[j] - 1
        /*we need to subtract one because we didn't subtract it before we execute the kernel*/
        {
            prg[j] += 1;
            inp_ptr += strides[j];
            break;
        } else {
            prg[j] = 0;
            inp_ptr += -strides[j] * (shape[j] - 1);
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
        $vec_preop:ident,
        $vec_cumulate:ident,
        $inp_strides:ident,
        $inp_shape:ident,
        $prg:ident,
        $vec_post:ident,
        [$($idx:expr),*]
    ) => {
        let origin_ptr = $inp_ptr.clone();
        for i in 0..$num_largest_vecs {
            $inp_ptr = origin_ptr.clone();
            $inp_ptr += i as i64 * ($unroll_num * $vec_size) as i64;
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
                        [<res_vec $idx>] = $vec_cumulate([<res_vec $idx>], $vec_preop([<inp_vec $idx>]));
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
pub(crate) fn fast_reduce_simd<T, O, F, F2, F3, F4, F5, F6>(
    inner_loop_size: isize,
    outer_loop_size: isize,
    mut inp_ptr: hpt_common::utils::pointer::Pointer<T>,
    mut res_ptr: hpt_common::utils::pointer::Pointer<O>,
    inp_strides: &[i64],
    inp_shape: &[i64],
    vec_size: isize,
    preop: F,
    cumulate: F2,
    post: Option<F3>,
    vec_preop: F4,
    vec_cumulate: F5,
    vec_post: Option<F6>,
) where
    T: CommonBounds,
    O: CommonBounds,
    F: Fn(T) -> O,
    F2: Fn(O, O) -> O,
    F3: Fn(O) -> O,
    F4: Fn(T::Vec) -> O::Vec,
    F5: Fn(O::Vec, O::Vec) -> O::Vec,
    F6: Fn(O::Vec) -> O::Vec,
{
    use hpt_types::REGNUM;

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
        vec_preop,
        vec_cumulate,
        inp_strides,
        inp_shape,
        prg,
        vec_post,
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
        vec_preop,
        vec_cumulate,
        inp_strides,
        inp_shape,
        prg,
        vec_post,
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32
        ]
    ); // prettier-ignore
    #[cfg(all(target_feature = "sse", not(target_feature = "avx2")))]
    gen_kernel!(
        num_largest_vecs,
        REGNUM as isize,
        inp_ptr,
        res_ptr,
        vec_size,
        outer_loop_size,
        vec_preop,
        vec_cumulate,
        inp_strides,
        inp_shape,
        prg,
        vec_post,
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
                res_ptr[idx] = cumulate(res_ptr[idx], preop(inp));
            }
            update_prg(&mut prg, &mut inp_ptr, inp_strides, inp_shape);
        }
        if let Some(post) = &post {
            for i in inner..inner_loop_size {
                res_ptr[i] = post(res_ptr[i]);
            }
        }
    }
}

#[inline]
pub(crate) fn fast_reduce_no_simd<T, O, F, F2, F3>(
    inner_loop_size: isize,
    outer_loop_size: isize,
    mut inp_ptr: hpt_common::utils::pointer::Pointer<T>,
    mut res_ptr: hpt_common::utils::pointer::Pointer<O>,
    inp_strides: &[i64],
    inp_shape: &[i64],
    preop: F,
    cumulate: F2,
    post: Option<F3>,
) where
    T: CommonBounds,
    O: CommonBounds,
    F: Fn(T) -> O,
    F2: Fn(O, O) -> O,
    F3: Fn(O) -> O,
{
    let ndim = inp_strides.len();
    let mut prg = vec![0; ndim]; // intialize loop progress
    for _ in 0..outer_loop_size {
        for idx in 0..inner_loop_size {
            let inp = inp_ptr[idx];
            res_ptr[idx] = cumulate(res_ptr[idx], preop(inp));
        }
        update_prg(&mut prg, &mut inp_ptr, inp_strides, inp_shape);
    }
    if let Some(post) = post {
        for i in 0..inner_loop_size {
            res_ptr[i] = post(res_ptr[i]);
        }
    }
}

macro_rules! gen_kernel2 {
    (
        $num_largest_vecs:expr,
        $unroll_num:expr,
        $inp_ptr:ident,
        $res_ptr:ident,
        $vec_size:expr,
        $intermediate_size:ident,
        $vec_preop:ident,
        $vec_cumulate:ident,
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
                $inp_ptr += i as i64 * ($unroll_num * $vec_size) as i64;
                    paste! {
                        $(
                        let mut [<res_vec $idx>] = O::Vec::from_ptr($res_ptr.ptr.offset((i * $unroll_num + ($idx - 1)) * O::Vec::SIZE as isize));
                        )*
                    }
                for _ in 0..$intermediate_size {
                        paste! {
                            $(
                                let [<inp_vec $idx>] = T::Vec::from_ptr($inp_ptr.ptr.offset(($idx - 1) * O::Vec::SIZE as isize));
                                [<res_vec $idx>] = $vec_cumulate([<res_vec $idx>], $vec_preop([<inp_vec $idx>]));
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
        $vec_preop:ident,
        $vec_cumulate:ident,
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
                $vec_preop,
                $vec_cumulate,
                $inp_strides,
                $inp_shape,
                $prg1,
                $shape_len,
                $vec_post,
                [$($idx),*]
            );
            update_prg3($prg2, $shape_len, &mut $inp_ptr, $inp_strides, $inp_shape);
            $res_ptr += $inner_loop_size as i64;
            $prg1.iter_mut().for_each(|x| {
                *x = 0;
            });
        }
    };
}

// case when reduce doesn't contain fastest dim, inner loop stride is always 1
#[inline]
pub(crate) fn reduce_dim_not_include_simd<T, O, F, F2, F3, F4, F5, F6>(
    inner_loop_size: isize,
    outer_loop_size: isize,
    intermediate_size: isize,
    mut inp_ptr: hpt_common::utils::pointer::Pointer<T>,
    mut res_ptr: hpt_common::utils::pointer::Pointer<O>,
    inp_strides: &[i64],
    inp_shape: &[i64],
    prg1: &mut [i64],
    prg2: &mut [i64],
    shape_len: i64,
    preop: F,
    cumulate: F2,
    post: Option<F3>,
    vec_preop: F4,
    vec_cumulate: F5,
    vec_post: Option<F6>,
) where
    T: CommonBounds,
    O: CommonBounds,
    F: Fn(T) -> O,
    F2: Fn(O, O) -> O,
    F3: Fn(O) -> O,
    F4: Fn(T::Vec) -> O::Vec,
    F5: Fn(O::Vec, O::Vec) -> O::Vec,
    F6: Fn(O::Vec) -> O::Vec,
{
    use std::ops::IndexMut;

    use hpt_types::REGNUM;

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
                vec_preop,
                vec_cumulate,
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
                vec_preop,
                vec_cumulate,
                inp_strides,
                inp_shape,
                prg1,
                shape_len,
                vec_post,
                [
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                    23, 24, 25, 26, 27, 28, 29, 30, 31, 32
                ]
            ); // prettier-ignore
            #[cfg(all(target_feature = "sse", not(target_feature = "avx2")))]
            gen_kernel2!(
                num_largest_vecs,
                8,
                inp_ptr,
                res_ptr,
                O::Vec::SIZE as isize,
                intermediate_size,
                vec_preop,
                vec_cumulate,
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
                            res_ptr
                                .ptr
                                .offset((i * (REGNUM as isize) + j) * (O::Vec::SIZE as isize))
                                as *mut O::Vec
                        };
                        unsafe {
                            mut_ref.write_unaligned(vec_post(mut_ref.read_unaligned()));
                        }
                    }
                }
            }
            res_ptr += inner_loop_size as i64;
            prg1.iter_mut().for_each(|x| {
                *x = 0;
            });
        }
    }
    let remain_vec = remain_vec as u32;
    inp_ptr = origin.clone(); // reset inp_ptr
    res_ptr = origin_res.clone(); // reset res_ptr
    inp_ptr += (num_largest_vecs as i64) * (REGNUM as i64) * (O::Vec::SIZE as i64);
    res_ptr += (num_largest_vecs as i64) * (REGNUM as i64) * (O::Vec::SIZE as i64);
    origin_prg2.iter().enumerate().for_each(|(i, x)| {
        *prg2.index_mut(i) = *x;
    });
    gen_reduce_dim_not_include_simd_helper!(remain_vec);
    origin_prg2.iter().enumerate().for_each(|(i, x)| {
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
                    *mut_ref = cumulate(*mut_ref, preop(a_val));
                }
                update_prg2(prg1, shape_len, &mut inp_ptr, inp_strides, inp_shape);
            }
            update_prg3(prg2, shape_len, &mut inp_ptr, inp_strides, inp_shape);
            if let Some(post) = &post {
                for i in inner..inner_loop_size {
                    let mut_ref = unsafe { &mut *res_ptr.ptr.offset(i) };
                    *mut_ref = post(*mut_ref);
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
pub(crate) fn reduce_dim_not_include<T, O, F, F2, F3>(
    inner_loop_size: isize,
    outer_loop_size: isize,
    intermediate_size: isize,
    mut inp_ptr: hpt_common::utils::pointer::Pointer<T>,
    mut res_ptr: hpt_common::utils::pointer::Pointer<O>,
    inp_strides: &[i64],
    inp_shape: &[i64],
    prg1: &mut [i64],
    prg2: &mut [i64],
    shape_len: i64,
    preop: F,
    cumulate: F2,
    post: Option<F3>,
) where
    T: CommonBounds,
    O: CommonBounds,
    F: Fn(T) -> O,
    F2: Fn(O, O) -> O,
    F3: Fn(O) -> O,
{
    for _ in 0..outer_loop_size {
        for _ in 0..intermediate_size {
            for i in 0..inner_loop_size as i64 {
                let a_val = inp_ptr[i];
                let result_val = res_ptr[i];
                let mut_ref = unsafe { &mut *res_ptr.ptr.offset(i as isize) };
                *mut_ref = cumulate(result_val, preop(a_val));
            }
            update_prg2(prg1, shape_len, &mut inp_ptr, inp_strides, inp_shape);
        }
        update_prg3(prg2, shape_len, &mut inp_ptr, inp_strides, inp_shape);
        if let Some(post) = &post {
            for i in 0..inner_loop_size {
                let mut_ref = unsafe { &mut *res_ptr.ptr.offset(i) };
                *mut_ref = post(*mut_ref);
            }
        }
        res_ptr += inner_loop_size as usize;
        prg1.iter_mut().for_each(|x| {
            *x = 0;
        });
    }
}

#[inline]
pub(crate) fn contiguous_reduce_dim_include_simd<T, F, F2, F3, F4, F5>(
    init: T,
    inner_loop_size: isize,
    outer_loop_size: isize,
    intermediate_size: isize,
    mut inp_ptr: hpt_common::utils::pointer::Pointer<T>,
    mut res_ptr: hpt_common::utils::pointer::Pointer<T>,
    inp_strides: &[i64],
    inp_shape: &[i64],
    prg1: &mut [i64],
    shape_len: i64,
    scalar_preop: F,
    scalar_cumulate: F2,
    vec_preop: F3,
    vec_cumulate: F4,
    scalar_post: Option<F5>,
) where
    T: CommonBounds,
    F: Fn(T) -> T,
    F2: Fn(T, T) -> T,
    F3: Fn(T::Vec) -> T::Vec,
    F4: Fn(T::Vec, T::Vec) -> T::Vec,
    F5: Fn(T) -> T,
{
    for _ in 0..outer_loop_size {
        let mut tmp = init;
        for _ in 0..intermediate_size {
            let inp_arr = unsafe {
                std::slice::from_raw_parts(inp_ptr.ptr as *const T, inner_loop_size as usize)
            };
            tmp = scalar_cumulate(
                tmp,
                array_vec_reduce(
                    inp_arr,
                    init,
                    &scalar_preop,
                    &scalar_cumulate,
                    &vec_preop,
                    &vec_cumulate,
                ),
            );
            update_prg3(prg1, shape_len, &mut inp_ptr, inp_strides, inp_shape);
        }
        if let Some(op_post) = &scalar_post {
            res_ptr[0isize] = op_post(tmp);
        } else {
            res_ptr[0isize] = tmp;
        }
        res_ptr += 1usize;
    }
}

#[inline]
pub(crate) fn contiguous_reduce_dim_include<T, O, F, F2, F3>(
    inner_loop_size: isize,
    outer_loop_size: isize,
    intermediate_size: isize,
    mut inp_ptr: hpt_common::utils::pointer::Pointer<T>,
    mut res_ptr: hpt_common::utils::pointer::Pointer<O>,
    inp_strides: &[i64],
    inp_shape: &[i64],
    prg1: &mut [i64],
    shape_len: i64,
    preop: F,
    cumulate: F2,
    postop: Option<F3>,
) where
    T: CommonBounds,
    O: CommonBounds,
    F: Fn(T) -> O,
    F2: Fn(O, O) -> O,
    F3: Fn(O) -> O,
{
    for _ in 0..outer_loop_size {
        let mut tmp = res_ptr[0isize];
        for _ in 0..intermediate_size {
            for i in 0..inner_loop_size as i64 {
                let a_val = inp_ptr[i];
                tmp = cumulate(tmp, preop(a_val));
            }
            update_prg3(prg1, shape_len, &mut inp_ptr, inp_strides, inp_shape);
        }
        if let Some(op_post) = &postop {
            res_ptr[0isize] = op_post(tmp);
        } else {
            res_ptr[0isize] = tmp;
        }
        res_ptr += 1usize;
    }
}

#[inline]
pub(crate) fn uncontiguous_reduce_dim_include<T, O, F, F2, F3>(
    inner_loop_size: isize,
    outer_loop_size: isize,
    intermediate_size: isize,
    mut inp_ptr: hpt_common::utils::pointer::Pointer<T>,
    mut res_ptr: hpt_common::utils::pointer::Pointer<O>,
    inp_strides: &[i64],
    inp_shape: &[i64],
    prg1: &mut [i64],
    prg3: &mut [i64],
    res_strides: &[i64],
    res_shape: &[i64],
    shape_len: i64,
    inp_last_stride: isize,
    preop: F,
    cumulate: F2,
    postop: Option<F3>,
) where
    T: CommonBounds,
    O: CommonBounds,
    F: Fn(T) -> O,
    F2: Fn(O, O) -> O,
    F3: Fn(O) -> O,
{
    for _ in 0..outer_loop_size {
        let mut tmp = res_ptr[0isize];
        for _ in 0..intermediate_size {
            for i in 0..inner_loop_size {
                let a_val = inp_ptr[i * inp_last_stride];
                tmp = cumulate(tmp, preop(a_val));
            }
            update_prg3(prg1, shape_len, &mut inp_ptr, inp_strides, inp_shape);
        }
        if let Some(postop) = &postop {
            res_ptr[0isize] = postop(tmp);
        } else {
            res_ptr[0isize] = tmp;
        }
        update_prg4(prg3, &mut res_ptr, res_strides, res_shape);
    }
}

#[inline]
pub(crate) fn uncontiguous_reduce_dim_not_include<T, O, F, F2, F3>(
    inner_loop_size: isize,
    outer_loop_size: isize,
    intermediate_size: isize,
    mut inp_ptr: hpt_common::utils::pointer::Pointer<T>,
    mut res_ptr: hpt_common::utils::pointer::Pointer<O>,
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
    preop: F,
    cumulate: F2,
    postop: Option<F3>,
) where
    T: CommonBounds,
    O: CommonBounds,
    F: Fn(T) -> O,
    F2: Fn(O, O) -> O,
    F3: Fn(O) -> O,
{
    for _ in 0..outer_loop_size {
        for _ in 0..intermediate_size {
            for i in 0..inner_loop_size {
                res_ptr[i * res_last_strides] = cumulate(
                    res_ptr[i * res_last_strides],
                    preop(inp_ptr[i * inp_last_stride]),
                );
            }
            update_prg2(prg1, shape_len, &mut inp_ptr, inp_strides, inp_shape);
        }
        update_prg3(prg2, shape_len, &mut inp_ptr, inp_strides, inp_shape);
        if let Some(postop) = &postop {
            for i in 0..inner_loop_size {
                let mut_ref = unsafe { &mut *res_ptr.ptr.offset(i * res_last_strides) };
                *mut_ref = postop(*mut_ref);
            }
        }
        update_prg(prg3, &mut res_ptr, res_strides, res_shape);
        prg1.iter_mut().for_each(|x| {
            *x = 0;
        });
    }
}
