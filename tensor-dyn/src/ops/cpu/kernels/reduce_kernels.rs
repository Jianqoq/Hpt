use tensor_common::pointer::Pointer;
use tensor_traits::CommonBounds;

#[cfg(feature = "simd")]
use paste::paste;
#[cfg(feature = "simd")]
use tensor_macros::{gen_fast_reduce_simd_helper, gen_reduce_dim_not_include_simd_helper};
#[cfg(feature = "simd")]
use tensor_types::dtype::TypeCommon;
#[cfg(feature = "simd")]
use tensor_types::vectors::traits::*;

#[inline]
fn update_prg<T>(prg: &mut [i64], inp_ptr: &mut Pointer<T>, strides: &[i64], shape: &[i64]) {
    for j in (0..strides.len() - 1).rev() {
        if prg[j] < shape[j] - 1
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
#[cfg(feature = "simd")]
#[inline]
fn update_prg2<T>(
    prg: &mut [i64],
    shape_len: i64,
    inp_ptr: &mut tensor_common::pointer::Pointer<T>,
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

/// used for updating prg and inp_ptr for case2, second next
#[cfg(feature = "simd")]
#[inline]
fn update_prg3<T>(
    prg: &mut [i64],
    shape_len: i64,
    inp_ptr: &mut Pointer<T>,
    strides: &[i64],
    shape: &[i64],
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
fn update_prg4<T>(prg: &mut [i64], inp_ptr: &mut Pointer<T>, strides: &[i64], shape: &[i64]) {
    for j in (0..strides.len()).rev() {
        if prg[j] < shape[j] - 1
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

#[cfg(feature = "simd")]
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
                        <O as TypeCommon>::Vec::from_ptr($res_ptr.ptr.offset((i * $unroll_num + ($idx - 1)) * <O as TypeCommon>::Vec::SIZE as isize))
                    };
                    )*
                }
            for _ in 0..$outer_loop_size {
                paste! {
                    $(
                        let [<inp_vec $idx>] = unsafe { T::Vec::from_ptr($inp_ptr.ptr.offset(($idx - 1) * <O as TypeCommon>::Vec::SIZE as isize)) };
                        [<res_vec $idx>] = $vec_op([<res_vec $idx>], [<inp_vec $idx>]);
                    )*
                }
                update_prg(&mut $prg, &mut $inp_ptr, $inp_strides, $inp_shape);
            }
            if let Some(vec_post) = &$vec_post {
                paste! {
                    $(
                        let mut_ref = unsafe { $res_ptr.ptr.offset((i * $unroll_num + ($idx - 1)) * <O as TypeCommon>::Vec::SIZE as isize) as *mut <O as TypeCommon>::Vec };
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
                                $res_ptr.ptr.offset((i * $unroll_num + ($idx - 1)) * <O as TypeCommon>::Vec::SIZE as isize),
                                <O as TypeCommon>::Vec::SIZE
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
#[cfg(feature = "simd")]
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
    vec_op_post: Option<F4>,
) where
    T: CommonBounds,
    O: CommonBounds,
    F: Fn(O, T) -> O,
    F2: Fn(<O as TypeCommon>::Vec, T::Vec) -> <O as TypeCommon>::Vec,
    F3: Fn(O) -> O,
    F4: Fn(<O as TypeCommon>::Vec) -> <O as TypeCommon>::Vec,
{
    use crate::REGNUM;

    let origin = inp_ptr.clone(); // save original inp_ptr
    let origin_res = res_ptr.clone(); // save original res_ptr
    let ndim = inp_strides.len();
    let mut prg = vec![0; ndim]; // intialize loop progress
    let remain = inner_loop_size % vec_size; // get inner loop size remainder
    let inner = inner_loop_size - remain; // get inner loop size that is multiple of vec_size
    let num_vecs = inner / vec_size; // get number of vectors
    let remain_vec = num_vecs % REGNUM as isize;
    let num_largest_vecs = (num_vecs - remain_vec) / REGNUM as isize;
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
            25, 26, 27, 28, 29, 30, 31, 32
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
    inp_ptr.offset((num_largest_vecs as i64) * (REGNUM as i64) * (vec_size as i64));
    res_ptr.offset((num_largest_vecs as i64) * (REGNUM as i64) * (vec_size as i64));
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
pub(crate) fn fast_reduce_no_simd<T, O, F, F2>(
    inner_loop_size: isize,
    outer_loop_size: isize,
    mut inp_ptr: tensor_common::pointer::Pointer<T>,
    mut res_ptr: tensor_common::pointer::Pointer<O>,
    inp_strides: &[i64],
    inp_shape: &[i64],
    op: F,
    op_post: Option<F2>,
) where
    T: CommonBounds,
    O: CommonBounds,
    F: Fn(O, T) -> O,
    F2: Fn(O) -> O,
{
    let ndim = inp_strides.len();
    let mut prg = vec![0; ndim]; // intialize loop progress
    for _ in 0..outer_loop_size {
        for idx in 0..inner_loop_size {
            let inp = inp_ptr[idx];
            res_ptr[idx] = op(res_ptr[idx], inp);
        }
        update_prg(&mut prg, &mut inp_ptr, inp_strides, inp_shape);
    }
    if let Some(op_post) = op_post {
        for i in 0..inner_loop_size {
            res_ptr[i] = op_post(res_ptr[i]);
        }
    }
}

#[cfg(feature = "simd")]
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
                        let mut [<res_vec $idx>] = <O as TypeCommon>::Vec::from_ptr($res_ptr.ptr.offset((i * $unroll_num + ($idx - 1)) * <O as TypeCommon>::Vec::SIZE as isize));
                        )*
                    }
                for _ in 0..$intermediate_size {
                        paste! {
                            $(
                                let [<inp_vec $idx>] = T::Vec::from_ptr($inp_ptr.ptr.offset(($idx - 1) * <O as TypeCommon>::Vec::SIZE as isize));
                                [<res_vec $idx>] = $vec_op([<res_vec $idx>], [<inp_vec $idx>]);
                            )*
                        }
                    update_prg2($prg, $shape_len, &mut $inp_ptr, $inp_strides, $inp_shape);
                }
                if let Some(vec_post) = &$vec_post {
                    paste! {
                        $(
                            let mut_ref = $res_ptr.ptr.offset((i * $unroll_num + ($idx - 1)) * <O as TypeCommon>::Vec::SIZE as isize) as *mut <O as TypeCommon>::Vec;
                            mut_ref.write_unaligned(vec_post([<res_vec $idx>]));
                        )*
                    }
                } else {
                    paste! {
                        $(
                            core::ptr::copy_nonoverlapping(
                                [<res_vec $idx>].as_ptr(),
                                $res_ptr.ptr.offset((i * $unroll_num + ($idx - 1)) * <O as TypeCommon>::Vec::SIZE as isize),
                                <O as TypeCommon>::Vec::SIZE
                            );
                        )*
                    }
                }
            }
        }
        $inp_ptr = origin_ptr; // reset inp_ptr
    };
}

#[cfg(feature = "simd")]
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
#[cfg(feature = "simd")]
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
    vec_post: Option<F4>,
) where
    T: CommonBounds,
    O: CommonBounds,
    F: Fn(O, T) -> O,
    F2: Fn(<O as TypeCommon>::Vec, T::Vec) -> <O as TypeCommon>::Vec,
    F3: Fn(O) -> O,
    F4: Fn(<O as TypeCommon>::Vec) -> <O as TypeCommon>::Vec,
{
    use std::ops::IndexMut;

    use crate::REGNUM;

    let origin = inp_ptr.clone(); // save original inp_ptr
    let origin_res = res_ptr.clone(); // save original res_ptr
    let remain = inner_loop_size % (<O as TypeCommon>::Vec::SIZE as isize); // get inner loop size remainder
    let inner = inner_loop_size - remain; // get inner loop size that is multiple of vec_size
    let num_vecs = inner / (<O as TypeCommon>::Vec::SIZE as isize); // get number of vectors
    let remain_vec = num_vecs % REGNUM as isize;
    let num_largest_vecs = (num_vecs - remain_vec) / REGNUM as isize;
    let origin_prg2 = prg2.iter().cloned().collect::<Vec<_>>();
    if num_largest_vecs > 0 {
        for _ in 0..outer_loop_size {
            #[cfg(target_feature = "avx2")]
            gen_kernel2!(
                num_largest_vecs,
                16,
                inp_ptr,
                res_ptr,
                <O as TypeCommon>::Vec::SIZE as isize,
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
                <O as TypeCommon>::Vec::SIZE as isize,
                intermediate_size,
                vec_op,
                inp_strides,
                inp_shape,
                prg1,
                shape_len,
                vec_post,
                [
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                    23, 24, 25, 26, 27, 28, 29, 30, 31, 32
                ]
            );
            #[cfg(all(target_feature = "sse", not(target_feature = "avx2")))]
            gen_kernel2!(
                num_largest_vecs,
                8,
                inp_ptr,
                res_ptr,
                <O as TypeCommon>::Vec::SIZE as isize,
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
                                (i * REGNUM as isize + j) * (<O as TypeCommon>::Vec::SIZE as isize),
                            ) as *mut <O as TypeCommon>::Vec
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
    inp_ptr.offset(
        (num_largest_vecs as i64) * (REGNUM as i64) * (<O as TypeCommon>::Vec::SIZE as i64),
    );
    res_ptr.offset(
        (num_largest_vecs as i64) * (REGNUM as i64) * (<O as TypeCommon>::Vec::SIZE as i64),
    );
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
            res_ptr.add(inner_loop_size as usize);
            prg1.iter_mut().for_each(|x| {
                *x = 0;
            });
        }
    }
}

// #[cfg(not(feature = "simd"))]
#[inline]
pub(crate) fn reduce_dim_not_include<T, O, F, F2>(
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
    op_post: Option<F2>,
) where
    T: CommonBounds,
    O: CommonBounds,
    F: Fn(O, T) -> O,
    F2: Fn(O) -> O,
{
    for _i in 0..outer_loop_size {
        for _ in 0..intermediate_size {
            for i in 0..inner_loop_size as i64 {
                let a_val = inp_ptr[i];
                let result_val = res_ptr[i];
                let mut_ref = unsafe { &mut *res_ptr.ptr.offset(i as isize) };
                *mut_ref = op(result_val, a_val);
            }
            update_prg2(prg1, shape_len, &mut inp_ptr, inp_strides, inp_shape);
        }
        update_prg3(prg2, shape_len, &mut inp_ptr, inp_strides, inp_shape);
        if let Some(op_post) = &op_post {
            for i in 0..inner_loop_size {
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

#[inline]
pub(crate) fn contiguous_reduce_dim_include<T, O, F, F2>(
    inner_loop_size: isize,
    outer_loop_size: isize,
    intermediate_size: isize,
    mut inp_ptr: tensor_common::pointer::Pointer<T>,
    mut res_ptr: tensor_common::pointer::Pointer<O>,
    inp_strides: &[i64],
    inp_shape: &[i64],
    prg1: &mut [i64],
    shape_len: i64,
    op: F,
    op_post: Option<F2>,
) where
    T: CommonBounds,
    O: CommonBounds,
    F: Fn(O, T) -> O,
    F2: Fn(O) -> O,
{
    for _ in 0..outer_loop_size {
        for _ in 0..intermediate_size {
            let mut tmp = res_ptr[0isize];
            for i in 0..inner_loop_size as i64 {
                let a_val = inp_ptr[i];
                tmp = op(tmp, a_val);
            }
            res_ptr[0isize] = tmp;
            update_prg3(prg1, shape_len, &mut inp_ptr, inp_strides, inp_shape);
        }
        if let Some(op_post) = &op_post {
            let tmp = res_ptr[0isize];
            let tmp = op_post(tmp);
            res_ptr[0isize] = tmp;
        }
        res_ptr.add(1);
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
    op_post: Option<F2>,
) where
    T: CommonBounds,
    O: CommonBounds,
    F: Fn(O, T) -> O,
    F2: Fn(O) -> O,
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
    op_post: Option<F2>,
) where
    T: CommonBounds,
    O: CommonBounds,
    F: Fn(O, T) -> O,
    F2: Fn(O) -> O,
{
    for _ in 0..outer_loop_size {
        for _ in 0..intermediate_size {
            for i in 0..inner_loop_size {
                res_ptr[i * res_last_strides] =
                    op(res_ptr[i * res_last_strides], inp_ptr[i * inp_last_stride]);
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
