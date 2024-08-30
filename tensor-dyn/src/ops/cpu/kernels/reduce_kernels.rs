use tensor_macros::gen_fast_reduce_simd_helper;
use tensor_macros::gen_reduce_dim_not_include_simd_helper;
use tensor_traits::CommonBounds;
use tensor_types::dtype::TypeCommon;
use paste::paste;
use tensor_types::vectors::traits::VecSize;
use tensor_types::vectors::traits::Init;
use tensor_types::vectors::traits::VecTrait;

#[inline]
fn update_prg<T>(
    prg: &mut [i64],
    inp_ptr: &mut tensor_common::pointer::Pointer<T>,
    strides: &[i64],
    shape: &[i64]
) {
    for j in (0..strides.len() - 1).rev() {
        if
            prg[j] <
            shape[j] -
                1 /*we need to subtract one because we didn't subtract it before we execute the kernel*/
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

// used for updating prg and inp_ptr for case2, first next
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

// used for updating prg and inp_ptr for case2, second next
#[inline]
fn update_prg3<T>(
    prg: &mut [i64],
    shape_len: i64,
    inp_ptr: &mut tensor_common::pointer::Pointer<T>,
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
        [$($idx:expr),*]
    ) => {
        let origin_ptr = $inp_ptr;
        for i in 0..$num_largest_vecs {
            $inp_ptr = origin_ptr;
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
                        let [<inp_vec $idx>] = unsafe { <T as TypeCommon>::Vec::from_ptr($inp_ptr.ptr.offset(($idx - 1) * <O as TypeCommon>::Vec::SIZE as isize)) };
                        [<res_vec $idx>] = $vec_op([<res_vec $idx>], [<inp_vec $idx>]);
                    )*
                }
                update_prg(&mut $prg, &mut $inp_ptr, $inp_strides, $inp_shape);
            }
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
        $prg.iter_mut().for_each(|x| {
            *x = 0;
        });
    };
}

// case when reduce along all axes except the fastest dimension, this case, inner loop stride is always 1
#[inline]
pub(crate) fn fast_reduce_simd<T, O, F, F2>(
    inner_loop_size: isize,
    outer_loop_size: isize,
    mut inp_ptr: tensor_common::pointer::Pointer<T>,
    mut res_ptr: tensor_common::pointer::Pointer<O>,
    inp_strides: &[i64],
    inp_shape: &[i64],
    vec_size: isize,
    op: F,
    vec_op: F2
)
    where
        T: CommonBounds,
        O: CommonBounds,
        F: Fn(O, T) -> O,
        F2: Fn(<O as TypeCommon>::Vec, <T as TypeCommon>::Vec) -> <O as TypeCommon>::Vec
{
    let origin = inp_ptr; // save original inp_ptr
    let origin_res = res_ptr; // save original res_ptr
    let ndim = inp_strides.len();
    let mut prg = vec![0; ndim]; // intialize loop progress
    let remain = inner_loop_size % vec_size; // get inner loop size remainder
    let inner = inner_loop_size - remain; // get inner loop size that is multiple of vec_size
    let num_vecs = inner / vec_size; // get number of vectors
    #[cfg(target_feature = "avx2")]
    let largest_num_vec = 16;
    #[cfg(target_feature = "avx512f")]
    let largest_num_vec = 32;
    #[cfg(all(target_feature = "sse", not(target_feature = "avx2")))]
    let largest_num_vec = 8;
    let remain_vec = num_vecs % largest_num_vec;
    let num_largest_vecs = (num_vecs - remain_vec) / largest_num_vec;
    gen_kernel!(
        num_largest_vecs,
        16,
        inp_ptr,
        res_ptr,
        vec_size,
        outer_loop_size,
        vec_op,
        inp_strides,
        inp_shape,
        prg,
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    );
    let remain_vec = remain_vec as u32;
    inp_ptr = origin; // reset inp_ptr
    res_ptr = origin_res; // reset res_ptr
    inp_ptr.offset((num_largest_vecs as i64) * (largest_num_vec as i64) * (vec_size as i64));
    res_ptr.offset((num_largest_vecs as i64) * (largest_num_vec as i64) * (vec_size as i64));
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
    }
}

#[inline]
pub(crate) fn fast_reduce_no_simd<T, O, F>(
    inner_loop_size: isize,
    outer_loop_size: isize,
    mut inp_ptr: tensor_common::pointer::Pointer<T>,
    mut res_ptr: tensor_common::pointer::Pointer<O>,
    inp_strides: &[i64],
    inp_shape: &[i64],
    op: F
)
    where T: CommonBounds, O: CommonBounds, F: Fn(O, T) -> O
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
        [$($idx:expr),*]
    ) => {
        let origin_ptr = $inp_ptr;
        unsafe {
            for i in 0..$num_largest_vecs {
                $inp_ptr = origin_ptr;
                $inp_ptr.offset(i as i64 * ($unroll_num * $vec_size) as i64);
                    paste! {
                        $(
                        let mut [<res_vec $idx>] = <O as TypeCommon>::Vec::from_ptr($res_ptr.ptr.offset((i * $unroll_num + ($idx - 1)) * <O as TypeCommon>::Vec::SIZE as isize));
                        )*
                    }
                for _ in 0..$intermediate_size {
                        paste! {
                            $(
                                let [<inp_vec $idx>] = <T as TypeCommon>::Vec::from_ptr($inp_ptr.ptr.offset(($idx - 1) * <O as TypeCommon>::Vec::SIZE as isize));
                                [<res_vec $idx>] = $vec_op([<res_vec $idx>], [<inp_vec $idx>]);
                            )*
                        }
                    update_prg2($prg, $shape_len, &mut $inp_ptr, $inp_strides, $inp_shape);
                }
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
pub(crate) fn reduce_dim_not_include_simd<T, O, F, F2>(
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
    vec_op: F2
)
    where
        T: CommonBounds,
        O: CommonBounds,
        F: Fn(O, T) -> O,
        F2: Fn(<O as TypeCommon>::Vec, <T as TypeCommon>::Vec) -> <O as TypeCommon>::Vec
{
    let origin = inp_ptr; // save original inp_ptr
    let origin_res = res_ptr; // save original res_ptr
    let remain = inner_loop_size % (<O as TypeCommon>::Vec::SIZE as isize); // get inner loop size remainder
    let inner = inner_loop_size - remain; // get inner loop size that is multiple of vec_size
    let num_vecs = inner / (<O as TypeCommon>::Vec::SIZE as isize); // get number of vectors
    #[cfg(target_feature = "avx2")]
    let largest_num_vec = 16;
    #[cfg(target_feature = "avx512f")]
    let largest_num_vec = 32;
    #[cfg(all(target_feature = "sse", not(target_feature = "avx2")))]
    let largest_num_vec = 8;
    let remain_vec = num_vecs % largest_num_vec;
    let num_largest_vecs = (num_vecs - remain_vec) / largest_num_vec;

    if num_largest_vecs > 0 {
        for _ in 0..outer_loop_size {
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
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            );
            update_prg3(prg2, shape_len, &mut inp_ptr, inp_strides, inp_shape);
            res_ptr.offset(inner_loop_size as i64);
            prg1.iter_mut().for_each(|x| {
                *x = 0;
            });
        }
    }
    let remain_vec = remain_vec as u32;
    inp_ptr = origin; // reset inp_ptr
    res_ptr = origin_res; // reset res_ptr
    inp_ptr.offset(
        (num_largest_vecs as i64) * (largest_num_vec as i64) * (<O as TypeCommon>::Vec::SIZE as i64)
    );
    res_ptr.offset(
        (num_largest_vecs as i64) * (largest_num_vec as i64) * (<O as TypeCommon>::Vec::SIZE as i64)
    );
    gen_reduce_dim_not_include_simd_helper!(remain_vec);
    if remain > 0 {
        inp_ptr = origin; // reset inp_ptr
        res_ptr = origin_res; // reset res_ptr
        for _i in 0..outer_loop_size {
            for _ in 0..intermediate_size {
                for i in inner..inner_loop_size {
                    let a_val = inp_ptr[i];
                    let mut_ref = unsafe { &mut *res_ptr.ptr.offset(i as isize) };
                    *mut_ref = op(*mut_ref, a_val);
                }
                update_prg2(prg1, shape_len, &mut inp_ptr, inp_strides, inp_shape);
            }
            update_prg3(prg2, shape_len, &mut inp_ptr, inp_strides, inp_shape);
            res_ptr.add(inner_loop_size as usize);
            prg1.iter_mut().for_each(|x| {
                *x = 0;
            });
        }
    }
}
