use tensor_macros::gen_fast_reduce_simd_helper;
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
        if prg[j] < shape[j] - 1 {
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
    inp_ptr.offset(num_largest_vecs as i64 * 16 * vec_size as i64);
    res_ptr.offset(num_largest_vecs as i64 * 16 * vec_size as i64);
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
