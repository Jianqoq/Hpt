use crate::tensor_base::_Tensor;
use crate::Cpu;
use crate::Tensor;
use hpt_allocator::traits::Allocator;
use hpt_allocator::traits::AllocatorOutputRetrive;
use hpt_common::error::base::TensorError;
use hpt_common::error::shape::ShapeError;
use hpt_iterator::iterator_traits::ParStridedIteratorZip;
use hpt_iterator::TensorIterator;
use hpt_traits::tensor::CommonBounds;
use hpt_traits::tensor::TensorCreator;
use hpt_traits::tensor::TensorInfo;
use hpt_traits::TensorLike;
use hpt_types::dtype::TypeCommon;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use std::borrow::Borrow;

/// Performs a binary operation on two tensors with optional SIMD optimization and an output tensor.
///
/// This method applies a binary function element-wise on two tensors (`lhs` and `rhs`) and returns
/// a new tensor with the result. Optionally, SIMD (Single Instruction, Multiple Data) can be used
/// for vectorized operations if the sizes of the underlying data vectors align. Additionally,
/// the user can provide an output tensor to store the result, allowing in-place computations
/// and reducing memory allocations.
///
/// # Arguments
///
/// * `lhs` - A reference to the left-hand side tensor involved in the binary operation.
/// * `rhs` - A reference to the right-hand side tensor involved in the binary operation.
/// * `f` - A binary function applied to elements of the tensors during the operation. This function
///   is used when SIMD is not applicable.
/// * `f2` - A binary function that operates on vectorized data (SIMD). This function is used when
///   SIMD is applicable.
/// * `out` - An optional output tensor that, if provided, will store the result of the operation.
///   If not provided, a new tensor will be created to hold the result.
///
/// # Returns
///
/// Returns a `Result` containing a new tensor with the result of the binary operation. If any error occurs
/// (e.g., shape mismatch or allocation issues), an `anyhow::Result` with an error message is returned.
///
/// # SIMD Optimization
///
/// If the vector sizes of the input tensors match and SIMD is enabled, the `f2` function is applied to
/// perform vectorized operations for faster computation. If not, the scalar function `f` is applied to each element.
#[track_caller]
pub(crate) fn binary_fn_with_out_simd<A, B, O, K, F, F2, const DEVICE: usize, Al>(
    lhs: &_Tensor<A, Cpu, DEVICE, Al>,
    rhs: &_Tensor<B, Cpu, DEVICE, Al>,
    f: F,
    f2: F2,
    out: Option<O>,
) -> std::result::Result<_Tensor<K, Cpu, DEVICE, Al>, TensorError>
where
    A: CommonBounds,
    B: CommonBounds,
    O: Borrow<_Tensor<K, Cpu, DEVICE, Al>>,
    K: CommonBounds,
    F: Fn(A, B) -> K + Sync + Send + Copy,
    F2: Fn(<A as TypeCommon>::Vec, <B as TypeCommon>::Vec) -> <K as TypeCommon>::Vec
        + Sync
        + Send
        + Copy,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    use hpt_types::traits::*;
    use rayon::slice::{ParallelSlice, ParallelSliceMut};
    if lhs.size() == 1 {
        let val = lhs.as_raw()[0];
        let val_vec = <A as TypeCommon>::Vec::splat(val);
        let mut res = if let Some(out) = out {
            ShapeError::check_inplace_out_layout_valid(rhs.shape(), &out.borrow().layout())?;
            let out: &_Tensor<K, Cpu, DEVICE, Al> = out.borrow();
            out.clone()
        } else {
            _Tensor::<K, Cpu, DEVICE, Al>::empty(rhs.shape())?
        };
        if rhs.is_contiguous() {
            if <A as TypeCommon>::Vec::SIZE == <B as TypeCommon>::Vec::SIZE
                && <B as TypeCommon>::Vec::SIZE == <K as TypeCommon>::Vec::SIZE
            {
                let remain = res.size() % <A as TypeCommon>::Vec::SIZE;
                res.as_raw_mut()
                    .par_chunks_exact_mut(<A as TypeCommon>::Vec::SIZE)
                    .zip(rhs.as_raw().par_chunks_exact(<A as TypeCommon>::Vec::SIZE))
                    .for_each(|(a, b)| {
                        let inp = unsafe { <B as TypeCommon>::Vec::from_ptr(b.as_ptr()) };
                        let res: *const K = f2(val_vec, inp).as_ptr();
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                res,
                                a.as_mut_ptr(),
                                <A as TypeCommon>::Vec::SIZE,
                            );
                        }
                    });
                if remain > 0 {
                    let ret_size = res.size();
                    res.as_raw_mut()[ret_size - remain..]
                        .iter_mut()
                        .zip(rhs.as_raw()[ret_size - remain..].iter())
                        .for_each(|(a, b)| {
                            *a = f(val, *b);
                        });
                }
            } else {
                res.as_raw_mut()
                    .par_chunks_exact_mut(<K as TypeCommon>::Vec::SIZE)
                    .zip(rhs.as_raw().par_chunks_exact(<K as TypeCommon>::Vec::SIZE))
                    .for_each(|(a, b)| {
                        a.iter_mut().zip(b.iter()).for_each(|(a, b)| {
                            *a = f(val, *b);
                        });
                    });
                let remain = res.size() % <K as TypeCommon>::Vec::SIZE;
                if remain > 0 {
                    let ret_size = res.size();
                    res.as_raw_mut()[ret_size - remain..]
                        .iter_mut()
                        .zip(rhs.as_raw()[ret_size - remain..].iter())
                        .for_each(|(a, b)| {
                            *a = f(val, *b);
                        });
                }
            }
        } else {
            res.par_iter_mut().zip(rhs.par_iter()).for_each(|(a, b)| {
                *a = f(val, b);
            });
        }
        Ok(res)
    } else if rhs.size() == 1 {
        let val = rhs.as_raw()[0];
        let val_vec = <B as TypeCommon>::Vec::splat(val);
        let mut res = if let Some(out) = out {
            ShapeError::check_inplace_out_layout_valid(lhs.shape(), &out.borrow().layout())?;
            let out: &_Tensor<K, Cpu, DEVICE, Al> = out.borrow();
            out.clone()
        } else {
            _Tensor::<K, Cpu, DEVICE, Al>::empty(lhs.shape())?
        };
        if lhs.is_contiguous() {
            if <A as TypeCommon>::Vec::SIZE == <B as TypeCommon>::Vec::SIZE
                && <B as TypeCommon>::Vec::SIZE == <K as TypeCommon>::Vec::SIZE
            {
                let remain = res.size() % <A as TypeCommon>::Vec::SIZE;
                res.as_raw_mut()
                    .par_chunks_exact_mut(<A as TypeCommon>::Vec::SIZE)
                    .zip(lhs.as_raw().par_chunks_exact(<A as TypeCommon>::Vec::SIZE))
                    .for_each(|(a, lhs)| {
                        let inp = unsafe { <A as TypeCommon>::Vec::from_ptr(lhs.as_ptr()) };
                        let res: *const K = f2(inp, val_vec).as_ptr();
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                res,
                                a.as_mut_ptr(),
                                <A as TypeCommon>::Vec::SIZE,
                            );
                        }
                    });
                if remain > 0 {
                    let ret_size = res.size();
                    res.as_raw_mut()[ret_size - remain..]
                        .iter_mut()
                        .zip(lhs.as_raw()[ret_size - remain..].iter())
                        .for_each(|(a, lhs)| {
                            *a = f(*lhs, val);
                        });
                }
            } else {
                res.as_raw_mut()
                    .par_chunks_exact_mut(<K as TypeCommon>::Vec::SIZE)
                    .zip(lhs.as_raw().par_chunks_exact(<K as TypeCommon>::Vec::SIZE))
                    .for_each(|(a, lhs)| {
                        a.iter_mut().zip(lhs.iter()).for_each(|(a, lhs)| {
                            *a = f(*lhs, val);
                        });
                    });
                let remain = res.size() % <K as TypeCommon>::Vec::SIZE;
                if remain > 0 {
                    let ret_size = res.size();
                    res.as_raw_mut()[ret_size - remain..]
                        .iter_mut()
                        .zip(lhs.as_raw()[ret_size - remain..].iter())
                        .for_each(|(a, lhs)| {
                            *a = f(*lhs, val);
                        });
                }
            }
        } else {
            res.par_iter_mut().zip(lhs.par_iter()).for_each(|(a, lhs)| {
                *a = f(lhs, val);
            });
        }
        Ok(res)
    } else {
        if rhs.is_contiguous() && lhs.is_contiguous() && rhs.shape() == lhs.shape() {
            let mut ret = if let Some(out) = out {
                ShapeError::check_inplace_out_layout_valid(rhs.shape(), &out.borrow().layout())?;
                let out: &_Tensor<K, Cpu, DEVICE, Al> = out.borrow();
                out.clone()
            } else {
                _Tensor::<K, Cpu, DEVICE, Al>::empty(rhs.shape())?
            };
            if <A as TypeCommon>::Vec::SIZE == <B as TypeCommon>::Vec::SIZE
                && <B as TypeCommon>::Vec::SIZE == <K as TypeCommon>::Vec::SIZE
            {
                let remain = ret.size() % <K as TypeCommon>::Vec::SIZE;
                ret.as_raw_mut()
                    .par_chunks_exact_mut(<K as TypeCommon>::Vec::SIZE)
                    .zip(lhs.as_raw().par_chunks_exact(<K as TypeCommon>::Vec::SIZE))
                    .zip(rhs.as_raw().par_chunks_exact(<K as TypeCommon>::Vec::SIZE))
                    .for_each(|((ret, lhs), rhs)| {
                        let a = unsafe { <A as TypeCommon>::Vec::from_ptr(lhs.as_ptr()) };
                        let b = unsafe { <B as TypeCommon>::Vec::from_ptr(rhs.as_ptr()) };
                        let res = f2(a, b);
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                res.as_ptr(),
                                ret.as_mut_ptr(),
                                <K as TypeCommon>::Vec::SIZE,
                            );
                        }
                    });
                if remain > 0 {
                    let ret_size = ret.size();
                    ret.as_raw_mut()[ret_size - remain..]
                        .iter_mut()
                        .zip(lhs.as_raw()[ret_size - remain..].iter())
                        .zip(rhs.as_raw()[ret_size - remain..].iter())
                        .for_each(|((a, &lhs), &rhs)| {
                            *a = f(lhs, rhs);
                        });
                }
            } else {
                let min_len: usize =
                    ret.size() / (((rayon::current_num_threads() as f64) * 1.3) as usize);
                ret.as_raw_mut()
                    .par_iter_mut()
                    .with_min_len(min_len)
                    .zip(lhs.as_raw().par_iter().with_min_len(min_len))
                    .zip(rhs.as_raw().par_iter().with_min_len(min_len))
                    .for_each(|((ret, &lhs), &rhs)| {
                        *ret = f(lhs, rhs);
                    });
            }
            Ok(ret)
        } else {
            let ret = lhs
                .par_iter()
                .zip(rhs.par_iter())
                .strided_map(|(res, (x, y))| *res = f(x, y))
                .collect::<_Tensor<K, Cpu, DEVICE, Al>>();
            Ok(ret)
        }
    }
}

/// Perform binary operation with output tensor
#[track_caller]
pub fn binary_with_out<A, B, O, K, F, F2, const DEVICE: usize, Al>(
    lhs: &Tensor<A, Cpu, DEVICE, Al>,
    rhs: &Tensor<B, Cpu, DEVICE, Al>,
    f: F,
    f2: F2,
    out: Option<O>,
) -> std::result::Result<Tensor<K, Cpu, DEVICE, Al>, TensorError>
where
    A: CommonBounds,
    B: CommonBounds,
    O: Borrow<Tensor<K, Cpu, DEVICE, Al>>,
    K: CommonBounds,
    F: Fn(A, B) -> K + Sync + Send + Copy,
    F2: Fn(<A as TypeCommon>::Vec, <B as TypeCommon>::Vec) -> <K as TypeCommon>::Vec
        + Sync
        + Send
        + Copy,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    let out: Option<_Tensor<K, Cpu, DEVICE, Al>> = out.map(|x| x.borrow().inner.as_ref().clone());
    Ok(binary_fn_with_out_simd(lhs.inner.as_ref(), rhs.inner.as_ref(), f, f2, out)?.into())
}

#[track_caller]
pub(crate) fn binary_fn_with_out_simd_3<A, B, C, O, K, F, F2, const DEVICE: usize, Al>(
    a: &_Tensor<A, Cpu, DEVICE, Al>,
    b: &_Tensor<B, Cpu, DEVICE, Al>,
    c: &_Tensor<C, Cpu, DEVICE, Al>,
    f: F,
    f2: F2,
    out: Option<O>,
) -> std::result::Result<_Tensor<K, Cpu, DEVICE, Al>, TensorError>
where
    A: CommonBounds,
    B: CommonBounds,
    C: CommonBounds,
    O: Borrow<_Tensor<K, Cpu, DEVICE, Al>>,
    K: CommonBounds,
    F: Fn(A, B, C) -> K + Sync + Send + Copy,
    F2: Fn(
            <A as TypeCommon>::Vec,
            <B as TypeCommon>::Vec,
            <C as TypeCommon>::Vec,
        ) -> <K as TypeCommon>::Vec
        + Sync
        + Send
        + Copy,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    use hpt_types::traits::*;
    use rayon::slice::{ParallelSlice, ParallelSliceMut};
    if b.is_contiguous() && a.is_contiguous() && b.shape() == a.shape() {
        let mut ret = if let Some(out) = out {
            ShapeError::check_inplace_out_layout_valid(b.shape(), &out.borrow().layout())?;
            let out: &_Tensor<K, Cpu, DEVICE, Al> = out.borrow();
            out.clone()
        } else {
            _Tensor::<K, Cpu, DEVICE, Al>::empty(b.shape())?
        };
        if <A as TypeCommon>::Vec::SIZE == <B as TypeCommon>::Vec::SIZE
            && <B as TypeCommon>::Vec::SIZE == <K as TypeCommon>::Vec::SIZE
            && <B as TypeCommon>::Vec::SIZE == <C as TypeCommon>::Vec::SIZE
        {
            let remain = ret.size() % <K as TypeCommon>::Vec::SIZE;
            ret.as_raw_mut()
                .par_chunks_exact_mut(<K as TypeCommon>::Vec::SIZE)
                .zip(a.as_raw().par_chunks_exact(<K as TypeCommon>::Vec::SIZE))
                .zip(b.as_raw().par_chunks_exact(<K as TypeCommon>::Vec::SIZE))
                .zip(c.as_raw().par_chunks_exact(<K as TypeCommon>::Vec::SIZE))
                .for_each(|(((ret, a), b), c)| {
                    let a = unsafe { <A as TypeCommon>::Vec::from_ptr(a.as_ptr()) };
                    let b = unsafe { <B as TypeCommon>::Vec::from_ptr(b.as_ptr()) };
                    let c = unsafe { <C as TypeCommon>::Vec::from_ptr(c.as_ptr()) };
                    let res = f2(a, b, c);
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            res.as_ptr(),
                            ret.as_mut_ptr(),
                            <K as TypeCommon>::Vec::SIZE,
                        );
                    }
                });
            if remain > 0 {
                let ret_size = ret.size();
                ret.as_raw_mut()[ret_size - remain..]
                    .iter_mut()
                    .zip(a.as_raw()[ret_size - remain..].iter())
                    .zip(b.as_raw()[ret_size - remain..].iter())
                    .zip(c.as_raw()[ret_size - remain..].iter())
                    .for_each(|(((a, &lhs), &rhs), &c)| {
                        *a = f(lhs, rhs, c);
                    });
            }
        } else {
            let min_len: usize =
                ret.size() / (((rayon::current_num_threads() as f64) * 1.3) as usize);
            ret.as_raw_mut()
                .par_iter_mut()
                .with_min_len(min_len)
                .zip(a.as_raw().par_iter().with_min_len(min_len))
                .zip(b.as_raw().par_iter().with_min_len(min_len))
                .zip(c.as_raw().par_iter().with_min_len(min_len))
                .for_each(|(((ret, &lhs), &rhs), &c)| {
                    *ret = f(lhs, rhs, c);
                });
        }
        Ok(ret)
    } else {
        let ret = a
            .par_iter()
            .zip(b.par_iter())
            .zip(c.par_iter())
            .strided_map(|(res, ((x, y), z))| *res = f(x, y, z))
            .collect::<_Tensor<K, Cpu, DEVICE, Al>>();
        Ok(ret)
    }
}
