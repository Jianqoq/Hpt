use crate::backend::Cpu;
use crate::tensor_base::_Tensor;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use tensor_traits::tensor::CommonBounds;
use tensor_traits::tensor::TensorCreator;
use tensor_traits::tensor::TensorInfo;

use std::borrow::Borrow;
use tensor_types::dtype::TypeCommon;

#[cfg_attr(feature = "track_caller", track_caller)]
pub fn binary_fn_with_out_simd<A, B, O, Q, K, F, F2>(
    lhs: &_Tensor<A>,
    rhs: &_Tensor<B>,
    f: F,
    f2: F2,
    out: Option<O>,
) -> anyhow::Result<_Tensor<K>>
where
    A: CommonBounds,
    B: CommonBounds,
    O: Borrow<_Tensor<Q>>,
    K: CommonBounds,
    Q: CommonBounds,
    F: Fn(A, B) -> K + Sync + Send + Copy,
    F2: Fn(<A as TypeCommon>::Vec, <B as TypeCommon>::Vec) -> <K as TypeCommon>::Vec
        + Sync
        + Send
        + Copy,
{
    use rayon::slice::{ParallelSlice, ParallelSliceMut};
    use tensor_types::traits::*;
    if lhs.size() == 1 {
        let val = lhs.as_raw()[0];
        let val_vec = <A as TypeCommon>::Vec::splat(val);
        let res = if let Some(out) = out {
            if out.borrow().size() * size_of::<Q>() != rhs.size() * size_of::<B>() {
                _Tensor::<K, Cpu>::empty(rhs.shape())?
            } else {
                out.borrow().static_cast::<K>()?
            }
        } else {
            _Tensor::<K, Cpu>::empty(rhs.shape())?
        };
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
                res.as_raw_mut()[res.size() - remain..]
                    .iter_mut()
                    .zip(rhs.as_raw()[res.size() - remain..].iter())
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
                res.as_raw_mut()[res.size() - remain..]
                    .iter_mut()
                    .zip(rhs.as_raw()[res.size() - remain..].iter())
                    .for_each(|(a, b)| {
                        *a = f(val, *b);
                    });
            }
        }
        Ok(res)
    } else if rhs.size() == 1 {
        let val = rhs.as_raw()[0];
        let val_vec = <B as TypeCommon>::Vec::splat(val);
        let res = if let Some(out) = out {
            if out.borrow().size() * size_of::<Q>() != lhs.size() * size_of::<B>() {
                _Tensor::<K, Cpu>::empty(lhs.shape())?
            } else {
                _Tensor::<K, Cpu>::empty(lhs.shape())?
            }
        } else {
            _Tensor::<K, Cpu>::empty(lhs.shape())?
        };
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
                res.as_raw_mut()[res.size() - remain..]
                    .iter_mut()
                    .zip(lhs.as_raw()[res.size() - remain..].iter())
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
                res.as_raw_mut()[res.size() - remain..]
                    .iter_mut()
                    .zip(lhs.as_raw()[res.size() - remain..].iter())
                    .for_each(|(a, lhs)| {
                        *a = f(*lhs, val);
                    });
            }
        }
        Ok(res)
    } else {
        if rhs.is_contiguous() && lhs.is_contiguous() && rhs.shape() == lhs.shape() {
            let ret = if let Some(out) = out {
                if out.borrow().size() * size_of::<Q>() != rhs.size() * size_of::<B>() {
                    _Tensor::<K, Cpu>::empty(rhs.shape())?
                } else {
                    out.borrow().static_cast::<K>()?
                }
            } else {
                _Tensor::<K, Cpu>::empty(rhs.shape())?
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
                    ret.as_raw_mut()[ret.size() - remain..]
                        .iter_mut()
                        .zip(lhs.as_raw()[ret.size() - remain..].iter())
                        .zip(rhs.as_raw()[ret.size() - remain..].iter())
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
            if <A as TypeCommon>::Vec::SIZE == <B as TypeCommon>::Vec::SIZE
                && <B as TypeCommon>::Vec::SIZE == <K as TypeCommon>::Vec::SIZE
            {
                let ret = lhs
                    .par_iter_simd()
                    .zip(rhs.par_iter_simd())
                    .strided_map(
                        |(res, (x, y))| {
                            *res = f(x, y);
                        },
                        |(res, (x, y))| {
                            let x_ptr = x.as_ptr();
                            let y_ptr = y.as_ptr();
                            *res = f2(unsafe { <A as TypeCommon>::Vec::from_ptr(x_ptr) }, unsafe {
                                <B as TypeCommon>::Vec::from_ptr(y_ptr)
                            });
                        },
                    )
                    .collect::<_Tensor<K>>();
                Ok(ret)
            } else {
                let ret = lhs
                    .par_iter()
                    .zip(rhs.par_iter())
                    .strided_map(|(x, y)| f(x, y))
                    .collect::<_Tensor<K>>();
                Ok(ret)
            }
        }
    }
}