use rayon::iter::{
    IndexedParallelIterator,
    IntoParallelRefIterator,
    IntoParallelRefMutIterator,
    ParallelIterator,
};
use std::panic::Location;
use crate::backend::Cpu;
use tensor_traits::tensor::CommonBounds;
use tensor_common::shape_utils::predict_broadcast_shape;
use tensor_traits::tensor::TensorInfo;
use tensor_traits::tensor::TensorLike;
use crate::tensor_base::_Tensor;
use tensor_traits::tensor::TensorCreator;

#[cfg(not(feature = "simd"))]
#[cfg_attr(feature = "track_caller", track_caller)]
pub fn binary_fn<A, B, K, F>(lhs: &_Tensor<A>, rhs: &_Tensor<B>, f: F) -> anyhow::Result<_Tensor<K>>
    where A: CommonBounds, B: CommonBounds, K: CommonBounds, F: Fn(A, B) -> K + Sync + Send + Copy
{
    if lhs.size() == 1 {
        let val = lhs.as_raw()[0];
        let res = _Tensor::<K, Cpu>::empty(rhs.shape())?;
        res.as_raw_mut()
            .par_iter_mut()
            .zip(rhs.as_raw().par_iter())
            .for_each(|(a, &b)| {
                *a = f(val, b);
            });
        Ok(res)
    } else if rhs.size() == 1 {
        let val = rhs.as_raw()[0];
        let res = _Tensor::<K, Cpu>::empty(lhs.shape())?;
        res.as_raw_mut()
            .par_iter_mut()
            .zip(lhs.as_raw().par_iter())
            .for_each(|(a, &b)| {
                *a = f(b, val);
            });
        Ok(res)
    } else {
        if rhs.is_contiguous() && lhs.is_contiguous() && rhs.shape() == lhs.shape() {
            let res_shape = predict_broadcast_shape(lhs.shape(), rhs.shape(), Location::caller())?;
            let ret;
            ret = _Tensor::<K, Cpu>::empty(res_shape)?;
            ret.as_raw_mut()
                .par_iter_mut()
                .zip(lhs.as_raw().par_iter())
                .zip(rhs.as_raw().par_iter())
                .for_each(|((ret, &lhs), &rhs)| {
                    *ret = f(lhs, rhs);
                });
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

#[cfg(feature = "simd")]
#[cfg_attr(feature = "track_caller", track_caller)]
pub fn binary_fn<A, B, K, F>(lhs: &_Tensor<A>, rhs: &_Tensor<B>, f: F) -> anyhow::Result<_Tensor<K>>
    where A: CommonBounds, B: CommonBounds, K: CommonBounds, F: Fn(A, B) -> K + Sync + Send + Copy
{
    if lhs.size() == 1 {
        let val = lhs.as_raw()[0];
        let res = _Tensor::<K, Cpu>::empty(rhs.shape())?;
        res.as_raw_mut()
            .par_iter_mut()
            .zip(rhs.as_raw().par_iter())
            .for_each(|(a, &b)| {
                *a = f(val, b);
            });
        Ok(res)
    } else if rhs.size() == 1 {
        let val = rhs.as_raw()[0];
        let res = _Tensor::<K, Cpu>::empty(lhs.shape())?;
        res.as_raw_mut()
            .par_iter_mut()
            .zip(lhs.as_raw().par_iter())
            .for_each(|(a, &b)| {
                *a = f(b, val);
            });
        Ok(res)
    } else {
        if rhs.is_contiguous() && lhs.is_contiguous() && rhs.shape() == lhs.shape() {
            let res_shape = predict_broadcast_shape(lhs.shape(), rhs.shape(), Location::caller())?;
            let ret;
            ret = _Tensor::<K, Cpu>::empty(res_shape)?;
            ret.as_raw_mut()
                .par_iter_mut()
                .zip(lhs.as_raw().par_iter())
                .zip(rhs.as_raw().par_iter())
                .for_each(|((ret, &lhs), &rhs)| {
                    *ret = f(lhs, rhs);
                });
            Ok(ret)
        } else {
            todo!();
        }
    }
}

pub fn binary_fn_with_out<A, B, O, Q, K, F>(
    lhs: &_Tensor<A>,
    rhs: &_Tensor<B>,
    f: F,
    out: O,
    location: &'static Location<'static>
)
    -> anyhow::Result<_Tensor<K>>
    where
        A: CommonBounds,
        B: CommonBounds,
        O: TensorLike<Q, Output = _Tensor<K>> + TensorInfo<Q>,
        K: CommonBounds,
        Q: CommonBounds,
        F: Fn(A, B) -> K + Sync + Send + Copy
{
    if lhs.size() == 1 {
        let val = lhs.as_raw()[0];
        let ret;
        if out.size() * std::mem::size_of::<Q>() != rhs.size() * std::mem::size_of::<B>() {
            ret = _Tensor::<K, Cpu>::empty(rhs.shape())?;
        } else {
            ret = _Tensor::<K, Cpu>::empty(rhs.shape())?;
        }
        ret.as_raw_mut()
            .par_iter_mut()
            .zip(rhs.as_raw().par_iter())
            .for_each(|(a, &b)| {
                *a = f(val, b);
            });
        Ok(ret)
    } else if rhs.size() == 1 {
        let val = rhs.as_raw()[0];
        let ret;
        if out.size() * std::mem::size_of::<Q>() != lhs.size() * std::mem::size_of::<A>() {
            ret = _Tensor::<K, Cpu>::empty(lhs.shape())?;
        } else {
            ret = _Tensor::<K, Cpu>::empty(lhs.shape())?;
        }
        ret.as_raw_mut()
            .par_iter_mut()
            .zip(lhs.as_raw().par_iter())
            .for_each(|(a, &b)| {
                *a = f(b, val);
            });
        Ok(ret)
    } else {
        let res_shape = predict_broadcast_shape(lhs.shape(), rhs.shape(), location)?;
        let ret;
        let ret_size: usize = res_shape.iter().product::<i64>() as usize;
        if out.size() * std::mem::size_of::<Q>() != ret_size * std::mem::size_of::<A>() {
            ret = _Tensor::<K, Cpu>::empty(res_shape)?;
        } else {
            ret = _Tensor::<K, Cpu>::empty(res_shape)?;
        }
        if rhs.is_contiguous() && lhs.is_contiguous() && rhs.shape() == lhs.shape() {
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
        } else {
            ret.par_iter_mut()
                .zip(lhs.par_iter().zip(rhs.par_iter()))
                .for_each(|(res, (x, y))| {
                    *res = f(x, y);
                });
        }
        Ok(ret)
    }
}

#[cfg(feature = "simd")]
#[cfg_attr(feature = "track_caller", track_caller)]
pub fn binary_fn_with_out_simd<A, B, O, Q, K, F, F2>(
    lhs: &_Tensor<A>,
    rhs: &_Tensor<B>,
    f: F,
    f2: F2,
    out: O,
)
    -> anyhow::Result<_Tensor<K>>
    where
        A: CommonBounds,
        B: CommonBounds,
        O: TensorLike<Q, Output = _Tensor<K>> + TensorInfo<Q>,
        K: CommonBounds,
        Q: CommonBounds,
        F: Fn(A, B) -> K + Sync + Send + Copy,
        F2: Fn(<A as TypeCommon>::Vec, <B as TypeCommon>::Vec) -> <K as TypeCommon>::Vec +
        Sync +
        Send +
        Copy
{
    use rayon::slice::{ParallelSlice, ParallelSliceMut};

    if lhs.size() == 1 {
        let val = lhs.as_raw()[0];
        let val_vec = <A as TypeCommon>::Vec::splat(val);
        let res = if out.size() * std::mem::size_of::<Q>() != rhs.size() * std::mem::size_of::<B>() {
            _Tensor::<K, Cpu>::empty(rhs.shape())?
        } else {
            _Tensor::<K, Cpu>::empty(rhs.shape())?
        };
        if
            <A as TypeCommon>::Vec::SIZE == <B as TypeCommon>::Vec::SIZE &&
            <B as TypeCommon>::Vec::SIZE == <K as TypeCommon>::Vec::SIZE
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
                            <A as TypeCommon>::Vec::SIZE
                        );
                    }
                });
            if remain > 0 {
                res.as_raw_mut()
                    [res.size() - remain..].iter_mut()
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
                    a.iter_mut()
                        .zip(b.iter())
                        .for_each(|(a, b)| {
                            *a = f(val, *b);
                        });
                });
            let remain = res.size() % <K as TypeCommon>::Vec::SIZE;
            if remain > 0 {
                res.as_raw_mut()
                    [res.size() - remain..].iter_mut()
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
        let res = if out.size() * std::mem::size_of::<Q>() != lhs.size() * std::mem::size_of::<B>() {
            _Tensor::<K, Cpu>::empty(lhs.shape())?
        } else {
            _Tensor::<K, Cpu>::empty(lhs.shape())?
        };
        if
            <A as TypeCommon>::Vec::SIZE == <B as TypeCommon>::Vec::SIZE &&
            <B as TypeCommon>::Vec::SIZE == <K as TypeCommon>::Vec::SIZE
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
                            <A as TypeCommon>::Vec::SIZE
                        );
                    }
                });
            if remain > 0 {
                res.as_raw_mut()
                    [res.size() - remain..].iter_mut()
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
                    a.iter_mut()
                        .zip(lhs.iter())
                        .for_each(|(a, lhs)| {
                            *a = f(*lhs, val);
                        });
                });
            let remain = res.size() % <K as TypeCommon>::Vec::SIZE;
            if remain > 0 {
                res.as_raw_mut()
                    [res.size() - remain..].iter_mut()
                    .zip(lhs.as_raw()[res.size() - remain..].iter())
                    .for_each(|(a, lhs)| {
                        *a = f(*lhs, val);
                    });
            }
        }
        Ok(res)
    } else {
        if rhs.is_contiguous() && lhs.is_contiguous() && rhs.shape() == lhs.shape() {
            let ret = if out.size() * std::mem::size_of::<Q>() != rhs.size() * std::mem::size_of::<B>() {
                _Tensor::<K, Cpu>::empty(rhs.shape())?
            } else {
                _Tensor::<K, Cpu>::empty(rhs.shape())?
            };
            if
                <A as TypeCommon>::Vec::SIZE == <B as TypeCommon>::Vec::SIZE &&
                <B as TypeCommon>::Vec::SIZE == <K as TypeCommon>::Vec::SIZE
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
                                <K as TypeCommon>::Vec::SIZE
                            );
                        }
                    });
                if remain > 0 {
                    ret.as_raw_mut()
                        [ret.size() - remain..].iter_mut()
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
            if
                <A as TypeCommon>::Vec::SIZE == <B as TypeCommon>::Vec::SIZE &&
                <B as TypeCommon>::Vec::SIZE == <K as TypeCommon>::Vec::SIZE
            {
                let ret = lhs
                    .par_iter_simd()
                    .zip(rhs.par_iter_simd())
                    .strided_map(
                        |(x, y)| f(x, y),
                        |(x, y)| {
                            let x_ptr = x.as_ptr();
                            let y_ptr = y.as_ptr();
                            f2(
                                unsafe {
                                    <A as TypeCommon>::Vec::from_ptr(x_ptr)
                                },
                                unsafe {
                                    <B as TypeCommon>::Vec::from_ptr(y_ptr)
                                }
                            )
                        }
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


use tensor_types::dtype::TypeCommon;
use tensor_types::vectors::traits::*;
#[cfg(feature = "simd")]
#[cfg_attr(feature = "track_caller", track_caller)]
pub fn binary_fn_simd<A, B, K, F, F2>(
    lhs: &_Tensor<A>,
    rhs: &_Tensor<B>,
    f: F,
    f2: F2
)
    -> anyhow::Result<_Tensor<K>>
    where
        A: CommonBounds,
        B: CommonBounds,
        K: CommonBounds,
        F: Fn(A, B) -> K + Sync + Send + Copy,
        F2: Fn(<A as TypeCommon>::Vec, <B as TypeCommon>::Vec) -> <K as TypeCommon>::Vec +
            Sync +
            Send +
            Copy
{
    use rayon::slice::{ ParallelSlice, ParallelSliceMut };

    if lhs.size() == 1 {
        let val = lhs.as_raw()[0];
        let val_vec = <A as TypeCommon>::Vec::splat(val);
        let res = _Tensor::<K, Cpu>::empty(rhs.shape())?;
        if
            <A as TypeCommon>::Vec::SIZE == <B as TypeCommon>::Vec::SIZE &&
            <B as TypeCommon>::Vec::SIZE == <K as TypeCommon>::Vec::SIZE
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
                            <A as TypeCommon>::Vec::SIZE
                        );
                    }
                });
            if remain > 0 {
                res.as_raw_mut()
                    [res.size() - remain..].iter_mut()
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
                    a.iter_mut()
                        .zip(b.iter())
                        .for_each(|(a, b)| {
                            *a = f(val, *b);
                        });
                });
            let remain = res.size() % <K as TypeCommon>::Vec::SIZE;
            if remain > 0 {
                res.as_raw_mut()
                    [res.size() - remain..].iter_mut()
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
        let res = _Tensor::<K, Cpu>::empty(lhs.shape())?;
        if
            <A as TypeCommon>::Vec::SIZE == <B as TypeCommon>::Vec::SIZE &&
            <B as TypeCommon>::Vec::SIZE == <K as TypeCommon>::Vec::SIZE
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
                            <A as TypeCommon>::Vec::SIZE
                        );
                    }
                });
            if remain > 0 {
                res.as_raw_mut()
                    [res.size() - remain..].iter_mut()
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
                    a.iter_mut()
                        .zip(lhs.iter())
                        .for_each(|(a, lhs)| {
                            *a = f(*lhs, val);
                        });
                });
            let remain = res.size() % <K as TypeCommon>::Vec::SIZE;
            if remain > 0 {
                res.as_raw_mut()
                    [res.size() - remain..].iter_mut()
                    .zip(lhs.as_raw()[res.size() - remain..].iter())
                    .for_each(|(a, lhs)| {
                        *a = f(*lhs, val);
                    });
            }
        }
        Ok(res)
    } else {
        if rhs.is_contiguous() && lhs.is_contiguous() && rhs.shape() == lhs.shape() {
            let ret;
            ret = _Tensor::<K, Cpu>::empty(rhs.shape())?;
            if
                <A as TypeCommon>::Vec::SIZE == <B as TypeCommon>::Vec::SIZE &&
                <B as TypeCommon>::Vec::SIZE == <K as TypeCommon>::Vec::SIZE
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
                                <K as TypeCommon>::Vec::SIZE
                            );
                        }
                    });
                if remain > 0 {
                    ret.as_raw_mut()
                        [ret.size() - remain..].iter_mut()
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
            if
                <A as TypeCommon>::Vec::SIZE == <B as TypeCommon>::Vec::SIZE &&
                <B as TypeCommon>::Vec::SIZE == <K as TypeCommon>::Vec::SIZE
            {
                let ret = lhs
                    .par_iter_simd()
                    .zip(rhs.par_iter_simd())
                    .strided_map(
                        |(x, y)| f(x, y),
                        |(x, y)| {
                            let x_ptr = x.as_ptr();
                            let y_ptr = y.as_ptr();
                            f2(
                                unsafe {
                                    <A as TypeCommon>::Vec::from_ptr(x_ptr)
                                },
                                unsafe {
                                    <B as TypeCommon>::Vec::from_ptr(y_ptr)
                                }
                            )
                        }
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
