use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use std::panic::Location;
use crate::backend::Cpu;
use tensor_traits::tensor::CommonBounds;
use tensor_common::shape_utils::predict_broadcast_shape;
use tensor_traits::tensor::TensorInfo;
use tensor_traits::tensor::TensorLike;
use crate::tensor_base::_Tensor;
use tensor_traits::tensor::TensorCreator;

macro_rules! impl_binary_fn {
    ($tensor_type:ident, $func_name:ident) => {
        pub fn $func_name<A, B, K, F>(
            lhs: &$tensor_type<A>,
            rhs: &$tensor_type<B>,
            f: F,
        ) -> anyhow::Result<$tensor_type<K>>
        where
            A: CommonBounds,
            B: CommonBounds,
            K: CommonBounds,
            F: Fn(A, B) -> K + Sync + Send + Copy,
        {
            if lhs.size == 1 {
                let val = lhs.as_raw()[0];
                let res = $tensor_type::empty(rhs.shape().inner()).unwrap();
                res.as_raw_mut()
                    .par_iter_mut()
                    .zip(rhs.as_raw().par_iter())
                    .for_each(|(a, &b)| {
                        *a = f(val, b);
                    });
                Ok(res)
            } else if rhs.size == 1 {
                let val = rhs.as_raw()[0];
                let res = $tensor_type::empty(lhs.shape().inner()).unwrap();
                res.as_raw_mut()
                    .par_iter_mut()
                    .zip(lhs.as_raw().par_iter())
                    .for_each(|(a, &b)| {
                        *a = f(b, val);
                    });
                Ok(res)
            } else {
                if rhs.is_contiguous() && lhs.is_contiguous() && rhs.shape() == lhs.shape() {
                    let res_shape = predict_broadcast_shape(lhs.shape(), rhs.shape())?;
                    let ret;
                    ret = $tensor_type::empty(res_shape.inner()).unwrap();
                    let min_len: usize =
                        ret.size / (((rayon::current_num_threads() as f64) * 1.3) as usize);
                    ret.as_raw_mut()
                        .par_iter_mut()
                        .with_min_len(min_len)
                        .zip(lhs.as_raw().par_iter().with_min_len(min_len))
                        .zip(rhs.as_raw().par_iter().with_min_len(min_len))
                        .for_each(|((ret, &lhs), &rhs)| {
                            *ret = f(lhs, rhs);
                        });
                    Ok(ret)
                } else {
                    let ret = lhs
                        .strided_par_iter()
                        .zip(rhs.strided_par_iter())
                        .strided_map(|(x, y)| f(x, y))
                        .collect::<$tensor_type<K>>();
                    Ok(ret)
                }
            }
        }
    };
    ($tensor_type:ident, $func_name:ident, out) => {
        pub fn $func_name<A, B, O, Q, K, F>(
            lhs: &$tensor_type<A>,
            rhs: &$tensor_type<B>,
            f: F,
            out: O, location: &'static Location<'static>
        ) -> anyhow::Result<$tensor_type<K>>
        where
            A: CommonBounds,
            B: CommonBounds,
            O: TensorLike<Q, Output = $tensor_type<K>> + TensorInfo<Q>,
            K: CommonBounds,
            Q: CommonBounds,
            F: Fn(A, B) -> K + Sync + Send + Copy,
        {
            if lhs.size() == 1 {
                let val = lhs.as_raw()[0];
                let ret;
                if out.size() * std::mem::size_of::<Q>() != rhs.size() * std::mem::size_of::<B>() {
                    ret = $tensor_type::<K, Cpu>::empty(rhs.shape()).unwrap();
                } else {
                    ret = $tensor_type::<K, Cpu>::empty(rhs.shape()).unwrap();
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
                    ret = $tensor_type::<K, Cpu>::empty(lhs.shape()).unwrap();
                } else {
                    ret = $tensor_type::<K, Cpu>::empty(lhs.shape()).unwrap();
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
                    ret = $tensor_type::<K, Cpu>::empty(res_shape).unwrap();
                } else {
                    ret = $tensor_type::<K, Cpu>::empty(res_shape).unwrap();
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
    };
}

#[cfg_attr(feature = "track_caller", track_caller)]
pub fn binary_fn<A, B, K, F>(lhs: &_Tensor<A>, rhs: &_Tensor<B>, f: F) -> anyhow::Result<_Tensor<K>>
where
    A: CommonBounds,
    B: CommonBounds,
    K: CommonBounds,
    F: Fn(A, B) -> K + Sync + Send + Copy,
{
    if lhs.size() == 1 {
        let val = lhs.as_raw()[0];
        let res = _Tensor::<K, Cpu>::empty(rhs.shape()).unwrap();
        res.as_raw_mut()
            .par_iter_mut()
            .zip(rhs.as_raw().par_iter())
            .for_each(|(a, &b)| {
                *a = f(val, b);
            });
        Ok(res)
    } else if rhs.size() == 1 {
        let val = rhs.as_raw()[0];
        let res = _Tensor::<K, Cpu>::empty(lhs.shape()).unwrap();
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
            ret = _Tensor::<K, Cpu>::empty(res_shape).unwrap();
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
impl_binary_fn!(_Tensor, binary_fn_with_out, out);
