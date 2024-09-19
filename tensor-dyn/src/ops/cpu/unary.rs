use std::borrow::Borrow;

use crate::backend::Cpu;
use crate::tensor_base::_Tensor;
use crate::THREAD_POOL;
use rayon::iter::{
    IndexedParallelIterator, ParallelIterator,
};
use rayon::slice::{ParallelSlice, ParallelSliceMut};
use tensor_common::err_handler::ErrHandler;
use tensor_common::shape_utils::mt_intervals;
use tensor_traits::tensor::TensorCreator;
use tensor_traits::tensor::{CommonBounds, TensorInfo, TensorLike};
use tensor_traits::{FloatUaryOps, Neg, NormalUaryOps};
use tensor_types::dtype::TypeCommon;
use tensor_types::into_scalar::IntoScalar;
use tensor_types::type_promote::{Eval, FloatOutBinary, FloatOutUnary, NormalOut};
use tensor_types::vectors::traits::*;
use threadpool::ThreadPool;

pub(crate) fn uary_fn_with_out_simd<A, O, K, F, F2>(
    inp: &_Tensor<A, Cpu>,
    f: F,
    f2: F2,
    out: Option<O>,
) -> anyhow::Result<_Tensor<K, Cpu>>
where
    A: CommonBounds,
    K: CommonBounds,
    O: Borrow<_Tensor<K, Cpu>>,
    F: Fn(<A as TypeCommon>::Vec) -> <K as TypeCommon>::Vec + Sync + Send,
    F2: Fn(A) -> K + Sync + Send,
{
    let ret = if let Some(out) = out {
        if out.borrow().size() * size_of::<K>() == inp.size() * size_of::<A>() {
            out.borrow().static_cast()?
        } else {
            _Tensor::<K, Cpu>::empty(inp.shape())?
        }
    } else {
        _Tensor::<K, Cpu>::empty(inp.shape())?
    };
    if inp.parent().is_some() {
        ret.par_iter_mut_simd()
            .zip(inp.par_iter_simd())
            .for_each(|(a, b)| {
                *a = f2(b);
            });
        return Ok(ret);
    }
    let per_thread_len = ret.size() / rayon::current_num_threads();
    let per_thread_remain = per_thread_len % K::Vec::SIZE;
    let total_remain = rayon::current_num_threads() * per_thread_remain
        + (ret.size() % rayon::current_num_threads());
    let per_thread_real_len = per_thread_len - per_thread_remain;
    if per_thread_real_len > 0 {
        ret.as_raw_mut()
            .par_chunks_exact_mut(per_thread_real_len)
            .zip(inp.as_raw().par_chunks_exact(per_thread_real_len))
            .for_each(|(ret, lhs)| {
                assert_eq!(lhs.len() % A::Vec::SIZE, 0);
                assert_eq!(ret.len() % K::Vec::SIZE, 0);
                ret.chunks_exact_mut(A::Vec::SIZE)
                    .zip(lhs.chunks_exact(A::Vec::SIZE))
                    .for_each(|(ret, lhs)| {
                        let a = unsafe { A::Vec::from_ptr(lhs.as_ptr()) };
                        let res = f(a);
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                res.as_ptr(),
                                ret.as_mut_ptr(),
                                K::Vec::SIZE,
                            );
                        }
                    });
            });
    }
    if total_remain > 0 {
        ret.as_raw_mut()[ret.size() - total_remain..]
            .iter_mut()
            .zip(inp.as_raw()[ret.size() - total_remain..].iter())
            .for_each(|(a, &lhs)| {
                *a = f2(lhs);
            });
    }
    Ok(ret)
}

pub(crate) type FloatUnaryType<T> = <T as FloatOutUnary>::Output;
pub(crate) type FloatBinaryType<T> = <T as FloatOutBinary>::Output;

impl<T> FloatUaryOps for _Tensor<T>
where
    T: FloatOutUnary<Base = FloatUnaryType<T>> + CommonBounds,
    FloatUnaryType<T>: CommonBounds,
    f64: IntoScalar<<T as FloatOutUnary>::Output>,
    T::Vec:
        FloatOutUnary<Output = <FloatUnaryType<T> as TypeCommon>::Vec, Base = FloatUnaryType<T>>,
    <FloatUnaryType<T> as TypeCommon>::Vec: Send + Copy + Sync,
{
    type Output = _Tensor<FloatUnaryType<T>>;

    type InplaceOutput = _Tensor<FloatUnaryType<T>>;

    type OutputMeta = FloatUnaryType<T>;

    fn sin(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._sin(),
            |x| x._sin(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn cos(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._cos(),
            |x| x._cos(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn tan(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._tan(),
            |x| x._tan(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn asin(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._asin(),
            |x| x._asin(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn acos(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._acos(),
            |x| x._acos(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn atan(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._atan(),
            |x| x._atan(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn sinh(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._sinh(),
            |x| x._sinh(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn cosh(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._cosh(),
            |x| x._cosh(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn erf(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._erf(),
            |x| x._erf(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn tanh(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._tanh(),
            |x| x._tanh(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn asinh(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._asinh(),
            |x| x._asinh(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn acosh(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._acosh(),
            |x| x._acosh(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn atanh(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._atanh(),
            |x| x._atanh(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn sin_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<_Tensor<FloatUnaryType<T>>>,
    {
        uary_fn_with_out_simd(self, |x| x._sin(), |x| x._sin(), Some(out))
    }

    fn cos_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._cos(), |x| x._cos(), Some(out))
    }

    fn tan_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._tan(), |x| x._tan(), Some(out))
    }

    fn asin_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._asin(), |x| x._asin(), Some(out))
    }

    fn acos_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._acos(), |x| x._acos(), Some(out))
    }

    fn atan_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._atan(), |x| x._atan(), Some(out))
    }

    fn sinh_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._sinh(), |x| x._sinh(), Some(out))
    }

    fn cosh_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._cosh(), |x| x._cosh(), Some(out))
    }

    fn tanh_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._tanh(), |x| x._tanh(), Some(out))
    }

    fn asinh_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._asinh(), |x| x._asinh(), Some(out))
    }

    fn acosh_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._acosh(), |x| x._acosh(), Some(out))
    }

    fn atanh_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._atanh(), |x| x._atanh(), Some(out))
    }

    fn exp(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._exp(),
            |x| x._exp(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn exp_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._exp(), |x| x._exp(), Some(out))
    }

    fn exp2(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._exp2(),
            |x| x._exp2(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn exp2_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._exp2(), |x| x._exp2(), Some(out))
    }

    fn sqrt(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._sqrt(),
            |x| x._sqrt(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn sqrt_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._sqrt(), |x| x._sqrt(), Some(out))
    }

    fn recip(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._recip(),
            |x| x._recip(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn recip_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._recip(), |x| x._recip(), Some(out))
    }

    fn ln(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._ln(),
            |x| x._ln(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn ln_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._ln(), |x| x._ln(), Some(out))
    }

    fn log2(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._log2(),
            |x| x._log2(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn log2_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._log2(), |x| x._log2(), Some(out))
    }

    fn log10(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._log10(),
            |x| x._log10(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn log10_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._log10(), |x| x._log10(), Some(out))
    }

    fn celu(&self, alpha: FloatUnaryType<T>) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._celu(alpha),
            |x| x._celu(alpha),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn celu_<U>(
        &self,
        alpha: FloatUnaryType<T>,
        out: U,
    ) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._celu(alpha), |x| x._celu(alpha), Some(out))
    }

    fn sigmoid(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._sigmoid(),
            |x| x._sigmoid(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn sigmoid_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._sigmoid(), |x| x._sigmoid(), Some(out))
    }

    fn elu(&self, alpha: FloatUnaryType<T>) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._elu(alpha),
            |x| x._elu(alpha),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn elu_<U>(
        &self,
        alpha: FloatUnaryType<T>,
        out: U,
    ) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._elu(alpha), |x| x._elu(alpha), Some(out))
    }

    fn leaky_relu(&self, alpha: FloatUnaryType<T>) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._leaky_relu(alpha),
            |x| x._leaky_relu(alpha),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn leaky_relu_<U>(
        &self,
        alpha: FloatUnaryType<T>,
        out: U,
    ) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            |x| x._leaky_relu(alpha),
            |x| x._leaky_relu(alpha),
            Some(out),
        )
    }

    fn gelu(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._gelu(),
            |x| x._gelu(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn gelu_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._gelu(), |x| x._gelu(), Some(out))
    }

    fn selu(
        &self,
        alpha: Option<FloatUnaryType<T>>,
        gamma: Option<FloatUnaryType<T>>,
    ) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        let alpha = alpha.unwrap_or(1.6732632423543772848170429916717.into_scalar());
        let gamma = gamma.unwrap_or(1.0507009873554804934193349852946.into_scalar());
        uary_fn_with_out_simd(
            self,
            |x| x._selu(alpha, gamma),
            |x| x._selu(alpha, gamma),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn selu_<U>(
        &self,
        alpha: Option<FloatUnaryType<T>>,
        gamma: Option<FloatUnaryType<T>>,
        out: U,
    ) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        let alpha = alpha.unwrap_or(1.67326319217681884765625.into_scalar());
        let gamma = gamma.unwrap_or(1.05070102214813232421875.into_scalar());
        uary_fn_with_out_simd(
            self,
            |x| x._selu(alpha, gamma),
            |x| x._selu(alpha, gamma),
            Some(out),
        )
    }

    fn hard_sigmoid(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._hard_sigmoid(),
            |x| x._hard_sigmoid(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn fast_hard_sigmoid(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._fast_hard_sigmoid(),
            |x| x._fast_hard_sigmoid(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn hard_sigmoid_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(
            self,
            |x| x._hard_sigmoid(),
            |x| x._hard_sigmoid(),
            Some(out),
        )
    }

    fn hard_swish(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._hard_swish(),
            |x| x._hard_swish(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn hard_swish_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._hard_swish(), |x| x._hard_swish(), Some(out))
    }

    fn relu6(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._relu6(),
            |x| x._relu6(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn relu6_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._relu6(), |x| x._relu6(), Some(out))
    }

    fn softplus(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._softplus(),
            |x| x._softplus(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn softplus_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._softplus(), |x| x._softplus(), Some(out))
    }

    fn softsign(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._softsign(),
            |x| x._softsign(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn softsign_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._softsign(), |x| x._softsign(), Some(out))
    }

    fn mish(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._mish(),
            |x| x._mish(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn mish_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._mish(), |x| x._mish(), Some(out))
    }

    fn relu(&self) -> anyhow::Result<_Tensor<FloatUnaryType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._relu(),
            |x| x._relu(),
            None::<_Tensor<FloatUnaryType<T>>>,
        )
    }

    fn relu_<U>(&self, out: U) -> anyhow::Result<_Tensor<FloatUnaryType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._relu(), |x| x._relu(), Some(out))
    }
}

pub(crate) type NormalType<T> = <T as NormalOut>::Output;

impl<T> NormalUaryOps for _Tensor<T>
where
    T: NormalOut<Output = T> + CommonBounds + IntoScalar<T>,
    NormalType<T>: CommonBounds,
    _Tensor<NormalType<T>>: TensorLike<NormalType<T>>,
    T::Vec: NormalOut<Output = T::Vec>,
{
    type Output = _Tensor<NormalType<T>>;

    type InplaceOutput = _Tensor<NormalType<T>>;

    type OutputMeta = NormalType<T>;

    fn floor(&self) -> anyhow::Result<_Tensor<NormalType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._floor(),
            |x| x._floor(),
            None::<_Tensor<NormalType<T>>>,
        )
    }

    fn floor_<U>(&self, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._floor(), |x| x._floor(), Some(out))
    }

    fn square(&self) -> anyhow::Result<_Tensor<NormalType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._square(),
            |x| x._square(),
            None::<_Tensor<NormalType<T>>>,
        )
    }

    fn square_<U>(&self, out: U) -> anyhow::Result<_Tensor<NormalType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._square(), |x| x._square(), Some(out))
    }

    fn abs(&self) -> anyhow::Result<Self> {
        uary_fn_with_out_simd(self, |x| x._abs(), |x| x._abs(), None::<Self::Output>)
    }

    fn abs_<U>(&self, out: U) -> anyhow::Result<Self>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._abs(), |x| x._abs(), Some(out))
    }

    fn ceil(&self) -> anyhow::Result<_Tensor<NormalType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._ceil(),
            |x| x._ceil(),
            None::<_Tensor<NormalType<T>>>,
        )
    }
    fn ceil_<U>(&self, out: U) -> anyhow::Result<_Tensor<NormalType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._ceil(), |x| x._ceil(), Some(out))
    }

    fn sign(&self) -> anyhow::Result<_Tensor<NormalType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._sign(),
            |x| x._sign(),
            None::<_Tensor<NormalType<T>>>,
        )
    }
    fn sign_<U>(&self, out: U) -> anyhow::Result<_Tensor<NormalType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._sign(), |x| x._sign(), Some(out))
    }
    fn clip(
        &self,
        min: NormalType<T>,
        max: NormalType<T>,
    ) -> anyhow::Result<_Tensor<NormalType<T>>> {
        let min_vec = T::Vec::splat(min);
        let max_vec = T::Vec::splat(max);
        uary_fn_with_out_simd(
            self,
            |x| x._clip(min_vec, max_vec),
            |x| x._clip(min, max),
            None::<_Tensor<NormalType<T>>>,
        )
    }
    fn clip_<U>(
        &self,
        min: NormalType<T>,
        max: NormalType<T>,
        out: U,
    ) -> anyhow::Result<_Tensor<NormalType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        let min_vec = T::Vec::splat(min);
        let max_vec = T::Vec::splat(max);
        uary_fn_with_out_simd(
            self,
            |x| x._clip(min_vec, max_vec),
            |x| x._clip(min, max),
            Some(out),
        )
    }
    fn round(&self) -> anyhow::Result<_Tensor<NormalType<T>>> {
        uary_fn_with_out_simd(
            self,
            |x| x._round(),
            |x| x._round(),
            None::<_Tensor<NormalType<T>>>,
        )
    }
    fn round_<U>(&self, out: U) -> anyhow::Result<_Tensor<NormalType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| x._round(), |x| x._round(), Some(out))
    }
}

type NegType<T> = <T as std::ops::Neg>::Output;

impl<T> Neg for _Tensor<T>
where
    T: std::ops::Neg<Output = T> + CommonBounds,
    NegType<T>: CommonBounds,
    _Tensor<NegType<T>>: TensorLike<NegType<T>>,
    T::Vec: std::ops::Neg<Output = T::Vec>,
{
    type Output = _Tensor<NegType<T>>;

    type InplaceOutput = _Tensor<NegType<T>>;

    type OutputMeta = NegType<T>;

    fn neg(&self) -> anyhow::Result<_Tensor<NegType<T>>> {
        uary_fn_with_out_simd(self, |x| -x, |x| -x, None::<_Tensor<NegType<T>>>)
    }

    fn neg_<U>(&self, out: U) -> anyhow::Result<_Tensor<NegType<T>>>
    where
        U: Borrow<Self::InplaceOutput>,
    {
        uary_fn_with_out_simd(self, |x| -x, |x| -x, Some(out))
    }
}

impl<T> _Tensor<T>
where
    T: CommonBounds + Eval,
    <T as Eval>::Output: CommonBounds,
    T::Vec: Eval<Output = <<T as Eval>::Output as TypeCommon>::Vec>,
{
    pub fn is_inf(&self) -> anyhow::Result<_Tensor<<T as Eval>::Output>> {
        uary_fn_with_out_simd(
            self,
            |x| x._is_inf(),
            |x| x._is_inf(),
            None::<_Tensor<<T as Eval>::Output>>,
        )
    }
    pub fn is_nan(&self) -> anyhow::Result<_Tensor<<T as Eval>::Output>> {
        uary_fn_with_out_simd(
            self,
            |x| x._is_nan(),
            |x| x._is_nan(),
            None::<_Tensor<<T as Eval>::Output>>,
        )
    }
}

impl<T> _Tensor<T>
where
    T: CommonBounds,
{
    #[allow(unused)]
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn cumsum(&self, axis: Option<i64>) -> anyhow::Result<Self>
    where
        T: NormalOut<T, Output = T>,
    {
        match axis {
            Some(axis) => {
                let mut _axis = axis;
                ErrHandler::check_index_in_range_mut(self.ndim(), &mut _axis)?;
                let stride = self.strides()[_axis as usize];
                let inner_loop = self.shape()[_axis as usize] as usize;
                let outer_loop = self.size() / inner_loop;
                let mut shape = self.shape().to_vec();
                shape.iter_mut().for_each(|x| {
                    *x -= 1;
                });
                shape.swap(_axis as usize, self.shape().len() - 1);
                let mut strides = self.strides().to_vec();
                strides.swap(_axis as usize, self.strides().len() - 1);
                let res = self.empty_like()?;
                let res_stride = res.strides()[_axis as usize];
                let mut res_strides = res.strides().to_vec();
                res_strides.swap(_axis as usize, res.strides().len() - 1);
                THREAD_POOL.with_borrow_mut(|pool: &mut ThreadPool| {
                    let num_threads;
                    if outer_loop < pool.max_count() {
                        num_threads = 1;
                    } else {
                        num_threads = 1;
                    }
                    let mut intervals = mt_intervals(outer_loop, num_threads);

                    let mut prgs = Vec::with_capacity(num_threads);
                    let mut ptrs = Vec::with_capacity(num_threads);
                    let mut res_ptrs = Vec::with_capacity(num_threads);
                    let mut shapes = Vec::with_capacity(num_threads);
                    let mut __res_strides = Vec::with_capacity(num_threads);
                    let mut __inp_strides = Vec::with_capacity(num_threads);
                    let mut amount = 0i64;
                    for i in 0..num_threads {
                        let (start, end) = intervals[i];
                        let mut amount_cpy = amount;
                        let mut prg_tmp = vec![0; self.shape().len() - 1];
                        let mut ptr_tmp = self.ptr();
                        let mut res_ptr_tmp = res.ptr();
                        for j in (0..(self.shape().len() as i64) - 1).rev() {
                            prg_tmp[j as usize] = amount_cpy % (shape[j as usize] + 1);
                            amount_cpy /= self.shape()[j as usize];
                            let inp_gap = prg_tmp[j as usize] * strides[j as usize];
                            let res_gap = prg_tmp[j as usize] * res_strides[j as usize];
                            ptr_tmp.offset(inp_gap);
                            res_ptr_tmp.offset(res_gap);
                        }
                        amount += ((end - start) as i64) * (shape[shape.len() - 1] + 1);
                        prgs.push(prg_tmp);
                        ptrs.push(ptr_tmp);
                        res_ptrs.push(res_ptr_tmp);
                        shapes.push(shape.clone());
                        __res_strides.push(res_strides.clone());
                        __inp_strides.push(strides.clone());
                    }
                    for _ in 0..num_threads {
                        let (start, end) = intervals.pop().unwrap();
                        let mut prg = prgs.pop().unwrap();
                        let mut ptr = ptrs.pop().unwrap();
                        let mut res_ptr = res_ptrs.pop().unwrap();
                        let current_size = end - start;
                        let __shape = shapes.pop().unwrap();
                        let __res_strides = __res_strides.pop().unwrap();
                        let __strides = __inp_strides.pop().unwrap();
                        pool.execute(move || {
                            for _ in 0..current_size {
                                let mut tmp = T::ZERO;
                                for i in 0..inner_loop as i64 {
                                    tmp = tmp._add(ptr[i * stride]);
                                    res_ptr.modify(i * res_stride, tmp);
                                }
                                for j in (0..(__shape.len() as i64) - 1).rev() {
                                    let j = j as usize;
                                    if prg[j] < __shape[j] {
                                        prg[j] += 1;
                                        res_ptr.offset(__res_strides[j]);
                                        ptr.offset(__strides[j]);
                                        break;
                                    } else {
                                        prg[j] = 0;
                                        res_ptr.offset(-__shape[j] * __res_strides[j]);
                                        ptr.offset(-__shape[j] * __strides[j]);
                                    }
                                }
                            }
                        });
                    }
                });
                Ok(res)
            }
            None => {
                let res = _Tensor::<T, Cpu>::empty(vec![self.size() as i64])?;
                let mut tmp = T::ZERO;
                if self.is_contiguous() {
                    let raw = self.as_raw();
                    let res_raw = res.as_raw_mut();
                    for i in 0..self.size() {
                        tmp = tmp._add(raw[i]);
                        res_raw[i] = tmp;
                    }
                    Ok(res)
                } else {
                    let new_self = self.contiguous()?;
                    let raw = new_self.as_raw();
                    let mut tmp = T::ZERO;
                    let res_raw = res.as_raw_mut();
                    for i in 0..self.size() {
                        tmp = tmp._add(raw[i]);
                        res_raw[i] = tmp;
                    }
                    Ok(res)
                }
            }
        }
    }
    #[allow(unused)]
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn cumprod(&self, axis: Option<i64>) -> anyhow::Result<Self>
    where
        T: NormalOut<T, Output = T>,
    {
        match axis {
            Some(axis) => {
                let mut _axis = axis;
                ErrHandler::check_index_in_range_mut(self.ndim(), &mut _axis)?;
                let stride = self.strides()[_axis as usize];
                let inner_loop = self.shape()[_axis as usize] as usize;
                let outer_loop = self.size() / inner_loop;
                let mut shape = self.shape().to_vec();
                shape.iter_mut().for_each(|x| {
                    *x -= 1;
                });
                shape.swap(_axis as usize, self.shape().len() - 1);
                let mut strides = self.strides().to_vec();
                strides.swap(_axis as usize, self.strides().len() - 1);
                let res = self.empty_like()?;
                let res_stride = res.strides()[_axis as usize];
                let mut res_strides = res.strides().to_vec();
                res_strides.swap(_axis as usize, res.strides().len() - 1);
                THREAD_POOL.with_borrow_mut(|pool: &mut ThreadPool| {
                    let num_threads;
                    if outer_loop < pool.max_count() {
                        num_threads = outer_loop;
                    } else {
                        num_threads = pool.max_count();
                    }
                    let mut intervals = mt_intervals(outer_loop, num_threads);

                    let mut prgs = Vec::with_capacity(num_threads);
                    let mut ptrs = Vec::with_capacity(num_threads);
                    let mut res_ptrs = Vec::with_capacity(num_threads);
                    let mut shapes = Vec::with_capacity(num_threads);
                    let mut __res_strides = Vec::with_capacity(num_threads);
                    let mut __inp_strides = Vec::with_capacity(num_threads);
                    let mut amount = 0i64;
                    for i in 0..num_threads {
                        let (start, end) = intervals[i];
                        let mut amount_cpy = amount;
                        let mut prg_tmp = vec![0; self.shape().len() - 1];
                        let mut ptr_tmp = self.ptr();
                        let mut res_ptr_tmp = res.ptr();
                        for j in (0..(self.shape().len() as i64) - 1).rev() {
                            prg_tmp[j as usize] = amount_cpy % (shape[j as usize] + 1);
                            amount_cpy /= self.shape()[j as usize];
                            let inp_gap = prg_tmp[j as usize] * strides[j as usize];
                            let res_gap = prg_tmp[j as usize] * res_strides[j as usize];
                            ptr_tmp.offset(inp_gap);
                            res_ptr_tmp.offset(res_gap);
                        }
                        amount += ((end - start) as i64) * (shape[shape.len() - 1] + 1);
                        prgs.push(prg_tmp);
                        ptrs.push(ptr_tmp);
                        res_ptrs.push(res_ptr_tmp);
                        shapes.push(shape.clone());
                        __res_strides.push(res_strides.clone());
                        __inp_strides.push(strides.clone());
                    }
                    for _ in 0..num_threads {
                        let (start, end) = intervals.pop().unwrap();
                        let mut prg = prgs.pop().unwrap();
                        let mut ptr = ptrs.pop().unwrap();
                        let mut res_ptr = res_ptrs.pop().unwrap();
                        let current_size = end - start;
                        let __shape = shapes.pop().unwrap();
                        let __res_strides = __res_strides.pop().unwrap();
                        let __strides = __inp_strides.pop().unwrap();
                        pool.execute(move || {
                            for _ in 0..current_size {
                                let mut tmp = T::ONE;
                                for i in 0..inner_loop as i64 {
                                    tmp = tmp._mul(ptr[i * stride]);
                                    res_ptr.modify(i * res_stride, tmp);
                                }
                                for j in (0..(__shape.len() as i64) - 1).rev() {
                                    let j = j as usize;
                                    if prg[j] < __shape[j] {
                                        prg[j] += 1;
                                        res_ptr.offset(__res_strides[j]);
                                        ptr.offset(__strides[j]);
                                        break;
                                    } else {
                                        prg[j] = 0;
                                        res_ptr.offset(-__shape[j] * __res_strides[j]);
                                        ptr.offset(-__shape[j] * __strides[j]);
                                    }
                                }
                            }
                        });
                    }
                });
                Ok(res)
            }
            None => {
                let res = _Tensor::<T, Cpu>::empty(vec![self.size() as i64])?;
                let mut tmp = T::ONE;
                if self.is_contiguous() {
                    let raw = self.as_raw();
                    let res_raw = res.as_raw_mut();
                    for i in 0..self.size() {
                        tmp = tmp._mul(raw[i]);
                        res_raw[i] = tmp;
                    }
                    Ok(res)
                } else {
                    let new_self = self.contiguous()?;
                    let raw = new_self.as_raw();
                    let mut tmp = T::ONE;
                    let res_raw = res.as_raw_mut();
                    for i in 0..self.size() {
                        tmp = tmp._mul(raw[i]);
                        res_raw[i] = tmp;
                    }
                    Ok(res)
                }
            }
        }
    }
}
