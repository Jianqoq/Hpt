use rayon::iter::{
    IndexedParallelIterator,
    IntoParallelRefIterator,
    IntoParallelRefMutIterator,
    ParallelIterator,
};
use tensor_types::vectors::traits::*;
use rayon::slice::{ ParallelSlice, ParallelSliceMut };
use tensor_common::err_handler::ErrHandler;
use tensor_common::shape_utils::mt_intervals;
use tensor_traits::BaseTensor;
use tensor_types::dtype::TypeCommon;
use tensor_types::into_scalar::IntoScalar;
use threadpool::ThreadPool;
use crate::backend::Cpu;
use crate::THREAD_POOL;
use tensor_traits::ops::uary::{ Cum, FloatUaryOps, Neg, NormalUaryOps };
use tensor_traits::tensor::{ CommonBounds, TensorInfo, TensorLike };
use tensor_traits::tensor::TensorCreator;
use tensor_types::type_promote::{ FloatOutBinary, FloatOutUnary, NormalOut };
use crate::tensor_base::_Tensor;

/// Applies a unary function to a tensor, returning a tensor with float type elements.
///
/// This function applies a unary function `f` to each element of the input tensor `inp` and
/// returns a new tensor with the result of the function. The result tensor has float type elements.
///
/// # Arguments
/// - `inp`: Input tensor.
/// - `f`: Unary function to apply to each element.
///
/// # Returns
/// `anyhow::Result<_Tensor<BFLOAT<A, A>>>`: A tensor with float type elements, with the result of applying `f`.

#[inline(always)]
pub fn uary_fn<A, F, O>(inp: &_Tensor<A>, f: F) -> anyhow::Result<_Tensor<O>>
    where A: CommonBounds, O: CommonBounds, F: Fn(A) -> O + Sync + Send
{
    let ret: _Tensor<O>;
    ret = _Tensor::<O, Cpu>::empty(inp.shape()).unwrap();
    let new_f = &f;
    let remain = inp.size() % 8;
    let exect_size = inp.size() - remain;
    ret.as_raw_mut()
        .par_chunks_exact_mut(8)
        .zip(inp.as_raw().par_chunks_exact(8))
        .for_each(|(a, b)| {
            a[0] = new_f(b[0]);
            a[1] = new_f(b[1]);
            a[2] = new_f(b[2]);
            a[3] = new_f(b[3]);
            a[4] = new_f(b[4]);
            a[5] = new_f(b[5]);
            a[6] = new_f(b[6]);
            a[7] = new_f(b[7]);
        });
    ret.as_raw_mut()
        [exect_size..].iter_mut()
        .zip(inp.as_raw()[exect_size..].iter())
        .for_each(|(a, &b)| {
            *a = f(b);
        });
    Ok(ret)
}

#[inline(always)]
pub fn uary_fn_simd<A, F, O, F2>(inp: &_Tensor<A>, f: F, f2: F2) -> anyhow::Result<_Tensor<O>>
    where
        A: CommonBounds,
        O: CommonBounds,
        F: Fn(<A as TypeCommon>::Vec) -> <O as TypeCommon>::Vec + Sync + Send,
        F2: Fn(A) -> O + Sync + Send
{
    let ret: _Tensor<O>;
    ret = _Tensor::<O, Cpu>::empty(inp.shape()).unwrap();
    let per_thread_len = ret.size() / rayon::current_num_threads();
    let per_thread_remain = per_thread_len % <O as TypeCommon>::Vec::SIZE;
    let total_remain = rayon::current_num_threads() * per_thread_remain;
    let per_thread_real_len = per_thread_len - per_thread_remain;
    ret.as_raw_mut()
        .par_chunks_exact_mut(per_thread_real_len)
        .zip(inp.as_raw().par_chunks_exact(per_thread_real_len))
        .for_each(|(ret, lhs)| {
            assert_eq!(lhs.len() % <A as TypeCommon>::Vec::SIZE, 0);
            assert_eq!(ret.len() % <O as TypeCommon>::Vec::SIZE, 0);
            ret.chunks_exact_mut(<A as TypeCommon>::Vec::SIZE)
                .zip(lhs.chunks_exact(<A as TypeCommon>::Vec::SIZE))
                .for_each(|(ret, lhs)| {
                    let a = unsafe { <A as TypeCommon>::Vec::from_ptr(lhs.as_ptr()) };
                    let res = f(a);
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            res.as_ptr(),
                            ret.as_mut_ptr(),
                            <O as TypeCommon>::Vec::SIZE
                        );
                    }
                });
        });
    if total_remain > 0 {
        ret.as_raw_mut()
            [ret.size() - total_remain..].iter_mut()
            .zip(inp.as_raw()[ret.size() - total_remain..].iter())
            .for_each(|(a, &lhs)| {
                *a = f2(lhs);
            });
    }
    Ok(ret)
}

/// Applies a unary function to a tensor with a specified output tensor.
///
/// Similar to `uary_fn_float`, but allows specifying an output tensor. The output tensor's memory
/// is reused if it matches the required size and reference count conditions; otherwise, a new tensor is created.
///
/// # Arguments
/// - `inp`: Input tensor.
/// - `f`: Unary function to apply to each element.
///
/// # Returns
/// `anyhow::Result<_Tensor<BFLOAT<A, A>>>`: A tensor with float type elements, with the result of applying `f`.
pub fn uary_fn_with_out<A, O, K, Q, F>(
    inp: &_Tensor<A, Cpu>,
    f: F,
    out: O
)
    -> anyhow::Result<_Tensor<K, Cpu>>
    where
        A: CommonBounds,
        O: TensorLike<Q, Output = _Tensor<K, Cpu>> + TensorInfo<Q>,
        K: CommonBounds,
        Q: CommonBounds,
        F: Fn(A) -> K + Sync + Send
{
    let ret_size: usize = inp.size();
    let ret = if out.size() * std::mem::size_of::<Q>() != ret_size * std::mem::size_of::<K>() {
        _Tensor::<K, Cpu>::empty(inp.shape())?
    } else {
        _Tensor::<K, Cpu>::empty(inp.shape())?
    };
    ret.as_raw_mut()
        .par_iter_mut()
        .zip(inp.as_raw().par_iter())
        .for_each(|(a, &b)| {
            *a = f(b);
        });
    Ok(ret)
}

fn uary_fn_with_out_simd<A, O, K, Q, F, F2>(
    inp: &_Tensor<A, Cpu>,
    f: F,
    f2: F2,
    out: O
)
    -> anyhow::Result<_Tensor<K, Cpu>>
    where
        A: CommonBounds,
        O: TensorLike<Q, Output = _Tensor<K, Cpu>> + TensorInfo<Q>,
        K: CommonBounds,
        Q: CommonBounds,
        F: Fn(<A as TypeCommon>::Vec) -> <K as TypeCommon>::Vec + Sync + Send,
        F2: Fn(A) -> K + Sync + Send
{
    let ret_size: usize = inp.size();
    let ret = if out.size() * std::mem::size_of::<Q>() != ret_size * std::mem::size_of::<K>() {
        _Tensor::<K, Cpu>::empty(inp.shape())?
    } else {
        _Tensor::<K, Cpu>::empty(inp.shape())?
    };
    let remain = inp.size() % <A as TypeCommon>::Vec::SIZE;
    let exect_size = inp.size() - remain;
    ret.as_raw_mut()
        .par_chunks_exact_mut(<A as TypeCommon>::Vec::SIZE)
        .zip(inp.as_raw().par_chunks_exact(<A as TypeCommon>::Vec::SIZE))
        .for_each(|(a, b)| {
            let b_ptr = b.as_ptr() as *const A;
            let inp = unsafe { A::Vec::from_ptr(b_ptr) };
            let res = f(inp);
            let res_ptr = res.as_ptr() as *mut K;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    res_ptr,
                    a.as_mut_ptr(),
                    <A as TypeCommon>::Vec::SIZE
                );
            }
        });
    ret.as_raw_mut()
        [exect_size..].iter_mut()
        .zip(inp.as_raw()[exect_size..].iter())
        .for_each(|(a, &b)| {
            *a = f2(b);
        });
    Ok(ret)
}

pub(crate) type FloatUnaryType<T> = <T as FloatOutUnary>::Output;
pub(crate) type FloatBinaryType<T> = <T as FloatOutBinary>::Output;

impl<T> FloatUaryOps
    for _Tensor<T>
    where
        T: FloatOutUnary + CommonBounds,
        FloatUnaryType<T>: CommonBounds,
        f64: IntoScalar<<T as FloatOutUnary>::Base>,
        _Tensor<<T as FloatOutUnary>::Output>: TensorLike<
            <T as FloatOutUnary>::Output,
            Output = _Tensor<<T as FloatOutUnary>::Output>
        >,
        <T as TypeCommon>::Vec: FloatOutUnary<
            Output = <FloatUnaryType<T> as TypeCommon>::Vec,
            Base = <T as FloatOutUnary>::Base
        >,
        <FloatUnaryType<T> as TypeCommon>::Vec: Send + Copy + Sync,
        <T as FloatOutUnary>::Base: CommonBounds
{
    type Output = _Tensor<FloatUnaryType<T>>;

    type InplaceOutput = _Tensor<FloatUnaryType<T>>;

    type OutputMeta = <T as FloatOutUnary>::Base;

    fn sin(&self) -> anyhow::Result<Self::Output> {
        #[cfg(feature = "simd")]
        let ret = uary_fn_simd(
            self,
            |x| x._sin(),
            |x| x._sin()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn(self, |x| x._sin());
        ret
    }

    fn cos(&self) -> anyhow::Result<Self::Output> {
        #[cfg(feature = "simd")]
        let ret = uary_fn_simd(
            self,
            |x| x._cos(),
            |x| x._cos()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn(self, |x| x._cos());
        ret
    }

    fn tan(&self) -> anyhow::Result<Self::Output> {
        #[cfg(feature = "simd")]
        let ret = uary_fn_simd(
            self,
            |x| x._tan(),
            |x| x._tan()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn(self, |x| x._tan());
        ret
    }

    fn asin(&self) -> anyhow::Result<Self::Output> {
        #[cfg(feature = "simd")]
        let ret = uary_fn_simd(
            self,
            |x| x._asin(),
            |x| x._asin()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn(self, |x| x._asin());
        ret
    }

    fn acos(&self) -> anyhow::Result<Self::Output> {
        #[cfg(feature = "simd")]
        let ret = uary_fn_simd(
            self,
            |x| x._acos(),
            |x| x._acos()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn(self, |x| x._acos());
        ret
    }

    fn atan(&self) -> anyhow::Result<Self::Output> {
        #[cfg(feature = "simd")]
        let ret = uary_fn_simd(
            self,
            |x| x._atan(),
            |x| x._atan()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn(self, |x| x._atan());
        ret
    }

    fn sinh(&self) -> anyhow::Result<Self::Output> {
        #[cfg(feature = "simd")]
        let ret = uary_fn_simd(
            self,
            |x| x._sinh(),
            |x| x._sinh()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn(self, |x| x._sinh());
        ret
    }

    fn cosh(&self) -> anyhow::Result<Self::Output> {
        #[cfg(feature = "simd")]
        let ret = uary_fn_simd(
            self,
            |x| x._cosh(),
            |x| x._cosh()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn(self, |x| x._cosh());
        ret
    }

    fn tanh(&self) -> anyhow::Result<Self::Output> {
        #[cfg(feature = "simd")]
        let ret = uary_fn_simd(
            self,
            |x| x._tanh(),
            |x| x._tanh()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn(self, |x| x._tanh());
        ret
    }

    fn asinh(&self) -> anyhow::Result<Self::Output> {
        #[cfg(feature = "simd")]
        let ret = uary_fn_simd(
            self,
            |x| x._asinh(),
            |x| x._asinh()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn(self, |x| x._asinh());
        ret
    }

    fn acosh(&self) -> anyhow::Result<Self::Output> {
        #[cfg(feature = "simd")]
        let ret = uary_fn_simd(
            self,
            |x| x._acosh(),
            |x| x._acosh()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn(self, |x| x._acosh());
        ret
    }

    fn atanh(&self) -> anyhow::Result<Self::Output> {
        #[cfg(feature = "simd")]
        let ret = uary_fn_simd(
            self,
            |x| x._atanh(),
            |x| x._atanh()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn(self, |x| x._atanh());
        ret
    }

    fn sin_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        #[cfg(feature = "simd")]
        let ret = uary_fn_with_out_simd(
            self,
            |x| x._sin(),
            |x| x._sin(),
            out.base().clone()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn_with_out(self, |x| x._sin(), out.base().clone());
        ret
    }

    fn cos_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        #[cfg(feature = "simd")]
        let ret = uary_fn_with_out_simd(
            self,
            |x| x._cos(),
            |x| x._cos(),
            out.base().clone()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn_with_out(self, |x| x._cos(), out.base().clone());
        ret
    }

    fn tan_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        #[cfg(feature = "simd")]
        let ret = uary_fn_with_out_simd(
            self,
            |x| x._tan(),
            |x| x._tan(),
            out.base().clone()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn_with_out(self, |x| x._tan(), out.base().clone());
        ret
    }

    fn asin_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        #[cfg(feature = "simd")]
        let ret = uary_fn_with_out_simd(
            self,
            |x| x._asin(),
            |x| x._asin(),
            out.base().clone()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn_with_out(self, |x| x._asin(), out.base().clone());
        ret
    }

    fn acos_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        #[cfg(feature = "simd")]
        let ret = uary_fn_with_out_simd(
            self,
            |x| x._acos(),
            |x| x._acos(),
            out.base().clone()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn_with_out(self, |x| x._acos(), out.base().clone());
        ret
    }

    fn atan_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        #[cfg(feature = "simd")]
        let ret = uary_fn_with_out_simd(
            self,
            |x| x._atan(),
            |x| x._atan(),
            out.base().clone()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn_with_out(self, |x| x._atan(), out.base().clone());
        ret
    }

    fn sinh_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        #[cfg(feature = "simd")]
        let ret = uary_fn_with_out_simd(
            self,
            |x| x._sinh(),
            |x| x._sinh(),
            out.base().clone()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn_with_out(self, |x| x._sinh(), out.base().clone());
        ret
    }

    fn cosh_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        #[cfg(feature = "simd")]
        let ret = uary_fn_with_out_simd(
            self,
            |x| x._cosh(),
            |x| x._cosh(),
            out.base().clone()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn_with_out(self, |x| x._cosh(), out.base().clone());
        ret
    }

    fn tanh_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        #[cfg(feature = "simd")]
        let ret = uary_fn_with_out_simd(
            self,
            |x| x._tanh(),
            |x| x._tanh(),
            out.base().clone()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn_with_out(self, |x| x._tanh(), out.base().clone());
        ret
    }

    fn asinh_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        #[cfg(feature = "simd")]
        let ret = uary_fn_with_out_simd(
            self,
            |x| x._asinh(),
            |x| x._asinh(),
            out.base().clone()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn_with_out(self, |x| x._asinh(), out.base().clone());
        ret
    }

    fn acosh_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        #[cfg(feature = "simd")]
        let ret = uary_fn_with_out_simd(
            self,
            |x| x._acosh(),
            |x| x._acosh(),
            out.base().clone()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn_with_out(self, |x| x._acosh(), out.base().clone());
        ret
    }

    fn atanh_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        #[cfg(feature = "simd")]
        let ret = uary_fn_with_out_simd(
            self,
            |x| x._atanh(),
            |x| x._atanh(),
            out.base().clone()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn_with_out(self, |x| x._atanh(), out.base().clone());
        ret
    }

    fn exp(&self) -> anyhow::Result<Self::Output> {
        #[cfg(feature = "simd")]
        let ret = uary_fn_simd(
            self,
            |x| x._exp(),
            |x| x._exp()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn(self, |x| x._exp());
        ret
    }

    fn exp_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        #[cfg(feature = "simd")]
        let ret = uary_fn_with_out_simd(
            self,
            |x| x._exp(),
            |x| x._exp(),
            out.base().clone()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn_with_out(self, |x| x._exp(), out.base().clone());
        ret
    }

    fn exp2(&self) -> anyhow::Result<Self::Output> {
        #[cfg(feature = "simd")]
        let ret = uary_fn_simd(
            self,
            |x| x._exp2(),
            |x| x._exp2()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn(self, |x| x._exp2());
        ret
    }

    fn exp2_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        #[cfg(feature = "simd")]
        let ret = uary_fn_with_out_simd(
            self,
            |x| x._exp2(),
            |x| x._exp2(),
            out.base().clone()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn_with_out(self, |x| x._exp2(), out.base().clone());
        ret
    }

    fn sqrt(&self) -> anyhow::Result<Self::Output> {
        #[cfg(feature = "simd")]
        let ret = uary_fn_simd(
            self,
            |x| x._sqrt(),
            |x| x._sqrt()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn(self, |x| x._sqrt());
        ret
    }

    fn sqrt_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        #[cfg(feature = "simd")]
        let ret = uary_fn_with_out_simd(
            self,
            |x| x._sqrt(),
            |x| x._sqrt(),
            out.base().clone()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn_with_out(self, |x| x._sqrt(), out.base().clone());
        ret
    }

    fn recip(&self) -> anyhow::Result<Self::Output> {
        #[cfg(feature = "simd")]
        let ret = uary_fn_simd(
            self,
            |x| x._recip(),
            |x| x._recip()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn(self, |x| x._recip());
        ret
    }

    fn recip_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        #[cfg(feature = "simd")]
        let ret = uary_fn_with_out_simd(
            self,
            |x| x._recip(),
            |x| x._recip(),
            out.base().clone()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn_with_out(self, |x| x._recip(), out.base().clone());
        ret
    }

    fn ln(&self) -> anyhow::Result<Self::Output> {
        #[cfg(feature = "simd")]
        let ret = uary_fn_simd(
            self,
            |x| x._ln(),
            |x| x._ln()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn(self, |x| x._ln());
        ret
    }

    fn ln_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        #[cfg(feature = "simd")]
        let ret = uary_fn_with_out_simd(
            self,
            |x| x._ln(),
            |x| x._ln(),
            out.base().clone()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn_with_out(self, |x| x._ln(), out.base().clone());
        ret
    }

    fn log2(&self) -> anyhow::Result<Self::Output> {
        #[cfg(feature = "simd")]
        let ret = uary_fn_simd(
            self,
            |x| x._log2(),
            |x| x._log2()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn(self, |x| x._log2());
        ret
    }

    fn log2_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        #[cfg(feature = "simd")]
        let ret = uary_fn_with_out_simd(
            self,
            |x| x._log2(),
            |x| x._log2(),
            out.base().clone()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn_with_out(self, |x| x._log2(), out.base().clone());
        ret
    }

    fn log10(&self) -> anyhow::Result<Self::Output> {
        #[cfg(feature = "simd")]
        let ret = uary_fn_simd(
            self,
            |x| x._log10(),
            |x| x._log10()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn(self, |x| x._log10());
        ret
    }

    fn log10_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        #[cfg(feature = "simd")]
        let ret = uary_fn_with_out_simd(
            self,
            |x| x._log10(),
            |x| x._log10(),
            out.base().clone()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn_with_out(self, |x| x._log10(), out.base().clone());
        ret
    }

    fn celu(&self, alpha: Self::OutputMeta) -> anyhow::Result<Self::Output> {
        #[cfg(feature = "simd")]
        let ret = uary_fn_simd(
            self,
            |x| x._celu(alpha),
            |x| x._celu(alpha)
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn(self, |x| x._celu(alpha));
        ret
    }

    fn celu_<U>(&self, alpha: Self::OutputMeta, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        #[cfg(feature = "simd")]
        let ret = uary_fn_with_out_simd(
            self,
            |x| x._celu(alpha),
            |x| x._celu(alpha),
            out.base().clone()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn_with_out(self, |x| x._celu(alpha), out.base().clone());
        ret
    }

    fn sigmoid(&self) -> anyhow::Result<Self::Output> {
        #[cfg(feature = "simd")]
        let ret = uary_fn_simd(
            self,
            |x| x._sigmoid(),
            |x| x._sigmoid()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn(self, |x| x._sigmoid());
        ret
    }

    fn sigmoid_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        #[cfg(feature = "simd")]
        let ret = uary_fn_with_out_simd(
            self,
            |x| x._sigmoid(),
            |x| x._sigmoid(),
            out.base().clone()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn_with_out(self, |x| x._sigmoid(), out.base().clone());
        ret
    }

    fn elu(&self, alpha: Self::OutputMeta) -> anyhow::Result<Self::Output> {
        #[cfg(feature = "simd")]
        let ret = uary_fn_simd(
            self,
            |x| x._elu(alpha),
            |x| x._elu(alpha)
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn(self, |x| x._elu(alpha));
        ret
    }

    fn elu_<U>(&self, alpha: Self::OutputMeta, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        #[cfg(feature = "simd")]
        let ret = uary_fn_with_out_simd(
            self,
            |x| x._elu(alpha),
            |x| x._elu(alpha),
            out.base().clone()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn_with_out(self, |x| x._elu(alpha), out.base().clone());
        ret
    }

    fn leaky_relu(&self, alpha: Self::OutputMeta) -> anyhow::Result<Self::Output> {
        #[cfg(feature = "simd")]
        let ret = uary_fn_simd(
            self,
            |x| x._leaky_relu(alpha),
            |x| x._leaky_relu(alpha)
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn(self, |x| x._leaky_relu(alpha));
        ret
    }

    fn leaky_relu_<U>(&self, alpha: Self::OutputMeta, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        #[cfg(feature = "simd")]
        let ret = uary_fn_with_out_simd(
            self,
            |x| x._leaky_relu(alpha),
            |x| x._leaky_relu(alpha),
            out.base().clone()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn_with_out(self, |x| x._leaky_relu(alpha), out.base().clone());
        ret
    }

    fn gelu(&self) -> anyhow::Result<Self::Output> {
        #[cfg(feature = "simd")]
        let ret = uary_fn_simd(
            self,
            |x| x._gelu(),
            |x| x._gelu()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn(self, |x| x._gelu());
        ret
    }

    fn gelu_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        #[cfg(feature = "simd")]
        let ret = uary_fn_with_out_simd(
            self,
            |x| x._gelu(),
            |x| x._gelu(),
            out.base().clone()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn_with_out(self, |x| x._gelu(), out.base().clone());
        ret
    }

    fn selu(
        &self,
        alpha: Option<Self::OutputMeta>,
        gamma: Option<Self::OutputMeta>
    ) -> anyhow::Result<Self::Output> {
        let alpha = alpha.unwrap_or((1.6732632423543772848170429916717).into_scalar());
        let gamma = gamma.unwrap_or((1.0507009873554804934193349852946).into_scalar());
        #[cfg(feature = "simd")]
        let ret = uary_fn_simd(
            self,
            |x| x._selu(alpha, gamma),
            move |x| x._selu(alpha, gamma)
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn(self, |x| x._selu(alpha, gamma));
        ret
    }

    fn selu_<U>(
        &self,
        alpha: Option<Self::OutputMeta>,
        gamma: Option<Self::OutputMeta>,
        out: U
    ) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        let alpha = alpha.unwrap_or((1.67326319217681884765625).into_scalar());
        let gamma = gamma.unwrap_or((1.05070102214813232421875).into_scalar());
        #[cfg(feature = "simd")]
        let ret = uary_fn_with_out_simd(
            self,
            |x| x._selu(alpha, gamma),
            |x| x._selu(alpha, gamma),
            out.base().clone()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn_with_out(self, |x| x._selu(alpha, gamma), out.base().clone());
        ret
    }

    fn hard_sigmoid(&self) -> anyhow::Result<Self::Output> {
        #[cfg(feature = "simd")]
        let ret = uary_fn_simd(
            self,
            |x| x._hard_sigmoid(),
            |x| x._hard_sigmoid()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn(self, |x| x._hard_sigmoid());
        ret
    }

    fn hard_sigmoid_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        #[cfg(feature = "simd")]
        let ret = uary_fn_with_out_simd(
            self,
            |x| x._hard_sigmoid(),
            |x| x._hard_sigmoid(),
            out.base().clone()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn_with_out(self, |x| x._hard_sigmoid(), out.base().clone());
        ret
    }

    fn hard_swish(&self) -> anyhow::Result<Self::Output> {
        #[cfg(feature = "simd")]
        let ret = uary_fn_simd(
            self,
            |x| x._hard_swish(),
            |x| x._hard_swish()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn(self, |x| x._hard_swish());
        ret
    }

    fn hard_swish_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        #[cfg(feature = "simd")]
        let ret = uary_fn_with_out_simd(
            self,
            |x| x._hard_swish(),
            |x| x._hard_swish(),
            out.base().clone()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn_with_out(self, |x| x._hard_swish(), out.base().clone());
        ret
    }

    fn relu6(&self) -> anyhow::Result<Self::Output> {
        #[cfg(feature = "simd")]
        let ret = uary_fn_simd(
            self,
            |x| x._relu6(),
            |x| x._relu6()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn(self, |x| x._relu6());
        ret
    }

    fn relu6_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        #[cfg(feature = "simd")]
        let ret = uary_fn_with_out_simd(
            self,
            |x| x._relu6(),
            |x| x._relu6(),
            out.base().clone()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn_with_out(self, |x| x._relu6(), out.base().clone());
        ret
    }

    fn softplus(&self) -> anyhow::Result<Self::Output> {
        #[cfg(feature = "simd")]
        let ret = uary_fn_simd(
            self,
            |x| x._softplus(),
            |x| x._softplus()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn(self, |x| x._softplus());
        ret
    }

    fn softplus_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        #[cfg(feature = "simd")]
        let ret = uary_fn_with_out_simd(
            self,
            |x| x._softplus(),
            |x| x._softplus(),
            out.base().clone()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn_with_out(self, |x| x._softplus(), out.base().clone());
        ret
    }

    fn softsign(&self) -> anyhow::Result<Self::Output> {
        #[cfg(feature = "simd")]
        let ret = uary_fn_simd(
            self,
            |x| x._softsign(),
            |x| x._softsign()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn(self, |x| x._softsign());
        ret
    }

    fn softsign_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        #[cfg(feature = "simd")]
        let ret = uary_fn_with_out_simd(
            self,
            |x| x._softsign(),
            |x| x._softsign(),
            out.base().clone()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn_with_out(self, |x| x._softsign(), out.base().clone());
        ret
    }

    fn mish(&self) -> anyhow::Result<Self::Output> {
        #[cfg(feature = "simd")]
        let ret = uary_fn_simd(
            self,
            |x| x._mish(),
            |x| x._mish()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn(self, |x| x._mish());
        ret
    }

    fn mish_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        #[cfg(feature = "simd")]
        let ret = uary_fn_with_out_simd(
            self,
            |x| x._mish(),
            |x| x._mish(),
            out.base().clone()
        );
        #[cfg(not(feature = "simd"))]
        let ret = uary_fn_with_out(self, |x| x._mish(), out.base().clone());
        ret
    }
}

pub(crate) type NormalType<T> = <T as NormalOut>::Output;

impl<T> NormalUaryOps
    for _Tensor<T>
    where
        T: NormalOut<Output = T> + CommonBounds + IntoScalar<T>,
        NormalType<T>: CommonBounds,
        _Tensor<NormalType<T>>: TensorLike<NormalType<T>, Output = _Tensor<NormalType<T>>>
{
    type Output = _Tensor<NormalType<T>>;

    type InplaceOutput = _Tensor<NormalType<T>>;

    type OutputMeta = NormalType<T>;

    fn square(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._square())
    }

    fn square_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._square(), out.base().clone())
    }

    fn abs(&self) -> anyhow::Result<Self> {
        uary_fn(self, |x| x._abs())
    }

    fn abs_<U>(&self, out: U) -> anyhow::Result<Self> where U: BaseTensor<Output = Self> {
        uary_fn_with_out(self, |x| x._abs(), out.base().clone())
    }

    fn ceil(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._ceil())
    }

    fn ceil_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._ceil(), out.base().clone())
    }

    fn sign(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._sign())
    }

    fn sign_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._sign(), out.base().clone())
    }

    fn clip(&self, min: Self::OutputMeta, max: Self::OutputMeta) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._clip(min, max))
    }

    fn clip_<U>(
        &self,
        min: Self::OutputMeta,
        max: Self::OutputMeta,
        out: U
    ) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._clip(min, max), out.base().clone())
    }

    fn round(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._round())
    }

    fn round_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._round(), out.base().clone())
    }
}

type NegType<T> = <T as std::ops::Neg>::Output;

impl<T> Neg
    for _Tensor<T>
    where
        T: std::ops::Neg + CommonBounds,
        NegType<T>: CommonBounds,
        _Tensor<NegType<T>>: TensorLike<NegType<T>, Output = _Tensor<NegType<T>>>
{
    type Output = _Tensor<NegType<T>>;

    type InplaceOutput = _Tensor<NegType<T>>;

    type OutputMeta = NegType<T>;

    fn neg(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| -x)
    }

    fn neg_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| -x, out.base().clone())
    }
}

impl<T> Cum for _Tensor<T> where T: CommonBounds {
    type Meta = T;

    fn cumsum(&self, axis: Option<i64>) -> anyhow::Result<Self>
        where Self::Meta: NormalOut<Self::Meta, Output = Self::Meta>
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

    fn cumprod(&self, axis: Option<i64>) -> anyhow::Result<Self>
        where Self::Meta: NormalOut<Self::Meta, Output = Self::Meta>
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
