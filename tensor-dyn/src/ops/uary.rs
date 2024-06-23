use rayon::iter::{
    IndexedParallelIterator,
    IntoParallelRefIterator,
    IntoParallelRefMutIterator,
    ParallelIterator,
};
use tensor_traits::ops::uary::TensorUaryOps;
use tensor_traits::tensor::{ CommonBounds, TensorInfo, TensorLike };
use tensor_traits::tensor::TensorCreator;
use tensor_types::type_promote::FloatOut;
use crate::tensor::_Tensor;

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
pub fn uary_fn<A, F, O>(inp: &_Tensor<A>, f: F) -> anyhow::Result<_Tensor<O>>
    where A: CommonBounds, O: CommonBounds, F: Fn(A) -> O + Sync + Send
{
    let ret: _Tensor<O>;
    ret = _Tensor::empty(inp.shape()).unwrap();
    let new_f = &f;
    ret.as_raw_mut()
        .par_iter_mut()
        .zip(inp.as_raw().par_iter())
        .for_each(|(a, &b)| {
            *a = new_f(b);
        });
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
pub fn uary_fn_with_out<A, O, K, Q, F>(inp: &_Tensor<A>, f: F, out: O) -> anyhow::Result<_Tensor<K>>
    where
        A: CommonBounds,
        O: TensorLike<Q, Output = _Tensor<K>> + TensorInfo<Q>,
        K: CommonBounds,
        Q: CommonBounds,
        F: Fn(A) -> K + Sync + Send
{
    let ret: <O as TensorLike<Q>>::Output;
    let ret_size: usize = inp.size();
    if out.size() * std::mem::size_of::<Q>() != ret_size * std::mem::size_of::<K>() {
        ret = _Tensor::empty(inp.shape()).unwrap();
    } else {
        ret = _Tensor::empty(inp.shape()).unwrap();
    }
    ret.as_raw_mut()
        .par_iter_mut()
        .zip(inp.as_raw().par_iter())
        .for_each(|(a, &b)| {
            *a = f(b);
        });
    Ok(ret)
}

type FloatType<T> = <T as FloatOut>::Output;

impl<T> TensorUaryOps for _Tensor<T> where T: FloatOut + CommonBounds, FloatType<T>: CommonBounds {
    type Output = _Tensor<FloatType<T>>;

    type InplaceOutput = _Tensor<FloatType<T>>;

    type OutputMeta = FloatType<T>;

    fn sin(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._sin())
    }

    fn cos(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._cos())
    }

    fn tan(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._tan())
    }

    fn asin(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._asin())
    }

    fn acos(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._acos())
    }

    fn atan(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._atan())
    }

    fn sinh(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._sinh())
    }

    fn cosh(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._cosh())
    }

    fn tanh(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._tanh())
    }

    fn asinh(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._asinh())
    }

    fn acosh(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._acosh())
    }

    fn atanh(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._atanh())
    }

    fn sin_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where
            U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                TensorInfo<Self::OutputMeta>
    {
        uary_fn_with_out(self, |x| x._sin(), out)
    }

    fn cos_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where
            U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                TensorInfo<Self::OutputMeta>
    {
        uary_fn_with_out(self, |x| x._cos(), out)
    }

    fn tan_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where
            U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                TensorInfo<Self::OutputMeta>
    {
        uary_fn_with_out(self, |x| x._tan(), out)
    }

    fn asin_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where
            U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                TensorInfo<Self::OutputMeta>
    {
        uary_fn_with_out(self, |x| x._asin(), out)
    }

    fn acos_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where
            U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                TensorInfo<Self::OutputMeta>
    {
        uary_fn_with_out(self, |x| x._acos(), out)
    }

    fn atan_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where
            U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                TensorInfo<Self::OutputMeta>
    {
        uary_fn_with_out(self, |x| x._atan(), out)
    }

    fn sinh_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where
            U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                TensorInfo<Self::OutputMeta>
    {
        uary_fn_with_out(self, |x| x._sinh(), out)
    }

    fn cosh_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where
            U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                TensorInfo<Self::OutputMeta>
    {
        uary_fn_with_out(self, |x| x._cosh(), out)
    }

    fn tanh_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where
            U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                TensorInfo<Self::OutputMeta>
    {
        uary_fn_with_out(self, |x| x._tanh(), out)
    }

    fn asinh_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where
            U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                TensorInfo<Self::OutputMeta>
    {
        uary_fn_with_out(self, |x| x._asinh(), out)
    }

    fn acosh_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where
            U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                TensorInfo<Self::OutputMeta>
    {
        uary_fn_with_out(self, |x| x._acosh(), out)
    }

    fn atanh_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where
            U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                TensorInfo<Self::OutputMeta>
    {
        uary_fn_with_out(self, |x| x._atanh(), out)
    }

    fn exp(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._exp())
    }

    fn exp_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where
            U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                TensorInfo<Self::OutputMeta>
    {
        uary_fn_with_out(self, |x| x._exp(), out)
    }

    fn exp2(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._exp2())
    }

    fn exp2_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where
            U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                TensorInfo<Self::OutputMeta>
    {
        uary_fn_with_out(self, |x| x._exp2(), out)
    }

    fn sqrt(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._sqrt())
    }

    fn sqrt_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where
            U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                TensorInfo<Self::OutputMeta>
    {
        uary_fn_with_out(self, |x| x._sqrt(), out)
    }

    fn recip(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._recip())
    }

    fn recip_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where
            U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                TensorInfo<Self::OutputMeta>
    {
        uary_fn_with_out(self, |x| x._recip(), out)
    }

    fn ln(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._ln())
    }

    fn ln_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where
            U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                TensorInfo<Self::OutputMeta>
    {
        uary_fn_with_out(self, |x| x._ln(), out)
    }

    fn log10(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._log10())
    }

    fn log10_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where
            U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                TensorInfo<Self::OutputMeta>
    {
        uary_fn_with_out(self, |x| x._log10(), out)
    }

    fn log2(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._log2())
    }

    fn log2_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where
            U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                TensorInfo<Self::OutputMeta>
    {
        uary_fn_with_out(self, |x| x._log2(), out)
    }
}
