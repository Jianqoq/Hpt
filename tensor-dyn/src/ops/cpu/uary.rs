use rayon::iter::{
    IndexedParallelIterator,
    IntoParallelRefIterator,
    IntoParallelRefMutIterator,
    ParallelIterator,
};
use tensor_common::shape_utils::mt_intervals;
use tensor_traits::BaseTensor;
use tensor_types::into_scalar::IntoScalar;
use threadpool::ThreadPool;
use crate::backend::Cpu;
use crate::THREAD_POOL;
use tensor_traits::ops::uary::{ Cum, FloatUaryOps, Neg, NormalUaryOps };
use tensor_traits::tensor::{ CommonBounds, TensorInfo, TensorLike };
use tensor_traits::tensor::TensorCreator;
use tensor_types::type_promote::{ FloatOut, NormalOut };
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

pub(crate) type FloatType<T> = <T as FloatOut>::Output;

impl<T> FloatUaryOps
    for _Tensor<T>
    where
        T: FloatOut + CommonBounds,
        FloatType<T>: CommonBounds,
        f64: IntoScalar<FloatType<T>>,
        _Tensor<<T as FloatOut>::Output>: TensorLike<
            <T as FloatOut>::Output,
            Output = _Tensor<<T as FloatOut>::Output>
        >
{
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
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._sin(), out.base().clone())
    }

    fn cos_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._cos(), out.base().clone())
    }

    fn tan_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._tan(), out.base().clone())
    }

    fn asin_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._asin(), out.base().clone())
    }

    fn acos_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._acos(), out.base().clone())
    }

    fn atan_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._atan(), out.base().clone())
    }

    fn sinh_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._sinh(), out.base().clone())
    }

    fn cosh_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._cosh(), out.base().clone())
    }

    fn tanh_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._tanh(), out.base().clone())
    }

    fn asinh_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._asinh(), out.base().clone())
    }

    fn acosh_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._acosh(), out.base().clone())
    }

    fn atanh_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._atanh(), out.base().clone())
    }

    fn exp(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._exp())
    }

    fn exp_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._exp(), out.base().clone())
    }

    fn exp2(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._exp2())
    }

    fn exp2_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._exp2(), out.base().clone())
    }

    fn sqrt(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._sqrt())
    }

    fn sqrt_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._sqrt(), out.base().clone())
    }

    fn recip(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._recip())
    }

    fn recip_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._recip(), out.base().clone())
    }

    fn ln(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._ln())
    }

    fn ln_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._ln(), out.base().clone())
    }

    fn log2(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._log2())
    }

    fn log2_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._log2(), out.base().clone())
    }

    fn log10(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._log10())
    }

    fn log10_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._log10(), out.base().clone())
    }

    fn celu(&self, alpha: Self::OutputMeta) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._celu(alpha))
    }

    fn celu_<U>(&self, alpha: Self::OutputMeta, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._celu(alpha), out.base().clone())
    }

    fn sigmoid(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._sigmoid())
    }

    fn sigmoid_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._sigmoid(), out.base().clone())
    }

    fn elu(&self, alpha: Self::OutputMeta) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._elu(alpha))
    }

    fn elu_<U>(&self, alpha: Self::OutputMeta, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._elu(alpha), out.base().clone())
    }

    fn leaky_relu(&self, alpha: Self::OutputMeta) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._leaky_relu(alpha))
    }

    fn leaky_relu_<U>(&self, alpha: Self::OutputMeta, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._leaky_relu(alpha), out.base().clone())
    }

    fn gelu(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._gelu())
    }

    fn gelu_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._gelu(), out.base().clone())
    }

    fn selu(
        &self,
        alpha: Option<Self::OutputMeta>,
        gamma: Option<Self::OutputMeta>
    ) -> anyhow::Result<Self::Output> {
        let alpha = alpha.unwrap_or((1.67326319217681884765625).into_scalar());
        let gamma = gamma.unwrap_or((1.05070102214813232421875).into_scalar());
        uary_fn(self, |x| x._selu(alpha, gamma))
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
        uary_fn_with_out(self, |x| x._selu(alpha, gamma), out.base().clone())
    }

    fn hard_sigmoid(
        &self,
        alpha: Option<Self::OutputMeta>,
        beta: Option<Self::OutputMeta>
    ) -> anyhow::Result<Self::Output> {
        let alpha = alpha.unwrap_or((0.2).into_scalar());
        let beta = beta.unwrap_or((0.5).into_scalar());
        uary_fn(self, |x| x._hard_sigmoid(alpha, beta))
    }

    fn hard_sigmoid_<U>(
        &self,
        alpha: Option<Self::OutputMeta>,
        beta: Option<Self::OutputMeta>,
        out: U
    ) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        let alpha = alpha.unwrap_or((0.2).into_scalar());
        let beta = beta.unwrap_or((0.5).into_scalar());
        uary_fn_with_out(self, |x| x._hard_sigmoid(alpha, beta), out.base().clone())
    }

    fn hard_swish(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._hard_swish())
    }

    fn hard_swish_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._hard_swish(), out.base().clone())
    }

    fn relu6(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._relu6())
    }

    fn relu6_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._relu6(), out.base().clone())
    }

    fn softplus(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._softplus())
    }

    fn softplus_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._softplus(), out.base().clone())
    }

    fn softsign(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._softsign())
    }

    fn softsign_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._softsign(), out.base().clone())
    }

    fn mish(&self) -> anyhow::Result<Self::Output> {
        uary_fn(self, |x| x._mish())
    }

    fn mish_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>
    {
        uary_fn_with_out(self, |x| x._mish(), out.base().clone())
    }
}

pub(crate) type NormalType<T> = <T as NormalOut>::Output;

impl<T> NormalUaryOps
    for _Tensor<T>
    where
        T: NormalOut + CommonBounds + IntoScalar<T>,
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
        return match axis {
            Some(axis) => {
                let mut _axis = axis;
                if axis < 0 {
                    _axis = (self.ndim() as i64) + axis;
                    if _axis < 0 {
                        anyhow::bail!("axis out of range");
                    }
                }
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
        };
    }

    fn cumprod(&self, axis: Option<i64>) -> anyhow::Result<Self>
        where Self::Meta: NormalOut<Self::Meta, Output = Self::Meta>
    {
        return match axis {
            Some(axis) => {
                let mut _axis = axis;
                if axis < 0 {
                    _axis = (self.ndim() as i64) + axis;
                    if _axis < 0 {
                        anyhow::bail!("axis out of range");
                    }
                }
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
        };
    }
}
