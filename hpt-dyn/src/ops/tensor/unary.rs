use std::borrow::Borrow;

use crate::Tensor;
use hpt_common::error::base::TensorError;
use hpt_common::error::shape::ShapeError;
use hpt_iterator::TensorIterator;
use hpt_iterator::iterator_traits::ParStridedIteratorSimd;
use hpt_iterator::iterator_traits::ParStridedIteratorSimdZip;
use hpt_traits::tensor::CommonBounds;
use hpt_traits::tensor::TensorInfo;
use hpt_types::dtype::ToDType;
use hpt_types::dtype::TypeCommon;
use hpt_types::into_scalar::Cast;
use hpt_types::traits::VecTrait;
use hpt_types::type_promote::FloatOutUnary;
use hpt_types::type_promote::NormalOutUnary;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::slice::ParallelSlice;
use rayon::slice::ParallelSliceMut;

use half::{bf16, f16};

#[inline(never)]
pub(crate) fn unary_map<A, K, F, F2>(slice_a: &[A], slice_o: &mut [K], f: F, f2: F2)
where
    A: CommonBounds,
    K: CommonBounds,
    F: Fn(A::Vec) -> K::Vec + Sync + Send,
    F2: Fn(A) -> K + Sync + Send,
{
    if K::BYTE_SIZE == A::BYTE_SIZE {
        let mut chunk_o = slice_o.par_chunks_exact_mut(4 * A::Vec::SIZE);
        let chunk_a = slice_a.par_chunks_exact(4 * A::Vec::SIZE);
        chunk_o
            .remainder()
            .into_iter()
            .zip(chunk_a.remainder().into_iter())
            .for_each(|(out, buffer)| {
                *out = f2(*buffer);
            });
        chunk_o
            .into_par_iter()
            .zip(chunk_a.into_par_iter())
            .for_each(|(out, buffer)| {
                let out_ptr = out.as_mut_ptr() as *mut K::Vec;
                let buffer_ptr = buffer.as_ptr() as *const A::Vec;
                unsafe {
                    out_ptr.write_unaligned(f(buffer_ptr.read_unaligned()));
                    out_ptr
                        .add(1)
                        .write_unaligned(f(buffer_ptr.add(1).read_unaligned()));
                    out_ptr
                        .add(2)
                        .write_unaligned(f(buffer_ptr.add(2).read_unaligned()));
                    out_ptr
                        .add(3)
                        .write_unaligned(f(buffer_ptr.add(3).read_unaligned()));
                }
            });
    } else {
        slice_o
            .par_iter_mut()
            .zip(slice_a.par_iter())
            .for_each(|(out, buffer)| {
                *out = f2(*buffer);
            });
    }
}

/// Perform unary operation with output tensor
pub(crate) fn unary_fn_with_out<A, K, O, F, F2>(
    inp: &Tensor,
    f: F,
    f2: F2,
    out: Option<O>,
) -> std::result::Result<Tensor, TensorError>
where
    A: CommonBounds,
    K: CommonBounds,
    O: Borrow<Tensor>,
    F: Fn(A::Vec) -> K::Vec + Sync + Send,
    F2: Fn(A) -> K + Sync + Send,
{
    let mut ret = if let Some(out) = out {
        ShapeError::check_inplace_out_layout_valid(inp.shape(), &out.borrow().layout())?;
        let ret: Tensor = out.borrow().clone();
        ret
    } else {
        inp.empty_like()?
    };
    if inp.parent::<A>().is_some() || !inp.is_contiguous() {
        let iter = ret.par_iter_mut_simd().zip(inp.par_iter_simd());
        ParStridedIteratorSimd::for_each(
            iter,
            |(a, b)| {
                *a = f2(b);
            },
            |(a, b)| {
                a.write_unaligned(f(b));
            },
        );
    } else {
        unary_map(inp.as_slice(), ret.as_slice_mut(), f, f2);
    }
    Ok(ret)
}

impl Tensor {
    #[duplicate::duplicate_item(
        func_name       kernel              description                                                                     test_code;
        [sin]           [_sin]              ["Computes sine of each element in the input tensor."]                          ["let y = x.sin()?"];
        [cos]           [_cos]              ["Computes cosine of each element in the input tensor."]                        ["let y = x.cos()?"];
        [tan]           [_tan]              ["Computes tangent of each element in the input tensor."]                       ["let y = x.tan()?"];
        [asin]          [_asin]             ["Computes arcsine of each element in the input tensor."]                       ["let y = x.asin()?"];
        [acos]          [_acos]             ["Computes arccosine of each element in the input tensor."]                     ["let y = x.acos()?"];
        [atan]          [_atan]             ["Computes arctangent of each element in the input tensor."]                    ["let y = x.atan()?"];
        [sinh]          [_sinh]             ["Computes hyperbolic sine of each element in the input tensor."]               ["let y = x.sinh()?"];
        [cosh]          [_cosh]             ["Computes hyperbolic cosine of each element in the input tensor."]             ["let y = x.cosh()?"];
        [tanh]          [_tanh]             ["Computes hyperbolic tangent of each element in the input tensor."]            ["let y = x.tanh()?"];
        [asinh]         [_asinh]            ["Computes inverse hyperbolic sine of each element in the input tensor."]       ["let y = x.asinh()?"];
        [acosh]         [_acosh]            ["Computes inverse hyperbolic cosine of each element in the input tensor."]     ["let y = x.acosh()?"];
        [atanh]         [_atanh]            ["Computes inverse hyperbolic tangent of each element in the input tensor."]    ["let y = x.atanh()?"];
        [exp]           [_exp]              ["Computes exponential of each element in the input tensor."]                   ["let y = x.exp()?"];
        [exp2]          [_exp2]             ["Computes 2 to the power of each element in the input tensor."]                ["let y = x.exp2()?"];
        [expm1]         [_expm1]            ["Computes exp(x) - 1 of each element in the input tensor."]                    ["let y = x.expm1()?"];
        [ln]            [_ln]               ["Computes natural logarithm of each element in the input tensor."]             ["let y = x.ln()?"];
        [log1p]         [_log1p]            ["Computes natural logarithm of each element in the input tensor."]             ["let y = x.log1p()?"];
        [log2]          [_log2]             ["Computes base 2 logarithm of each element in the input tensor."]              ["let y = x.log2()?"];
        [log10]         [_log10]            ["Computes base 10 logarithm of each element in the input tensor."]             ["let y = x.log10()?"];
        [sqrt]          [_sqrt]             ["Computes square root of each element in the input tensor."]                   ["let y = x.sqrt()?"];
        [cbrt]          [_cbrt]             ["Computes cube root of each element in the input tensor."]                     ["let y = x.cbrt()?"];
        [recip]         [_recip]            ["Computes reciprocal of each element in the input tensor."]                    ["let y = x.recip()?"];
        [erf]           [_erf]              ["Computes error function of each element in the input tensor."]                ["let y = x.erf()?"];
        [sigmoid]       [_sigmoid]          ["Computes sigmoid of each element in the input tensor."]                       ["let y = x.sigmoid()?"];
        [gelu]          [_gelu]             ["Computes GELU of each element in the input tensor."]                          ["let y = x.gelu()?"];
        [hard_sigmoid]  [_hard_sigmoid]     ["Computes hard sigmoid of each element in the input tensor."]                  ["let y = x.hard_sigmoid()?"];
        [hard_swish]    [_hard_swish]       ["Computes hard swish of each element in the input tensor."]                    ["let y = x.hard_swish()?"];
        [softplus]      [_softplus]         ["Computes softplus of each element in the input tensor."]                      ["let y = x.softplus()?"];
        [softsign]      [_softsign]         ["Computes softsign of each element in the input tensor."]                      ["let y = x.softsign()?"];
        [mish]          [_mish]             ["Computes mish of each element in the input tensor."]                          ["let y = x.mish()?"];
    )]
    #[doc = description]
    /// # Parameters:
    /// `x`: Input tensor
    ///
    /// # Example:
    /// ```rust
    /// let x = Tensor::randn(/*mean*/0.0, /*std*/1.0, /*shape*/&[2, 3, 4], DType::F32, Device::Cpu)?;
    #[doc = test_code]
    /// ```
    ///
    /// # Returns:
    /// A new tensor with the same shape as the input tensor.
    pub fn func_name(&self) -> Result<Self, TensorError> {
        macro_rules! unary {
            ($dtype:ty) => {{
                type T = $dtype;
                type O = <T as FloatOutUnary>::Output;
                type InVec = <T as TypeCommon>::Vec;
                let res = Tensor::empty(&self.layout.shape(), O::to_dtype(), self.device.clone())?;

                unary_fn_with_out(self, |x: InVec| x.kernel(), |x: T| x.kernel(), Some(res))
            }};
        }
        match self.dtype {
            hpt_types::dtype::DType::I8 => unary!(i8),
            hpt_types::dtype::DType::U8 => unary!(u8),
            hpt_types::dtype::DType::F32 => unary!(f32),
            hpt_types::dtype::DType::F16 => unary!(f16),
            hpt_types::dtype::DType::BF16 => unary!(bf16),
            _ => unimplemented!(),
        }
    }

    #[duplicate::duplicate_item(
        func_name       kernel              description                                                                     test_code;
        [floor]         [_floor]            ["Computes floor of each element in the input tensor."]                         ["let y = x.floor()?"];
        [ceil]          [_ceil]             ["Computes ceil of each element in the input tensor."]                          ["let y = x.ceil()?"];
        [round]         [_round]            ["Computes round of each element in the input tensor."]                         ["let y = x.round()?"];
        [trunc]         [_trunc]            ["Computes trunc of each element in the input tensor."]                         ["let y = x.trunc()?"];
        [abs]           [_abs]              ["Computes absolute value of each element in the input tensor."]                ["let y = x.abs()?"];
        [neg]           [_neg]              ["Computes negative value of each element in the input tensor."]                ["let y = x.neg()?"];
        [signum]        [_signum]           ["Computes sign of each element in the input tensor."]                          ["let y = x.sign()?"];
        [relu]          [_relu]             ["Computes relu of each element in the input tensor."]                          ["let y = x.relu()?"];
        [relu6]         [_relu6]            ["Computes relu6 of each element in the input tensor."]                         ["let y = x.relu6()?"];
        [square]        [_square]           ["Computes square of each element in the input tensor."]                        ["let y = x.square()?"];
    )]
    #[doc = description]
    /// # Parameters:
    /// `x`: Input tensor
    ///
    /// # Example:
    /// ```rust
    /// let x = Tensor::randn(/*mean*/0.0, /*std*/1.0, /*shape*/&[2, 3, 4], DType::F32, Device::Cpu)?;
    #[doc = test_code]
    /// ```
    ///
    /// # Returns:
    /// A new tensor with the same shape as the input tensor.
    pub fn func_name(&self) -> Result<Self, TensorError> {
        macro_rules! unary {
            ($dtype:ty) => {{
                type T = $dtype;
                type InVec = <T as TypeCommon>::Vec;
                let res = Tensor::empty(&self.layout.shape(), T::to_dtype(), self.device.clone())?;

                unary_fn_with_out(self, |x: InVec| x.kernel(), |x: T| x.kernel(), Some(res))
            }};
        }
        match self.dtype {
            hpt_types::dtype::DType::I8 => unary!(i8),
            hpt_types::dtype::DType::U8 => unary!(u8),
            hpt_types::dtype::DType::F32 => unary!(f32),
            hpt_types::dtype::DType::F16 => unary!(f16),
            hpt_types::dtype::DType::BF16 => unary!(bf16),
            _ => unimplemented!(),
        }
    }

    #[duplicate::duplicate_item(
        func_name       kernel                  description                                                                     test_code;
        [celu]          [_celu]                 ["Computes celu of each element in the input tensor."]                         ["let y = x.celu(/*alpha*/1.0)?"];
        [elu]           [_elu]                  ["Computes elu of each element in the input tensor."]                          ["let y = x.elu(/*alpha*/1.0)?"];
    )]
    #[doc = description]
    /// # Parameters:
    /// `x`: Input tensor
    /// `alpha`: Alpha value
    ///
    /// # Example:
    /// ```rust
    /// let x = Tensor::randn(/*mean*/0.0, /*std*/1.0, /*shape*/&[2, 3, 4], DType::F32, Device::Cpu)?;
    #[doc = test_code]
    /// ```
    ///
    /// # Returns:
    /// A new tensor with the same shape as the input tensor.
    pub fn func_name(&self, alpha: f64) -> Result<Self, TensorError> {
        macro_rules! unary {
            ($dtype:ty) => {{
                type T = $dtype;
                type O = <T as FloatOutUnary>::Output;
                type InVec = <T as TypeCommon>::Vec;
                type OVec = <O as TypeCommon>::Vec;
                let res = Tensor::empty(&self.layout.shape(), O::to_dtype(), self.device.clone())?;
                let alpha: O = alpha.cast();
                let alpha_vec = OVec::splat(alpha);
                unary_fn_with_out(
                    self,
                    |x: InVec| x.kernel(alpha_vec),
                    |x: T| x.kernel(alpha),
                    Some(res),
                )
            }};
        }
        match self.dtype {
            hpt_types::dtype::DType::I8 => unary!(i8),
            hpt_types::dtype::DType::U8 => unary!(u8),
            hpt_types::dtype::DType::F32 => unary!(f32),
            hpt_types::dtype::DType::F16 => unary!(f16),
            hpt_types::dtype::DType::BF16 => unary!(bf16),
            _ => unimplemented!(),
        }
    }

    pub fn selu(&self) -> Result<Self, TensorError> {
        macro_rules! unary {
            ($dtype:ty) => {{
                type T = $dtype;
                type O = <T as FloatOutUnary>::Output;
                type InVec = <T as TypeCommon>::Vec;
                type OVec = <O as TypeCommon>::Vec;
                let res = Tensor::empty(&self.layout.shape(), O::to_dtype(), self.device.clone())?;
                let arg1 = 1.6732632423543772848170429916717;
                let arg2 = 1.0507009873554804934193349852946;
                let arg1: O = arg1.cast();
                let arg2: O = arg2.cast();
                let arg1_vec = OVec::splat(arg1);
                let arg2_vec = OVec::splat(arg2);
                unary_fn_with_out(
                    self,
                    |x: InVec| x._selu(arg1_vec, arg2_vec),
                    |x: T| x._selu(arg1, arg2),
                    Some(res),
                )
            }};
        }
        match self.dtype {
            hpt_types::dtype::DType::I8 => unary!(i8),
            hpt_types::dtype::DType::U8 => unary!(u8),
            hpt_types::dtype::DType::F32 => unary!(f32),
            hpt_types::dtype::DType::F16 => unary!(f16),
            hpt_types::dtype::DType::BF16 => unary!(bf16),
            _ => unimplemented!(),
        }
    }

    pub fn contiguous(&self) -> Result<Self, TensorError> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }
        let res = Tensor::empty(
            &self.layout.shape(),
            self.dtype.clone(),
            self.device.clone(),
        )?;
        macro_rules! contiguous {
            ($dtype:ty) => {{
                type T = $dtype;
                type InVec = <T as TypeCommon>::Vec;
                unary_fn_with_out(self, |x: InVec| x, |x: T| x, Some(res.clone()))
            }};
        }
        match self.dtype {
            #[cfg(feature = "bool")]
            hpt_types::dtype::DType::Bool => contiguous!(bool),
            #[cfg(feature = "i8")]
            hpt_types::dtype::DType::I8 => contiguous!(i8),
            #[cfg(feature = "u8")]
            hpt_types::dtype::DType::U8 => contiguous!(u8),
            #[cfg(feature = "i16")]
            hpt_types::dtype::DType::I16 => contiguous!(i16),
            #[cfg(feature = "u16")]
            hpt_types::dtype::DType::U16 => contiguous!(u16),
            #[cfg(feature = "i32")]
            hpt_types::dtype::DType::I32 => contiguous!(i32),
            #[cfg(feature = "u32")]
            hpt_types::dtype::DType::U32 => contiguous!(u32),
            #[cfg(feature = "i64")]
            hpt_types::dtype::DType::I64 => contiguous!(i64),
            #[cfg(feature = "f32")]
            hpt_types::dtype::DType::F32 => contiguous!(f32),
            #[cfg(feature = "f16")]
            hpt_types::dtype::DType::F16 => contiguous!(f16),
            #[cfg(feature = "bf16")]
            hpt_types::dtype::DType::BF16 => contiguous!(bf16),
            #[cfg(feature = "u64")]
            hpt_types::dtype::DType::U64 => contiguous!(u64),
            #[cfg(feature = "f64")]
            hpt_types::dtype::DType::F64 => contiguous!(f64),
            _ => panic!("unsupported dtype {:?}", self.dtype),
        }
    }
}
