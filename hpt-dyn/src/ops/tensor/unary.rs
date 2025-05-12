use std::borrow::Borrow;

use crate::Tensor;
use hpt_common::error::base::TensorError;
use hpt_common::error::shape::ShapeError;
use hpt_iterator::TensorIterator;
use hpt_iterator::iterator_traits::ParStridedIteratorSimd;
use hpt_iterator::iterator_traits::ParStridedIteratorSimdZip;
use hpt_traits::tensor::CommonBounds;
use hpt_traits::tensor::TensorInfo;
use hpt_types::dtype::DType;
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
        self.kernel(None)
    }

    #[duplicate::duplicate_item(
        func_name       kernel              description                                                                     test_code;
        [sin_]           [_sin]              ["Computes sine of each element in the input tensor."]                          ["let y = x.sin_(&mut x.clone())?"];
        [cos_]           [_cos]              ["Computes cosine of each element in the input tensor."]                        ["let y = x.cos_(&mut x.clone())?"];
        [tan_]           [_tan]              ["Computes tangent of each element in the input tensor."]                       ["let y = x.tan_(&mut x.clone())?"];
        [asin_]          [_asin]             ["Computes arcsine of each element in the input tensor."]                       ["let y = x.asin_(&mut x.clone())?"];
        [acos_]          [_acos]             ["Computes arccosine of each element in the input tensor."]                     ["let y = x.acos_(&mut x.clone())?"];
        [atan_]          [_atan]             ["Computes arctangent of each element in the input tensor."]                    ["let y = x.atan_(&mut x.clone())?"];
        [sinh_]          [_sinh]             ["Computes hyperbolic sine of each element in the input tensor."]               ["let y = x.sinh_(&mut x.clone())?"];
        [cosh_]          [_cosh]             ["Computes hyperbolic cosine of each element in the input tensor."]             ["let y = x.cosh_(&mut x.clone())?"];
        [tanh_]          [_tanh]             ["Computes hyperbolic tangent of each element in the input tensor."]            ["let y = x.tanh_(&mut x.clone())?"];
        [asinh_]         [_asinh]            ["Computes inverse hyperbolic sine of each element in the input tensor."]       ["let y = x.asinh_(&mut x.clone())?"];
        [acosh_]         [_acosh]            ["Computes inverse hyperbolic cosine of each element in the input tensor."]     ["let y = x.acosh_(&mut x.clone())?"];
        [atanh_]         [_atanh]            ["Computes inverse hyperbolic tangent of each element in the input tensor."]    ["let y = x.atanh_(&mut x.clone())?"];
        [exp_]           [_exp]              ["Computes exponential of each element in the input tensor."]                   ["let y = x.exp_(&mut x.clone())?"];
        [exp2_]          [_exp2]             ["Computes 2 to the power of each element in the input tensor."]                ["let y = x.exp2_(&mut x.clone())?"];
        [expm1_]         [_expm1]            ["Computes exp(x) - 1 of each element in the input tensor."]                    ["let y = x.expm1_(&mut x.clone())?"];
        [ln_]            [_ln]               ["Computes natural logarithm of each element in the input tensor."]             ["let y = x.ln_(&mut x.clone())?"];
        [log1p_]         [_log1p]            ["Computes natural logarithm of each element in the input tensor."]             ["let y = x.log1p_(&mut x.clone())?"];
        [log2_]          [_log2]             ["Computes base 2 logarithm of each element in the input tensor."]              ["let y = x.log2_(&mut x.clone())?"];
        [log10_]         [_log10]            ["Computes base 10 logarithm of each element in the input tensor."]             ["let y = x.log10_(&mut x.clone())?"];
        [sqrt_]          [_sqrt]             ["Computes square root of each element in the input tensor."]                   ["let y = x.sqrt_(&mut x.clone())?"];
        [cbrt_]          [_cbrt]             ["Computes cube root of each element in the input tensor."]                     ["let y = x.cbrt_(&mut x.clone())?"];
        [recip_]         [_recip]            ["Computes reciprocal of each element in the input tensor."]                    ["let y = x.recip_(&mut x.clone())?"];
        [erf_]           [_erf]              ["Computes error function of each element in the input tensor."]                ["let y = x.erf_(&mut x.clone())?"];
        [sigmoid_]       [_sigmoid]          ["Computes sigmoid of each element in the input tensor."]                       ["let y = x.sigmoid_(&mut x.clone())?"];
        [gelu_]          [_gelu]             ["Computes GELU of each element in the input tensor."]                          ["let y = x.gelu_(&mut x.clone())?"];
        [hard_sigmoid_]  [_hard_sigmoid]     ["Computes hard sigmoid of each element in the input tensor."]                  ["let y = x.hard_sigmoid_(&mut x.clone())?"];
        [hard_swish_]    [_hard_swish]       ["Computes hard swish of each element in the input tensor."]                    ["let y = x.hard_swish_(&mut x.clone())?"];
        [softplus_]      [_softplus]         ["Computes softplus of each element in the input tensor."]                      ["let y = x.softplus_(&mut x.clone())?"];
        [softsign_]      [_softsign]         ["Computes softsign of each element in the input tensor."]                      ["let y = x.softsign_(&mut x.clone())?"];
        [mish_]          [_mish]             ["Computes mish of each element in the input tensor."]                          ["let y = x.mish_(&mut x.clone())?"];
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
    pub fn func_name(&self, out: &mut Tensor) -> Result<Self, TensorError> {
        self.kernel(Some(out.clone()))
    }

    #[duplicate::duplicate_item(
        func_name       kernel;
        [_sin]           [_sin];
        [_cos]           [_cos];
        [_tan]           [_tan];
        [_asin]          [_asin];
        [_acos]          [_acos];
        [_atan]          [_atan];
        [_sinh]          [_sinh];
        [_cosh]          [_cosh];
        [_tanh]          [_tanh];
        [_asinh]         [_asinh];
        [_acosh]         [_acosh];
        [_atanh]         [_atanh];
        [_exp]           [_exp];
        [_exp2]          [_exp2];
        [_expm1]         [_expm1];
        [_ln]            [_ln];
        [_log1p]         [_log1p];
        [_log2]          [_log2];
        [_log10]         [_log10];
        [_sqrt]          [_sqrt];
        [_cbrt]          [_cbrt];
        [_recip]         [_recip];
        [_erf]           [_erf];
        [_sigmoid]       [_sigmoid];
        [_gelu]          [_gelu];
        [_hard_sigmoid]  [_hard_sigmoid];
        [_hard_swish]    [_hard_swish];
        [_softplus]      [_softplus];
        [_softsign]      [_softsign];
        [_mish]          [_mish];
    )]
    pub fn func_name(&self, out: Option<Tensor>) -> Result<Self, TensorError> {
        macro_rules! unary {
            ($dtype:ty) => {{
                type T = $dtype;
                type O = <T as FloatOutUnary>::Output;
                type InVec = <T as TypeCommon>::Vec;
                let res = if let Some(out) = out {
                    out
                } else {
                    Tensor::empty(&self.layout.shape(), O::to_dtype(), self.device.clone())?
                };

                unary_fn_with_out(self, |x: InVec| x.kernel(), |x: T| x.kernel(), Some(res))
            }};
        }
        match self.dtype {
            #[cfg(feature = "i8")]
            hpt_types::dtype::DType::I8 => unary!(i8),
            #[cfg(feature = "u8")]
            hpt_types::dtype::DType::U8 => unary!(u8),
            #[cfg(feature = "f32")]
            hpt_types::dtype::DType::F32 => unary!(f32),
            #[cfg(feature = "f16")]
            hpt_types::dtype::DType::F16 => unary!(f16),
            #[cfg(feature = "bf16")]
            hpt_types::dtype::DType::BF16 => unary!(bf16),
            #[cfg(feature = "f64")]
            hpt_types::dtype::DType::F64 => unary!(f64),
            #[cfg(feature = "bool")]
            hpt_types::dtype::DType::Bool => unary!(bool),
            #[cfg(feature = "u64")]
            hpt_types::dtype::DType::U64 => unary!(u64),
            #[cfg(feature = "i16")]
            hpt_types::dtype::DType::I16 => unary!(i16),
            #[cfg(feature = "u16")]
            hpt_types::dtype::DType::U16 => unary!(u16),
            #[cfg(feature = "i32")]
            hpt_types::dtype::DType::I32 => unary!(i32),
            #[cfg(feature = "u32")]
            hpt_types::dtype::DType::U32 => unary!(u32),
            #[cfg(feature = "i64")]
            hpt_types::dtype::DType::I64 => unary!(i64),
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
            #[cfg(feature = "i8")]
            hpt_types::dtype::DType::I8 => unary!(i8),
            #[cfg(feature = "u8")]
            hpt_types::dtype::DType::U8 => unary!(u8),
            #[cfg(feature = "f32")]
            hpt_types::dtype::DType::F32 => unary!(f32),
            #[cfg(feature = "f16")]
            hpt_types::dtype::DType::F16 => unary!(f16),
            #[cfg(feature = "bf16")]
            hpt_types::dtype::DType::BF16 => unary!(bf16),
            #[cfg(feature = "f64")]
            hpt_types::dtype::DType::F64 => unary!(f64),
            #[cfg(feature = "bool")]
            hpt_types::dtype::DType::Bool => unary!(bool),
            #[cfg(feature = "u64")]
            hpt_types::dtype::DType::U64 => unary!(u64),
            #[cfg(feature = "i16")]
            hpt_types::dtype::DType::I16 => unary!(i16),
            #[cfg(feature = "u16")]
            hpt_types::dtype::DType::U16 => unary!(u16),
            #[cfg(feature = "i32")]
            hpt_types::dtype::DType::I32 => unary!(i32),
            #[cfg(feature = "u32")]
            hpt_types::dtype::DType::U32 => unary!(u32),
            #[cfg(feature = "i64")]
            hpt_types::dtype::DType::I64 => unary!(i64),
            _ => unimplemented!(),
        }
    }

    #[duplicate::duplicate_item(
        func_name        kernel              description                                                                     test_code;
        [floor_]         [_floor]            ["Computes floor of each element in the input tensor."]                         ["let y = x.floor_(&mut x.clone())?"];
        [ceil_]          [_ceil]             ["Computes ceil of each element in the input tensor."]                          ["let y = x.ceil_(&mut x.clone())?"];
        [round_]         [_round]            ["Computes round of each element in the input tensor."]                         ["let y = x.round_(&mut x.clone())?"];
        [trunc_]         [_trunc]            ["Computes trunc of each element in the input tensor."]                         ["let y = x.trunc_(&mut x.clone())?"];
        [abs_]           [_abs]              ["Computes absolute value of each element in the input tensor."]                ["let y = x.abs_(&mut x.clone())?"];
        [neg_]           [_neg]              ["Computes negative value of each element in the input tensor."]                ["let y = x.neg_(&mut x.clone())?"];
        [signum_]        [_signum]           ["Computes sign of each element in the input tensor."]                          ["let y = x.sign_(&mut x.clone())?"];
        [relu_]          [_relu]             ["Computes relu of each element in the input tensor."]                          ["let y = x.relu_(&mut x.clone())?"];
        [relu6_]         [_relu6]            ["Computes relu6 of each element in the input tensor."]                         ["let y = x.relu6_(&mut x.clone())?"];
        [square_]        [_square]           ["Computes square of each element in the input tensor."]                        ["let y = x.square_(&mut x.clone())?"];
    )]
    #[doc = description]
    /// # Parameters:
    /// `x`: Input tensor
    ///
    /// `out`: Output tensor
    ///
    /// # Example:
    /// ```rust
    /// let x = Tensor::randn(/*mean*/0.0, /*std*/1.0, /*shape*/&[2, 3, 4], DType::F32, Device::Cpu)?;
    #[doc = test_code]
    /// ```
    ///
    /// # Returns:
    /// A new tensor with the same shape as the input tensor.
    pub fn func_name(&self, out: &mut Tensor) -> Result<Self, TensorError> {
        macro_rules! unary {
            ($dtype:ty) => {{
                type T = $dtype;
                type InVec = <T as TypeCommon>::Vec;
                unary_fn_with_out(self, |x: InVec| x.kernel(), |x: T| x.kernel(), Some(out))
            }};
        }
        match self.dtype {
            #[cfg(feature = "i8")]
            hpt_types::dtype::DType::I8 => unary!(i8),
            #[cfg(feature = "u8")]
            hpt_types::dtype::DType::U8 => unary!(u8),
            #[cfg(feature = "f32")]
            hpt_types::dtype::DType::F32 => unary!(f32),
            #[cfg(feature = "f16")]
            hpt_types::dtype::DType::F16 => unary!(f16),
            #[cfg(feature = "bf16")]
            hpt_types::dtype::DType::BF16 => unary!(bf16),
            #[cfg(feature = "f64")]
            hpt_types::dtype::DType::F64 => unary!(f64),
            #[cfg(feature = "bool")]
            hpt_types::dtype::DType::Bool => unary!(bool),
            #[cfg(feature = "u64")]
            hpt_types::dtype::DType::U64 => unary!(u64),
            #[cfg(feature = "i16")]
            hpt_types::dtype::DType::I16 => unary!(i16),
            #[cfg(feature = "u16")]
            hpt_types::dtype::DType::U16 => unary!(u16),
            #[cfg(feature = "i32")]
            hpt_types::dtype::DType::I32 => unary!(i32),
            #[cfg(feature = "u32")]
            hpt_types::dtype::DType::U32 => unary!(u32),
            #[cfg(feature = "i64")]
            hpt_types::dtype::DType::I64 => unary!(i64),
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
            #[cfg(feature = "i8")]
            hpt_types::dtype::DType::I8 => unary!(i8),
            #[cfg(feature = "u8")]
            hpt_types::dtype::DType::U8 => unary!(u8),
            #[cfg(feature = "f32")]
            hpt_types::dtype::DType::F32 => unary!(f32),
            #[cfg(feature = "f16")]
            hpt_types::dtype::DType::F16 => unary!(f16),
            #[cfg(feature = "bf16")]
            hpt_types::dtype::DType::BF16 => unary!(bf16),
            #[cfg(feature = "f64")]
            hpt_types::dtype::DType::F64 => unary!(f64),
            #[cfg(feature = "bool")]
            hpt_types::dtype::DType::Bool => unary!(bool),
            #[cfg(feature = "u64")]
            hpt_types::dtype::DType::U64 => unary!(u64),
            #[cfg(feature = "i16")]
            hpt_types::dtype::DType::I16 => unary!(i16),
            #[cfg(feature = "u16")]
            hpt_types::dtype::DType::U16 => unary!(u16),
            #[cfg(feature = "i32")]
            hpt_types::dtype::DType::I32 => unary!(i32),
            #[cfg(feature = "u32")]
            hpt_types::dtype::DType::U32 => unary!(u32),
            #[cfg(feature = "i64")]
            hpt_types::dtype::DType::I64 => unary!(i64),
            _ => unimplemented!(),
        }
    }

    pub fn selu(&self, alpha: f64, gamma: f64) -> Result<Self, TensorError> {
        self._selu(alpha, gamma, None)
    }

    pub fn selu_(&self, alpha: f64, gamma: f64, out: &mut Tensor) -> Result<Self, TensorError> {
        self._selu(alpha, gamma, Some(out.clone()))
    }

    pub(crate) fn _selu(
        &self,
        alpha: f64,
        gamma: f64,
        out: Option<Tensor>,
    ) -> Result<Self, TensorError> {
        macro_rules! unary {
            ($dtype:ty) => {{
                type T = $dtype;
                type O = <T as FloatOutUnary>::Output;
                type InVec = <T as TypeCommon>::Vec;
                type OVec = <O as TypeCommon>::Vec;
                let out = if let Some(out) = out {
                    out
                } else {
                    Tensor::empty(&self.layout.shape(), O::to_dtype(), self.device.clone())?
                };
                let arg1 = alpha;
                let arg2 = gamma;
                let arg1: O = arg1.cast();
                let arg2: O = arg2.cast();
                let arg1_vec = OVec::splat(arg1);
                let arg2_vec = OVec::splat(arg2);
                unary_fn_with_out(
                    self,
                    |x: InVec| x._selu(arg1_vec, arg2_vec),
                    |x: T| x._selu(arg1, arg2),
                    Some(out),
                )
            }};
        }
        match self.dtype {
            #[cfg(feature = "i8")]
            hpt_types::dtype::DType::I8 => unary!(i8),
            #[cfg(feature = "u8")]
            hpt_types::dtype::DType::U8 => unary!(u8),
            #[cfg(feature = "f32")]
            hpt_types::dtype::DType::F32 => unary!(f32),
            #[cfg(feature = "f16")]
            hpt_types::dtype::DType::F16 => unary!(f16),
            #[cfg(feature = "bf16")]
            hpt_types::dtype::DType::BF16 => unary!(bf16),
            #[cfg(feature = "f64")]
            hpt_types::dtype::DType::F64 => unary!(f64),
            #[cfg(feature = "bool")]
            hpt_types::dtype::DType::Bool => unary!(bool),
            #[cfg(feature = "u64")]
            hpt_types::dtype::DType::U64 => unary!(u64),
            #[cfg(feature = "i16")]
            hpt_types::dtype::DType::I16 => unary!(i16),
            #[cfg(feature = "u16")]
            hpt_types::dtype::DType::U16 => unary!(u16),
            #[cfg(feature = "i32")]
            hpt_types::dtype::DType::I32 => unary!(i32),
            #[cfg(feature = "u32")]
            hpt_types::dtype::DType::U32 => unary!(u32),
            #[cfg(feature = "i64")]
            hpt_types::dtype::DType::I64 => unary!(i64),
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

    pub fn cast(&self, dtype: DType) -> Result<Self, TensorError> {
        use hpt_types::into_vec::IntoVec;

        if self.dtype == dtype {
            return Ok(self.clone());
        }
        let res = Tensor::empty(&self.layout.shape(), dtype, self.device.clone())?;
        macro_rules! cast {
            ($dtype:ty, $target:ty) => {{
                type T = $dtype;
                type InVec = <T as TypeCommon>::Vec;
                type OutVec = <$target as TypeCommon>::Vec;
                unary_fn_with_out(
                    self,
                    |x: InVec| {
                        let out: OutVec = x.into_vec();
                        out
                    },
                    |x: T| {
                        let out: $target = x.cast();
                        out
                    },
                    Some(res.clone()),
                )
            }};
        }
        match (self.dtype, dtype) {
            #[cfg(feature = "i8")]
            (hpt_types::dtype::DType::I8, hpt_types::dtype::DType::I8) => cast!(i8, i8),
            #[cfg(all(feature = "i8", feature = "u8"))]
            (hpt_types::dtype::DType::I8, hpt_types::dtype::DType::U8) => cast!(i8, u8),
            #[cfg(all(feature = "i8", feature = "f32"))]
            (hpt_types::dtype::DType::I8, hpt_types::dtype::DType::F32) => cast!(i8, f32),
            #[cfg(all(feature = "i8", feature = "f16"))]
            (hpt_types::dtype::DType::I8, hpt_types::dtype::DType::F16) => cast!(i8, f16),
            #[cfg(all(feature = "i8", feature = "bf16"))]
            (hpt_types::dtype::DType::I8, hpt_types::dtype::DType::BF16) => cast!(i8, bf16),
            #[cfg(all(feature = "u8", feature = "i8"))]
            (hpt_types::dtype::DType::U8, hpt_types::dtype::DType::I8) => cast!(u8, i8),
            #[cfg(feature = "u8")]
            (hpt_types::dtype::DType::U8, hpt_types::dtype::DType::U8) => cast!(u8, u8),
            #[cfg(all(feature = "u8", feature = "f32"))]
            (hpt_types::dtype::DType::U8, hpt_types::dtype::DType::F32) => cast!(u8, f32),
            #[cfg(all(feature = "u8", feature = "f16"))]
            (hpt_types::dtype::DType::U8, hpt_types::dtype::DType::F16) => cast!(u8, f16),
            #[cfg(all(feature = "u8", feature = "bf16"))]
            (hpt_types::dtype::DType::U8, hpt_types::dtype::DType::BF16) => cast!(u8, bf16),
            #[cfg(all(feature = "f32", feature = "i8"))]
            (hpt_types::dtype::DType::F32, hpt_types::dtype::DType::I8) => cast!(f32, i8),
            #[cfg(all(feature = "f32", feature = "u8"))]
            (hpt_types::dtype::DType::F32, hpt_types::dtype::DType::U8) => cast!(f32, u8),
            #[cfg(feature = "f32")]
            (hpt_types::dtype::DType::F32, hpt_types::dtype::DType::F32) => cast!(f32, f32),
            #[cfg(all(feature = "f32", feature = "f16"))]
            (hpt_types::dtype::DType::F32, hpt_types::dtype::DType::F16) => cast!(f32, f16),
            #[cfg(all(feature = "f32", feature = "bf16"))]
            (hpt_types::dtype::DType::F32, hpt_types::dtype::DType::BF16) => cast!(f32, bf16),
            #[cfg(all(feature = "f16", feature = "i8"))]
            (hpt_types::dtype::DType::F16, hpt_types::dtype::DType::I8) => cast!(f16, i8),
            #[cfg(all(feature = "f16", feature = "u8"))]
            (hpt_types::dtype::DType::F16, hpt_types::dtype::DType::U8) => cast!(f16, u8),
            #[cfg(all(feature = "f16", feature = "f32"))]
            (hpt_types::dtype::DType::F16, hpt_types::dtype::DType::F32) => cast!(f16, f32),
            #[cfg(feature = "f16")]
            (hpt_types::dtype::DType::F16, hpt_types::dtype::DType::F16) => cast!(f16, f16),
            #[cfg(all(feature = "f16", feature = "bf16"))]
            (hpt_types::dtype::DType::F16, hpt_types::dtype::DType::BF16) => cast!(f16, bf16),
            #[cfg(all(feature = "bf16", feature = "i8"))]
            (hpt_types::dtype::DType::BF16, hpt_types::dtype::DType::I8) => cast!(bf16, i8),
            #[cfg(all(feature = "bf16", feature = "u8"))]
            (hpt_types::dtype::DType::BF16, hpt_types::dtype::DType::U8) => cast!(bf16, u8),
            #[cfg(all(feature = "bf16", feature = "f32"))]
            (hpt_types::dtype::DType::BF16, hpt_types::dtype::DType::F32) => cast!(bf16, f32),
            #[cfg(all(feature = "bf16", feature = "f16"))]
            (hpt_types::dtype::DType::BF16, hpt_types::dtype::DType::F16) => cast!(bf16, f16),
            #[cfg(feature = "bf16")]
            (hpt_types::dtype::DType::BF16, hpt_types::dtype::DType::BF16) => cast!(bf16, bf16),
            #[cfg(all(feature = "bool", feature = "bool"))]
            (hpt_types::dtype::DType::Bool, hpt_types::dtype::DType::Bool) => cast!(bool, bool),
            #[cfg(all(feature = "bool", feature = "i8"))]
            (hpt_types::dtype::DType::Bool, hpt_types::dtype::DType::I8) => cast!(bool, i8),
            #[cfg(all(feature = "bool", feature = "u8"))]
            (hpt_types::dtype::DType::Bool, hpt_types::dtype::DType::U8) => cast!(bool, u8),
            #[cfg(all(feature = "bool", feature = "i16"))]
            (hpt_types::dtype::DType::Bool, hpt_types::dtype::DType::I16) => cast!(bool, i16),
            #[cfg(all(feature = "bool", feature = "u16"))]
            (hpt_types::dtype::DType::Bool, hpt_types::dtype::DType::U16) => cast!(bool, u16),
            #[cfg(all(feature = "bool", feature = "i32"))]
            (hpt_types::dtype::DType::Bool, hpt_types::dtype::DType::I32) => cast!(bool, i32),
            #[cfg(all(feature = "bool", feature = "u32"))]
            (hpt_types::dtype::DType::Bool, hpt_types::dtype::DType::U32) => cast!(bool, u32),
            #[cfg(all(feature = "bool", feature = "i64"))]
            (hpt_types::dtype::DType::Bool, hpt_types::dtype::DType::I64) => cast!(bool, i64),
            #[cfg(all(feature = "bool", feature = "u64"))]
            (hpt_types::dtype::DType::Bool, hpt_types::dtype::DType::U64) => cast!(bool, u64),
            #[cfg(all(feature = "bool", feature = "f32"))]
            (hpt_types::dtype::DType::Bool, hpt_types::dtype::DType::F32) => cast!(bool, f32),
            #[cfg(all(feature = "bool", feature = "f16"))]
            (hpt_types::dtype::DType::Bool, hpt_types::dtype::DType::F16) => cast!(bool, f16),
            #[cfg(all(feature = "bool", feature = "bf16"))]
            (hpt_types::dtype::DType::Bool, hpt_types::dtype::DType::BF16) => cast!(bool, bf16),
            #[cfg(all(feature = "bool", feature = "f64"))]
            (hpt_types::dtype::DType::Bool, hpt_types::dtype::DType::F64) => cast!(bool, f64),
            #[cfg(all(feature = "i8", feature = "bool"))]
            (hpt_types::dtype::DType::I8, hpt_types::dtype::DType::Bool) => cast!(i8, bool),
            #[cfg(all(feature = "i8", feature = "i16"))]
            (hpt_types::dtype::DType::I8, hpt_types::dtype::DType::I16) => cast!(i8, i16),
            #[cfg(all(feature = "i8", feature = "u16"))]
            (hpt_types::dtype::DType::I8, hpt_types::dtype::DType::U16) => cast!(i8, u16),
            #[cfg(all(feature = "i8", feature = "i32"))]
            (hpt_types::dtype::DType::I8, hpt_types::dtype::DType::I32) => cast!(i8, i32),
            #[cfg(all(feature = "i8", feature = "u32"))]
            (hpt_types::dtype::DType::I8, hpt_types::dtype::DType::U32) => cast!(i8, u32),
            #[cfg(all(feature = "i8", feature = "i64"))]
            (hpt_types::dtype::DType::I8, hpt_types::dtype::DType::I64) => cast!(i8, i64),
            #[cfg(all(feature = "i8", feature = "u64"))]
            (hpt_types::dtype::DType::I8, hpt_types::dtype::DType::U64) => cast!(i8, u64),
            #[cfg(all(feature = "i8", feature = "f64"))]
            (hpt_types::dtype::DType::I8, hpt_types::dtype::DType::F64) => cast!(i8, f64),
            #[cfg(all(feature = "u8", feature = "bool"))]
            (hpt_types::dtype::DType::U8, hpt_types::dtype::DType::Bool) => cast!(u8, bool),
            #[cfg(all(feature = "u8", feature = "i16"))]
            (hpt_types::dtype::DType::U8, hpt_types::dtype::DType::I16) => cast!(u8, i16),
            #[cfg(all(feature = "u8", feature = "u16"))]
            (hpt_types::dtype::DType::U8, hpt_types::dtype::DType::U16) => cast!(u8, u16),
            #[cfg(all(feature = "u8", feature = "i32"))]
            (hpt_types::dtype::DType::U8, hpt_types::dtype::DType::I32) => cast!(u8, i32),
            #[cfg(all(feature = "u8", feature = "u32"))]
            (hpt_types::dtype::DType::U8, hpt_types::dtype::DType::U32) => cast!(u8, u32),
            #[cfg(all(feature = "u8", feature = "i64"))]
            (hpt_types::dtype::DType::U8, hpt_types::dtype::DType::I64) => cast!(u8, i64),
            #[cfg(all(feature = "u8", feature = "u64"))]
            (hpt_types::dtype::DType::U8, hpt_types::dtype::DType::U64) => cast!(u8, u64),
            #[cfg(all(feature = "u8", feature = "f64"))]
            (hpt_types::dtype::DType::U8, hpt_types::dtype::DType::F64) => cast!(u8, f64),
            #[cfg(all(feature = "i16", feature = "bool"))]
            (hpt_types::dtype::DType::I16, hpt_types::dtype::DType::Bool) => cast!(i16, bool),
            #[cfg(all(feature = "i16", feature = "i8"))]
            (hpt_types::dtype::DType::I16, hpt_types::dtype::DType::I8) => cast!(i16, i8),
            #[cfg(all(feature = "i16", feature = "u8"))]
            (hpt_types::dtype::DType::I16, hpt_types::dtype::DType::U8) => cast!(i16, u8),
            #[cfg(all(feature = "i16", feature = "i16"))]
            (hpt_types::dtype::DType::I16, hpt_types::dtype::DType::I16) => cast!(i16, i16),
            #[cfg(all(feature = "i16", feature = "u16"))]
            (hpt_types::dtype::DType::I16, hpt_types::dtype::DType::U16) => cast!(i16, u16),
            #[cfg(all(feature = "i16", feature = "i32"))]
            (hpt_types::dtype::DType::I16, hpt_types::dtype::DType::I32) => cast!(i16, i32),
            #[cfg(all(feature = "i16", feature = "u32"))]
            (hpt_types::dtype::DType::I16, hpt_types::dtype::DType::U32) => cast!(i16, u32),
            #[cfg(all(feature = "i16", feature = "i64"))]
            (hpt_types::dtype::DType::I16, hpt_types::dtype::DType::I64) => cast!(i16, i64),
            #[cfg(all(feature = "i16", feature = "u64"))]
            (hpt_types::dtype::DType::I16, hpt_types::dtype::DType::U64) => cast!(i16, u64),
            #[cfg(all(feature = "i16", feature = "f32"))]
            (hpt_types::dtype::DType::I16, hpt_types::dtype::DType::F32) => cast!(i16, f32),
            #[cfg(all(feature = "i16", feature = "f16"))]
            (hpt_types::dtype::DType::I16, hpt_types::dtype::DType::F16) => cast!(i16, f16),
            #[cfg(all(feature = "i16", feature = "bf16"))]
            (hpt_types::dtype::DType::I16, hpt_types::dtype::DType::BF16) => cast!(i16, bf16),
            #[cfg(all(feature = "i16", feature = "f64"))]
            (hpt_types::dtype::DType::I16, hpt_types::dtype::DType::F64) => cast!(i16, f64),
            #[cfg(all(feature = "u16", feature = "bool"))]
            (hpt_types::dtype::DType::U16, hpt_types::dtype::DType::Bool) => cast!(u16, bool),
            #[cfg(all(feature = "u16", feature = "i8"))]
            (hpt_types::dtype::DType::U16, hpt_types::dtype::DType::I8) => cast!(u16, i8),
            #[cfg(all(feature = "u16", feature = "u8"))]
            (hpt_types::dtype::DType::U16, hpt_types::dtype::DType::U8) => cast!(u16, u8),
            #[cfg(all(feature = "u16", feature = "i16"))]
            (hpt_types::dtype::DType::U16, hpt_types::dtype::DType::I16) => cast!(u16, i16),
            #[cfg(all(feature = "u16", feature = "u16"))]
            (hpt_types::dtype::DType::U16, hpt_types::dtype::DType::U16) => cast!(u16, u16),
            #[cfg(all(feature = "u16", feature = "i32"))]
            (hpt_types::dtype::DType::U16, hpt_types::dtype::DType::I32) => cast!(u16, i32),
            #[cfg(all(feature = "u16", feature = "u32"))]
            (hpt_types::dtype::DType::U16, hpt_types::dtype::DType::U32) => cast!(u16, u32),
            #[cfg(all(feature = "u16", feature = "i64"))]
            (hpt_types::dtype::DType::U16, hpt_types::dtype::DType::I64) => cast!(u16, i64),
            #[cfg(all(feature = "u16", feature = "u64"))]
            (hpt_types::dtype::DType::U16, hpt_types::dtype::DType::U64) => cast!(u16, u64),
            #[cfg(all(feature = "u16", feature = "f32"))]
            (hpt_types::dtype::DType::U16, hpt_types::dtype::DType::F32) => cast!(u16, f32),
            #[cfg(all(feature = "u16", feature = "f16"))]
            (hpt_types::dtype::DType::U16, hpt_types::dtype::DType::F16) => cast!(u16, f16),
            #[cfg(all(feature = "u16", feature = "bf16"))]
            (hpt_types::dtype::DType::U16, hpt_types::dtype::DType::BF16) => cast!(u16, bf16),
            #[cfg(all(feature = "u16", feature = "f64"))]
            (hpt_types::dtype::DType::U16, hpt_types::dtype::DType::F64) => cast!(u16, f64),
            #[cfg(all(feature = "i32", feature = "bool"))]
            (hpt_types::dtype::DType::I32, hpt_types::dtype::DType::Bool) => cast!(i32, bool),
            #[cfg(all(feature = "i32", feature = "i8"))]
            (hpt_types::dtype::DType::I32, hpt_types::dtype::DType::I8) => cast!(i32, i8),
            #[cfg(all(feature = "i32", feature = "u8"))]
            (hpt_types::dtype::DType::I32, hpt_types::dtype::DType::U8) => cast!(i32, u8),
            #[cfg(all(feature = "i32", feature = "i16"))]
            (hpt_types::dtype::DType::I32, hpt_types::dtype::DType::I16) => cast!(i32, i16),
            #[cfg(all(feature = "i32", feature = "u16"))]
            (hpt_types::dtype::DType::I32, hpt_types::dtype::DType::U16) => cast!(i32, u16),
            #[cfg(all(feature = "i32", feature = "i32"))]
            (hpt_types::dtype::DType::I32, hpt_types::dtype::DType::I32) => cast!(i32, i32),
            #[cfg(all(feature = "i32", feature = "u32"))]
            (hpt_types::dtype::DType::I32, hpt_types::dtype::DType::U32) => cast!(i32, u32),
            #[cfg(all(feature = "i32", feature = "i64"))]
            (hpt_types::dtype::DType::I32, hpt_types::dtype::DType::I64) => cast!(i32, i64),
            #[cfg(all(feature = "i32", feature = "u64"))]
            (hpt_types::dtype::DType::I32, hpt_types::dtype::DType::U64) => cast!(i32, u64),
            #[cfg(all(feature = "i32", feature = "f32"))]
            (hpt_types::dtype::DType::I32, hpt_types::dtype::DType::F32) => cast!(i32, f32),
            #[cfg(all(feature = "i32", feature = "f16"))]
            (hpt_types::dtype::DType::I32, hpt_types::dtype::DType::F16) => cast!(i32, f16),
            #[cfg(all(feature = "i32", feature = "bf16"))]
            (hpt_types::dtype::DType::I32, hpt_types::dtype::DType::BF16) => cast!(i32, bf16),
            #[cfg(all(feature = "i32", feature = "f64"))]
            (hpt_types::dtype::DType::I32, hpt_types::dtype::DType::F64) => cast!(i32, f64),
            #[cfg(all(feature = "u32", feature = "bool"))]
            (hpt_types::dtype::DType::U32, hpt_types::dtype::DType::Bool) => cast!(u32, bool),
            #[cfg(all(feature = "u32", feature = "i8"))]
            (hpt_types::dtype::DType::U32, hpt_types::dtype::DType::I8) => cast!(u32, i8),
            #[cfg(all(feature = "u32", feature = "u8"))]
            (hpt_types::dtype::DType::U32, hpt_types::dtype::DType::U8) => cast!(u32, u8),
            #[cfg(all(feature = "u32", feature = "i16"))]
            (hpt_types::dtype::DType::U32, hpt_types::dtype::DType::I16) => cast!(u32, i16),
            #[cfg(all(feature = "u32", feature = "u16"))]
            (hpt_types::dtype::DType::U32, hpt_types::dtype::DType::U16) => cast!(u32, u16),
            #[cfg(all(feature = "u32", feature = "i32"))]
            (hpt_types::dtype::DType::U32, hpt_types::dtype::DType::I32) => cast!(u32, i32),
            #[cfg(all(feature = "u32", feature = "u32"))]
            (hpt_types::dtype::DType::U32, hpt_types::dtype::DType::U32) => cast!(u32, u32),
            #[cfg(all(feature = "u32", feature = "i64"))]
            (hpt_types::dtype::DType::U32, hpt_types::dtype::DType::I64) => cast!(u32, i64),
            #[cfg(all(feature = "u32", feature = "u64"))]
            (hpt_types::dtype::DType::U32, hpt_types::dtype::DType::U64) => cast!(u32, u64),
            #[cfg(all(feature = "u32", feature = "f32"))]
            (hpt_types::dtype::DType::U32, hpt_types::dtype::DType::F32) => cast!(u32, f32),
            #[cfg(all(feature = "u32", feature = "f16"))]
            (hpt_types::dtype::DType::U32, hpt_types::dtype::DType::F16) => cast!(u32, f16),
            #[cfg(all(feature = "u32", feature = "bf16"))]
            (hpt_types::dtype::DType::U32, hpt_types::dtype::DType::BF16) => cast!(u32, bf16),
            #[cfg(all(feature = "u32", feature = "f64"))]
            (hpt_types::dtype::DType::U32, hpt_types::dtype::DType::F64) => cast!(u32, f64),
            #[cfg(all(feature = "i64", feature = "bool"))]
            (hpt_types::dtype::DType::I64, hpt_types::dtype::DType::Bool) => cast!(i64, bool),
            #[cfg(all(feature = "i64", feature = "i8"))]
            (hpt_types::dtype::DType::I64, hpt_types::dtype::DType::I8) => cast!(i64, i8),
            #[cfg(all(feature = "i64", feature = "u8"))]
            (hpt_types::dtype::DType::I64, hpt_types::dtype::DType::U8) => cast!(i64, u8),
            #[cfg(all(feature = "i64", feature = "i16"))]
            (hpt_types::dtype::DType::I64, hpt_types::dtype::DType::I16) => cast!(i64, i16),
            #[cfg(all(feature = "i64", feature = "u16"))]
            (hpt_types::dtype::DType::I64, hpt_types::dtype::DType::U16) => cast!(i64, u16),
            #[cfg(all(feature = "i64", feature = "i32"))]
            (hpt_types::dtype::DType::I64, hpt_types::dtype::DType::I32) => cast!(i64, i32),
            #[cfg(all(feature = "i64", feature = "u32"))]
            (hpt_types::dtype::DType::I64, hpt_types::dtype::DType::U32) => cast!(i64, u32),
            #[cfg(all(feature = "i64", feature = "i64"))]
            (hpt_types::dtype::DType::I64, hpt_types::dtype::DType::I64) => cast!(i64, i64),
            #[cfg(all(feature = "i64", feature = "u64"))]
            (hpt_types::dtype::DType::I64, hpt_types::dtype::DType::U64) => cast!(i64, u64),
            #[cfg(all(feature = "i64", feature = "f32"))]
            (hpt_types::dtype::DType::I64, hpt_types::dtype::DType::F32) => cast!(i64, f32),
            #[cfg(all(feature = "i64", feature = "f16"))]
            (hpt_types::dtype::DType::I64, hpt_types::dtype::DType::F16) => cast!(i64, f16),
            #[cfg(all(feature = "i64", feature = "bf16"))]
            (hpt_types::dtype::DType::I64, hpt_types::dtype::DType::BF16) => cast!(i64, bf16),
            #[cfg(all(feature = "i64", feature = "f64"))]
            (hpt_types::dtype::DType::I64, hpt_types::dtype::DType::F64) => cast!(i64, f64),
            #[cfg(all(feature = "u64", feature = "bool"))]
            (hpt_types::dtype::DType::U64, hpt_types::dtype::DType::Bool) => cast!(u64, bool),
            #[cfg(all(feature = "u64", feature = "i8"))]
            (hpt_types::dtype::DType::U64, hpt_types::dtype::DType::I8) => cast!(u64, i8),
            #[cfg(all(feature = "u64", feature = "u8"))]
            (hpt_types::dtype::DType::U64, hpt_types::dtype::DType::U8) => cast!(u64, u8),
            #[cfg(all(feature = "u64", feature = "i16"))]
            (hpt_types::dtype::DType::U64, hpt_types::dtype::DType::I16) => cast!(u64, i16),
            #[cfg(all(feature = "u64", feature = "u16"))]
            (hpt_types::dtype::DType::U64, hpt_types::dtype::DType::U16) => cast!(u64, u16),
            #[cfg(all(feature = "u64", feature = "i32"))]
            (hpt_types::dtype::DType::U64, hpt_types::dtype::DType::I32) => cast!(u64, i32),
            #[cfg(all(feature = "u64", feature = "u32"))]
            (hpt_types::dtype::DType::U64, hpt_types::dtype::DType::U32) => cast!(u64, u32),
            #[cfg(all(feature = "u64", feature = "i64"))]
            (hpt_types::dtype::DType::U64, hpt_types::dtype::DType::I64) => cast!(u64, i64),
            #[cfg(all(feature = "u64", feature = "u64"))]
            (hpt_types::dtype::DType::U64, hpt_types::dtype::DType::U64) => cast!(u64, u64),
            #[cfg(all(feature = "u64", feature = "f32"))]
            (hpt_types::dtype::DType::U64, hpt_types::dtype::DType::F32) => cast!(u64, f32),
            #[cfg(all(feature = "u64", feature = "f16"))]
            (hpt_types::dtype::DType::U64, hpt_types::dtype::DType::F16) => cast!(u64, f16),
            #[cfg(all(feature = "u64", feature = "bf16"))]
            (hpt_types::dtype::DType::U64, hpt_types::dtype::DType::BF16) => cast!(u64, bf16),
            #[cfg(all(feature = "u64", feature = "f64"))]
            (hpt_types::dtype::DType::U64, hpt_types::dtype::DType::F64) => cast!(u64, f64),
            #[cfg(all(feature = "f32", feature = "bool"))]
            (hpt_types::dtype::DType::F32, hpt_types::dtype::DType::Bool) => cast!(f32, bool),
            #[cfg(all(feature = "f32", feature = "i16"))]
            (hpt_types::dtype::DType::F32, hpt_types::dtype::DType::I16) => cast!(f32, i16),
            #[cfg(all(feature = "f32", feature = "u16"))]
            (hpt_types::dtype::DType::F32, hpt_types::dtype::DType::U16) => cast!(f32, u16),
            #[cfg(all(feature = "f32", feature = "i32"))]
            (hpt_types::dtype::DType::F32, hpt_types::dtype::DType::I32) => cast!(f32, i32),
            #[cfg(all(feature = "f32", feature = "u32"))]
            (hpt_types::dtype::DType::F32, hpt_types::dtype::DType::U32) => cast!(f32, u32),
            #[cfg(all(feature = "f32", feature = "i64"))]
            (hpt_types::dtype::DType::F32, hpt_types::dtype::DType::I64) => cast!(f32, i64),
            #[cfg(all(feature = "f32", feature = "u64"))]
            (hpt_types::dtype::DType::F32, hpt_types::dtype::DType::U64) => cast!(f32, u64),
            #[cfg(all(feature = "f32", feature = "f64"))]
            (hpt_types::dtype::DType::F32, hpt_types::dtype::DType::F64) => cast!(f32, f64),
            #[cfg(all(feature = "f16", feature = "bool"))]
            (hpt_types::dtype::DType::F16, hpt_types::dtype::DType::Bool) => cast!(f16, bool),
            #[cfg(all(feature = "f16", feature = "i16"))]
            (hpt_types::dtype::DType::F16, hpt_types::dtype::DType::I16) => cast!(f16, i16),
            #[cfg(all(feature = "f16", feature = "u16"))]
            (hpt_types::dtype::DType::F16, hpt_types::dtype::DType::U16) => cast!(f16, u16),
            #[cfg(all(feature = "f16", feature = "i32"))]
            (hpt_types::dtype::DType::F16, hpt_types::dtype::DType::I32) => cast!(f16, i32),
            #[cfg(all(feature = "f16", feature = "u32"))]
            (hpt_types::dtype::DType::F16, hpt_types::dtype::DType::U32) => cast!(f16, u32),
            #[cfg(all(feature = "f16", feature = "i64"))]
            (hpt_types::dtype::DType::F16, hpt_types::dtype::DType::I64) => cast!(f16, i64),
            #[cfg(all(feature = "f16", feature = "u64"))]
            (hpt_types::dtype::DType::F16, hpt_types::dtype::DType::U64) => cast!(f16, u64),
            #[cfg(all(feature = "f16", feature = "f64"))]
            (hpt_types::dtype::DType::F16, hpt_types::dtype::DType::F64) => cast!(f16, f64),
            #[cfg(all(feature = "bf16", feature = "bool"))]
            (hpt_types::dtype::DType::BF16, hpt_types::dtype::DType::Bool) => cast!(bf16, bool),
            #[cfg(all(feature = "bf16", feature = "i16"))]
            (hpt_types::dtype::DType::BF16, hpt_types::dtype::DType::I16) => cast!(bf16, i16),
            (hpt_types::dtype::DType::BF16, hpt_types::dtype::DType::U16) => todo!(),
            #[cfg(all(feature = "bf16", feature = "i32"))]
            (hpt_types::dtype::DType::BF16, hpt_types::dtype::DType::I32) => cast!(bf16, i32),
            #[cfg(all(feature = "bf16", feature = "u32"))]
            (hpt_types::dtype::DType::BF16, hpt_types::dtype::DType::U32) => cast!(bf16, u32),
            #[cfg(all(feature = "bf16", feature = "i64"))]
            (hpt_types::dtype::DType::BF16, hpt_types::dtype::DType::I64) => cast!(bf16, i64),
            #[cfg(all(feature = "bf16", feature = "u64"))]
            (hpt_types::dtype::DType::BF16, hpt_types::dtype::DType::U64) => cast!(bf16, u64),
            #[cfg(all(feature = "bf16", feature = "f64"))]
            (hpt_types::dtype::DType::BF16, hpt_types::dtype::DType::F64) => cast!(bf16, f64),
            #[cfg(all(feature = "f64", feature = "bool"))]
            (hpt_types::dtype::DType::F64, hpt_types::dtype::DType::Bool) => cast!(f64, bool),
            #[cfg(all(feature = "f64", feature = "i8"))]
            (hpt_types::dtype::DType::F64, hpt_types::dtype::DType::I8) => cast!(f64, i8),
            #[cfg(all(feature = "f64", feature = "u8"))]
            (hpt_types::dtype::DType::F64, hpt_types::dtype::DType::U8) => cast!(f64, u8),
            #[cfg(all(feature = "f64", feature = "i16"))]
            (hpt_types::dtype::DType::F64, hpt_types::dtype::DType::I16) => cast!(f64, i16),
            #[cfg(all(feature = "f64", feature = "u16"))]
            (hpt_types::dtype::DType::F64, hpt_types::dtype::DType::U16) => cast!(f64, u16),
            #[cfg(all(feature = "f64", feature = "i32"))]
            (hpt_types::dtype::DType::F64, hpt_types::dtype::DType::I32) => cast!(f64, i32),
            #[cfg(all(feature = "f64", feature = "u32"))]
            (hpt_types::dtype::DType::F64, hpt_types::dtype::DType::U32) => cast!(f64, u32),
            #[cfg(all(feature = "f64", feature = "i64"))]
            (hpt_types::dtype::DType::F64, hpt_types::dtype::DType::I64) => cast!(f64, i64),
            #[cfg(all(feature = "f64", feature = "u64"))]
            (hpt_types::dtype::DType::F64, hpt_types::dtype::DType::U64) => cast!(f64, u64),
            #[cfg(all(feature = "f64", feature = "f32"))]
            (hpt_types::dtype::DType::F64, hpt_types::dtype::DType::F32) => cast!(f64, f32),
            #[cfg(all(feature = "f64", feature = "f16"))]
            (hpt_types::dtype::DType::F64, hpt_types::dtype::DType::F16) => cast!(f64, f16),
            #[cfg(all(feature = "f64", feature = "bf16"))]
            (hpt_types::dtype::DType::F64, hpt_types::dtype::DType::BF16) => cast!(f64, bf16),
            #[cfg(all(feature = "f64", feature = "f64"))]
            (hpt_types::dtype::DType::F64, hpt_types::dtype::DType::F64) => cast!(f64, f64),
            _ => unreachable!("unsupported dtype: {:?}", self.dtype),
        }
    }
}
