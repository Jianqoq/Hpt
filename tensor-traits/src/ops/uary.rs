use anyhow::Result;
use tensor_types::type_promote::NormalOut;

use crate::{ tensor::CommonBounds, BaseTensor };

pub trait FloatUaryOps {
    type Output;
    type InplaceOutput;
    type OutputMeta;
    /// Compute sine, element-wise.
    ///
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([0f32, 1f32, 2f32]);
    /// let b = a.sin().unwrap();
    /// assert!(b.allclose(&Tensor::<f32>::new([0f32, 0.8414709848078965f32, 0.9092974268256817f32])));
    /// ```
    fn sin(&self) -> Result<Self::Output>;

    /// Compute cosine, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([0f32, 1f32, 2f32]);
    /// let b = a.cos().unwrap();
    /// assert!(b.allclose(&Tensor::<f32>::new([1f32, 0.5403023058681398f32, -0.4161468365471424f32])));
    /// ```
    fn cos(&self) -> Result<Self::Output>;

    /// Compute tangent, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([0f32, 1f32, 2f32]);
    /// let b = a.tan().unwrap();
    /// assert!(b.allclose(&Tensor::<f32>::new([0f32, 1.5574077246549023f32, -2.185039863261519f32])));
    /// ```
    fn tan(&self) -> Result<Self::Output>;

    /// Compute inverse sine, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([0f32, 0.5f32, 1f32]);
    /// let b = a.asin().unwrap();
    /// assert!(b.allclose(&Tensor::<f32>::new([0f32, 0.5235987755982989f32, 1.5707963267948966f32])));
    /// ```
    fn asin(&self) -> Result<Self::Output>;

    /// Compute inverse cosine, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([0f32, 0.5f32, 1f32]);
    /// let b = a.acos().unwrap();
    /// assert!(b.allclose(&Tensor::<f32>::new([1.5707963267948966f32, 1.0471975511965979f32, 0f32])));
    /// ```
    fn acos(&self) -> Result<Self::Output>;

    /// Compute inverse tangent, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([0f32, 0.5f32, 1f32]);
    /// let b = a.atan().unwrap();
    /// assert!(b.allclose(&Tensor::<f32>::new([0f32, 0.4636476090008061f32, 0.7853981633974483f32])));
    /// ```
    fn atan(&self) -> Result<Self::Output>;

    /// Compute hyperbolic sine, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([0f32, 1f32, 2f32]);
    /// let b = a.sinh().unwrap();
    /// assert!(b.allclose(&Tensor::<f32>::new([0f32, 1.1752011936438014f32, 3.626860407847019f32])));
    /// ```
    fn sinh(&self) -> Result<Self::Output>;

    /// Compute hyperbolic cosine, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([0f32, 1f32, 2f32]);
    /// let b = a.cosh().unwrap();
    /// assert!(b.allclose(&Tensor::<f32>::new([1f32, 1.5430806348152437f32, 3.7621956910836314f32])));
    /// ```
    fn cosh(&self) -> Result<Self::Output>;

    /// Compute hyperbolic tangent, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([0f32, 1f32, 2f32]);
    /// let b = a.tanh().unwrap();
    /// assert!(b.allclose(&Tensor::<f32>::new([0f32, 0.7615941559557649f32, 0.9640275800758169f32])));
    /// ```
    fn tanh(&self) -> Result<Self::Output>;

    /// Compute inverse hyperbolic sine, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([0.5f32, 1f32, 2f32]);
    /// let b = a.asinh().unwrap();
    /// assert!(b.allclose(&Tensor::<f32>::new([0.48121182505960347f32, 0.881373587019543f32, 1.4436354751788103f32])));
    /// ```
    fn asinh(&self) -> Result<Self::Output>;

    /// Compute inverse hyperbolic cosine, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([1f32, 2f32, 3f32]);
    /// let b = a.acosh().unwrap();
    /// assert!(b.allclose(&Tensor::<f32>::new([0f32, 1.3169578969248166f32, 1.762747174039086f32])));
    /// ```
    fn acosh(&self) -> Result<Self::Output>;

    /// Compute inverse hyperbolic tangent, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([0.5f32, 0.8f32, 0.9f32]);
    /// let b = a.atanh().unwrap();
    /// assert!(b.allclose(&Tensor::<f32>::new([0.5493061443340549f32, 1.0986122886681098f32, 1.4722194895832204f32])));
    /// ```
    fn atanh(&self) -> Result<Self::Output>;

    /// Inplace Version of sin. Compute sine, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::<f32>::new([0f32, 1f32, 2f32]);
    /// let c = Tensor::<f32>::new([0f32, 1f32, 2f32]);
    /// let res = a.sin_(c).unwrap(); // note: c moved into sin_ and out as res
    /// assert!(res.allclose(&Tensor::<f32>::new([0f32, 0.8414709848078965f32, 0.9092974268256817f32])));
    /// ```
    fn sin_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    /// Inplace Version of cos. Compute cosine, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::<f32>::new([0f32, 1f32, 2f32]);
    /// let c = Tensor::<f32>::new([0f32, 1f32, 2f32]);
    /// let res = a.cos_(c).unwrap(); // note: c moved into sin_ and out as res
    /// assert!(res.allclose(&Tensor::<f32>::new([1f32, 0.5403023058681398f32, -0.4161468365471424f32])));
    /// ```
    fn cos_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    /// Inplace Version of tan. Compute tangent, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::<f32>::new([0f32, 1f32, 2f32]);
    /// let c = Tensor::<f32>::new([0f32, 1f32, 2f32]);
    /// let res = a.tan_(c).unwrap(); // note: c moved into sin_ and out as res
    /// assert!(res.allclose(&Tensor::<f32>::new([0f32, 1.5574077246549023f32, -2.185039863261519f32])));
    /// ```
    fn tan_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    /// Inplace Version of asin. Compute inverse sine, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::<f32>::new([0f32, 0.5f32, 1f32]);
    /// let c = Tensor::<f32>::new([0f32, 0.5f32, 1f32]);
    /// let res = a.asin_(c).unwrap(); // note: c moved into sin_ and out as res
    /// assert!(res.allclose(&Tensor::<f32>::new([0f32, 0.5235987755982989f32, 1.5707963267948966f32])));
    /// ```
    fn asin_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    /// Inplace Version of acos. Compute inverse cosine, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::<f32>::new([0f32, 0.5f32, 1f32]);
    /// let c = Tensor::<f32>::new([0f32, 0.5f32, 1f32]);
    /// let res = a.acos_(c).unwrap(); // note: c moved into sin_ and out as res
    /// assert!(res.allclose(&Tensor::<f32>::new([1.5707963267948966f32, 1.0471975511965979f32, 0f32])));
    /// ```
    fn acos_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    /// Inplace Version of atan. Compute inverse tangent, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::<f32>::new([0f32, 0.5f32, 1f32]);
    /// let c = Tensor::<f32>::new([0f32, 0.5f32, 1f32]);
    /// let res = a.atan_(c).unwrap(); // note: c moved into sin_ and out as res
    /// assert!(res.allclose(&Tensor::<f32>::new([0f32, 0.4636476090008061f32, 0.7853981633974483f32])));
    /// ```
    fn atan_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    /// Inplace Version of sinh. Compute hyperbolic sine, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::<f32>::new([0f32, 1f32, 2f32]);
    /// let c = Tensor::<f32>::new([0f32, 1f32, 2f32]);
    /// let res = a.sinh_(c).unwrap(); // note: c moved into sin_ and out as res
    /// assert!(res.allclose(&Tensor::<f32>::new([0f32, 1.1752011936438014f32, 3.626860407847019f32])));
    /// ```
    fn sinh_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    /// Inplace Version of cosh. Compute hyperbolic cosine, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::<f32>::new([0f32, 1f32, 2f32]);
    /// let c = Tensor::<f32>::new([0f32, 1f32, 2f32]);
    /// let res = a.cosh_(c).unwrap(); // note: c moved into sin_ and out as res
    /// assert!(res.allclose(&Tensor::<f32>::new([1f32, 1.5430806348152437f32, 3.7621956910836314f32])));
    /// ```
    fn cosh_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    /// Inplace Version of tanh. Compute hyperbolic tangent, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::<f32>::new([0f32, 1f32, 2f32]);
    /// let c = Tensor::<f32>::new([0f32, 1f32, 2f32]);
    /// let res = a.tanh_(c).unwrap(); // note: c moved into sin_ and out as res
    /// assert!(res.allclose(&Tensor::<f32>::new([0f32, 0.7615941559557649f32, 0.9640275800758169f32])));
    /// ```
    fn tanh_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    /// Inplace Version of asinh. Compute inverse hyperbolic sine, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::<f32>::new([0.5f32, 1f32, 2f32]);
    /// let c = Tensor::<f32>::new([0.5f32, 1f32, 2f32]);
    /// let res = a.asinh_(c).unwrap(); // note: c moved into sin_ and out as res
    /// assert!(res.allclose(&Tensor::<f32>::new([0.48121182505960347f32, 0.881373587019543f32, 1.4436354751788103f32])));
    /// ```
    fn asinh_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    /// Inplace Version of acosh. Compute inverse hyperbolic cosine, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::<f32>::new([1f32, 2f32, 3f32]);
    /// let c = Tensor::<f32>::new([1f32, 2f32, 3f32]);
    /// let res = a.acosh_(c).unwrap(); // note: c moved into sin_ and out as res
    /// assert!(res.allclose(&Tensor::<f32>::new([0f32, 1.3169578969248166f32, 1.762747174039086f32])));
    /// ```
    fn acosh_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    /// Inplace Version of atanh. Compute inverse hyperbolic tangent, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::<f32>::new([0.5f32, 0.8f32, 0.9f32]);
    /// let c = Tensor::<f32>::new([0.5f32, 0.8f32, 0.9f32]);
    /// let res = a.atanh_(c).unwrap(); // note: c moved into sin_ and out as res
    /// assert!(res.allclose(&Tensor::<f32>::new([0.5493061443340549f32, 1.0986122886681098f32, 1.4722194895832204f32])));
    /// ```
    fn atanh_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    /// Compute exponential, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([0f64, 1f64, 2f64]);
    /// let b = a.exp().unwrap();
    /// assert!(b.allclose(&Tensor::<f64>::new([1f64, 2.718281828459045f64, 7.38905609893065f64])));
    /// ```
    fn exp(&self) -> Result<Self::Output>;

    /// Compute exponential, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([0f64, 1f64, 2f64]);
    /// let b = a.exp().unwrap();
    /// assert!(b.allclose(&Tensor::<f64>::new([1f64, 2.718281828459045f64, 7.38905609893065f64])));
    /// ```
    fn exp_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    /// Compute exponential minus one, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([0.5f64, 1f64, 2f64]);
    /// let b = a.expm1().unwrap();
    /// assert!(b.allclose(&Tensor::<f64>::new([0.6487212707001282f64, 1.718281828459045f64, 6.38905609893065f64])));
    /// ```
    fn exp2(&self) -> Result<Self::Output>;

    /// Compute exponential minus one, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([0.5f64, 1f64, 2f64]);
    /// let b = a.expm1().unwrap();
    /// assert!(b.allclose(&Tensor::<f64>::new([0.6487212707001282f64, 1.718281828459045f64, 6.38905609893065f64])));
    /// ```
    fn exp2_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    /// Compute square root, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([4f64, 9f64, 16f64]);
    /// let b = a.sqrt().unwrap();
    /// assert!(b.allclose(&Tensor::<f64>::new([2f64, 3f64, 4f64])));
    /// ```
    fn sqrt(&self) -> Result<Self::Output>;

    /// Compute square root, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([4f64, 9f64, 16f64]);
    /// let b = a.sqrt().unwrap();
    /// assert!(b.allclose(&Tensor::<f64>::new([2f64, 3f64, 4f64])));
    /// ```
    fn sqrt_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    /// Compute reciprocal, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([-4.2f64, 9.8f64, 16.3f64]);
    /// let b = a.recip().unwrap();
    /// assert!(b.allclose(&Tensor::<f64>::new([-0.23809523809523808f64, 0.10204081632653061f64, 0.06134969325153374f64])));
    /// ```
    fn recip(&self) -> Result<Self::Output>;

    /// Inplace Version of reciprocal. Compute reciprocal, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::<f64>::new([-4.2f64, 9.8f64, 16.3f64]);
    /// let c = Tensor::<f64>::new([-4.2f64, 9.8f64, 16.3f64]);
    /// a.recip_(c).unwrap();
    /// assert!(a.allclose(&Tensor::<f64>::new([-0.23809523809523808f64, 0.10204081632653061f64, 0.06134969325153374f64])));
    /// ```
    fn recip_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    /// Compute natural logarithm, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([1f64, 2f64, 3f64]);
    /// let b = a.ln().unwrap();
    /// assert!(b.allclose(&Tensor::<f64>::new([0f64, 0.6931471805599453f64, 1.0986122886681098f64])));
    /// ```
    fn ln(&self) -> Result<Self::Output>;

    /// Inplace Version of ln. Compute natural logarithm, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::<f64>::new([1f64, 2f64, 3f64]);
    /// let c = Tensor::<f64>::new([1f64, 2f64, 3f64]);
    /// a.ln_(c).unwrap();
    /// assert!(a.allclose(&Tensor::<f64>::new([0f64, 0.6931471805599453f64, 1.0986122886681098f64])));
    /// ```
    fn ln_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    /// Compute logarithm base 2, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([1f64, 2f64, 3f64]);
    /// let b = a.log2().unwrap();
    /// assert!(b.allclose(&Tensor::<f64>::new([0f64, 1f64, 1.5849625007211563f64])));
    /// ```
    fn log2(&self) -> Result<Self::Output>;

    /// Inplace Version of log2. Compute logarithm base 2, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::<f64>::new([1f64, 2f64, 3f64]);
    /// let c = Tensor::<f64>::new([1f64, 2f64, 3f64]);
    /// a.log2_(c).unwrap();
    /// assert!(a.allclose(&Tensor::<f64>::new([0f64, 1f64, 1.5849625007211563f64])));
    /// ```
    fn log2_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    /// Compute logarithm base 10, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([1f64, 2f64, 3f64]);
    /// let b = a.log10().unwrap();
    /// assert!(b.allclose(&Tensor::<f64>::new([0f64, 0.3010299956639812f64, 0.47712125471966244f64])));
    /// ```
    fn log10(&self) -> Result<Self::Output>;

    /// Inplace Version of log10. Compute logarithm base 10, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::<f64>::new([1f64, 2f64, 3f64]);
    /// let c = Tensor::<f64>::new([1f64, 2f64, 3f64]);
    /// a.log10_(c).unwrap();
    /// assert!(a.allclose(&Tensor::<f64>::new([0f64, 0.3010299956639812f64, 0.47712125471966244f64])));
    /// ```
    fn log10_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    fn celu(&self, alpha: Self::OutputMeta) -> anyhow::Result<Self::Output>;

    fn celu_<U>(&self, alpha: Self::OutputMeta, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    fn sigmoid(&self) -> Result<Self::Output>;

    fn sigmoid_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    fn elu(&self, alpha: Self::OutputMeta) -> anyhow::Result<Self::Output>;

    fn elu_<U>(&self, alpha: Self::OutputMeta, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    fn leaky_relu(&self, alpha: Self::OutputMeta) -> anyhow::Result<Self::Output>;

    fn leaky_relu_<U>(&self, alpha: Self::OutputMeta, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    fn gelu(&self) -> anyhow::Result<Self::Output>;

    fn gelu_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    fn selu(
        &self,
        alpha: Option<Self::OutputMeta>,
        gamma: Option<Self::OutputMeta>
    ) -> anyhow::Result<Self::Output>;

    fn selu_<U>(
        &self,
        alpha: Option<Self::OutputMeta>,
        gamma: Option<Self::OutputMeta>,
        out: U
    ) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    fn hard_sigmoid(
        &self,
        alpha: Option<Self::OutputMeta>,
        beta: Option<Self::OutputMeta>
    ) -> anyhow::Result<Self::Output>;

    fn hard_sigmoid_<U>(
        &self,
        alpha: Option<Self::OutputMeta>,
        beta: Option<Self::OutputMeta>,
        out: U
    ) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    fn hard_swish(&self) -> anyhow::Result<Self::Output>;

    fn hard_swish_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    fn relu6(&self) -> anyhow::Result<Self::Output>;

    fn relu6_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    fn softplus(&self) -> anyhow::Result<Self::Output>;

    fn softplus_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    fn softsign(&self) -> anyhow::Result<Self::Output>;

    fn softsign_<U>(&self, out: U) -> anyhow::Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;
}

pub trait NormalUaryOps {
    type Output;
    type InplaceOutput;
    type OutputMeta;
    /// Compute square, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([1f64, 2f64, 3f64]);
    /// let b = a.square().unwrap();
    /// assert!(b.allclose(&Tensor::<f64>::new([1f64, 4f64, 9f64])));
    /// ```
    fn square(&self) -> Result<Self::Output>;

    /// Inplace Version of square. Compute square, element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::<f64>::new([1f64, 2f64, 3f64]);
    /// let c = Tensor::<f64>::new([1f64, 2f64, 3f64]);
    /// a.square_(c).unwrap();
    /// assert!(a.allclose(&Tensor::<f64>::new([1f64, 4f64, 9f64])));
    /// ```
    fn square_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    /// Compute absolute value element-wise.
    ///
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([-1f32, 0f32, 3.]);
    /// let b = a.abs().unwrap();
    /// assert_eq!(b, Tensor::new([1f32, 0f32, 3.]));
    /// ```
    fn abs(&self) -> Result<Self::Output>;

    /// Inplace Version of abs. Compute absolute value element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::<f64>::new([-1f64, 0f64, 3f64]);
    /// let c = Tensor::<f64>::new([-1f64, 0f64, 3f64]);
    /// a.abs_(c).unwrap();
    /// assert!(a.allclose(&Tensor::<f64>::new([1f64, 0f64, 3f64])));
    /// ```
    fn abs_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    fn ceil(&self) -> Result<Self::Output>;

    fn ceil_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    fn sign(&self) -> Result<Self::Output>;

    fn sign_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    fn clip(&self, min: Self::OutputMeta, max: Self::OutputMeta) -> Result<Self::Output>;

    fn clip_<U>(&self, min: Self::OutputMeta, max: Self::OutputMeta, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;

    fn round(&self) -> Result<Self::Output>;

    fn round_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;
}

pub trait Cum where Self: Sized {
    type Meta: CommonBounds;

    /// Comput cumsum along specified axis.
    ///
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([[1f32, 2f32, 3f32], [4f32, 5f32, 6f32]]);
    /// let b = a.cumsum(Some(0)).unwrap();
    /// assert_eq!(b, Tensor::new([[1f32, 2f32, 3f32], [5f32, 7f32, 9f32]]));
    /// ```
    fn cumsum(&self, axis: Option<i64>) -> Result<Self>
        where Self::Meta: NormalOut<Self::Meta, Output = Self::Meta>;

    /// Comput cumprod along specified axis.
    ///
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([[1f32, 2f32, 3f32], [4f32, 5f32, 6f32]]);
    /// let b = a.cumprod(Some(0)).unwrap();
    /// assert_eq!(b, Tensor::new([[1f32, 2f32, 3f32], [4f32, 10f32, 18f32]]));
    /// ```
    fn cumprod(&self, axis: Option<i64>) -> Result<Self>
        where Self::Meta: NormalOut<Self::Meta, Output = Self::Meta>;
}

pub trait Neg {
    type Output;
    type InplaceOutput;
    type OutputMeta;
    /// Compute negative value element-wise.
    ///
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([-1f32, 0f32, 3.]);
    /// let b = a.neg().unwrap();
    /// assert_eq!(b, Tensor::new([1f32, 0f32, -3.]));
    /// ```
    fn neg(&self) -> Result<Self::Output>;

    /// Inplace Version of neg. Compute negative value element-wise.
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::<f64>::new([-1f64, 0f64, 3f64]);
    /// let c = Tensor::<f64>::new([-1f64, 0f64, 3f64]);
    /// a.neg_(c).unwrap();
    /// assert!(a.allclose(&Tensor::<f64>::new([1f64, 0f64, -3f64])));
    /// ```
    fn neg_<U>(&self, out: U) -> Result<Self::Output>
        where U: BaseTensor<Output = Self::InplaceOutput>;
}
