use super::{convertion::CudaConvertor, scalar::Scalar};
use crate::type_promote::{
    BitWiseOut2, Eval2, FloatOutBinary2, FloatOutUnary2, NormalOut2, NormalOutUnary2,
};
use half::f16;
impl FloatOutBinary2 for Scalar<f16> {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        Scalar::new(format!("__hdiv({},{})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __log(self, base: Self) -> Self {
        Scalar::new(format!(
            "__float2half_rn(logf({}) / logf({}))",
            self.to_f32().val,
            base.to_f32().val
        ))
    }
    #[inline(always)]
    fn __hypot(self, rhs: Self) -> Self {
        Scalar::new(format!(
            "__float2half_rn(hypotf({}, {}))",
            self.to_f32().val,
            rhs.to_f32().val
        ))
    }

    #[inline(always)]
    fn __pow(self, rhs: Self) -> Self {
        Scalar::new(format!(
            "__float2half_rn(powf({}, {}))",
            self.to_f32().val,
            rhs.to_f32().val
        ))
    }
}

impl NormalOut2 for Scalar<f16> {
    #[inline(always)]
    fn __add(self, rhs: Self) -> Self {
        Scalar::new(format!("__hadd({},{})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __sub(self, rhs: Self) -> Self {
        Scalar::new(format!("__hsub({},{})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __mul_add(self, a: Self, b: Self) -> Self {
        Scalar::new(format!("__hfma({},{},{})", self.val, a.val, b.val))
    }

    #[inline(always)]
    fn __mul(self, rhs: Self) -> Self {
        Scalar::new(format!("__hmulf({}, {})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __rem(self, rhs: Self) -> Self {
        Scalar::new(format!(
            "__float2half_rn(({} != 0) ? (({}) % ({})) : 0)",
            rhs.to_f32().val,
            self.to_f32().val,
            rhs.to_f32().val
        ))
    }

    #[inline(always)]
    fn __max(self, rhs: Self) -> Self {
        Scalar::new(format!("__hmax_nan({}, {})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __min(self, rhs: Self) -> Self {
        Scalar::new(format!("__hmin_nan({}, {})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __clamp(self, min: Self, max: Self) -> Self {
        Scalar::new(format!(
            "__hmin_nan(__hmax_nan({}, {}), {})",
            self.val, min.val, max.val
        ))
    }
}

impl NormalOutUnary2 for Scalar<f16> {
    #[inline(always)]
    fn __square(self) -> Self {
        Scalar::new(format!("__hmul({}, {})", self.val, self.val))
    }

    #[inline(always)]
    fn __abs(self) -> Self {
        Scalar::new(format!("__hfabs({})", self.val))
    }

    #[inline(always)]
    fn __ceil(self) -> Self {
        Scalar::new(format!("__float2half_rn(ceilf({}))", self.to_f32().val))
    }

    #[inline(always)]
    fn __floor(self) -> Self {
        Scalar::new(format!("__float2half_rn(floorf({}))", self.to_f32().val))
    }

    #[inline(always)]
    fn __neg(self) -> Self {
        Scalar::new(format!("(-{})", self.val))
    }

    #[inline(always)]
    fn __round(self) -> Self {
        Scalar::new(format!("__float2half_rn(roundf({}))", self.to_f32().val))
    }

    #[inline(always)]
    fn __signum(self) -> Self {
        Scalar::new(format!(
            "__float2half_rn(copysignf(1.0f, {}))",
            self.to_f32().val
        ))
    }

    #[inline(always)]
    fn __leaky_relu(self, alpha: Self) -> Self {
        Scalar::new(format!(
            "({} > __half(0.0)) ? {} : ({} * {})",
            self.val, self.val, alpha.val, self.val
        ))
    }

    #[inline(always)]
    fn __relu(self) -> Self {
        Scalar::new(format!("__hmax({}, __half(0.0))", self.val))
    }

    #[inline(always)]
    fn __relu6(self) -> Self {
        Scalar::new(format!(
            "__hmin(__hmax({}, __half(0.0)), __half(6.0))",
            self.val
        ))
    }

    #[inline(always)]
    fn __trunc(self) -> Self {
        Scalar::new(format!("__float2half_rn(truncf({}))", self.to_f32().val))
    }

    fn __copysign(self, rhs: Self) -> Self {
        Scalar::new(format!(
            "__float2half_rn(copysignf({}, {}))",
            self.val, rhs.val
        ))
    }
}

impl BitWiseOut2 for Scalar<f16> {
    #[inline(always)]
    fn __bitand(self, _: Self) -> Self {
        panic!("Bitwise operations are not supported for cuda f16")
    }

    #[inline(always)]
    fn __bitor(self, _: Self) -> Self {
        panic!("Bitwise operations are not supported for cuda f16")
    }

    #[inline(always)]
    fn __bitxor(self, _: Self) -> Self {
        panic!("Bitwise operations are not supported for cuda f16")
    }

    #[inline(always)]
    fn __not(self) -> Self {
        panic!("Bitwise operations are not supported for cuda f16")
    }

    #[inline(always)]
    fn __shl(self, _: Self) -> Self {
        panic!("Shift operations are not supported for cuda f16")
    }

    #[inline(always)]
    fn __shr(self, _: Self) -> Self {
        panic!("Shift operations are not supported for cuda f16")
    }
}

impl Eval2 for Scalar<f16> {
    type Output = Scalar<bool>;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        Scalar::new(format!("__hisnan({})", self.val))
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        Scalar::new(format!(
            "({} != __half(0.0) && !__hisnan({}))",
            self.val, self.val
        ))
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        Scalar::new(format!("__hisinf({})", self.val))
    }
}

impl FloatOutUnary2 for Scalar<f16> {
    #[inline(always)]
    fn __exp(self) -> Self {
        self.to_f32().__exp().to_f16()
    }
    #[inline(always)]
    fn __exp2(self) -> Self {
        self.to_f32().__exp2().to_f16()
    }
    #[inline(always)]
    fn __ln(self) -> Self {
        self.to_f32().__ln().to_f16()
    }
    #[inline(always)]
    fn __celu(self, alpha: Self) -> Self {
        self.to_f32().__celu(alpha.to_f32()).to_f16()
    }
    #[inline(always)]
    fn __log2(self) -> Self {
        self.to_f32().__log2().to_f16()
    }
    #[inline(always)]
    fn __log10(self) -> Self {
        self.to_f32().__log10().to_f16()
    }
    #[inline(always)]
    fn __sqrt(self) -> Self {
        self.to_f32().__sqrt().to_f16()
    }
    #[inline(always)]
    fn __sin(self) -> Self {
        self.to_f32().__sin().to_f16()
    }
    #[inline(always)]
    fn __cos(self) -> Self {
        self.to_f32().__cos().to_f16()
    }
    #[inline(always)]
    fn __tan(self) -> Self {
        self.to_f32().__tan().to_f16()
    }
    #[inline(always)]
    fn __asin(self) -> Self {
        self.to_f32().__asin().to_f16()
    }
    #[inline(always)]
    fn __acos(self) -> Self {
        self.to_f32().__acos().to_f16()
    }
    #[inline(always)]
    fn __atan(self) -> Self {
        self.to_f32().__atan().to_f16()
    }
    #[inline(always)]
    fn __sinh(self) -> Self {
        self.to_f32().__sinh().to_f16()
    }
    #[inline(always)]
    fn __cosh(self) -> Self {
        self.to_f32().__cosh().to_f16()
    }
    #[inline(always)]
    fn __tanh(self) -> Self {
        self.to_f32().__tanh().to_f16()
    }
    #[inline(always)]
    fn __asinh(self) -> Self {
        self.to_f32().__asinh().to_f16()
    }
    #[inline(always)]
    fn __acosh(self) -> Self {
        self.to_f32().__acosh().to_f16()
    }
    #[inline(always)]
    fn __atanh(self) -> Self {
        self.to_f32().__atanh().to_f16()
    }
    #[inline(always)]
    fn __recip(self) -> Self {
        self.to_f32().__recip().to_f16()
    }
    #[inline(always)]
    fn __erf(self) -> Self {
        self.to_f32().__erf().to_f16()
    }

    #[inline(always)]
    fn __sigmoid(self) -> Self {
        self.to_f32().__sigmoid().to_f16()
    }

    fn __elu(self, alpha: Self) -> Self {
        self.to_f32().__elu(alpha.to_f32()).to_f16()
    }

    fn __gelu(self) -> Self {
        self.to_f32().__gelu().to_f16()
    }

    fn __selu(self, alpha: Self, scale: Self) -> Self {
        self.to_f32()
            .__selu(alpha.to_f32(), scale.to_f32())
            .to_f16()
    }

    fn __hard_sigmoid(self) -> Self {
        self.to_f32().__hard_sigmoid().to_f16()
    }

    fn __hard_swish(self) -> Self {
        self.to_f32().__hard_swish().to_f16()
    }

    fn __softplus(self) -> Self {
        self.to_f32().__softplus().to_f16()
    }

    fn __softsign(self) -> Self {
        self.to_f32().__softsign().to_f16()
    }

    fn __mish(self) -> Self {
        self.to_f32().__mish().to_f16()
    }

    fn __cbrt(self) -> Self {
        self.to_f32().__cbrt().to_f16()
    }

    fn __expm1(self) -> Self {
        self.to_f32().__expm1().to_f16()
    }

    fn __exp10(self) -> Self {
        self.to_f32().__exp10().to_f16()
    }

    fn __log1p(self) -> Self {
        self.to_f32().__log1p().to_f16()
    }

    fn __sincos(self) -> (Self, Self)
    where
        Self: Sized,
    {
        (
            self.to_f32().__sin().to_f16(),
            self.to_f32().__cos().to_f16(),
        )
    }

    fn __atan2(self, rhs: Self) -> Self {
        self.to_f32().__atan2(rhs.to_f32()).to_f16()
    }
}
