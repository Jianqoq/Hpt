use crate::type_promote::{
    BitWiseOut2, Eval2, FloatOutBinary2, FloatOutUnary2, NormalOut2, NormalOutUnary2,
};

use super::scalar::Scalar;
impl FloatOutBinary2 for Scalar<f64> {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        Scalar::new(format!("({} / {})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __log(self, base: Self) -> Self {
        Scalar::new(format!("(log({}) / log({}))", self.val, base.val))
    }

    #[inline(always)]
    fn __hypot(self, rhs: Self) -> Self {
        Scalar::new(format!("hypot({}, {})", self.val, rhs.val))
    }
}

impl NormalOut2 for Scalar<f64> {
    #[inline(always)]
    fn __add(self, rhs: Self) -> Self {
        Scalar::new(format!("({} + {})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __sub(self, rhs: Self) -> Self {
        Scalar::new(format!("({} - {})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __mul_add(self, a: Self, b: Self) -> Self {
        Scalar::new(format!("fma({}, {}, {})", self.val, a.val, b.val))
    }

    #[inline(always)]
    fn __mul(self, rhs: Self) -> Self {
        Scalar::new(format!("({} * {})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __pow(self, rhs: Self) -> Self {
        Scalar::new(format!("pow({}, {})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __rem(self, rhs: Self) -> Self {
        Scalar::new(format!(
            "(({} != 0) ? (({}) % ({})) : 0)",
            rhs.val, self.val, rhs.val
        ))
    }

    #[inline(always)]
    fn __max(self, rhs: Self) -> Self {
        Scalar::new(format!("fmax({}, {})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __min(self, rhs: Self) -> Self {
        Scalar::new(format!("fmin({}, {})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __clamp(self, min: Self, max: Self) -> Self {
        Scalar::new(format!(
            "fmin(fmax({}, {}), {})",
            self.val, min.val, max.val
        ))
    }
}

impl NormalOutUnary2 for Scalar<f64> {
    #[inline(always)]
    fn __square(self) -> Self {
        Scalar::new(format!("({} * {})", self.val, self.val))
    }

    #[inline(always)]
    fn __abs(self) -> Self {
        Scalar::new(format!("fabs({})", self.val))
    }

    #[inline(always)]
    fn __ceil(self) -> Self {
        Scalar::new(format!("ceil({})", self.val))
    }

    #[inline(always)]
    fn __floor(self) -> Self {
        Scalar::new(format!("floor({})", self.val))
    }

    #[inline(always)]
    fn __neg(self) -> Self {
        Scalar::new(format!("(-{})", self.val))
    }

    #[inline(always)]
    fn __round(self) -> Self {
        Scalar::new(format!("round({})", self.val))
    }

    #[inline(always)]
    fn __signum(self) -> Self {
        Scalar::new(format!("copysign(1.0, {})", self.val))
    }

    #[inline(always)]
    fn __leaky_relu(self, alpha: Self) -> Self {
        Scalar::new(format!(
            "({} > 0.0) ? {} : ({} * {})",
            self.val, self.val, alpha.val, self.val
        ))
    }

    #[inline(always)]
    fn __relu(self) -> Self {
        Scalar::new(format!("fmax({}, 0.0)", self.val))
    }

    #[inline(always)]
    fn __relu6(self) -> Self {
        Scalar::new(format!("fmin(fmax({}, 0.0), 6.0)", self.val))
    }

    #[inline(always)]
    fn __trunc(self) -> Self {
        Scalar::new(format!("trunc({})", self.val))
    }

    #[inline(always)]
    fn __copysign(self, rhs: Self) -> Self {
        Scalar::new(format!("copysign({}, {})", self.val, rhs.val))
    }
}

impl BitWiseOut2 for Scalar<f64> {
    #[inline(always)]
    fn __bitand(self, _: Self) -> Self {
        panic!("Bitwise operations are not supported for cuda f64")
    }

    #[inline(always)]
    fn __bitor(self, _: Self) -> Self {
        panic!("Bitwise operations are not supported for cuda f64")
    }

    #[inline(always)]
    fn __bitxor(self, _: Self) -> Self {
        panic!("Bitwise operations are not supported for cuda f64")
    }

    #[inline(always)]
    fn __not(self) -> Self {
        panic!("Bitwise operations are not supported for cuda f64")
    }

    #[inline(always)]
    fn __shl(self, _: Self) -> Self {
        panic!("Shift operations are not supported for cuda f64")
    }

    #[inline(always)]
    fn __shr(self, _: Self) -> Self {
        panic!("Shift operations are not supported for cuda f64")
    }
}

impl Eval2 for Scalar<f64> {
    type Output = Scalar<bool>;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        Scalar::new(format!("isnan({})", self.val))
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        Scalar::new(format!("({} != 0.0 && !isnan({}))", self.val, self.val))
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        Scalar::new(format!("isinf({})", self.val))
    }
}

impl FloatOutUnary2 for Scalar<f64> {
    #[inline(always)]
    fn __exp(self) -> Self {
        Scalar::new(format!("exp({})", self.val))
    }
    #[inline(always)]
    fn __exp2(self) -> Self {
        Scalar::new(format!("exp2({})", self.val))
    }
    #[inline(always)]
    fn __ln(self) -> Self {
        Scalar::new(format!("log({})", self.val))
    }
    #[inline(always)]
    fn __celu(self, alpha: Self) -> Self {
        Scalar::new(format!(
            "({} > 0.0 ? {} : {} * (exp({} / {}) - 1.0))",
            self.val, self.val, alpha.val, self.val, alpha.val
        ))
    }
    #[inline(always)]
    fn __log2(self) -> Self {
        Scalar::new(format!("log2({})", self.val))
    }
    #[inline(always)]
    fn __log10(self) -> Self {
        Scalar::new(format!("log10({})", self.val))
    }
    #[inline(always)]
    fn __sqrt(self) -> Self {
        Scalar::new(format!("sqrt({})", self.val))
    }
    #[inline(always)]
    fn __sin(self) -> Self {
        Scalar::new(format!("sin({})", self.val))
    }
    #[inline(always)]
    fn __cos(self) -> Self {
        Scalar::new(format!("cos({})", self.val))
    }
    #[inline(always)]
    fn __tan(self) -> Self {
        Scalar::new(format!("tan({})", self.val))
    }
    #[inline(always)]
    fn __asin(self) -> Self {
        Scalar::new(format!("asin({})", self.val))
    }
    #[inline(always)]
    fn __acos(self) -> Self {
        Scalar::new(format!("acos({})", self.val))
    }
    #[inline(always)]
    fn __atan(self) -> Self {
        Scalar::new(format!("atan({})", self.val))
    }
    #[inline(always)]
    fn __sinh(self) -> Self {
        Scalar::new(format!("sinh({})", self.val))
    }
    #[inline(always)]
    fn __cosh(self) -> Self {
        Scalar::new(format!("cosh({})", self.val))
    }
    #[inline(always)]
    fn __tanh(self) -> Self {
        Scalar::new(format!("tanh({})", self.val))
    }
    #[inline(always)]
    fn __asinh(self) -> Self {
        Scalar::new(format!("asinh({})", self.val))
    }
    #[inline(always)]
    fn __acosh(self) -> Self {
        Scalar::new(format!("acosh({})", self.val))
    }
    #[inline(always)]
    fn __atanh(self) -> Self {
        Scalar::new(format!("atanh({})", self.val))
    }
    #[inline(always)]
    fn __recip(self) -> Self {
        Scalar::new(format!("(1.0 / {})", self.val))
    }
    #[inline(always)]
    fn __erf(self) -> Self {
        Scalar::new(format!("erf({})", self.val))
    }

    #[inline(always)]
    fn __sigmoid(self) -> Self {
        Scalar::new(format!("(1.0 / (1.0 + exp(-{})))", self.val))
    }

    fn __elu(self, alpha: Self) -> Self {
        Scalar::new(format!(
            "({} > 0.0) ? {} : ({} * (exp({}) - 1.0))",
            self.val, self.val, alpha.val, self.val
        ))
    }

    fn __gelu(self) -> Self {
        Scalar::new(format!(
            "(0.5 * {} * (1.0 + erf({} * {})))",
            self.val,
            self.val,
            std::f64::consts::FRAC_1_SQRT_2
        ))
    }

    fn __selu(self, alpha: Self, scale: Self) -> Self {
        Scalar::new(format!(
            "({} * ({} > 0.0 ? {} : ({} * (exp({}) - 1.0))))",
            scale.val, self.val, self.val, alpha.val, self.val
        ))
    }

    fn __hard_sigmoid(self) -> Self {
        Scalar::new(format!(
            "fmax(fmin({} * (1.0 / 6.0) + 0.5, 1.0), 0.0)",
            self.val
        ))
    }

    fn __hard_swish(self) -> Self {
        Scalar::new(format!(
            "({} * (fmin(fmax({} + 3.0, 0.0), 6.0) / 6.0))",
            self.val, self.val
        ))
    }

    fn __softplus(self) -> Self {
        Scalar::new(format!("log(1.0 + exp({}))", self.val))
    }

    fn __softsign(self) -> Self {
        Scalar::new(format!("({} / (1.0 + fabs({})))", self.val, self.val))
    }

    fn __mish(self) -> Self {
        Scalar::new(format!(
            "({} * tanh(log(1.0 + exp({}))))",
            self.val, self.val
        ))
    }

    fn __cbrt(self) -> Self {
        Scalar::new(format!("cbrt({})", self.val))
    }

    fn __expm1(self) -> Self {
        Scalar::new(format!("expm1({})", self.val))
    }

    fn __exp10(self) -> Self {
        Scalar::new(format!("exp10({})", self.val))
    }

    fn __log1p(self) -> Self {
        Scalar::new(format!("log1p({})", self.val))
    }

    fn __sincos(self) -> (Self, Self)
    where
        Self: Sized,
    {
        (
            Scalar::new(format!("sin({})", self.val)),
            Scalar::new(format!("cos({})", self.val)),
        )
    }

    fn __atan2(self, rhs: Self) -> Self {
        Scalar::new(format!("atan2({}, {})", self.val, rhs.val))
    }
}
