use crate::type_promote::{
    BitWiseOut2, Eval2, FloatOutBinary2, FloatOutUnary2, NormalOut2, NormalOutUnary2,
};

use super::scalar::Scalar;
impl FloatOutBinary2 for Scalar<f32> {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        Scalar::new(format!("({} / {})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __log(self, base: Self) -> Self {
        Scalar::new(format!("(logf({}) / logf({}))", self.val, base.val))
    }
}

impl NormalOut2 for Scalar<f32> {
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
        Scalar::new(format!("fmaf({}, {}, {})", self.val, a.val, b.val))
    }

    #[inline(always)]
    fn __mul(self, rhs: Self) -> Self {
        Scalar::new(format!("({} * {})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __pow(self, rhs: Self) -> Self {
        Scalar::new(format!("powf({}, {})", self.val, rhs.val))
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
        Scalar::new(format!("fmaxf({}, {})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __min(self, rhs: Self) -> Self {
        Scalar::new(format!("fminf({}, {})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __clip(self, min: Self, max: Self) -> Self {
        Scalar::new(format!(
            "fminf(fmaxf({}, {}), {})",
            self.val, min.val, max.val
        ))
    }
}

impl NormalOutUnary2 for Scalar<f32> {
    #[inline(always)]
    fn __square(self) -> Self {
        Scalar::new(format!("({} * {})", self.val, self.val))
    }

    #[inline(always)]
    fn __abs(self) -> Self {
        Scalar::new(format!("fabsf({})", self.val))
    }

    #[inline(always)]
    fn __ceil(self) -> Self {
        Scalar::new(format!("ceilf({})", self.val))
    }

    #[inline(always)]
    fn __floor(self) -> Self {
        Scalar::new(format!("floorf({})", self.val))
    }

    #[inline(always)]
    fn __neg(self) -> Self {
        Scalar::new(format!("(-{})", self.val))
    }

    #[inline(always)]
    fn __round(self) -> Self {
        Scalar::new(format!("roundf({})", self.val))
    }

    #[inline(always)]
    fn __sign(self) -> Self {
        Scalar::new(format!("copysignf(1.0f, {})", self.val))
    }

    #[inline(always)]
    fn __leaky_relu(self, alpha: Self) -> Self {
        Scalar::new(format!(
            "({} > 0.0f) ? {} : ({} * {})",
            self.val, self.val, alpha.val, self.val
        ))
    }

    #[inline(always)]
    fn __relu(self) -> Self {
        Scalar::new(format!("fmaxf({}, 0.0f)", self.val))
    }

    #[inline(always)]
    fn __relu6(self) -> Self {
        Scalar::new(format!("fminf(fmaxf({}, 0.0f), 6.0f)", self.val))
    }
}

impl BitWiseOut2 for Scalar<f32> {
    #[inline(always)]
    fn __bitand(self, _: Self) -> Self {
        panic!("Bitwise operations are not supported for cuda f32")
    }

    #[inline(always)]
    fn __bitor(self, _: Self) -> Self {
        panic!("Bitwise operations are not supported for cuda f32")
    }

    #[inline(always)]
    fn __bitxor(self, _: Self) -> Self {
        panic!("Bitwise operations are not supported for cuda f32")
    }

    #[inline(always)]
    fn __not(self) -> Self {
        panic!("Bitwise operations are not supported for cuda f32")
    }

    #[inline(always)]
    fn __shl(self, _: Self) -> Self {
        panic!("Shift operations are not supported for cuda f32")
    }

    #[inline(always)]
    fn __shr(self, _: Self) -> Self {
        panic!("Shift operations are not supported for cuda f32")
    }
}

impl Eval2 for Scalar<f32> {
    type Output = Scalar<bool>;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        Scalar::new(format!("isnan({})", self.val))
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        Scalar::new(format!("({} != 0.0f && !isnan({}))", self.val, self.val))
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        Scalar::new(format!("isinf({})", self.val))
    }
}

impl FloatOutUnary2 for Scalar<f32> {
    #[inline(always)]
    fn __exp(self) -> Self {
        Scalar::new(format!("expf({})", self.val))
    }
    #[inline(always)]
    fn __exp2(self) -> Self {
        Scalar::new(format!("exp2f({})", self.val))
    }
    #[inline(always)]
    fn __ln(self) -> Self {
        Scalar::new(format!("logf({})", self.val))
    }
    #[inline(always)]
    fn __celu(self, alpha: Self) -> Self {
        Scalar::new(format!(
            "({} > 0.0f) ? {} : ({} * expf({}) - 0.0f)",
            self.val, self.val, alpha.val, self.val
        ))
    }
    #[inline(always)]
    fn __log2(self) -> Self {
        Scalar::new(format!("log2f({})", self.val))
    }
    #[inline(always)]
    fn __log10(self) -> Self {
        Scalar::new(format!("log10f({})", self.val))
    }
    #[inline(always)]
    fn __sqrt(self) -> Self {
        Scalar::new(format!("sqrtf({})", self.val))
    }
    #[inline(always)]
    fn __sin(self) -> Self {
        Scalar::new(format!("sinf({})", self.val))
    }
    #[inline(always)]
    fn __cos(self) -> Self {
        Scalar::new(format!("cosf({})", self.val))
    }
    #[inline(always)]
    fn __tan(self) -> Self {
        Scalar::new(format!("tanf({})", self.val))
    }
    #[inline(always)]
    fn __asin(self) -> Self {
        Scalar::new(format!("asinf({})", self.val))
    }
    #[inline(always)]
    fn __acos(self) -> Self {
        Scalar::new(format!("acosf({})", self.val))
    }
    #[inline(always)]
    fn __atan(self) -> Self {
        Scalar::new(format!("atanf({})", self.val))
    }
    #[inline(always)]
    fn __sinh(self) -> Self {
        Scalar::new(format!("sinf({})", self.val))
    }
    #[inline(always)]
    fn __cosh(self) -> Self {
        Scalar::new(format!("cosf({})", self.val))
    }
    #[inline(always)]
    fn __tanh(self) -> Self {
        Scalar::new(format!("tanf({})", self.val))
    }
    #[inline(always)]
    fn __asinh(self) -> Self {
        Scalar::new(format!("asinf({})", self.val))
    }
    #[inline(always)]
    fn __acosh(self) -> Self {
        Scalar::new(format!("acoshf({})", self.val))
    }
    #[inline(always)]
    fn __atanh(self) -> Self {
        Scalar::new(format!("atanhf({})", self.val))
    }
    #[inline(always)]
    fn __recip(self) -> Self {
        Scalar::new(format!("(1.0f / {})", self.val))
    }
    #[inline(always)]
    fn __erf(self) -> Self {
        Scalar::new(format!("erff({})", self.val))
    }

    #[inline(always)]
    fn __sigmoid(self) -> Self {
        Scalar::new(format!("(1.0f / (1.0f + expf(-{})))", self.val))
    }

    fn __elu(self, alpha: Self) -> Self {
        Scalar::new(format!(
            "({} > 0.0f) ? {} : ({} * (expf({}) - 1.0f))",
            self.val, self.val, alpha.val, self.val
        ))
    }

    fn __gelu(self) -> Self {
        Scalar::new(format!(
            "(0.5f * {} * (1.0f + erff({} * {} * {}f)))",
            self.val,
            self.val,
            self.val,
            std::f32::consts::FRAC_1_SQRT_2
        ))
    }

    fn __selu(self, alpha: Self, scale: Self) -> Self {
        Scalar::new(format!(
            "({} * ({} > 0.0f ? {} : ({} * (expf({}) - 1.0f))))",
            scale.val, self.val, self.val, alpha.val, self.val
        ))
    }

    fn __hard_sigmoid(self) -> Self {
        Scalar::new(format!(
            "fminf(fmaxf({} * 0.2f + 0.5f, 0.0f), 1.0f)",
            self.val
        ))
    }

    fn __fast_hard_sigmoid(self) -> Self {
        Scalar::new(format!(
            "(fminf(fmaxf({} + 1.0f, 0.0f), 2.0f) * 0.5f)",
            self.val
        ))
    }

    fn __hard_swish(self) -> Self {
        Scalar::new(format!(
            "({} * (fminf(fmaxf({} + 3.0f, 0.0f), 6.0f) / 6.0f))",
            self.val, self.val
        ))
    }

    fn __softplus(self) -> Self {
        Scalar::new(format!(
            "(fmaxf({}, 20.0f) + logf(1.0f + expf(-fabsf({}))))",
            self.val, self.val
        ))
    }

    fn __softsign(self) -> Self {
        Scalar::new(format!("({} / (1.0f + fabsf({})))", self.val, self.val))
    }

    fn __mish(self) -> Self {
        Scalar::new(format!(
            "({} * tanhf(logf(1.0f + expf({}))))",
            self.val, self.val
        ))
    }

    fn __cbrt(self) -> Self {
        Scalar::new(format!("cbrtf({})", self.val))
    }
}
