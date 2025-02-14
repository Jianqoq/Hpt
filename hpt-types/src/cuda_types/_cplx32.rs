use num_complex::Complex32;

use crate::type_promote::{
    BitWiseOut2, Eval2, FloatOutBinary2, FloatOutUnary2, NormalOut2, NormalOutUnary2,
};

use super::scalar::Scalar;
impl FloatOutBinary2 for Scalar<Complex32> {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        Scalar::new(format!("cuCdivf({}, {})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __log(self, _: Self) -> Self {
        panic!("Log operation is not well-defined for complex numbers")
    }

    #[inline(always)]
    fn __hypot(self, _: Self) -> Self {
        panic!("Hypotenuse operation is not well-defined for complex numbers")
    }
}

impl NormalOut2 for Scalar<Complex32> {
    #[inline(always)]
    fn __add(self, rhs: Self) -> Self {
        Scalar::new(format!("cuCaddf({}, {})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __sub(self, rhs: Self) -> Self {
        Scalar::new(format!("cuCsubf({}, {})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __mul_add(self, a: Self, b: Self) -> Self {
        Scalar::new(format!(
            "cuCaddf(cuCmulf({}, {}), {})",
            self.val, a.val, b.val
        ))
    }

    #[inline(always)]
    fn __mul(self, rhs: Self) -> Self {
        Scalar::new(format!("cuCmulf({}, {})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __pow(self, _: Self) -> Self {
        panic!("Power operation is not supported for complex numbers")
    }

    #[inline(always)]
    fn __rem(self, _: Self) -> Self {
        panic!("Remainder operation is not supported for complex numbers")
    }

    #[inline(always)]
    fn __max(self, _: Self) -> Self {
        panic!("Max operation is not well-defined for complex numbers")
    }

    #[inline(always)]
    fn __min(self, _: Self) -> Self {
        panic!("Min operation is not well-defined for complex numbers")
    }

    #[inline(always)]
    fn __clamp(self, _: Self, _: Self) -> Self {
        panic!("Clip operation is not well-defined for complex numbers")
    }
}

impl NormalOutUnary2 for Scalar<Complex32> {
    #[inline(always)]
    fn __square(self) -> Self {
        Scalar::new(format!("cuCmulf({}, {})", self.val, self.val))
    }

    #[inline(always)]
    fn __abs(self) -> Self {
        Scalar::new(format!("fabsf({})", self.val))
    }

    #[inline(always)]
    fn __ceil(self) -> Self {
        panic!("Ceil operation is not well-defined for complex numbers")
    }

    #[inline(always)]
    fn __floor(self) -> Self {
        panic!("Floor operation is not well-defined for complex numbers")
    }

    #[inline(always)]
    fn __neg(self) -> Self {
        Scalar::new(format!(
            "cuCmulf({}, make_cuComplex(-1.0f, 0.0f))",
            self.val
        ))
    }

    #[inline(always)]
    fn __round(self) -> Self {
        panic!("Round operation is not well-defined for complex numbers")
    }

    #[inline(always)]
    fn __signum(self) -> Self {
        panic!("Sign operation is not well-defined for complex numbers")
    }

    #[inline(always)]
    fn __leaky_relu(self, _: Self) -> Self {
        panic!("Leaky ReLU is not well-defined for complex numbers")
    }

    #[inline(always)]
    fn __relu(self) -> Self {
        panic!("ReLU is not well-defined for complex numbers")
    }

    #[inline(always)]
    fn __relu6(self) -> Self {
        panic!("ReLU6 is not well-defined for complex numbers")
    }

    #[inline(always)]
    fn __trunc(self) -> Self {
        panic!("Trunc operation is not well-defined for complex numbers")
    }

    #[inline(always)]
    fn __copysign(self, _: Self) -> Self {
        panic!("Copysign operation is not well-defined for complex numbers")
    }
}

impl BitWiseOut2 for Scalar<Complex32> {
    #[inline(always)]
    fn __bitand(self, _: Self) -> Self {
        panic!("Bitwise operations are not supported for cuda Complex32")
    }

    #[inline(always)]
    fn __bitor(self, _: Self) -> Self {
        panic!("Bitwise operations are not supported for cuda Complex32")
    }

    #[inline(always)]
    fn __bitxor(self, _: Self) -> Self {
        panic!("Bitwise operations are not supported for cuda Complex32")
    }

    #[inline(always)]
    fn __not(self) -> Self {
        panic!("Bitwise operations are not supported for cuda Complex32")
    }

    #[inline(always)]
    fn __shl(self, _: Self) -> Self {
        panic!("Shift operations are not supported for cuda Complex32")
    }

    #[inline(always)]
    fn __shr(self, _: Self) -> Self {
        panic!("Shift operations are not supported for cuda Complex32")
    }
}

impl Eval2 for Scalar<Complex32> {
    type Output = Scalar<bool>;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        Scalar::new(format!(
            "(isnan(cuCrealf({})) || isnan(cuCimagf({})))",
            self.val, self.val
        ))
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        Scalar::new(format!(
            "((cuCrealf({}) != 0.0f || cuCimagf({}) != 0.0f) && !isnan(cuCrealf({})) && !isnan(cuCimagf({})))",
            self.val, self.val, self.val, self.val
        ))
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        Scalar::new(format!(
            "(isinf(cuCrealf({})) || isinf(cuCimagf({})))",
            self.val, self.val
        ))
    }
}

impl FloatOutUnary2 for Scalar<Complex32> {
    #[inline(always)]
    fn __exp(self) -> Self {
        panic!("Exp operation is not well-defined for complex numbers")
    }
    #[inline(always)]
    fn __exp2(self) -> Self {
        panic!("Exp2 operation is not well-defined for complex numbers")
    }
    #[inline(always)]
    fn __ln(self) -> Self {
        panic!("Ln operation is not well-defined for complex numbers")
    }
    #[inline(always)]
    fn __celu(self, _: Self) -> Self {
        panic!("CElu operation is not well-defined for complex numbers")
    }
    #[inline(always)]
    fn __log2(self) -> Self {
        panic!("Log2 operation is not well-defined for complex numbers")
    }
    #[inline(always)]
    fn __log10(self) -> Self {
        panic!("Log10 operation is not well-defined for complex numbers")
    }
    #[inline(always)]
    fn __sqrt(self) -> Self {
        panic!("Sqrt operation is not well-defined for complex numbers")
    }
    #[inline(always)]
    fn __sin(self) -> Self {
        panic!("Sin operation is not well-defined for complex numbers")
    }
    #[inline(always)]
    fn __cos(self) -> Self {
        panic!("Cos operation is not well-defined for complex numbers")
    }
    #[inline(always)]
    fn __tan(self) -> Self {
        panic!("Tan operation is not well-defined for complex numbers")
    }
    #[inline(always)]
    fn __asin(self) -> Self {
        panic!("Asin operation is not well-defined for complex numbers")
    }
    #[inline(always)]
    fn __acos(self) -> Self {
        panic!("Acos operation is not well-defined for complex numbers")
    }
    #[inline(always)]
    fn __atan(self) -> Self {
        panic!("Atan operation is not well-defined for complex numbers")
    }
    #[inline(always)]
    fn __sinh(self) -> Self {
        panic!("Sinh operation is not well-defined for complex numbers")
    }
    #[inline(always)]
    fn __cosh(self) -> Self {
        panic!("Cosh operation is not well-defined for complex numbers")
    }
    #[inline(always)]
    fn __tanh(self) -> Self {
        panic!("Tanh operation is not well-defined for complex numbers")
    }
    #[inline(always)]
    fn __asinh(self) -> Self {
        panic!("Asinh operation is not well-defined for complex numbers")
    }
    #[inline(always)]
    fn __acosh(self) -> Self {
        panic!("Acosh operation is not well-defined for complex numbers")
    }
    #[inline(always)]
    fn __atanh(self) -> Self {
        panic!("Atanh operation is not well-defined for complex numbers")
    }
    #[inline(always)]
    fn __recip(self) -> Self {
        panic!("Recip operation is not well-defined for complex numbers")
    }
    #[inline(always)]
    fn __erf(self) -> Self {
        panic!("Erf operation is not well-defined for complex numbers")
    }

    #[inline(always)]
    fn __sigmoid(self) -> Self {
        panic!("Sigmoid operation is not well-defined for complex numbers")
    }

    fn __elu(self, _: Self) -> Self {
        panic!("Elu operation is not well-defined for complex numbers")
    }

    fn __gelu(self) -> Self {
        panic!("Gelu operation is not well-defined for complex numbers")
    }

    fn __selu(self, _: Self, _: Self) -> Self {
        panic!("Selu operation is not well-defined for complex numbers")
    }

    fn __hard_sigmoid(self) -> Self {
        panic!("Hard sigmoid operation is not well-defined for complex numbers")
    }

    fn __hard_swish(self) -> Self {
        panic!("Hard swish operation is not well-defined for complex numbers")
    }

    fn __softplus(self) -> Self {
        panic!("Softplus operation is not well-defined for complex numbers")
    }

    fn __softsign(self) -> Self {
        panic!("Softsign operation is not well-defined for complex numbers")
    }

    fn __mish(self) -> Self {
        panic!("Mish operation is not well-defined for complex numbers")
    }

    fn __cbrt(self) -> Self {
        panic!("Cbrt operation is not well-defined for complex numbers")
    }

    fn __expm1(self) -> Self {
        panic!("Expm1 operation is not well-defined for complex numbers")
    }

    fn __exp10(self) -> Self {
        panic!("Exp10 operation is not well-defined for complex numbers")
    }

    fn __log1p(self) -> Self {
        panic!("Log1p operation is not well-defined for complex numbers")
    }

    fn __sincos(self) -> (Self, Self)
    where
        Self: Sized,
    {
        panic!("Sincos operation is not well-defined for complex numbers")
    }

    fn __atan2(self, _: Self) -> Self {
        panic!("Atan2 operation is not well-defined for complex numbers")
    }
}
