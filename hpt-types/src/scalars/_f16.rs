use num_traits::real::Real;

use crate::{
    convertion::Convertor,
    type_promote::{
        BitWiseOut2, Eval2, FloatOutBinary2, FloatOutUnary2, NormalOut2, NormalOutUnary2,
    },
};

impl FloatOutBinary2 for half::f16 {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        self / rhs
    }
    #[inline(always)]
    fn __log(self, base: Self) -> Self {
        self.log(base)
    }
    #[inline(always)]
    fn __hypot(self, rhs: Self) -> Self {
        self.hypot(rhs)
    }
    #[inline(always)]
    fn __pow(self, rhs: Self) -> Self {
        self.powf(rhs)
    }
}

impl NormalOut2 for half::f16 {
    #[inline(always)]
    fn __add(self, rhs: Self) -> Self {
        self + rhs
    }

    #[inline(always)]
    fn __sub(self, rhs: Self) -> Self {
        self - rhs
    }

    #[inline(always)]
    fn __mul_add(self, a: Self, b: Self) -> Self {
        (self * a) + b
    }

    #[inline(always)]
    fn __mul(self, rhs: Self) -> Self {
        self * rhs
    }

    #[inline(always)]
    fn __rem(self, rhs: Self) -> Self {
        self % rhs
    }

    #[inline(always)]
    fn __max(self, rhs: Self) -> Self {
        self.max(rhs)
    }

    #[inline(always)]
    fn __min(self, rhs: Self) -> Self {
        self.min(rhs)
    }

    #[inline(always)]
    fn __clamp(self, min: Self, max: Self) -> Self {
        self.clamp(min, max)
    }
}

impl NormalOutUnary2 for half::f16 {
    #[inline(always)]
    fn __square(self) -> Self {
        self * self
    }

    #[inline(always)]
    fn __abs(self) -> Self {
        self.abs()
    }

    #[inline(always)]
    fn __ceil(self) -> Self {
        self.ceil()
    }

    #[inline(always)]
    fn __floor(self) -> Self {
        self.floor()
    }

    #[inline(always)]
    fn __neg(self) -> Self {
        -self
    }

    #[inline(always)]
    fn __round(self) -> Self {
        self.round()
    }

    #[inline(always)]
    fn __signum(self) -> Self {
        self.signum()
    }

    #[inline(always)]
    fn __trunc(self) -> Self {
        self.trunc()
    }

    #[inline(always)]
    fn __leaky_relu(self, alpha: Self) -> Self {
        self.max(half::f16::from_f32_const(0.0)) + alpha * self.min(half::f16::from_f32_const(0.0))
    }

    #[inline(always)]
    fn __relu(self) -> Self {
        self.max(half::f16::from_f32_const(0.0))
    }

    #[inline(always)]
    fn __relu6(self) -> Self {
        self.min(half::f16::from_f32_const(6.0))
            .max(half::f16::from_f32_const(0.0))
    }

    #[inline(always)]
    fn __copysign(self, rhs: Self) -> Self {
        self.copysign(rhs)
    }
}

impl BitWiseOut2 for half::f16 {
    #[inline(always)]
    fn __bitand(self, rhs: Self) -> Self {
        half::f16::from_bits(self.to_bits() & rhs.to_bits())
    }

    #[inline(always)]
    fn __bitor(self, rhs: Self) -> Self {
        half::f16::from_bits(self.to_bits() | rhs.to_bits())
    }

    #[inline(always)]
    fn __bitxor(self, rhs: Self) -> Self {
        half::f16::from_bits(self.to_bits() ^ rhs.to_bits())
    }

    #[inline(always)]
    fn __not(self) -> Self {
        half::f16::from_bits(!self.to_bits())
    }

    #[inline(always)]
    fn __shl(self, _: Self) -> Self {
        panic!("Shift operations are not supported for half::f16")
    }

    #[inline(always)]
    fn __shr(self, _: Self) -> Self {
        panic!("Shift operations are not supported for half::f16")
    }
}

impl Eval2 for half::f16 {
    type Output = bool;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        self.is_nan()
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        *self != half::f16::from_f32_const(0.0) && !self.is_nan()
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        self.is_infinite()
    }
}

impl FloatOutUnary2 for half::f16 {
    #[inline(always)]
    fn __exp(self) -> Self {
        self.exp()
    }
    #[inline(always)]
    fn __expm1(self) -> Self {
        self.to_f32().__expm1().to_f16()
    }
    #[inline(always)]
    fn __exp2(self) -> Self {
        self.exp2()
    }
    #[inline(always)]
    fn __ln(self) -> Self {
        self.ln()
    }
    #[inline(always)]
    fn __log1p(self) -> Self {
        self.to_f32().__log1p().to_f16()
    }
    #[inline(always)]
    fn __celu(self, alpha: Self) -> Self {
        self.to_f32().__celu(alpha.to_f32()).to_f16()
    }
    #[inline(always)]
    fn __log2(self) -> Self {
        self.log2()
    }
    #[inline(always)]
    fn __log10(self) -> Self {
        self.log10()
    }
    #[inline(always)]
    fn __sqrt(self) -> Self {
        self.sqrt()
    }
    #[inline(always)]
    fn __sin(self) -> Self {
        self.sin()
    }
    #[inline(always)]
    fn __cos(self) -> Self {
        self.cos()
    }
    #[inline(always)]
    fn __tan(self) -> Self {
        self.tan()
    }
    #[inline(always)]
    fn __asin(self) -> Self {
        self.asin()
    }
    #[inline(always)]
    fn __acos(self) -> Self {
        self.acos()
    }
    #[inline(always)]
    fn __atan(self) -> Self {
        self.atan()
    }
    #[inline(always)]
    fn __sinh(self) -> Self {
        self.sinh()
    }
    #[inline(always)]
    fn __cosh(self) -> Self {
        self.cosh()
    }
    #[inline(always)]
    fn __tanh(self) -> Self {
        self.tanh()
    }
    #[inline(always)]
    fn __asinh(self) -> Self {
        self.asinh()
    }
    #[inline(always)]
    fn __acosh(self) -> Self {
        self.acosh()
    }
    #[inline(always)]
    fn __atanh(self) -> Self {
        self.atanh()
    }
    #[inline(always)]
    fn __recip(self) -> Self {
        self.recip()
    }
    #[inline(always)]
    fn __erf(self) -> Self {
        self.to_f32().__erf().to_f16()
    }
    #[inline(always)]
    fn __sigmoid(self) -> Self {
        self.to_f32().__sigmoid().to_f16()
    }
    #[inline(always)]
    fn __elu(self, alpha: Self) -> Self {
        self.to_f32().__elu(alpha.to_f32()).to_f16()
    }
    #[inline(always)]
    fn __gelu(self) -> Self {
        self.to_f32().__gelu().to_f16()
    }
    #[inline(always)]
    fn __selu(self, alpha: Self, scale: Self) -> Self {
        self.to_f32()
            .__selu(alpha.to_f32(), scale.to_f32())
            .to_f16()
    }
    #[inline(always)]
    fn __hard_sigmoid(self) -> Self {
        self.to_f32().__hard_sigmoid().to_f16()
    }
    #[inline(always)]
    fn __hard_swish(self) -> Self {
        self.to_f32().__hard_swish().to_f16()
    }
    #[inline(always)]
    fn __softplus(self) -> Self {
        self.to_f32().__softplus().to_f16()
    }
    #[inline(always)]
    fn __softsign(self) -> Self {
        self.to_f32().__softsign().to_f16()
    }
    #[inline(always)]
    fn __mish(self) -> Self {
        self.to_f32().__mish().to_f16()
    }
    #[inline(always)]
    fn __cbrt(self) -> Self {
        self.to_f32().__cbrt().to_f16()
    }
    #[inline(always)]
    fn __sincos(self) -> (Self, Self) {
        let res = self.to_f32().sin_cos();
        (res.0.to_f16(), res.1.to_f16())
    }
    #[inline(always)]
    fn __atan2(self, rhs: Self) -> Self {
        self.atan2(rhs)
    }
    #[inline(always)]
    fn __exp10(self) -> Self {
        self.to_f32().__exp10().to_f16()
    }
}
