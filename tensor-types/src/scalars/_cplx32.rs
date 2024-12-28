use num_complex::{Complex32, ComplexFloat};

use crate::type_promote::{BitWiseOut2, Eval2, FloatOutBinary2, FloatOutUnary2, NormalOut2, NormalOutUnary2};

impl FloatOutBinary2 for Complex32 {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline(always)]
    fn __log(self, base: Self) -> Self {
        self.log(base.re)
    }
}

impl NormalOut2 for Complex32 {
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
    fn __pow(self, rhs: Self) -> Self {
        self.powf(rhs.re)
    }

    #[inline(always)]
    fn __rem(self, rhs: Self) -> Self {
        self % rhs
    }

    #[inline(always)]
    fn __max(self, rhs: Self) -> Self {
        if self.norm() >= rhs.norm() {
            self
        } else {
            rhs
        }
    }

    #[inline(always)]
    fn __min(self, rhs: Self) -> Self {
        if self.norm() <= rhs.norm() {
            self
        } else {
            rhs
        }
    }

    #[inline(always)]
    fn __clip(self, min: Self, max: Self) -> Self {
        let norm = self.norm();
        if norm < min.norm() {
            self * (min.norm() / norm)
        } else if norm > max.norm() {
            self * (max.norm() / norm)
        } else {
            self
        }
    }
}

impl NormalOutUnary2 for Complex32 {
    #[inline(always)]
    fn __square(self) -> Self {
        self * self
    }

    #[inline(always)]
    fn __abs(self) -> Self {
        self.abs().into()
    }

    #[inline(always)]
    fn __ceil(self) -> Self {
        Complex32::new(self.re.ceil(), self.im.ceil())
    }

    #[inline(always)]
    fn __floor(self) -> Self {
        Complex32::new(self.re.floor(), self.im.floor())
    }

    #[inline(always)]
    fn __neg(self) -> Self {
        -self
    }

    #[inline(always)]
    fn __round(self) -> Self {
        Complex32::new(self.re.round(), self.im.round())
    }

    #[inline(always)]
    fn __signum(self) -> Self {
        if self == Complex32::new(0.0, 0.0) {
            self
        } else {
            self / Complex32::from(self.norm())
        }
    }

    #[inline(always)]
    fn __leaky_relu(self, alpha: Self) -> Self {
        let norm = self.norm();
        if norm > 0.0 {
            self
        } else {
            self * alpha
        }
    }

    #[inline(always)]
    fn __relu(self) -> Self {
        let norm = self.norm();
        if norm > 0.0 {
            self
        } else {
            Complex32::new(0.0, 0.0)
        }
    }

    #[inline(always)]
    fn __relu6(self) -> Self {
        let norm = self.norm();
        if norm > 6.0 {
            self * (6.0 / norm)
        } else if norm > 0.0 {
            self
        } else {
            Complex32::new(0.0, 0.0)
        }
    }
}

impl BitWiseOut2 for Complex32 {
    #[inline(always)]
    fn __bitand(self, rhs: Self) -> Self {
        Complex32::new(
            f32::from_bits(self.re.to_bits() & rhs.re.to_bits()),
            f32::from_bits(self.im.to_bits() & rhs.im.to_bits()),
        )
    }

    #[inline(always)]
    fn __bitor(self, rhs: Self) -> Self {
        Complex32::new(
            f32::from_bits(self.re.to_bits() | rhs.re.to_bits()),
            f32::from_bits(self.im.to_bits() | rhs.im.to_bits()),
        )
    }

    #[inline(always)]
    fn __bitxor(self, rhs: Self) -> Self {
        Complex32::new(
            f32::from_bits(self.re.to_bits() ^ rhs.re.to_bits()),
            f32::from_bits(self.im.to_bits() ^ rhs.im.to_bits()),
        )
    }

    #[inline(always)]
    fn __not(self) -> Self {
        Complex32::new(
            f32::from_bits(!self.re.to_bits()),
            f32::from_bits(!self.im.to_bits()),
        )
    }

    #[inline(always)]
    fn __shl(self, _: Self) -> Self {
        panic!("Shift operations are not supported for Complex32")
    }

    #[inline(always)]
    fn __shr(self, _: Self) -> Self {
        panic!("Shift operations are not supported for Complex32")
    }
}

impl Eval2 for Complex32 {
    type Output = bool;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        self.is_nan()
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        self.norm() != 0.0 && !self.is_nan()
    }

    #[inline(always)]
    fn __is_inf(&self) -> bool {
        self.is_infinite()
    }
}

impl FloatOutUnary2 for Complex32 {
    #[inline(always)]
    fn __exp(self) -> Self {
        self.exp()
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
    fn __celu(self, _: Self) -> Self {
        panic!("celu is not supported for complex numbers")
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
        panic!("erf is not supported for complex numbers")
    }

    #[inline(always)]
    fn __sigmoid(self) -> Self {
        1.0 / (1.0 + (-self).exp())
    }

    fn __elu(self, _: Self) -> Self {
        panic!("elu is not supported for complex numbers")
    }

    fn __gelu(self) -> Self {
        panic!("gelu is not supported for complex numbers")
    }

    fn __selu(self, _: Self, _: Self) -> Self {
        panic!("selu is not supported for complex numbers")
    }

    fn __hard_sigmoid(self) -> Self {
        panic!("hard sigmoid is not supported for complex numbers")
    }

    fn __fast_hard_sigmoid(self) -> Self {
        panic!("fast hard sigmoid is not supported for complex numbers")
    }

    fn __hard_swish(self) -> Self {
        panic!("hard swish is not supported for complex numbers")
    }

    fn __softplus(self) -> Self {
        panic!("softplus is not supported for complex numbers")
    }

    fn __softsign(self) -> Self {
        self / (1.0 + self.abs())
    }

    fn __mish(self) -> Self {
        self * ((1.0 + self.exp()).ln()).tanh()
    }

    fn __cbrt(self) -> Self {
        panic!("cbrt is not supported for complex numbers")
    }
}
