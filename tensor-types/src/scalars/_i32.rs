use crate::type_promote::{BitWiseOut2, Eval2, FloatOutBinary2, NormalOut2, NormalOutUnary2};

impl FloatOutBinary2 for i32 {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        if rhs == 0 {
            panic!("Division by zero");
        } else {
            self / rhs
        }
    }

    #[inline(always)]
    fn __log(self, _: Self) -> Self {
        panic!("Logarithm operation is not supported for i32")
    }
}

impl NormalOut2 for i32 {
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
        self.pow(rhs as u32)
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
    fn __clip(self, min: Self, max: Self) -> Self {
        self.clamp(min, max)
    }
}

impl NormalOutUnary2 for i32 {
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
        self
    }

    #[inline(always)]
    fn __floor(self) -> Self {
        self
    }

    #[inline(always)]
    fn __neg(self) -> Self {
        -self
    }

    #[inline(always)]
    fn __round(self) -> Self {
        self
    }

    #[inline(always)]
    fn __sign(self) -> Self {
        self.signum()
    }

    #[inline(always)]
    fn __leaky_relu(self, alpha: Self) -> Self {
        self.max(0) + alpha * self.min(0)
    }

    #[inline(always)]
    fn __relu(self) -> Self {
        self.max(0)
    }

    #[inline(always)]
    fn __relu6(self) -> Self {
        self.min(6).max(0)
    }
}

impl BitWiseOut2 for i32 {
    #[inline(always)]
    fn __bitand(self, rhs: Self) -> Self {
        self & rhs
    }

    #[inline(always)]
    fn __bitor(self, rhs: Self) -> Self {
        self | rhs
    }

    #[inline(always)]
    fn __bitxor(self, rhs: Self) -> Self {
        self ^ rhs
    }

    #[inline(always)]
    fn __not(self) -> Self {
        !self
    }

    #[inline(always)]
    fn __shl(self, rhs: Self) -> Self {
        self << rhs
    }

    #[inline(always)]
    fn __shr(self, rhs: Self) -> Self {
        self >> rhs
    }
}

impl Eval2 for i32 {
    type Output = bool;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        false
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        *self != 0
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        false
    }
}
