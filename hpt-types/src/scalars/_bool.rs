use crate::type_promote::{BitWiseOut2, Eval2, FloatOutBinary2, NormalOut2, NormalOutUnary2};

impl FloatOutBinary2 for bool {
    #[inline(always)]
    fn __div(self, _: Self) -> Self {
        panic!("Division operation is not supported for boolean type")
    }

    #[inline(always)]
    fn __log(self, _: Self) -> Self {
        panic!("Logarithm operation is not supported for boolean type")
    }

    #[inline(always)]
    fn __hypot(self, _: Self) -> Self {
        panic!("Hypot operation is not supported for boolean type")
    }
}

impl NormalOut2 for bool {
    #[inline(always)]
    fn __add(self, rhs: Self) -> Self {
        self || rhs
    }

    #[inline(always)]
    fn __sub(self, _: Self) -> Self {
        panic!("Subtraction is not supported for boolean type")
    }

    #[inline(always)]
    fn __mul_add(self, a: Self, b: Self) -> Self {
        (self && a) || b
    }

    #[inline(always)]
    fn __mul(self, rhs: Self) -> Self {
        self && rhs
    }

    #[inline(always)]
    fn __pow(self, _: Self) -> Self {
        panic!("Power operation is not supported for boolean type")
    }

    #[inline(always)]
    fn __rem(self, _: Self) -> Self {
        panic!("Remainder operation is not supported for boolean type")
    }

    #[inline(always)]
    fn __max(self, rhs: Self) -> Self {
        self || rhs
    }

    #[inline(always)]
    fn __min(self, rhs: Self) -> Self {
        self && rhs
    }

    #[inline(always)]
    fn __clamp(self, _: Self, _: Self) -> Self {
        self
    }
}

impl NormalOutUnary2 for bool {
    #[inline(always)]
    fn __square(self) -> Self {
        self
    }

    #[inline(always)]
    fn __abs(self) -> Self {
        self
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
        !self
    }

    #[inline(always)]
    fn __round(self) -> Self {
        self
    }

    #[inline(always)]
    fn __signum(self) -> Self {
        self
    }

    #[inline(always)]
    fn __trunc(self) -> Self {
        self
    }

    #[inline(always)]
    fn __leaky_relu(self, _: Self) -> Self {
        self
    }

    #[inline(always)]
    fn __relu(self) -> Self {
        self
    }

    #[inline(always)]
    fn __relu6(self) -> Self {
        self
    }

    #[inline(always)]
    fn __copysign(self, sign: Self) -> Self {
        if sign {
            self
        } else {
            !self
        }
    }
}

impl BitWiseOut2 for bool {
    #[inline(always)]
    fn __bitand(self, rhs: Self) -> Self {
        self && rhs
    }

    #[inline(always)]
    fn __bitor(self, rhs: Self) -> Self {
        self || rhs
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
    fn __shl(self, _: Self) -> Self {
        self
    }

    #[inline(always)]
    fn __shr(self, _: Self) -> Self {
        self
    }
}

impl Eval2 for bool {
    type Output = bool;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        false
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        *self
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        false
    }
}
