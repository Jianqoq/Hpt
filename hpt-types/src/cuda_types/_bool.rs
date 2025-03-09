use crate::type_promote::{BitWiseOut2, Eval2, FloatOutBinary2, NormalOut2, NormalOutUnary2};

use super::scalar::Scalar;
impl FloatOutBinary2 for Scalar<bool> {
    #[inline(always)]
    fn __div(self, _: Self) -> Self {
        panic!("Division operation is not supported for bool")
    }

    #[inline(always)]
    fn __log(self, _: Self) -> Self {
        panic!("Logarithm operation is not supported for bool")
    }

    #[inline(always)]
    fn __hypot(self, _: Self) -> Self {
        panic!("Hypotenuse operation is not supported for bool")
    }

    #[inline(always)]
    fn __pow(self, _: Self) -> Self {
        panic!("Power operation is not supported for boolean type")
    }
}

impl NormalOut2 for Scalar<bool> {
    #[inline(always)]
    fn __add(self, rhs: Self) -> Self {
        Scalar::new(format!("({} || {})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __sub(self, _: Self) -> Self {
        panic!("Subtraction is not supported for boolean type")
    }

    #[inline(always)]
    fn __mul_add(self, a: Self, b: Self) -> Self {
        Scalar::new(format!("(({} && {}) || {})", self.val, a.val, b.val))
    }

    #[inline(always)]
    fn __mul(self, rhs: Self) -> Self {
        Scalar::new(format!("({} && {})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __rem(self, _: Self) -> Self {
        panic!("Remainder operation is not supported for boolean type")
    }

    #[inline(always)]
    fn __max(self, rhs: Self) -> Self {
        Scalar::new(format!("({} || {})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __min(self, rhs: Self) -> Self {
        Scalar::new(format!("({} && {})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __clamp(self, _: Self, _: Self) -> Self {
        self
    }
}

impl NormalOutUnary2 for Scalar<bool> {
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
        Scalar::new(format!("(~{})", self.val))
    }

    #[inline(always)]
    fn __round(self) -> Self {
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
    fn __signum(self) -> Self {
        self
    }

    #[inline(always)]
    fn __trunc(self) -> Self {
        self
    }

    #[inline(always)]
    fn __copysign(self, _: Self) -> Self {
        self
    }
}

impl BitWiseOut2 for Scalar<bool> {
    #[inline(always)]
    fn __bitand(self, rhs: Self) -> Self {
        Scalar::new(format!("({} & {})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __bitor(self, rhs: Self) -> Self {
        Scalar::new(format!("({} | {})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __bitxor(self, rhs: Self) -> Self {
        Scalar::new(format!("({} ^ {})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __not(self) -> Self {
        Scalar::new(format!("(~{})", self.val))
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

impl Eval2 for Scalar<bool> {
    type Output = Scalar<bool>;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        Scalar::new(format!("false"))
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        self.clone()
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        Scalar::new(format!("false"))
    }
}
