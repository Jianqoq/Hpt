use crate::type_promote::{BitWiseOut2, Eval2, FloatOutBinary2, NormalOut2, NormalOutUnary2};

use super::scalar::Scalar;
impl FloatOutBinary2 for Scalar<i8> {
    #[inline(always)]
    fn __div(self, rhs: Self) -> Self {
        Scalar::new(format!("({} / {})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __log(self, _: Self) -> Self {
        panic!("Logarithm operation is not supported for i8")
    }

    #[inline(always)]
    fn __hypot(self, _: Self) -> Self {
        panic!("Hypot operation is not supported for i8")
    }

    #[inline(always)]
    fn __pow(self, rhs: Self) -> Self {
        Scalar::new(format!("pow({}, {})", self.val, rhs.val))
    }
}

impl NormalOut2 for Scalar<i8> {
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
        Scalar::new(format!("({} * {} + {})", self.val, a.val, b.val))
    }

    #[inline(always)]
    fn __mul(self, rhs: Self) -> Self {
        Scalar::new(format!("({} * {})", self.val, rhs.val))
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
        Scalar::new(format!("max({}, {})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __min(self, rhs: Self) -> Self {
        Scalar::new(format!("min({}, {})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __clamp(self, min: Self, max: Self) -> Self {
        Scalar::new(format!("min(max({}, {}), {})", self.val, min.val, max.val))
    }
}

impl NormalOutUnary2 for Scalar<i8> {
    #[inline(always)]
    fn __square(self) -> Self {
        Scalar::new(format!("({} * {})", self.val, self.val))
    }

    #[inline(always)]
    fn __abs(self) -> Self {
        Scalar::new(format!("abs({})", self.val))
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
        Scalar::new(format!("(-{})", self.val))
    }

    #[inline(always)]
    fn __round(self) -> Self {
        self
    }

    #[inline(always)]
    fn __signum(self) -> Self {
        Scalar::new(format!(
            "(({} > 0) ? 1 : ({} < 0) ? -1 : 0)",
            self.val, self.val
        ))
    }

    #[inline(always)]
    fn __leaky_relu(self, alpha: Self) -> Self {
        Scalar::new(format!(
            "({} > 0) ? {} : ({} * {})",
            self.val, self.val, alpha.val, self.val
        ))
    }

    #[inline(always)]
    fn __relu(self) -> Self {
        Scalar::new(format!("max({}, 0)", self.val))
    }

    #[inline(always)]
    fn __relu6(self) -> Self {
        Scalar::new(format!("min(max({}, 0), 6)", self.val))
    }

    #[inline(always)]
    fn __trunc(self) -> Self {
        self
    }

    #[inline(always)]
    fn __copysign(self, rhs: Self) -> Self {
        Scalar::new(format!(
            "({} >= 0 ? abs({}) : -abs({}))",
            rhs.val, self.val, self.val
        ))
    }
}

impl BitWiseOut2 for Scalar<i8> {
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
    fn __shl(self, rhs: Self) -> Self {
        Scalar::new(format!("({} << {})", self.val, rhs.val))
    }

    #[inline(always)]
    fn __shr(self, rhs: Self) -> Self {
        Scalar::new(format!("({} >> {})", self.val, rhs.val))
    }
}

impl Eval2 for Scalar<i8> {
    type Output = Scalar<bool>;
    #[inline(always)]
    fn __is_nan(&self) -> Self::Output {
        Scalar::new(format!("false"))
    }

    #[inline(always)]
    fn __is_true(&self) -> Self::Output {
        Scalar::new(format!("({} != 0)", self.val))
    }

    #[inline(always)]
    fn __is_inf(&self) -> Self::Output {
        Scalar::new(format!("false"))
    }
}
