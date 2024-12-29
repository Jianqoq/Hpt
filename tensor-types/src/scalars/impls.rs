use crate::type_promote::{NormalOutUnary2, NormalOut2, FloatOutBinary2, BitWiseOut2, Eval2};
macro_rules! impl_int_traits {
    ($type:ty, [$($abs:tt)*], [$($neg:tt)*], [$($signum:tt)*]) => {
        impl FloatOutBinary2 for $type {
            #[inline(always)]
            fn __div(self, rhs: Self) -> Self {
                if rhs == 0 {
                    panic!("Division by zero for {}", stringify!($type));
                } else {
                    self / rhs
                }
            }
        
            #[inline(always)]
            fn __log(self, _: Self) -> Self {
                panic!("Logarithm operation is not supported for {}", stringify!($type));
            }
        }
        
        impl NormalOut2 for $type {
            #[inline(always)]
            fn __add(self, rhs: Self) -> Self {
                self.wrapping_add(rhs)
            }
        
            #[inline(always)]
            fn __sub(self, rhs: Self) -> Self {
                self.wrapping_sub(rhs)
            }
        
            #[inline(always)]
            fn __mul_add(self, a: Self, b: Self) -> Self {
                (self * a) + b
            }
        
            #[inline(always)]
            fn __mul(self, rhs: Self) -> Self {
                self.wrapping_mul(rhs)
            }
        
            #[inline(always)]
            fn __pow(self, rhs: Self) -> Self {
                self.pow(rhs as u32)
            }
        
            #[inline(always)]
            fn __rem(self, rhs: Self) -> Self {
                self.wrapping_rem(rhs)
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
        
        impl NormalOutUnary2 for $type {
            #[inline(always)]
            fn __square(self) -> Self {
                self.wrapping_mul(self)
            }
        
            #[inline(always)]
            fn __abs(self) -> Self {
                self$($abs)*
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
                $($neg)*self
            }
        
            #[inline(always)]
            fn __round(self) -> Self {
                self
            }
        
            #[inline(always)]
            fn __signum(self) -> Self {
                self$($signum)*
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
        
        impl BitWiseOut2 for $type {
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
                self.wrapping_shl(rhs as u32)
            }
        
            #[inline(always)]
            fn __shr(self, rhs: Self) -> Self {
                self.wrapping_shr(rhs as u32)
            }
        }
        
        impl Eval2 for $type {
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
    };
}

impl_int_traits!(i8, [.abs()], [-], [.signum()]);
impl_int_traits!(i16, [.abs()], [-], [.signum()]);
impl_int_traits!(i32, [.abs()], [-], [.signum()]);
impl_int_traits!(i64, [.abs()], [-], [.signum()]);
impl_int_traits!(i128, [.abs()], [-], [.signum()]);
impl_int_traits!(isize, [.abs()], [-], [.signum()]);
impl_int_traits!(u8, [], [], []);
impl_int_traits!(u16, [], [], []);
impl_int_traits!(u32, [], [], []);
impl_int_traits!(u64, [], [], []);
impl_int_traits!(u128, [], [], []);
impl_int_traits!(usize, [], [], []);

