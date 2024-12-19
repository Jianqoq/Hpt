// use crate::type_promote::NormalOutUnary;

// impl NormalOutUnary for bool {
//     type Base = Self;
//     #[inline(always)]
//     fn _square(self) -> Self {
//         self._mul(self)
//     }
//     #[inline(always)]
//     fn _neg(self) -> Self {
//         !self
//     }
//     #[inline(always)]
//     fn _abs(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _ceil(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _floor(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _sign(self) -> Self {
//         bool::ZERO
//     }
//     #[inline(always)]
//     fn _round(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _relu(self) -> Self {
//         self._max(Self::ZERO)
//     }
//     #[inline(always)]
//     fn _relu6(self) -> Self {
//         self._max(Self::ZERO)._min(Self::SIX)
//     }
//     #[inline(always)]
//     fn _leaky_relu(self,alpha:Self::Base) -> Self {
//         self._max(Self::ZERO)._add(alpha._mul(self._min(Self::ZERO)))
//     }

//     }
// impl NormalOutUnary for f16 {
//     type Base = Self;
//     #[inline(always)]
//     fn _square(self) -> Self {
//         self._mul(self)
//     }
//     #[inline(always)]
//     fn _neg(self) -> Self {
//         -self
//     }
//     #[inline(always)]
//     fn _abs(self) -> Self {
//         self.abs()
//     }
//     #[inline(always)]
//     fn _ceil(self) -> Self {
//         self.ceil()
//     }
//     #[inline(always)]
//     fn _floor(self) -> Self {
//         self.floor()
//     }
//     #[inline(always)]
//     fn _sign(self) -> Self {
//         self.signum()
//     }
//     #[inline(always)]
//     fn _round(self) -> Self {
//         self.round()
//     }
//     #[inline(always)]
//     fn _relu(self) -> Self {
//         self._max(Self::ZERO)
//     }
//     #[inline(always)]
//     fn _relu6(self) -> Self {
//         self._max(Self::ZERO)._min(Self::SIX)
//     }
//     #[inline(always)]
//     fn _leaky_relu(self,alpha:Self::Base) -> Self {
//         self._max(Self::ZERO)._add(alpha._mul(self._min(Self::ZERO)))
//     }

//     }
// impl NormalOutUnary for f32 {
//     type Base = Self;
//     #[inline(always)]
//     fn _square(self) -> Self {
//         self._mul(self)
//     }
//     #[inline(always)]
//     fn _neg(self) -> Self {
//         -self
//     }
//     #[inline(always)]
//     fn _abs(self) -> Self {
//         self.abs()
//     }
//     #[inline(always)]
//     fn _ceil(self) -> Self {
//         self.ceil()
//     }
//     #[inline(always)]
//     fn _floor(self) -> Self {
//         self.floor()
//     }
//     #[inline(always)]
//     fn _sign(self) -> Self {
//         self.signum()
//     }
//     #[inline(always)]
//     fn _round(self) -> Self {
//         self.round()
//     }
//     #[inline(always)]
//     fn _relu(self) -> Self {
//         self._max(Self::ZERO)
//     }
//     #[inline(always)]
//     fn _relu6(self) -> Self {
//         self._max(Self::ZERO)._min(Self::SIX)
//     }
//     #[inline(always)]
//     fn _leaky_relu(self,alpha:Self::Base) -> Self {
//         self._max(Self::ZERO)._add(alpha._mul(self._min(Self::ZERO)))
//     }

//     }
// impl NormalOutUnary for f64 {
//     type Base = Self;
//     #[inline(always)]
//     fn _square(self) -> Self {
//         self._mul(self)
//     }
//     #[inline(always)]
//     fn _neg(self) -> Self {
//         -self
//     }
//     #[inline(always)]
//     fn _abs(self) -> Self {
//         self.abs()
//     }
//     #[inline(always)]
//     fn _ceil(self) -> Self {
//         self.ceil()
//     }
//     #[inline(always)]
//     fn _floor(self) -> Self {
//         self.floor()
//     }
//     #[inline(always)]
//     fn _sign(self) -> Self {
//         self.signum()
//     }
//     #[inline(always)]
//     fn _round(self) -> Self {
//         self.round()
//     }
//     #[inline(always)]
//     fn _relu(self) -> Self {
//         self._max(Self::ZERO)
//     }
//     #[inline(always)]
//     fn _relu6(self) -> Self {
//         self._max(Self::ZERO)._min(Self::SIX)
//     }
//     #[inline(always)]
//     fn _leaky_relu(self,alpha:Self::Base) -> Self {
//         self._max(Self::ZERO)._add(alpha._mul(self._min(Self::ZERO)))
//     }

//     }
// impl NormalOutUnary for i8 {
//     type Base = Self;
//     #[inline(always)]
//     fn _square(self) -> Self {
//         self._mul(self)
//     }
//     #[inline(always)]
//     fn _neg(self) -> Self {
//         -self
//     }
//     #[inline(always)]
//     fn _abs(self) -> Self {
//         self.abs()
//     }
//     #[inline(always)]
//     fn _ceil(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _floor(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _sign(self) -> Self {
//         self.signum()
//     }
//     #[inline(always)]
//     fn _round(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _relu(self) -> Self {
//         self._max(Self::ZERO)
//     }
//     #[inline(always)]
//     fn _relu6(self) -> Self {
//         self._max(Self::ZERO)._min(Self::SIX)
//     }
//     #[inline(always)]
//     fn _leaky_relu(self,alpha:Self::Base) -> Self {
//         self._max(Self::ZERO)._add(alpha._mul(self._min(Self::ZERO)))
//     }

//     }
// impl NormalOutUnary for i16 {
//     type Base = Self;
//     #[inline(always)]
//     fn _square(self) -> Self {
//         self._mul(self)
//     }
//     #[inline(always)]
//     fn _neg(self) -> Self {
//         -self
//     }
//     #[inline(always)]
//     fn _abs(self) -> Self {
//         self.abs()
//     }
//     #[inline(always)]
//     fn _ceil(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _floor(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _sign(self) -> Self {
//         self.signum()
//     }
//     #[inline(always)]
//     fn _round(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _relu(self) -> Self {
//         self._max(Self::ZERO)
//     }
//     #[inline(always)]
//     fn _relu6(self) -> Self {
//         self._max(Self::ZERO)._min(Self::SIX)
//     }
//     #[inline(always)]
//     fn _leaky_relu(self,alpha:Self::Base) -> Self {
//         self._max(Self::ZERO)._add(alpha._mul(self._min(Self::ZERO)))
//     }

//     }
// impl NormalOutUnary for i32 {
//     type Base = Self;
//     #[inline(always)]
//     fn _square(self) -> Self {
//         self._mul(self)
//     }
//     #[inline(always)]
//     fn _neg(self) -> Self {
//         -self
//     }
//     #[inline(always)]
//     fn _abs(self) -> Self {
//         self.abs()
//     }
//     #[inline(always)]
//     fn _ceil(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _floor(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _sign(self) -> Self {
//         self.signum()
//     }
//     #[inline(always)]
//     fn _round(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _relu(self) -> Self {
//         self._max(Self::ZERO)
//     }
//     #[inline(always)]
//     fn _relu6(self) -> Self {
//         self._max(Self::ZERO)._min(Self::SIX)
//     }
//     #[inline(always)]
//     fn _leaky_relu(self,alpha:Self::Base) -> Self {
//         self._max(Self::ZERO)._add(alpha._mul(self._min(Self::ZERO)))
//     }

//     }
// impl NormalOutUnary for i64 {
//     type Base = Self;
//     #[inline(always)]
//     fn _square(self) -> Self {
//         self._mul(self)
//     }
//     #[inline(always)]
//     fn _neg(self) -> Self {
//         -self
//     }
//     #[inline(always)]
//     fn _abs(self) -> Self {
//         self.abs()
//     }
//     #[inline(always)]
//     fn _ceil(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _floor(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _sign(self) -> Self {
//         self.signum()
//     }
//     #[inline(always)]
//     fn _round(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _relu(self) -> Self {
//         self._max(Self::ZERO)
//     }
//     #[inline(always)]
//     fn _relu6(self) -> Self {
//         self._max(Self::ZERO)._min(Self::SIX)
//     }
//     #[inline(always)]
//     fn _leaky_relu(self,alpha:Self::Base) -> Self {
//         self._max(Self::ZERO)._add(alpha._mul(self._min(Self::ZERO)))
//     }

//     }
// impl NormalOutUnary for u8 {
//     type Base = Self;
//     #[inline(always)]
//     fn _square(self) -> Self {
//         self._mul(self)
//     }
//     #[inline(always)]
//     fn _neg(self) -> Self {
//         !self+1
//     }
//     #[inline(always)]
//     fn _abs(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _ceil(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _floor(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _sign(self) -> Self {
//         u8::ZERO
//     }
//     #[inline(always)]
//     fn _round(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _relu(self) -> Self {
//         self._max(Self::ZERO)
//     }
//     #[inline(always)]
//     fn _relu6(self) -> Self {
//         self._max(Self::ZERO)._min(Self::SIX)
//     }
//     #[inline(always)]
//     fn _leaky_relu(self,alpha:Self::Base) -> Self {
//         self._max(Self::ZERO)._add(alpha._mul(self._min(Self::ZERO)))
//     }

//     }
// impl NormalOutUnary for u16 {
//     type Base = Self;
//     #[inline(always)]
//     fn _square(self) -> Self {
//         self._mul(self)
//     }
//     #[inline(always)]
//     fn _neg(self) -> Self {
//         !self+1
//     }
//     #[inline(always)]
//     fn _abs(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _ceil(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _floor(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _sign(self) -> Self {
//         u16::ZERO
//     }
//     #[inline(always)]
//     fn _round(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _relu(self) -> Self {
//         self._max(Self::ZERO)
//     }
//     #[inline(always)]
//     fn _relu6(self) -> Self {
//         self._max(Self::ZERO)._min(Self::SIX)
//     }
//     #[inline(always)]
//     fn _leaky_relu(self,alpha:Self::Base) -> Self {
//         self._max(Self::ZERO)._add(alpha._mul(self._min(Self::ZERO)))
//     }

//     }
// impl NormalOutUnary for u32 {
//     type Base = Self;
//     #[inline(always)]
//     fn _square(self) -> Self {
//         self._mul(self)
//     }
//     #[inline(always)]
//     fn _neg(self) -> Self {
//         !self+1
//     }
//     #[inline(always)]
//     fn _abs(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _ceil(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _floor(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _sign(self) -> Self {
//         u32::ZERO
//     }
//     #[inline(always)]
//     fn _round(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _relu(self) -> Self {
//         self._max(Self::ZERO)
//     }
//     #[inline(always)]
//     fn _relu6(self) -> Self {
//         self._max(Self::ZERO)._min(Self::SIX)
//     }
//     #[inline(always)]
//     fn _leaky_relu(self,alpha:Self::Base) -> Self {
//         self._max(Self::ZERO)._add(alpha._mul(self._min(Self::ZERO)))
//     }

//     }
// impl NormalOutUnary for u64 {
//     type Base = Self;
//     #[inline(always)]
//     fn _square(self) -> Self {
//         self._mul(self)
//     }
//     #[inline(always)]
//     fn _neg(self) -> Self {
//         !self+1
//     }
//     #[inline(always)]
//     fn _abs(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _ceil(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _floor(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _sign(self) -> Self {
//         u64::ZERO
//     }
//     #[inline(always)]
//     fn _round(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _relu(self) -> Self {
//         self._max(Self::ZERO)
//     }
//     #[inline(always)]
//     fn _relu6(self) -> Self {
//         self._max(Self::ZERO)._min(Self::SIX)
//     }
//     #[inline(always)]
//     fn _leaky_relu(self,alpha:Self::Base) -> Self {
//         self._max(Self::ZERO)._add(alpha._mul(self._min(Self::ZERO)))
//     }

//     }
// impl NormalOutUnary for bf16 {
//     type Base = Self;
//     #[inline(always)]
//     fn _square(self) -> Self {
//         self._mul(self)
//     }
//     #[inline(always)]
//     fn _neg(self) -> Self {
//         -self
//     }
//     #[inline(always)]
//     fn _abs(self) -> Self {
//         self.abs()
//     }
//     #[inline(always)]
//     fn _ceil(self) -> Self {
//         self.ceil()
//     }
//     #[inline(always)]
//     fn _floor(self) -> Self {
//         self.floor()
//     }
//     #[inline(always)]
//     fn _sign(self) -> Self {
//         self.signum()
//     }
//     #[inline(always)]
//     fn _round(self) -> Self {
//         self.round()
//     }
//     #[inline(always)]
//     fn _relu(self) -> Self {
//         self._max(Self::ZERO)
//     }
//     #[inline(always)]
//     fn _relu6(self) -> Self {
//         self._max(Self::ZERO)._min(Self::SIX)
//     }
//     #[inline(always)]
//     fn _leaky_relu(self,alpha:Self::Base) -> Self {
//         self._max(Self::ZERO)._add(alpha._mul(self._min(Self::ZERO)))
//     }

//     }
// impl NormalOutUnary for isize {
//     type Base = Self;
//     #[inline(always)]
//     fn _square(self) -> Self {
//         self._mul(self)
//     }
//     #[inline(always)]
//     fn _neg(self) -> Self {
//         -self
//     }
//     #[inline(always)]
//     fn _abs(self) -> Self {
//         self.abs()
//     }
//     #[inline(always)]
//     fn _ceil(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _floor(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _sign(self) -> Self {
//         self.signum()
//     }
//     #[inline(always)]
//     fn _round(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _relu(self) -> Self {
//         self._max(Self::ZERO)
//     }
//     #[inline(always)]
//     fn _relu6(self) -> Self {
//         self._max(Self::ZERO)._min(Self::SIX)
//     }
//     #[inline(always)]
//     fn _leaky_relu(self,alpha:Self::Base) -> Self {
//         self._max(Self::ZERO)._add(alpha._mul(self._min(Self::ZERO)))
//     }

//     }
// impl NormalOutUnary for usize {
//     type Base = Self;
//     #[inline(always)]
//     fn _square(self) -> Self {
//         self._mul(self)
//     }
//     #[inline(always)]
//     fn _neg(self) -> Self {
//         !self+1
//     }
//     #[inline(always)]
//     fn _abs(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _ceil(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _floor(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _sign(self) -> Self {
//         usize::ZERO
//     }
//     #[inline(always)]
//     fn _round(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _relu(self) -> Self {
//         self._max(Self::ZERO)
//     }
//     #[inline(always)]
//     fn _relu6(self) -> Self {
//         self._max(Self::ZERO)._min(Self::SIX)
//     }
//     #[inline(always)]
//     fn _leaky_relu(self,alpha:Self::Base) -> Self {
//         self._max(Self::ZERO)._add(alpha._mul(self._min(Self::ZERO)))
//     }

//     }
// impl NormalOutUnary for Complex32 {
//     type Base = Self;
//     #[inline(always)]
//     fn _square(self) -> Self {
//         self._mul(self)
//     }
//     #[inline(always)]
//     fn _neg(self) -> Self {
//         -self
//     }
//     #[inline(always)]
//     fn _abs(self) -> Self {
//         panic!("abs method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _ceil(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _floor(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _sign(self) -> Self {
//         panic!("sign method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _round(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _relu(self) -> Self {
//         self._max(Self::ZERO)
//     }
//     #[inline(always)]
//     fn _relu6(self) -> Self {
//         self._max(Self::ZERO)._min(Self::SIX)
//     }
//     #[inline(always)]
//     fn _leaky_relu(self,alpha:Self::Base) -> Self {
//         self._max(Self::ZERO)._add(alpha._mul(self._min(Self::ZERO)))
//     }

//     }
// impl NormalOutUnary for Complex64 {
//     type Base = Self;
//     #[inline(always)]
//     fn _square(self) -> Self {
//         self._mul(self)
//     }
//     #[inline(always)]
//     fn _neg(self) -> Self {
//         -self
//     }
//     #[inline(always)]
//     fn _abs(self) -> Self {
//         panic!("abs method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _ceil(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _floor(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _sign(self) -> Self {
//         panic!("sign method is not supported for complex number")
//     }
//     #[inline(always)]
//     fn _round(self) -> Self {
//         self
//     }
//     #[inline(always)]
//     fn _relu(self) -> Self {
//         self._max(Self::ZERO)
//     }
//     #[inline(always)]
//     fn _relu6(self) -> Self {
//         self._max(Self::ZERO)._min(Self::SIX)
//     }
//     #[inline(always)]
//     fn _leaky_relu(self,alpha:Self::Base) -> Self {
//         self._max(Self::ZERO)._add(alpha._mul(self._min(Self::ZERO)))
//     }

//     }