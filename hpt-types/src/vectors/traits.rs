use crate::dtype::TypeCommon;

/// common trait for all vector types
pub trait VecTrait<T: Copy> {
    /// the size of the vector
    const SIZE: usize;
    /// the base type of the vector
    type Base: TypeCommon;
    /// peform self * a + b, fused multiply add
    fn mul_add(self, a: Self, b: Self) -> Self;
    /// convert self to a const pointer
    fn as_ptr(&self) -> *const T {
        self as *const _ as *const T
    }
    /// convert self to a mutable pointer
    fn as_mut_ptr(&mut self) -> *mut T {
        self as *mut _ as *mut T
    }
    /// convert self to a mutable pointer without check
    fn as_mut_ptr_uncheck(&self) -> *mut T {
        unsafe { std::mem::transmute(self.as_ptr()) }
    }
    /// extract a value from vector
    fn extract(&self, idx: usize) -> T {
        assert!(idx < Self::SIZE);
        unsafe { *self.as_ptr().add(idx) }
    }
    /// get the sum of all elements in vector
    fn sum(&self) -> T;
    /// write value to vector, this is unaligned write
    #[inline(always)]
    fn write_unaligned(&mut self, vec: T::Vec)
    where
        T: TypeCommon,
    {
        let ptr = self.as_mut_ptr() as *mut T::Vec;
        unsafe { ptr.write_unaligned(vec) }
    }
    /// read a value from vector
    #[inline(always)]
    fn read_unaligned(&self) -> T::Vec
    where
        T: TypeCommon,
    {
        let ptr = self.as_ptr() as *const T::Vec;
        unsafe { ptr.read_unaligned() }
    }
    /// create a vector with all elements set to the val
    fn splat(val: T) -> Self;
    /// load data to vector from pointer
    ///
    /// # Safety
    ///
    /// This function is unsafe because it can cause undefined behavior if the pointer is invalid or the data len is less than the vector size
    unsafe fn from_ptr(ptr: *const T) -> Self;
    /// mul add lane
    #[cfg(target_feature = "neon")]
    fn mul_add_lane<const LANE: i32>(self, a: Self, b: Self) -> Self;
}

/// a trait to select value from two vectors
pub trait SimdSelect<T> {
    /// select value based on mask
    fn select(&self, true_val: T, false_val: T) -> T;
}

/// A trait for vector comparison
pub trait SimdCompare {
    /// the mask type for the vector
    type SimdMask;
    /// compare two vectors to check if is equal and return a mask
    fn simd_eq(self, other: Self) -> Self::SimdMask;
    /// compare two vectors to check if is not equal and return a mask
    fn simd_ne(self, other: Self) -> Self::SimdMask;
    /// compare two vectors to check if is less than and return a mask
    fn simd_lt(self, other: Self) -> Self::SimdMask;
    /// compare two vectors to check if is less than or equal and return a mask
    fn simd_le(self, other: Self) -> Self::SimdMask;
    /// compare two vectors to check if is greater than and return a mask
    fn simd_gt(self, other: Self) -> Self::SimdMask;
    /// compare two vectors to check if is greater than or equal and return a mask
    fn simd_ge(self, other: Self) -> Self::SimdMask;
}

pub(crate) trait SimdMath<T>: Copy {
    /// compute the sine of the vector
    fn sin(self) -> Self {
        unreachable!()
    }
    /// compute the cosine of the vector
    fn cos(self) -> Self {
        unreachable!()
    }
    /// compute the sine and cosine of the vector
    fn sincos(self) -> (Self, Self) {
        unreachable!()
    }
    /// compute the tangent of the vector
    fn tan(self) -> Self {
        unreachable!()
    }
    /// coupute asin
    fn asin(self) -> Self {
        unreachable!()
    }
    /// coupute acos
    fn acos(self) -> Self {
        unreachable!()
    }
    /// coupute atan
    fn atan(self) -> Self {
        unreachable!()
    }
    /// coupute atan2
    fn atan2(self, _: Self) -> Self {
        unreachable!()
    }
    /// compute sinh
    fn sinh(self) -> Self {
        unreachable!()
    }
    /// compute cosh
    fn cosh(self) -> Self {
        unreachable!()
    }
    /// compute tanh
    fn tanh(self) -> Self {
        unreachable!()
    }
    /// compute asinh
    fn asinh(self) -> Self {
        unreachable!()
    }
    /// compute acosh
    fn acosh(self) -> Self {
        unreachable!()
    }
    /// compute atanh
    fn atanh(self) -> Self {
        unreachable!()
    }
    /// compute the absolute value of the vector
    fn abs(self) -> Self {
        unreachable!()
    }
    /// compute the floor of the vector
    fn floor(self) -> Self {
        unreachable!()
    }
    /// compute the ceil of the vector
    fn ceil(self) -> Self {
        unreachable!()
    }
    /// compute the negate of the vector
    fn neg(self) -> Self {
        unreachable!()
    }
    /// compute the round of the vector
    fn round(self) -> Self {
        unreachable!()
    }
    /// compute the signum of the vector
    fn signum(self) -> Self {
        unreachable!()
    }
    /// compute the copysign of the vector
    fn copysign(self, _: Self) -> Self {
        unreachable!()
    }
    /// compute the square root of the vector
    fn sqrt(self) -> Self {
        unreachable!()
    }
    /// leaky relu
    fn leaky_relu(self, _: Self) -> Self {
        unreachable!()
    }
    /// relu
    fn relu(self) -> Self {
        unreachable!()
    }
    /// relu6
    fn relu6(self) -> Self {
        unreachable!()
    }
    /// pow
    fn pow(self, _: Self) -> Self {
        unreachable!()
    }
    /// exp
    fn exp(self) -> Self {
        unreachable!()
    }
    /// exp2
    fn exp2(self) -> Self {
        unreachable!()
    }
    /// exp10
    fn exp10(self) -> Self {
        unreachable!()
    }
    /// expm1
    fn expm1(self) -> Self {
        unreachable!()
    }
    /// log10
    fn log10(self) -> Self {
        unreachable!()
    }
    /// log2
    fn log2(self) -> Self {
        unreachable!()
    }
    /// log1p
    fn log1p(self) -> Self {
        unreachable!()
    }
    /// hypot
    fn hypot(self, _: Self) -> Self {
        unreachable!()
    }
    /// trunc
    fn trunc(self) -> Self {
        unreachable!()
    }
    /// erf
    fn erf(self) -> Self {
        unreachable!()
    }
    /// cbrt
    fn cbrt(self) -> Self {
        unreachable!()
    }
    /// ln
    fn ln(self) -> Self {
        unreachable!()
    }
    /// min
    fn min(self, _: Self) -> Self {
        unreachable!()
    }
    /// max
    fn max(self, _: Self) -> Self {
        unreachable!()
    }
    /// reciprocal
    fn recip(self) -> Self {
        unreachable!()
    }
    /// sigmoid
    fn sigmoid(self) -> Self {
        unreachable!()
    }
    /// gelu
    fn gelu(self) -> Self {
        unreachable!()
    }
    /// softplus
    fn softplus(self) -> Self {
        unreachable!()
    }
    /// softsign
    fn softsign(self) -> Self {
        unreachable!()
    }
    /// mish
    fn mish(self) -> Self {
        unreachable!()
    }
    /// celu
    fn celu(self, _: Self) -> Self {
        unreachable!()
    }
    /// selu
    fn selu(self, _: Self, _: Self) -> Self {
        unreachable!()
    }
    /// elu
    fn elu(self, _: Self) -> Self {
        unreachable!()
    }
    /// hard sigmoid
    fn hard_sigmoid(self) -> Self {
        unreachable!()
    }
    /// hard swish
    fn hard_swish(self) -> Self {
        unreachable!()
    }
}
