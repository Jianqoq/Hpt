use crate::dtype::TypeCommon;

/// common trait for all vector types
pub trait VecTrait<T: Copy> {
    /// the size of the vector
    const SIZE: usize;
    /// the base type of the vector
    type Base: TypeCommon;
    /// peform self * a + b, fused multiply add
    fn mul_add(self, a: Self, b: Self) -> Self;
    /// copy data from slice to self
    fn copy_from_slice(&mut self, slice: &[T]);
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
        unsafe { *self.as_ptr().add(idx) }
    }
    /// get the sum of all elements in vector
    fn sum(&self) -> T;
    /// write value to vector, this is unaligned write
    #[inline(always)]
    fn write_unaligned(&mut self, vec: T::Vec) where T: TypeCommon {
        let ptr = self.as_mut_ptr() as *mut T::Vec;
        unsafe { ptr.write_unaligned(vec) }
    }
    /// read a value from vector
    #[inline(always)]
    fn read_unaligned(&self) -> T::Vec where T: TypeCommon {
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
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const T) -> Self where Self: Sized {
        let ptr = ptr as *const Self;
        unsafe { ptr.read_unaligned() }
    }
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

/// A trait for vector math operations
pub trait SimdMath<T>: Copy {
    /// compute the sine of the vector
    fn sin(self) -> Self;
    /// compute the cosine of the vector
    fn cos(self) -> Self;
    /// compute the tangent of the vector
    fn tan(self) -> Self;
    /// coupute asin
    fn asin(self) -> Self;
    /// coupute acos
    fn acos(self) -> Self;
    /// coupute atan
    fn atan(self) -> Self;
    /// compute sinh
    fn sinh(self) -> Self;
    /// compute cosh
    fn cosh(self) -> Self;
    /// compute tanh
    fn tanh(self) -> Self;
    /// compute asinh
    fn asinh(self) -> Self;
    /// compute acosh
    fn acosh(self) -> Self;
    /// compute atanh
    fn atanh(self) -> Self;
    /// compute the square of the vector
    fn square(self) -> Self;
    /// compute the absolute value of the vector
    fn abs(self) -> Self;
    /// compute the floor of the vector
    fn floor(self) -> Self;
    /// compute the ceil of the vector
    fn ceil(self) -> Self;
    /// compute the negate of the vector
    fn neg(self) -> Self;
    /// compute the round of the vector
    fn round(self) -> Self;
    /// compute the sign of the vector
    fn sign(self) -> Self;
    /// compute the square root of the vector
    fn sqrt(self) -> Self;
    /// leaky relu
    fn leaky_relu(self, alpha: T) -> Self;
    /// relu
    fn relu(self) -> Self;
    /// relu6
    fn relu6(self) -> Self;
    /// pow
    fn pow(self, exp: Self) -> Self;
    /// exp
    fn exp(self) -> Self;
    /// exp2
    fn exp2(self) -> Self;
    /// exp10
    fn exp10(self) -> Self;
    /// expm1
    fn expm1(self) -> Self;
    /// log10
    fn log10(self) -> Self;
    /// log2
    fn log2(self) -> Self;
    /// log1p
    fn log1p(self) -> Self;
    /// hypot
    fn hypot(self, other: Self) -> Self;
    /// trunc
    fn trunc(self) -> Self;
    /// erf
    fn erf(self) -> Self;
    /// cbrt
    fn cbrt(self) -> Self;
    /// ln
    fn ln(self) -> Self;
    /// log
    fn log(self) -> Self;
}
