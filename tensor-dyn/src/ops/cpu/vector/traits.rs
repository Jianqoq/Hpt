use wide::*;

pub trait VecTrait<T> {
    fn fma(&mut self, a: Self, b: Self);
    fn _mul_add(self, a: Self, b: Self) -> Self;
    fn copy_from_slice(&mut self, slice: &[T]);
    fn as_ptr(&self) -> *const T;
    fn as_mut_ptr(&mut self) -> *mut T;
}
pub trait Init<T> {
    fn splat(val: T) -> Self;
}
pub trait VecSize {
    const SIZE: usize;
}

macro_rules! impl_vectors {
    ($T:ident, $base:ident, $size:expr) => {
        impl Init<$base> for $T {
            fn splat(val: $base) -> $T {
                $T::splat(val)
            }
        }
        impl VecTrait<$base> for $T {
            #[inline(always)]
            fn fma(&mut self, a: Self, b: Self) {
                *self += a * b;
            }
            #[inline(always)]
            fn copy_from_slice(&mut self, slice: &[$base]) {
                self.as_array_mut().copy_from_slice(slice);
            }
            #[inline(always)]
            fn as_ptr(&self) -> *const $base {
                self.as_array_ref().as_ptr()
            }
            #[inline(always)]
            fn _mul_add(self, _: Self, _: Self) -> Self {
                todo!()
            }
            #[inline(always)]
            fn as_mut_ptr(&mut self) -> *mut $base {
                self.as_array_mut().as_mut_ptr()
            }
        }
        impl VecSize for $T {
            const SIZE: usize = $size;
        }
    };
}

impl_vectors!(i32x4, i32, 4);
impl_vectors!(i32x8, i32, 8);
impl_vectors!(f32x4, f32, 4);
impl Init<f32> for f32x8 {
    fn splat(val: f32) -> f32x8 {
        unsafe { std::mem::transmute(std::arch::x86_64::_mm256_set1_ps(val)) }
    }
}
impl VecTrait<f32> for f32x8 {
    #[inline(always)]
    fn fma(&mut self, a: Self, b: Self) {
        *self += a * b;
    }
    #[inline(always)]
    fn copy_from_slice(&mut self, slice: &[f32]) {
        self.as_array_mut().copy_from_slice(slice);
    }
    #[inline(always)]
    fn as_ptr(&self) -> *const f32 {
        self.as_array_ref().as_ptr()
    }
    #[inline(always)]
    fn _mul_add(self, a: Self, b: Self) -> Self {
        self.mul_add(a, b)
    }
    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut f32 {
        self.as_array_mut().as_mut_ptr()
    }
}
impl VecSize for f32x8 {
    const SIZE: usize = 8;
}
impl_vectors!(f64x2, f64, 2);
impl_vectors!(f64x4, f64, 4);
impl_vectors!(i64x2, i64, 2);
impl_vectors!(u64x2, u64, 2);
impl_vectors!(u64x4, u64, 4);
impl_vectors!(u32x4, u32, 4);
impl_vectors!(u32x8, u32, 8);
impl_vectors!(i16x8, i16, 8);
impl_vectors!(i16x16, i16, 16);
impl_vectors!(u16x8, u16, 8);
impl_vectors!(u16x16, u16, 16);
