use wide::*;

pub trait VecTrait<T> {
    fn fma(&mut self, a: &Self, b: &Self);
    fn copy_from_slice(&mut self, slice: &[T]);
    fn as_ptr(&self) -> *const T;
}
pub trait Init<T> {
    fn splat(val: T) -> Self;
}
pub trait VecSize {
    const SIZE: usize;
}

macro_rules! impl_vectors {
    ($T: ident, $base: ident, $size: expr) => {
        impl Init<$base> for $T {
            fn splat(val: $base) -> $T {
                $T::splat(val)
            }
        }
        impl VecTrait<$base> for $T {
            #[inline(always)]
            fn fma(&mut self, a: &Self, b: &Self) {
                *self += *a * *b;
            }
            #[inline(always)]
            fn copy_from_slice(&mut self, slice: &[$base]) {
                self.as_array_mut().copy_from_slice(slice);
            }
            #[inline(always)]
            fn as_ptr(&self) -> *const $base {
                self.as_array_ref().as_ptr()
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
impl_vectors!(f32x8, f32, 8);
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




