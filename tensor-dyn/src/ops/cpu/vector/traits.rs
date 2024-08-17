use wide::i32x4;

pub trait X4<T> {
    fn fma(self, a: Self, b: Self) -> Self;
    fn copy_from_slice(&mut self, slice: &[T]);
}

pub trait X8<T> {
    fn fma(self, a: Self, b: Self) -> Self;
}

pub trait InitX4<T> {
    fn splat(val: T) -> Self;
}

impl InitX4<i32> for i32x4 {
    fn splat(val: i32) -> i32x4 {
        i32x4::splat(val)
    }
}

impl X4<i32> for i32x4 {
    fn fma(self, a: Self, b: Self) -> Self {
        self + a * b
    }
    fn copy_from_slice(&mut self, slice: &[i32]) {
        self.as_array_mut().copy_from_slice(slice);
    }
}
