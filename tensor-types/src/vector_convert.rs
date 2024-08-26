use wide::f32x8;


pub trait StaticVecCvt<T> {
    type Output;
}

impl StaticVecCvt<[i32; 8]> for f32x8 {
    type Output = [i32; 8];
}

impl StaticVecCvt<f32x8> for f32x8 {
    type Output = f32x8;
}