#[cfg(feature = "cuda")]
use crate::cuda_types::scalar::Scalar;
use crate::type_promote::SimdCmpPromote;
use crate::type_promote::{FloatOutBinaryPromote, FloatOutUnaryPromote, NormalOutPromote};
use crate::vectors::vector_promote::*;
use crate::{
    impl_float_out_binary_promote, impl_float_out_unary_promote, impl_normal_out_promote,
    impl_simd_cmp_promote,
};
use half::{bf16, f16};
use num_complex::{Complex32, Complex64};
impl_float_out_binary_promote!(i8, bool, f16, f32);
impl_float_out_binary_promote!(i8, i8, f16, f32);
impl_float_out_binary_promote!(i8, i16, f16, f32);
impl_float_out_binary_promote!(i8, i32, f32, f32);
impl_float_out_binary_promote!(i8, i64, f64, f64);
#[cfg(target_pointer_width = "64")]
impl_float_out_binary_promote!(i8, isize, f64, f64);
#[cfg(target_pointer_width = "32")]
impl_float_out_binary_promote!(i8, isize, f32, f32);
impl_float_out_binary_promote!(i8, u8, f16, f32);
impl_float_out_binary_promote!(i8, u16, f16, f32);
impl_float_out_binary_promote!(i8, u32, f32, f32);
impl_float_out_binary_promote!(i8, u64, f64, f64);
#[cfg(target_pointer_width = "64")]
impl_float_out_binary_promote!(i8, usize, f64, f64);
#[cfg(target_pointer_width = "32")]
impl_float_out_binary_promote!(i8, usize, f32, f32);
impl_float_out_binary_promote!(i8, f16, f16, f32);
impl_float_out_binary_promote!(i8, bf16, bf16, f32);
impl_float_out_binary_promote!(i8, f32, f32, f32);
impl_float_out_binary_promote!(i8, f64, f64, f64);
impl_float_out_binary_promote!(i8, Complex32, Complex32, Complex32);
impl_float_out_binary_promote!(i8, Complex64, Complex64, Complex64);

impl_normal_out_promote!(i8, bool, i8, i8);
impl_normal_out_promote!(i8, i8, i8, i8);
impl_normal_out_promote!(i8, i16, i16, i16);
impl_normal_out_promote!(i8, i32, i32, i32);
impl_normal_out_promote!(i8, i64, i64, i64);
impl_normal_out_promote!(i8, isize, isize, isize);
impl_normal_out_promote!(i8, u8, i8, i8);
impl_normal_out_promote!(i8, u16, i16, i16);
impl_normal_out_promote!(i8, u32, i32, i32);
impl_normal_out_promote!(i8, u64, i64, i64);
impl_normal_out_promote!(i8, usize, isize, isize);
impl_normal_out_promote!(i8, f16, f16, f32);
impl_normal_out_promote!(i8, bf16, bf16, f32);
impl_normal_out_promote!(i8, f32, f32, f32);
impl_normal_out_promote!(i8, f64, f64, f64);
impl_normal_out_promote!(i8, Complex32, Complex32, Complex32);
impl_normal_out_promote!(i8, Complex64, Complex64, Complex64);

impl_simd_cmp_promote!(i8, bool, i8);
impl_simd_cmp_promote!(i8, i8, i8);
impl_simd_cmp_promote!(i8, i16, i16);
impl_simd_cmp_promote!(i8, i32, i32);
impl_simd_cmp_promote!(i8, i64, i64);
impl_simd_cmp_promote!(i8, isize, isize);
impl_simd_cmp_promote!(i8, u8, i8);
impl_simd_cmp_promote!(i8, u16, i16);
impl_simd_cmp_promote!(i8, u32, i32);
impl_simd_cmp_promote!(i8, u64, i64);
impl_simd_cmp_promote!(i8, usize, isize);
impl_simd_cmp_promote!(i8, f16, i16);
impl_simd_cmp_promote!(i8, bf16, i16);
impl_simd_cmp_promote!(i8, f32, i32);
impl_simd_cmp_promote!(i8, f64, i64);

impl_float_out_unary_promote!(i8, f16, f32);
