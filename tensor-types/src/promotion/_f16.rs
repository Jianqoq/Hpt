use crate::cuda_types::scalar::Scalar;
use crate::type_promote::{
    FloatOutBinaryPromote, FloatOutUnaryPromote, NormalOutPromote, SimdCmpPromote,
};
use crate::vectors::vector_promote::*;
use crate::{
    impl_float_out_binary_promote, impl_float_out_unary_promote, impl_normal_out_promote,
    impl_simd_cmp_promote,
};
use half::{bf16, f16};
use num_complex::{Complex32, Complex64};
impl_float_out_binary_promote!(f16, bool, f16);
impl_float_out_binary_promote!(f16, i8, f16);
impl_float_out_binary_promote!(f16, i16, f16);
impl_float_out_binary_promote!(f16, i32, f32);
impl_float_out_binary_promote!(f16, i64, f64);
#[cfg(target_pointer_width = "64")]
impl_float_out_binary_promote!(f16, isize, f64);
#[cfg(target_pointer_width = "32")]
impl_float_out_binary_promote!(f16, isize, f32);
impl_float_out_binary_promote!(f16, u8, f16);
impl_float_out_binary_promote!(f16, u16, f16);
impl_float_out_binary_promote!(f16, u32, f32);
impl_float_out_binary_promote!(f16, u64, f64);
#[cfg(target_pointer_width = "64")]
impl_float_out_binary_promote!(f16, usize, f64);
#[cfg(target_pointer_width = "32")]
impl_float_out_binary_promote!(f16, usize, f32);
impl_float_out_binary_promote!(f16, f16, f16);
impl_float_out_binary_promote!(f16, bf16, f16);
impl_float_out_binary_promote!(f16, f32, f32);
impl_float_out_binary_promote!(f16, f64, f64);
impl_float_out_binary_promote!(f16, Complex32, Complex32);
impl_float_out_binary_promote!(f16, Complex64, Complex64);

impl_normal_out_promote!(f16, bool, f16);
impl_normal_out_promote!(f16, i8, f16);
impl_normal_out_promote!(f16, i16, f16);
impl_normal_out_promote!(f16, i32, f32);
impl_normal_out_promote!(f16, i64, f64);
#[cfg(target_pointer_width = "64")]
impl_normal_out_promote!(f16, isize, f64);
#[cfg(target_pointer_width = "32")]
impl_normal_out_promote!(f16, isize, f32);
impl_normal_out_promote!(f16, u8, f16);
impl_normal_out_promote!(f16, u16, f16);
impl_normal_out_promote!(f16, u32, f32);
impl_normal_out_promote!(f16, u64, f64);
#[cfg(target_pointer_width = "64")]
impl_normal_out_promote!(f16, usize, f64);
#[cfg(target_pointer_width = "32")]
impl_normal_out_promote!(f16, usize, f32);
impl_normal_out_promote!(f16, f16, f16);
impl_normal_out_promote!(f16, bf16, f16);
impl_normal_out_promote!(f16, f32, f32);
impl_normal_out_promote!(f16, f64, f64);
impl_normal_out_promote!(f16, Complex32, Complex32);
impl_normal_out_promote!(f16, Complex64, Complex64);

impl_simd_cmp_promote!(f16, bool, i8);
impl_simd_cmp_promote!(f16, i8, i8);
impl_simd_cmp_promote!(f16, i16, i16);
impl_simd_cmp_promote!(f16, i32, i32);
impl_simd_cmp_promote!(f16, i64, i64);
#[cfg(target_pointer_width = "64")]
impl_simd_cmp_promote!(f16, isize, i64);
#[cfg(target_pointer_width = "32")]
impl_simd_cmp_promote!(f16, isize, i32);
impl_simd_cmp_promote!(f16, u8, i8);
impl_simd_cmp_promote!(f16, u16, i16);
impl_simd_cmp_promote!(f16, u32, i32);
impl_simd_cmp_promote!(f16, u64, i64);
#[cfg(target_pointer_width = "64")]
impl_simd_cmp_promote!(f16, usize, i64);
#[cfg(target_pointer_width = "32")]
impl_simd_cmp_promote!(f16, usize, i32);
impl_simd_cmp_promote!(f16, f16, i16);
impl_simd_cmp_promote!(f16, bf16, i16);
impl_simd_cmp_promote!(f16, f32, f32);
impl_simd_cmp_promote!(f16, f64, f64);

impl_float_out_unary_promote!(f16, f16);
