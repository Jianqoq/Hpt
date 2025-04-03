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
impl_float_out_binary_promote!(i32, bool, f32, f32);
impl_float_out_binary_promote!(i32, i8, f32, f32);
impl_float_out_binary_promote!(i32, i16, f32, f32);
impl_float_out_binary_promote!(i32, i32, f32, f32);
impl_float_out_binary_promote!(i32, i64, f64, f64);
#[cfg(target_pointer_width = "64")]
impl_float_out_binary_promote!(i32, isize, f64, f64);
#[cfg(target_pointer_width = "32")]
impl_float_out_binary_promote!(i32, isize, f32, f32);
impl_float_out_binary_promote!(i32, u8, f32, f32);
impl_float_out_binary_promote!(i32, u16, f32, f32);
impl_float_out_binary_promote!(i32, u32, f32, f32);
impl_float_out_binary_promote!(i32, u64, f64, f64);
#[cfg(target_pointer_width = "64")]
impl_float_out_binary_promote!(i32, usize, f64, f64);
#[cfg(target_pointer_width = "32")]
impl_float_out_binary_promote!(i32, usize, f32, f32);
impl_float_out_binary_promote!(i32, f16, f16, f32);
impl_float_out_binary_promote!(i32, bf16, bf16, f32);
impl_float_out_binary_promote!(i32, f32, f32, f32);
impl_float_out_binary_promote!(i32, f64, f64, f64);
impl_float_out_binary_promote!(i32, Complex32, Complex32, Complex32);
impl_float_out_binary_promote!(i32, Complex64, Complex64, Complex64);

impl_normal_out_promote!(i32, bool, i32, i32);
impl_normal_out_promote!(i32, i8, i32, i32);
impl_normal_out_promote!(i32, i16, i32, i32);
impl_normal_out_promote!(i32, i32, i32, i32);
impl_normal_out_promote!(i32, i64, i64, i64);
#[cfg(target_pointer_width = "64")]
impl_normal_out_promote!(i32, isize, i64, i64);
#[cfg(target_pointer_width = "32")]
impl_normal_out_promote!(i32, isize, i32, i32);
impl_normal_out_promote!(i32, u8, i32, i32);
impl_normal_out_promote!(i32, u16, i32, i32);
impl_normal_out_promote!(i32, u32, i32, i32);
impl_normal_out_promote!(i32, u64, i64, i64);
#[cfg(target_pointer_width = "64")]
impl_normal_out_promote!(i32, usize, i64, i64);
#[cfg(target_pointer_width = "32")]
impl_normal_out_promote!(i32, usize, i32, i32);
impl_normal_out_promote!(i32, f16, f16, f32);
impl_normal_out_promote!(i32, bf16, bf16, f32);
impl_normal_out_promote!(i32, f32, f32, f32);
impl_normal_out_promote!(i32, f64, f64, f64);
impl_normal_out_promote!(i32, Complex32, Complex32, Complex32);
impl_normal_out_promote!(i32, Complex64, Complex64, Complex64);

impl_simd_cmp_promote!(i32, bool, i32);
impl_simd_cmp_promote!(i32, i8, i32);
impl_simd_cmp_promote!(i32, i16, i32);
impl_simd_cmp_promote!(i32, i32, i32);
impl_simd_cmp_promote!(i32, i64, i64);
#[cfg(target_pointer_width = "64")]
impl_simd_cmp_promote!(i32, isize, i64);
#[cfg(target_pointer_width = "32")]
impl_simd_cmp_promote!(i32, isize, i32);
impl_simd_cmp_promote!(i32, u8, i32);
impl_simd_cmp_promote!(i32, u16, i32);
impl_simd_cmp_promote!(i32, u32, i32);
impl_simd_cmp_promote!(i32, u64, i64);
#[cfg(target_pointer_width = "64")]
impl_simd_cmp_promote!(i32, usize, i64);
#[cfg(target_pointer_width = "32")]
impl_simd_cmp_promote!(i32, usize, i32);
impl_simd_cmp_promote!(i32, f16, i16);
impl_simd_cmp_promote!(i32, bf16, i16);
impl_simd_cmp_promote!(i32, f32, i32);
impl_simd_cmp_promote!(i32, f64, i64);

impl_float_out_unary_promote!(i32, f32, f32);
