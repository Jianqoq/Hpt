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
impl_float_out_binary_promote!(i64, bool, f64);
impl_float_out_binary_promote!(i64, i8, f64);
impl_float_out_binary_promote!(i64, i16, f64);
impl_float_out_binary_promote!(i64, i32, f64);
impl_float_out_binary_promote!(i64, i64, f64);
impl_float_out_binary_promote!(i64, isize, f64);
impl_float_out_binary_promote!(i64, u8, f64);
impl_float_out_binary_promote!(i64, u16, f64);
impl_float_out_binary_promote!(i64, u32, f64);
impl_float_out_binary_promote!(i64, u64, f64);
impl_float_out_binary_promote!(i64, usize, f64);
impl_float_out_binary_promote!(i64, f16, f64);
impl_float_out_binary_promote!(i64, bf16, f64);
impl_float_out_binary_promote!(i64, f32, f64);
impl_float_out_binary_promote!(i64, f64, f64);
impl_float_out_binary_promote!(i64, Complex32, Complex64);
impl_float_out_binary_promote!(i64, Complex64, Complex64);

impl_normal_out_promote!(i64, bool, i64);
impl_normal_out_promote!(i64, i8, i64);
impl_normal_out_promote!(i64, i16, i64);
impl_normal_out_promote!(i64, i32, i64);
impl_normal_out_promote!(i64, i64, i64);
#[cfg(target_pointer_width = "64")]
impl_normal_out_promote!(i64, isize, i64);
#[cfg(target_pointer_width = "32")]
impl_normal_out_promote!(i64, isize, i32);
impl_normal_out_promote!(i64, u8, i64);
impl_normal_out_promote!(i64, u16, i64);
impl_normal_out_promote!(i64, u32, i64);
impl_normal_out_promote!(i64, u64, i64);
#[cfg(target_pointer_width = "64")]
impl_normal_out_promote!(i64, usize, i64);
#[cfg(target_pointer_width = "32")]
impl_normal_out_promote!(i64, usize, i32);
impl_normal_out_promote!(i64, f16, f16);
impl_normal_out_promote!(i64, bf16, bf16);
impl_normal_out_promote!(i64, f32, f64);
impl_normal_out_promote!(i64, f64, f64);
impl_normal_out_promote!(i64, Complex32, Complex64);
impl_normal_out_promote!(i64, Complex64, Complex64);

impl_simd_cmp_promote!(i64, bool, i64);
impl_simd_cmp_promote!(i64, i8, i64);
impl_simd_cmp_promote!(i64, i16, i64);
impl_simd_cmp_promote!(i64, i32, i64);
impl_simd_cmp_promote!(i64, i64, i64);
#[cfg(target_pointer_width = "64")]
impl_simd_cmp_promote!(i64, isize, i64);
#[cfg(target_pointer_width = "32")]
impl_simd_cmp_promote!(i64, isize, i32);
impl_simd_cmp_promote!(i64, u8, i64);
impl_simd_cmp_promote!(i64, u16, i64);
impl_simd_cmp_promote!(i64, u32, i64);
impl_simd_cmp_promote!(i64, u64, i64);
#[cfg(target_pointer_width = "64")]
impl_simd_cmp_promote!(i64, usize, i64);
#[cfg(target_pointer_width = "32")]
impl_simd_cmp_promote!(i64, usize, i32);
impl_simd_cmp_promote!(i64, f16, i16);
impl_simd_cmp_promote!(i64, bf16, i16);
impl_simd_cmp_promote!(i64, f32, i64);
impl_simd_cmp_promote!(i64, f64, i64);

impl_float_out_unary_promote!(i64, f64);
