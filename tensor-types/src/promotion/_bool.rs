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
impl_float_out_binary_promote!(bool, bool, f16);
impl_float_out_binary_promote!(bool, i8, f16);
impl_float_out_binary_promote!(bool, i16, f16);
impl_float_out_binary_promote!(bool, i32, f32);
impl_float_out_binary_promote!(bool, i64, f64);
#[cfg(target_pointer_width = "64")]
impl_float_out_binary_promote!(bool, isize, f64);
#[cfg(target_pointer_width = "32")]
impl_float_out_binary_promote!(bool, isize, f32);
impl_float_out_binary_promote!(bool, u8, f16);
impl_float_out_binary_promote!(bool, u16, f16);
impl_float_out_binary_promote!(bool, u32, f32);
impl_float_out_binary_promote!(bool, u64, f64);
#[cfg(target_pointer_width = "64")]
impl_float_out_binary_promote!(bool, usize, f64);
#[cfg(target_pointer_width = "32")]
impl_float_out_binary_promote!(bool, usize, f32);
impl_float_out_binary_promote!(bool, f16, f16);
impl_float_out_binary_promote!(bool, bf16, bf16);
impl_float_out_binary_promote!(bool, f32, f32);
impl_float_out_binary_promote!(bool, f64, f64);
impl_float_out_binary_promote!(bool, Complex32, Complex32);
impl_float_out_binary_promote!(bool, Complex64, Complex64);

impl_normal_out_promote!(bool, bool, bool);
impl_normal_out_promote!(bool, i8, i8);
impl_normal_out_promote!(bool, i16, i16);
impl_normal_out_promote!(bool, i32, i32);
impl_normal_out_promote!(bool, i64, i64);
impl_normal_out_promote!(bool, isize, isize);
impl_normal_out_promote!(bool, u8, u8);
impl_normal_out_promote!(bool, u16, u16);
impl_normal_out_promote!(bool, u32, u32);
impl_normal_out_promote!(bool, u64, u64);
impl_normal_out_promote!(bool, usize, usize);
impl_normal_out_promote!(bool, f16, f16);
impl_normal_out_promote!(bool, bf16, bf16);
impl_normal_out_promote!(bool, f32, f32);
impl_normal_out_promote!(bool, f64, f64);
impl_normal_out_promote!(bool, Complex32, Complex32);
impl_normal_out_promote!(bool, Complex64, Complex64);

impl_simd_cmp_promote!(bool, bool, i8);
impl_simd_cmp_promote!(bool, i8, i8);
impl_simd_cmp_promote!(bool, i16, i16);
impl_simd_cmp_promote!(bool, i32, i32);
impl_simd_cmp_promote!(bool, i64, i64);
impl_simd_cmp_promote!(bool, isize, isize);
impl_simd_cmp_promote!(bool, u8, i8);
impl_simd_cmp_promote!(bool, u16, i16);
impl_simd_cmp_promote!(bool, u32, i32);
impl_simd_cmp_promote!(bool, u64, i64);
impl_simd_cmp_promote!(bool, usize, isize);
impl_simd_cmp_promote!(bool, f16, i16);
impl_simd_cmp_promote!(bool, bf16, i16);
impl_simd_cmp_promote!(bool, f32, i32);
impl_simd_cmp_promote!(bool, f64, i64);

impl_float_out_unary_promote!(bool, f16);
