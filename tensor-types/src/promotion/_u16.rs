use crate::type_promote::{FloatOutBinaryPromote, NormalOutPromote, SimdCmpPromote, FloatOutUnaryPromote};
use crate::vectors::vector_promote::*;
use crate::{impl_float_out_binary_promote, impl_normal_out_promote, impl_simd_cmp_promote, impl_float_out_unary_promote};
use half::{bf16, f16};
use num_complex::{Complex32, Complex64};

impl_float_out_binary_promote!(u16, bool, f16);
impl_float_out_binary_promote!(u16, i8, f16);
impl_float_out_binary_promote!(u16, i16, f16);
impl_float_out_binary_promote!(u16, i32, f32);
impl_float_out_binary_promote!(u16, i64, f64);
#[cfg(target_pointer_width = "64")]
impl_float_out_binary_promote!(u16, isize, f64);
#[cfg(target_pointer_width = "32")]
impl_float_out_binary_promote!(u16, isize, f32);
impl_float_out_binary_promote!(u16, u8, f16);
impl_float_out_binary_promote!(u16, u16, f16);
impl_float_out_binary_promote!(u16, u32, f32);
impl_float_out_binary_promote!(u16, u64, f64);
#[cfg(target_pointer_width = "64")]
impl_float_out_binary_promote!(u16, usize, f64);
#[cfg(target_pointer_width = "32")]
impl_float_out_binary_promote!(u16, usize, f32);
impl_float_out_binary_promote!(u16, f16, f16);
impl_float_out_binary_promote!(u16, bf16, f16);
impl_float_out_binary_promote!(u16, f32, f32);
impl_float_out_binary_promote!(u16, f64, f64);
impl_float_out_binary_promote!(u16, Complex32, Complex32);
impl_float_out_binary_promote!(u16, Complex64, Complex64);

impl_normal_out_promote!(u16, bool, u16);
impl_normal_out_promote!(u16, i8, i16);
impl_normal_out_promote!(u16, i16, i16);
impl_normal_out_promote!(u16, i32, i32);
impl_normal_out_promote!(u16, i64, i64);
impl_normal_out_promote!(u16, isize, isize);
impl_normal_out_promote!(u16, u8, u16);
impl_normal_out_promote!(u16, u16, u16);
impl_normal_out_promote!(u16, u32, u32);
impl_normal_out_promote!(u16, u64, u64);
impl_normal_out_promote!(u16, usize, usize);
impl_normal_out_promote!(u16, f16, f16);
impl_normal_out_promote!(u16, bf16, bf16);
impl_normal_out_promote!(u16, f32, f32);
impl_normal_out_promote!(u16, f64, f64);
impl_normal_out_promote!(u16, Complex32, Complex32);
impl_normal_out_promote!(u16, Complex64, Complex64);

impl_simd_cmp_promote!(u16, bool, i16);
impl_simd_cmp_promote!(u16, i8, i16);
impl_simd_cmp_promote!(u16, i16, i16);
impl_simd_cmp_promote!(u16, i32, i32);
impl_simd_cmp_promote!(u16, i64, i64);
impl_simd_cmp_promote!(u16, isize, isize);
impl_simd_cmp_promote!(u16, u8, i16);
impl_simd_cmp_promote!(u16, u16, i16);
impl_simd_cmp_promote!(u16, u32, i32);
impl_simd_cmp_promote!(u16, u64, i64);
impl_simd_cmp_promote!(u16, usize, isize);
impl_simd_cmp_promote!(u16, f16, i16);
impl_simd_cmp_promote!(u16, bf16, i16);
impl_simd_cmp_promote!(u16, f32, i32);
impl_simd_cmp_promote!(u16, f64, i64);

impl_float_out_unary_promote!(u16, f16);