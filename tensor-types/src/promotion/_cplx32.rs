#[cfg(feature = "cuda")]
use crate::cuda_types::scalar::Scalar;
use crate::type_promote::{FloatOutBinaryPromote, FloatOutUnaryPromote, NormalOutPromote};
use crate::vectors::vector_promote::*;
use crate::{impl_float_out_binary_promote, impl_float_out_unary_promote, impl_normal_out_promote};
use half::{bf16, f16};
use num_complex::{Complex32, Complex64};
impl_float_out_binary_promote!(Complex32, bool, Complex32);
impl_float_out_binary_promote!(Complex32, i8, Complex32);
impl_float_out_binary_promote!(Complex32, i16, Complex32);
impl_float_out_binary_promote!(Complex32, i32, Complex32);
impl_float_out_binary_promote!(Complex32, i64, Complex64);
#[cfg(target_pointer_width = "64")]
impl_float_out_binary_promote!(Complex32, isize, Complex64);
#[cfg(target_pointer_width = "32")]
impl_float_out_binary_promote!(Complex32, isize, Complex32);
impl_float_out_binary_promote!(Complex32, u8, Complex32);
impl_float_out_binary_promote!(Complex32, u16, Complex32);
impl_float_out_binary_promote!(Complex32, u32, Complex32);
impl_float_out_binary_promote!(Complex32, u64, Complex64);
#[cfg(target_pointer_width = "64")]
impl_float_out_binary_promote!(Complex32, usize, Complex64);
#[cfg(target_pointer_width = "32")]
impl_float_out_binary_promote!(Complex32, usize, Complex32);
impl_float_out_binary_promote!(Complex32, f16, Complex32);
impl_float_out_binary_promote!(Complex32, bf16, Complex32);
impl_float_out_binary_promote!(Complex32, f32, Complex32);
impl_float_out_binary_promote!(Complex32, f64, Complex64);
impl_float_out_binary_promote!(Complex32, Complex32, Complex32);
impl_float_out_binary_promote!(Complex32, Complex64, Complex64);

impl_normal_out_promote!(Complex32, bool, Complex32);
impl_normal_out_promote!(Complex32, i8, Complex32);
impl_normal_out_promote!(Complex32, i16, Complex32);
impl_normal_out_promote!(Complex32, i32, Complex32);
impl_normal_out_promote!(Complex32, i64, Complex64);
#[cfg(target_pointer_width = "64")]
impl_normal_out_promote!(Complex32, isize, Complex64);
#[cfg(target_pointer_width = "32")]
impl_normal_out_promote!(Complex32, isize, Complex32);
impl_normal_out_promote!(Complex32, u8, Complex32);
impl_normal_out_promote!(Complex32, u16, Complex32);
impl_normal_out_promote!(Complex32, u32, Complex32);
impl_normal_out_promote!(Complex32, u64, Complex64);
#[cfg(target_pointer_width = "64")]
impl_normal_out_promote!(Complex32, usize, Complex64);
#[cfg(target_pointer_width = "32")]
impl_normal_out_promote!(Complex32, usize, Complex32);
impl_normal_out_promote!(Complex32, f16, Complex32);
impl_normal_out_promote!(Complex32, bf16, Complex32);
impl_normal_out_promote!(Complex32, f32, Complex32);
impl_normal_out_promote!(Complex32, f64, Complex64);
impl_normal_out_promote!(Complex32, Complex32, Complex32);
impl_normal_out_promote!(Complex32, Complex64, Complex64);

impl_float_out_unary_promote!(Complex32, Complex32);
