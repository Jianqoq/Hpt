#[cfg(target_pointer_width = "64")]
mod impl_isize {
    use crate::type_promote::{FloatOutBinaryPromote, NormalOutPromote, SimdCmpPromote, FloatOutUnaryPromote};
    use crate::vectors::vector_promote::*;
    use crate::{impl_float_out_binary_promote, impl_normal_out_promote, impl_simd_cmp_promote, impl_float_out_unary_promote};
    use half::{bf16, f16};
    use num_complex::{Complex32, Complex64};
    impl_float_out_binary_promote!(usize, bool, f64);
    impl_float_out_binary_promote!(usize, i8, f64);
    impl_float_out_binary_promote!(usize, i16, f64);
    impl_float_out_binary_promote!(usize, i32, f64);
    impl_float_out_binary_promote!(usize, i64, f64);
    impl_float_out_binary_promote!(usize, isize, f64);
    impl_float_out_binary_promote!(usize, u8, f64);
    impl_float_out_binary_promote!(usize, u16, f64);
    impl_float_out_binary_promote!(usize, u32, f64);
    impl_float_out_binary_promote!(usize, u64, f64);
    impl_float_out_binary_promote!(usize, usize, f64);
    impl_float_out_binary_promote!(usize, f16, f64);
    impl_float_out_binary_promote!(usize, bf16, f64);
    impl_float_out_binary_promote!(usize, f32, f64);
    impl_float_out_binary_promote!(usize, f64, f64);
    impl_float_out_binary_promote!(usize, Complex32, Complex64);
    impl_float_out_binary_promote!(usize, Complex64, Complex64);

    impl_normal_out_promote!(usize, bool, u64);
    impl_normal_out_promote!(usize, i8, i64);
    impl_normal_out_promote!(usize, i16, i64);
    impl_normal_out_promote!(usize, i32, i64);
    impl_normal_out_promote!(usize, i64, i64);
    impl_normal_out_promote!(usize, isize, i64);
    impl_normal_out_promote!(usize, u8, u64);
    impl_normal_out_promote!(usize, u16, u64);
    impl_normal_out_promote!(usize, u32, u64);
    impl_normal_out_promote!(usize, u64, u64);
    impl_normal_out_promote!(usize, usize, usize);
    impl_normal_out_promote!(usize, f16, f16);
    impl_normal_out_promote!(usize, bf16, bf16);
    impl_normal_out_promote!(usize, f32, f32);
    impl_normal_out_promote!(usize, f64, f64);
    impl_normal_out_promote!(usize, Complex32, Complex32);
    impl_normal_out_promote!(usize, Complex64, Complex64);

    impl_simd_cmp_promote!(usize, bool, isize);
    impl_simd_cmp_promote!(usize, i8, isize);
    impl_simd_cmp_promote!(usize, i16, isize);
    impl_simd_cmp_promote!(usize, i32, isize);
    impl_simd_cmp_promote!(usize, i64, isize);
    impl_simd_cmp_promote!(usize, isize, isize);
    impl_simd_cmp_promote!(usize, u8, isize);
    impl_simd_cmp_promote!(usize, u16, isize);
    impl_simd_cmp_promote!(usize, u32, isize);
    impl_simd_cmp_promote!(usize, u64, isize);
    impl_simd_cmp_promote!(usize, usize, isize);
    impl_simd_cmp_promote!(usize, f16, isize);
    impl_simd_cmp_promote!(usize, bf16, isize);
    impl_simd_cmp_promote!(usize, f32, isize);
    impl_simd_cmp_promote!(usize, f64, isize);

    impl_float_out_unary_promote!(usize, f64);
}

#[cfg(target_pointer_width = "32")]
mod impl_isize {
    use crate::type_promote::{FloatOutBinaryPromote, NormalOutPromote, SimdCmpPromote, FloatOutUnaryPromote};
    use crate::vectors::vector_promote::*;
    use crate::{impl_float_out_binary_promote, impl_normal_out_promote, impl_simd_cmp_promote, impl_float_out_unary_promote};
    use half::{bf16, f16};
    use num_complex::{Complex32, Complex64};
    impl_float_out_binary_promote!(usize, bool, f32);
    impl_float_out_binary_promote!(usize, i8, f32);
    impl_float_out_binary_promote!(usize, i16, f32);
    impl_float_out_binary_promote!(usize, i32, f32);
    impl_float_out_binary_promote!(usize, i64, f64);
    impl_float_out_binary_promote!(usize, isize, f32);
    impl_float_out_binary_promote!(usize, u8, f32);
    impl_float_out_binary_promote!(usize, u16, f32);
    impl_float_out_binary_promote!(usize, u32, f32);
    impl_float_out_binary_promote!(usize, u64, f64);
    impl_float_out_binary_promote!(usize, usize, f32);
    impl_float_out_binary_promote!(usize, f16, f32);
    impl_float_out_binary_promote!(usize, bf16, f32);
    impl_float_out_binary_promote!(usize, f32, f32);
    impl_float_out_binary_promote!(usize, f64, f64);
    impl_float_out_binary_promote!(usize, Complex32, Complex32);
    impl_float_out_binary_promote!(usize, Complex64, Complex64);

    impl_normal_out_promote!(usize, bool, u32);
    impl_normal_out_promote!(usize, i8, i32);
    impl_normal_out_promote!(usize, i16, i32);
    impl_normal_out_promote!(usize, i32, i32);
    impl_normal_out_promote!(usize, i64, i64);
    impl_normal_out_promote!(usize, isize, i32);
    impl_normal_out_promote!(usize, u8, u32);
    impl_normal_out_promote!(usize, u16, u32);
    impl_normal_out_promote!(usize, u32, u32);
    impl_normal_out_promote!(usize, u64, u64);
    impl_normal_out_promote!(usize, usize, usize);
    impl_normal_out_promote!(usize, f16, f16);
    impl_normal_out_promote!(usize, bf16, bf16);
    impl_normal_out_promote!(usize, f32, f32);
    impl_normal_out_promote!(usize, f64, f64);
    impl_normal_out_promote!(usize, Complex32, Complex32);
    impl_normal_out_promote!(usize, Complex64, Complex64);

    impl_simd_cmp_promote!(usize, bool, isize);
    impl_simd_cmp_promote!(usize, i8, isize);
    impl_simd_cmp_promote!(usize, i16, isize);
    impl_simd_cmp_promote!(usize, i32, isize);
    impl_simd_cmp_promote!(usize, i64, isize);
    impl_simd_cmp_promote!(usize, isize, isize);
    impl_simd_cmp_promote!(usize, u8, isize);
    impl_simd_cmp_promote!(usize, u16, isize);
    impl_simd_cmp_promote!(usize, u32, isize);
    impl_simd_cmp_promote!(usize, u64, i64);
    impl_simd_cmp_promote!(usize, usize, isize);
    impl_simd_cmp_promote!(usize, f16, isize);
    impl_simd_cmp_promote!(usize, bf16, isize);
    impl_simd_cmp_promote!(usize, f32, isize);
    impl_simd_cmp_promote!(usize, f64, i64);

    impl_float_out_unary_promote!(usize, f32);
}
