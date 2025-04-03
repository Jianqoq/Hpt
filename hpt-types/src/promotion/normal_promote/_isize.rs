#[cfg(target_pointer_width = "64")]
mod impl_isize {
    #[cfg(feature = "cuda")]
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
    impl_float_out_binary_promote!(isize, bool, f64, f64);
    impl_float_out_binary_promote!(isize, i8, f64, f64);
    impl_float_out_binary_promote!(isize, i16, f64, f64);
    impl_float_out_binary_promote!(isize, i32, f64, f64);
    impl_float_out_binary_promote!(isize, i64, f64, f64);
    impl_float_out_binary_promote!(isize, isize, f64, f64);
    impl_float_out_binary_promote!(isize, usize, f64, f64);
    impl_float_out_binary_promote!(isize, u8, f64, f64);
    impl_float_out_binary_promote!(isize, u16, f64, f64);
    impl_float_out_binary_promote!(isize, u32, f64, f64);
    impl_float_out_binary_promote!(isize, u64, f64, f64);
    impl_float_out_binary_promote!(isize, f16, f64, f64);
    impl_float_out_binary_promote!(isize, bf16, f64, f64);
    impl_float_out_binary_promote!(isize, f32, f64, f64);
    impl_float_out_binary_promote!(isize, f64, f64, f64);
    impl_float_out_binary_promote!(isize, Complex32, Complex64, Complex64);
    impl_float_out_binary_promote!(isize, Complex64, Complex64, Complex64);

    impl_normal_out_promote!(isize, bool, i64, i64);
    impl_normal_out_promote!(isize, i8, i64, i64);
    impl_normal_out_promote!(isize, i16, i64, i64);
    impl_normal_out_promote!(isize, i32, i64, i64);
    impl_normal_out_promote!(isize, i64, i64, i64);
    impl_normal_out_promote!(isize, isize, isize, isize);
    impl_normal_out_promote!(isize, usize, i64, i64);
    impl_normal_out_promote!(isize, u8, i64, i64);
    impl_normal_out_promote!(isize, u16, i64, i64);
    impl_normal_out_promote!(isize, u32, i64, i64);
    impl_normal_out_promote!(isize, u64, i64, i64);
    impl_normal_out_promote!(isize, f16, f64, f64);
    impl_normal_out_promote!(isize, bf16, f64, f64);
    impl_normal_out_promote!(isize, f32, f64, f64);
    impl_normal_out_promote!(isize, f64, f64, f64);
    impl_normal_out_promote!(isize, Complex32, Complex64, Complex64);
    impl_normal_out_promote!(isize, Complex64, Complex64, Complex64);

    impl_simd_cmp_promote!(isize, bool, i64);
    impl_simd_cmp_promote!(isize, i8, i64);
    impl_simd_cmp_promote!(isize, i16, i64);
    impl_simd_cmp_promote!(isize, i32, i64);
    impl_simd_cmp_promote!(isize, i64, i64);
    impl_simd_cmp_promote!(isize, isize, isize);
    impl_simd_cmp_promote!(isize, usize, i64);
    impl_simd_cmp_promote!(isize, u8, i64);
    impl_simd_cmp_promote!(isize, u16, i64);
    impl_simd_cmp_promote!(isize, u32, i64);
    impl_simd_cmp_promote!(isize, u64, i64);
    impl_simd_cmp_promote!(isize, f16, i16);
    impl_simd_cmp_promote!(isize, bf16, i16);
    impl_simd_cmp_promote!(isize, f32, i64);
    impl_simd_cmp_promote!(isize, f64, i64);

    impl_float_out_unary_promote!(isize, f64, f64);
}

#[cfg(target_pointer_width = "32")]
mod impl_isize {
    #[cfg(feature = "cuda")]
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
    impl_float_out_binary_promote!(isize, bool, f32, f32);
    impl_float_out_binary_promote!(isize, i8, f32, f32);
    impl_float_out_binary_promote!(isize, i16, f32, f32);
    impl_float_out_binary_promote!(isize, i32, f32, f32);
    impl_float_out_binary_promote!(isize, i64, f64, f64);
    impl_float_out_binary_promote!(isize, isize, f32, f32);
    impl_float_out_binary_promote!(isize, u8, f32, f32);
    impl_float_out_binary_promote!(isize, u16, f32, f32);
    impl_float_out_binary_promote!(isize, u32, f32, f32);
    impl_float_out_binary_promote!(isize, u64, f64, f64);
    impl_float_out_binary_promote!(isize, usize, f32, f32);
    impl_float_out_binary_promote!(isize, f16, f32, f32);
    impl_float_out_binary_promote!(isize, bf16, f32, f32);
    impl_float_out_binary_promote!(isize, f32, f32, f32);
    impl_float_out_binary_promote!(isize, f64, f64, f64);
    impl_float_out_binary_promote!(isize, Complex32, Complex32, Complex32);
    impl_float_out_binary_promote!(isize, Complex64, Complex64, Complex64);

    impl_normal_out_promote!(isize, bool, i32, i32);
    impl_normal_out_promote!(isize, i8, i32, i32);
    impl_normal_out_promote!(isize, i16, i32, i32);
    impl_normal_out_promote!(isize, i32, i32, i32);
    impl_normal_out_promote!(isize, i64, i64, i64);
    impl_normal_out_promote!(isize, isize, isize, isize);
    impl_normal_out_promote!(isize, u8, i32, i32);
    impl_normal_out_promote!(isize, u16, i32, i32);
    impl_normal_out_promote!(isize, u32, i32, i32);
    impl_normal_out_promote!(isize, u64, i64, i64);
    impl_normal_out_promote!(isize, usize, i32, i32);
    impl_normal_out_promote!(isize, f16, f32, f32);
    impl_normal_out_promote!(isize, bf16, f32, f32);
    impl_normal_out_promote!(isize, f32, f32, f32);
    impl_normal_out_promote!(isize, f64, f64, f64);
    impl_normal_out_promote!(isize, Complex32, Complex32, Complex32);
    impl_normal_out_promote!(isize, Complex64, Complex64, Complex64);

    impl_simd_cmp_promote!(isize, bool, i64);
    impl_simd_cmp_promote!(isize, i8, i64);
    impl_simd_cmp_promote!(isize, i16, i64);
    impl_simd_cmp_promote!(isize, i32, i64);
    impl_simd_cmp_promote!(isize, i64, i64);
    impl_simd_cmp_promote!(isize, isize, isize);
    impl_simd_cmp_promote!(isize, usize, i64);
    impl_simd_cmp_promote!(isize, u8, i64);
    impl_simd_cmp_promote!(isize, u16, i64);
    impl_simd_cmp_promote!(isize, u32, i64);
    impl_simd_cmp_promote!(isize, u64, i64);

    impl_float_out_unary_promote!(isize, f32, f32);
}
