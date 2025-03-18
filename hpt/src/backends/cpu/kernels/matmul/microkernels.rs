use hpt_traits::tensor::CommonBounds;
use hpt_types::dtype::TypeCommon;

use super::microkernel_trait::MatmulMicroKernel;

/// Define Matmul Micro Kernel function
///
/// # Arguments
///
/// * `$name`: The name of the function
/// * `T`: The type of the elements in the matrix
/// * `$nr`: The number of registers to use for the columns of B
/// * `$mr`: The number of rows of C to accumulate in registers
///
/// # Example
///
/// ```rust
/// define_matmul_micro_kernel!(f32x2x1, f32, 2, 1);
/// ```
///
#[macro_export]
macro_rules! define_matmul_micro_kernel {
    ($name:ident, $nr:expr, $mr:expr) => {
        fn $name<T: $crate::common::CommonBounds>(
            a: $crate::common::Pointer<T>,
            b: $crate::common::Pointer<T>,
            c: $crate::common::Pointer<T>,
            ldc: i64,
            lda: i64,
            kc: usize,
            jb: usize,
            ks: i64,
            first_kiter: bool,
        ) {
            use $crate::types::vectors::traits::VecTrait;
            use $crate::common::Pointer;
            use $crate::types::math::NormalOut;
            #[inline(always)]
            fn mma<T: $crate::common::CommonBounds>(mut a: Pointer<T>, mut b: Pointer<T>, lda: i64, kc: usize, ks: i64) -> [[<T as $crate::types::TypeCommon>::Vec; $nr]; $mr] {
                let mut c_local = [[<T as $crate::types::TypeCommon>::Vec::splat(<T>::ZERO); $nr]; $mr];
                for _ in 0..kc {
                    $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                            let b_vec~NR = unsafe {
                                *(b.ptr.add(NR * <T as $crate::types::TypeCommon>::Vec::SIZE)
                                    as *const <T as $crate::types::TypeCommon>::Vec)
                            };
                        }
                    );
                    #[allow(unused_mut)]
                    let mut a_vec;
                    $crate::re_exports::seq_macro::seq!(MR in 0..$mr {
                            a_vec = <T as $crate::types::TypeCommon>::Vec::splat(a[MR as i64 * lda]);
                            $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                                c_local[MR][NR] = a_vec._mul_add(b_vec~NR, c_local[MR][NR]);
                            });
                        }
                    );
                    b += $nr * <T as $crate::types::TypeCommon>::Vec::SIZE as i64;
                    a += ks;
                }
                c_local
            }

            let c_local = mma(a, b, lda, kc, ks);
            if jb == $nr * <T as $crate::types::TypeCommon>::Vec::SIZE {
                if first_kiter {
                    $crate::re_exports::seq_macro::seq!(MR in 0..$mr {
                        $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                                let res_ptr = unsafe {
                                    c.ptr.offset(
                                        (MR * ldc + NR * <T as $crate::types::TypeCommon>::Vec::SIZE as i64) as isize,
                                    )
                                } as *mut <T as $crate::types::TypeCommon>::Vec;
                                unsafe {res_ptr.write_unaligned(c_local[MR as usize][NR as usize]) };
                                }
                            );
                        }
                    );
                } else {
                    $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                        #[allow(unused_mut)]
                        let mut res~NR;
                        }
                    );
                    $crate::re_exports::seq_macro::seq!(MR in 0..$mr {
                        $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                                let res_ptr = unsafe {
                                    c.ptr.offset(
                                        (MR * ldc + NR * <T as $crate::types::TypeCommon>::Vec::SIZE as i64) as isize,
                                    )
                                } as *mut <T as $crate::types::TypeCommon>::Vec;
                                res~NR = unsafe {res_ptr.read_unaligned()};
                                unsafe {res_ptr.write_unaligned(res~NR._add(c_local[MR as usize][NR as usize])) };
                                }
                            );
                        }
                    );
                }
            } else {
                if first_kiter {
                    for ii in 0..$mr as i64 {
                        let c_local_ptr = c_local[ii as usize].as_ptr() as *const T;
                        for jj in 0..jb as i64 {
                            let res_ptr = unsafe { c.ptr.offset((ii * ldc + jj) as isize) } as *mut T;
                            unsafe {
                                *res_ptr = c_local_ptr.offset(jj as isize).read();
                            };
                        }
                    }
                } else {
                    for ii in 0..$mr as i64 {
                        let c_local_ptr = c_local[ii as usize].as_ptr() as *const T;
                        for jj in 0..jb as i64 {
                            let res_ptr = unsafe { c.ptr.offset((ii * ldc + jj) as isize) } as *mut T;
                            unsafe {
                                *res_ptr = (*res_ptr)._add(c_local_ptr.offset(jj as isize).read());
                            };
                        }
                    }
                }
            }
        }
    };
}

/// Define Matmul Micro Kernel function
///
/// # Arguments
///
/// * `$name`: The name of the function
/// * `T`: The type of the elements in the matrix
/// * `$nr`: The number of registers to use for the columns of B
/// * `$mr`: The number of rows of C to accumulate in registers
///
/// # Example
///
/// ```rust
/// define_matmul_micro_kernel!(f32x2x1, f32, 2, 1);
/// ```
///
#[macro_export]
macro_rules! define_mixed_precision_matmul_micro_kernel {
    ($name:ident, $nr:expr, $mr:expr, $nr2:expr) => {
        fn $name<T: $crate::common::CommonBounds, IM: $crate::common::CommonBounds>(
            a: $crate::common::Pointer<IM>,
            b: $crate::common::Pointer<IM>,
            c: $crate::common::Pointer<T>,
            ldc: i64,
            lda: i64,
            kc: usize,
            jb: usize,
            ks: i64,
            first_kiter: bool,
            vec_cast_back: fn(*const IM::Vec) -> T::Vec,
            cast_back: fn(IM) -> T,
        ) {
            use $crate::types::vectors::traits::VecTrait;
            use $crate::common::Pointer;
            use $crate::types::math::NormalOut;
            #[inline(always)]
            fn mma<T: $crate::common::CommonBounds>(mut a: Pointer<T>, mut b: Pointer<T>, lda: i64, kc: usize, ks: i64) -> [[<T as $crate::types::TypeCommon>::Vec; $nr2]; $mr] {
                let mut c_local = [[<T as $crate::types::TypeCommon>::Vec::splat(<T>::ZERO); $nr2]; $mr];
                for _ in 0..kc {
                    $crate::re_exports::seq_macro::seq!(NR in 0..$nr2 {
                            let b_vec~NR = unsafe {
                                *(b.ptr.add(NR * <T as $crate::types::TypeCommon>::Vec::SIZE)
                                    as *const <T as $crate::types::TypeCommon>::Vec)
                            };
                        }
                    );
                    #[allow(unused_mut)]
                    let mut a_vec;
                    $crate::re_exports::seq_macro::seq!(MR in 0..$mr {
                            a_vec = <T as $crate::types::TypeCommon>::Vec::splat(a[MR as i64 * lda]);
                            $crate::re_exports::seq_macro::seq!(NR in 0..$nr2 {
                                c_local[MR][NR] = a_vec._mul_add(b_vec~NR, c_local[MR][NR]);
                            });
                        }
                    );
                    b += $nr2 * <T as $crate::types::TypeCommon>::Vec::SIZE as i64;
                    a += ks;
                }
                c_local
            }

            let c_local = mma(a, b, lda, kc, ks);
            if jb == $nr * <T as $crate::types::TypeCommon>::Vec::SIZE {
                if first_kiter {
                    $crate::re_exports::seq_macro::seq!(MR in 0..$mr {
                        $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                                let res_ptr = unsafe {
                                    c.ptr.offset(
                                        (MR * ldc + NR * <T as $crate::types::TypeCommon>::Vec::SIZE as i64) as isize,
                                    )
                                } as *mut <T as $crate::types::TypeCommon>::Vec;
                                unsafe {res_ptr.write_unaligned(vec_cast_back(c_local[MR as usize].as_ptr())) };
                                }
                            );
                        }
                    );
                } else {
                    $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                        #[allow(unused_mut)]
                        let mut res~NR;
                        }
                    );
                    $crate::re_exports::seq_macro::seq!(MR in 0..$mr {
                        $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                                let res_ptr = unsafe {
                                    c.ptr.offset(
                                        (MR * ldc + NR * <T as $crate::types::TypeCommon>::Vec::SIZE as i64) as isize,
                                    )
                                } as *mut <T as $crate::types::TypeCommon>::Vec;
                                res~NR = unsafe {res_ptr.read_unaligned()};
                                unsafe {res_ptr.write_unaligned(res~NR._add(vec_cast_back(c_local[MR as usize].as_ptr()))) };
                                }
                            );
                        }
                    );
                }
            } else {
                if first_kiter {
                    for ii in 0..$mr as i64 {
                        let c_local_ptr = c_local[ii as usize].as_ptr() as *const IM;
                        for jj in 0..jb as i64 {
                            let res_ptr = unsafe { c.ptr.offset((ii * ldc + jj) as isize) } as *mut T;
                            unsafe {
                                *res_ptr = cast_back(c_local_ptr.offset(jj as isize).read());
                            };
                        }
                    }
                } else {
                    for ii in 0..$mr as i64 {
                        let c_local_ptr = c_local[ii as usize].as_ptr() as *const IM;
                        for jj in 0..jb as i64 {
                            let res_ptr = unsafe { c.ptr.offset((ii * ldc + jj) as isize) } as *mut T;
                            unsafe {
                                *res_ptr = (*res_ptr)._add(cast_back(c_local_ptr.offset(jj as isize).read()));
                            };
                        }
                    }
                }
            }
        }
    };
}

impl MatmulMicroKernel for bool {}
impl MatmulMicroKernel for i8 {}
impl MatmulMicroKernel for i16 {}
impl MatmulMicroKernel for i32 {}
impl MatmulMicroKernel for i64 {}
impl MatmulMicroKernel for u8 {}
impl MatmulMicroKernel for u16 {}
impl MatmulMicroKernel for u32 {}
impl MatmulMicroKernel for u64 {}
impl MatmulMicroKernel for f32 {}
impl MatmulMicroKernel for f64 {}
impl MatmulMicroKernel for usize {}
impl MatmulMicroKernel for isize {}
impl MatmulMicroKernel for crate::types::f16 {
    fn get_mixed_precision_kernel<MixedType>(
        nr: usize,
        mr: usize,
    ) -> fn(
        hpt_common::Pointer<MixedType>,
        hpt_common::Pointer<MixedType>,
        hpt_common::Pointer<Self>,
        i64,
        i64,
        usize,
        usize,
        i64,
        bool,
        fn(*const <MixedType as TypeCommon>::Vec) -> Self::Vec,
        fn(MixedType) -> Self,
    )
    where
        MixedType: CommonBounds,
    {
        assert_eq!(MixedType::BYTE_SIZE, 4);
        #[cfg(target_feature = "avx2")]
        {
            assert_eq!(nr, 1);
            // avx2 has 16 registers, each has 256 bits, assume cache line size is 512 bits
            define_mixed_precision_matmul_micro_kernel!(x1x1, 1, 1, 2);
            define_mixed_precision_matmul_micro_kernel!(x1x2, 1, 2, 2);
            define_mixed_precision_matmul_micro_kernel!(x1x3, 1, 3, 2);
            define_mixed_precision_matmul_micro_kernel!(x1x4, 1, 4, 2);
            define_mixed_precision_matmul_micro_kernel!(x1x5, 1, 5, 2);
            define_mixed_precision_matmul_micro_kernel!(x1x6, 1, 6, 2);
            return [x1x1, x1x2, x1x3, x1x4, x1x5, x1x6][mr - 1];
        }
        #[cfg(all(not(target_feature = "avx2"), target_feature = "sse"))]
        {
            assert_eq!(nr, 2);
            // sse has 16 registers, each has 128 bits, assume cache line size is 512 bits
            define_mixed_precision_matmul_micro_kernel!(x2x1, 4, 1, 4);
            define_mixed_precision_matmul_micro_kernel!(x2x2, 4, 2, 4);
            define_mixed_precision_matmul_micro_kernel!(x2x3, 4, 3, 4);
            return [x2x1, x2x2, x2x3][mr - 1];
        }
        #[cfg(target_feature = "neon")]
        {
            // neon has 32 registers, each has 128 bits, assume cache line size is 1024 bits
            unimplemented!()
        }
        #[cfg(all(
            not(target_feature = "avx2"),
            not(target_feature = "sse"),
            not(target_feature = "neon")
        ))]
        {
            unimplemented!()
        }
    }

    fn get_max_mixed_precision_nr() -> usize {
        #[cfg(target_feature = "avx2")]
        {
            1
        }
        #[cfg(all(not(target_feature = "avx2"), target_feature = "sse"))]
        {
            2
        }
        #[cfg(target_feature = "neon")]
        {
            // neon has 32 registers, each has 128 bits, assume cache line size is 1024 bits
            unimplemented!()
        }
        #[cfg(all(
            not(target_feature = "avx2"),
            not(target_feature = "sse"),
            not(target_feature = "neon")
        ))]
        {
            unimplemented!()
        }
    }
    fn get_max_mixed_precision_mr() -> usize {
        #[cfg(target_feature = "avx2")]
        {
            6
        }
        #[cfg(all(not(target_feature = "avx2"), target_feature = "sse"))]
        {
            3
        }
        #[cfg(target_feature = "neon")]
        {
            // neon has 32 registers, each has 128 bits, assume cache line size is 1024 bits
            unimplemented!()
        }
        #[cfg(all(
            not(target_feature = "avx2"),
            not(target_feature = "sse"),
            not(target_feature = "neon")
        ))]
        {
            unimplemented!()
        }
    }
}
impl MatmulMicroKernel for crate::types::bf16 {}
impl MatmulMicroKernel for crate::types::Complex32 {}
impl MatmulMicroKernel for crate::types::Complex64 {}
