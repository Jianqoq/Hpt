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
impl MatmulMicroKernel for crate::types::f16 {}
impl MatmulMicroKernel for crate::types::bf16 {}
