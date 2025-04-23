/// Define Matmul Micro Kernel function
///
/// # Arguments
///
/// * `$name`: The name of the function
/// * `$nr`: The number of registers to use for the columns of B
/// * `$mr`: The number of rows of C to accumulate in registers
///
/// Total registers = ($mr + 1) * $nr + 1
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
            fn mma<T: $crate::common::CommonBounds>(a: Pointer<T>, b: Pointer<T>, lda: i64, kc: usize, ks: i64) -> [[<T as $crate::types::TypeCommon>::Vec; $nr]; $mr] {
                let mut c_local = [[<T as $crate::types::TypeCommon>::Vec::splat(<T>::ZERO); $nr]; $mr];
                #[inline(always)]
                fn prefetch_b<T: crate::types::TypeCommon>(b: *const <T as crate::types::TypeCommon>::Vec, offset: usize) {
                    unsafe {
                        std::arch::x86_64::_mm_prefetch(
                            b.add(offset) as *const i8,
                            std::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                }
                fn prefetch_a<T: crate::types::TypeCommon>(a: *const T, offset: usize) {
                    unsafe {
                        std::arch::x86_64::_mm_prefetch(
                            a.add(offset) as *const i8,
                            std::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                }
                let a_ptr = a.ptr as *const T;
                let b_ptr = b.ptr as *const <T as crate::types::TypeCommon>::Vec;
                let rem = kc % 4;
                for k in 0..(kc / 4) as i64 {
                    $crate::re_exports::seq_macro::seq!(MR in 0..$mr {
                        prefetch_a::<T>(a_ptr, (k * (64 / 4) * ks + MR as i64 * lda) as usize);
                    });
                    $crate::re_exports::seq_macro::seq!(UNROLL in 0..4 {
                        $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                            prefetch_b::<T>(b_ptr, ((k * 4 + 4 + UNROLL) * $nr + NR) as usize);
                        });
                        $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                            let b_vec~NR = unsafe {*b_ptr.add(((k * 4 + UNROLL) * $nr + NR) as usize)};
                        });
                        #[allow(unused_mut)]
                        let mut a_vec;
                        $crate::re_exports::seq_macro::seq!(MR in 0..$mr {
                            a_vec = <T as $crate::types::TypeCommon>::Vec::splat(a[(k * 4 + UNROLL) * ks + MR as i64 * lda]);
                            $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                                c_local[MR][NR] = a_vec._mul_add(b_vec~NR, c_local[MR][NR]);
                            });
                        });
                    });
                }

                for k in (kc - rem) as i64..kc as i64 {
                    $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                        let b_vec~NR = unsafe {*b_ptr.add((k * $nr + NR) as usize)};
                    });
                    #[allow(unused_mut)]
                    let mut a_vec;
                    $crate::re_exports::seq_macro::seq!(MR in 0..$mr {
                        a_vec = <T as $crate::types::TypeCommon>::Vec::splat(a[k * ks + MR as i64 * lda]);
                        $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                            c_local[MR][NR] = a_vec._mul_add(b_vec~NR, c_local[MR][NR]);
                        });
                    });
                }

                c_local
            }

            #[inline(always)]
            fn mma_packed_a<T: $crate::common::CommonBounds>(a: Pointer<T>, b: Pointer<T>, kc: usize) -> [[<T as $crate::types::TypeCommon>::Vec; $nr]; $mr] {
                let mut c_local = [[<T as $crate::types::TypeCommon>::Vec::splat(<T>::ZERO); $nr]; $mr];
                #[inline(always)]
                fn prefetch_b<T: crate::types::TypeCommon>(b: *const <T as crate::types::TypeCommon>::Vec, offset: usize) {
                    unsafe {
                        std::arch::x86_64::_mm_prefetch(
                            b.add(offset) as *const i8,
                            std::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                }
                fn prefetch_a<T: crate::types::TypeCommon>(a: *const T, offset: usize) {
                    unsafe {
                        std::arch::x86_64::_mm_prefetch(
                            a.add(offset) as *const i8,
                            std::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                }
                let a_ptr = a.ptr as *const T;
                let b_ptr = b.ptr as *const <T as crate::types::TypeCommon>::Vec;
                let rem = kc % 4;
                prefetch_a::<T>(a_ptr, 0);
                for k in 0..(kc / 4) as i64 {
                    prefetch_a::<T>(a_ptr, ((k + 1) * 4 * $mr) as usize);
                    $crate::re_exports::seq_macro::seq!(UNROLL in 0..4 {
                        $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                            prefetch_b::<T>(b_ptr, ((k * 4 + 4 + UNROLL) * $nr + NR) as usize);
                        });
                        $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                            let b_vec~NR = unsafe {*b_ptr.add(((k * 4 + UNROLL) * $nr + NR) as usize)};
                        });
                        #[allow(unused_mut)]
                        let mut a_vec;
                        $crate::re_exports::seq_macro::seq!(MR in 0..$mr {
                            a_vec = <T as $crate::types::TypeCommon>::Vec::splat(a[(k * 4 + UNROLL) * $mr + MR]);
                            $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                                c_local[MR][NR] = a_vec._mul_add(b_vec~NR, c_local[MR][NR]);
                            });
                        });
                    });
                }

                for k in (kc - rem) as i64..kc as i64 {
                    $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                        let b_vec~NR = unsafe {*b_ptr.add((k * $nr + NR) as usize)};
                    });
                    #[allow(unused_mut)]
                    let mut a_vec;
                    $crate::re_exports::seq_macro::seq!(MR in 0..$mr {
                        a_vec = <T as $crate::types::TypeCommon>::Vec::splat(a[k * $mr + MR]);
                        $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                            c_local[MR][NR] = a_vec._mul_add(b_vec~NR, c_local[MR][NR]);
                        });
                    });
                }
                c_local
            }

            let c_local = if lda == 1 && ks == $mr {
                mma_packed_a::<T>(a, b, kc)
            }else {
                mma(a, b, lda, kc, ks)
            };
            if jb == $nr * <T as $crate::types::TypeCommon>::Vec::SIZE {
                if first_kiter {
                    $crate::re_exports::seq_macro::seq!(MR in 0..$mr {
                        let c_ptr = unsafe { c.ptr.offset(MR * ldc as isize) };
                        $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                                let res_ptr = unsafe {
                                    c_ptr.offset(
                                        (NR * <T as $crate::types::TypeCommon>::Vec::SIZE as i64) as isize,
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
                        let c_ptr = unsafe { c.ptr.offset(MR * ldc as isize) };
                        $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                                let res_ptr = unsafe {
                                    c_ptr.offset(
                                        (NR * <T as $crate::types::TypeCommon>::Vec::SIZE as i64) as isize,
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
/// * `$nr`: The number of registers to use for the columns of B
/// * `$mr`: The number of rows of C to accumulate in registers
///
/// Total registers = ($mr + 1) * $nr + 1
///
/// # Example
///
/// ```rust
/// define_matmul_micro_kernel_inline_asm!(f32x2x1, f32, 2, 1);
/// ```
///
#[macro_export]
macro_rules! define_matmul_micro_kernel_inline_asm {
    (
        $name:ident,
        $nr:expr,
        $mr:expr,
        $vec_size:expr,
        $vxor:ident,
        $vmov_aligned:ident,
        $vmov_unaligned:ident,
        $vbroadcast:ident,
        $vfma:ident,
        $vadd:ident,
        $c_regs:expr,
        $b_regs:expr,
        $a_regs:expr,
        $res_regs:expr
    ) => {
        #[allow(unused)]
        fn $name<T: crate::common::CommonBounds>(
            a: crate::common::Pointer<T>,
            b: crate::common::Pointer<T>,
            c: crate::common::Pointer<T>,
            mut ldc: i64,
            mut lda: i64,
            kc: usize,
            _: usize,
            mut ks: i64,
            first_kiter: bool,
        ) {
            use hpt_macros::gen_matmul_microkernel_asm;
            let b_ptr = b.ptr as *const T;
            let a_ptr = a.ptr as *const T;
            let c_ptr = c.ptr as *mut T;
            let first_kiter = first_kiter as i64;
            ldc *= 4;
            lda *= 4;
            ks *= 4;
            ks -= ($mr - 1) * lda;
            unsafe {
                gen_matmul_microkernel_asm!(
                    $nr,       /* nr */
                    $mr,       /* mr */
                    $vec_size, /* vec_size */
                    $vxor,
                    $vmov_aligned,
                    $vmov_unaligned,
                    $vbroadcast,
                    $vfma,
                    $vadd,
                    a_ptr,
                    b_ptr,
                    c_ptr,
                    lda,
                    ldc,
                    kc,
                    ks,
                    first_kiter,
                    $c_regs,
                    $b_regs,
                    $a_regs,
                    $res_regs
                )
            };
        }
    };
}

/// Define Matmul Micro Kernel function with inline asm
///
/// # Arguments
///
/// * `$name`: The name of the function
/// * `$nr`: The number of registers to use for the columns of B
#[macro_export]
macro_rules! define_matmul_micro_kernel_inline_asm_rem {
    (
        $name:ident,
        $nr:expr,
        $mr:expr
    ) => {
        #[allow(unused_variables)]
        fn $name<T: crate::common::CommonBounds>(
            a: crate::common::Pointer<T>,
            b: crate::common::Pointer<T>,
            c: crate::common::Pointer<T>,
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
    };
}

/// Define Matmul Micro Kernel function with post operation
///
/// # Arguments
///
/// * `$name`: The name of the function
/// * `$nr`: The number of registers to use for the columns of B
/// * `$mr`: The number of rows of C to accumulate in registers
///
/// Total registers = ($mr + 1) * $nr + 1
///
/// # Example
///
/// ```rust
/// define_post_op_matmul_micro_kernel!(f32x2x1, f32, 2, 1);
/// ```
///
#[macro_export]
macro_rules! define_post_op_matmul_micro_kernel {
    ($name:ident, $nr:expr, $mr:expr) => {
        fn $name<T: $crate::common::CommonBounds, F: Fn(T) -> T, F2: Fn(T::Vec) -> T::Vec>(
            a: $crate::common::Pointer<T>,
            b: $crate::common::Pointer<T>,
            c: $crate::common::Pointer<T>,
            ldc: i64,
            lda: i64,
            kc: usize,
            jb: usize,
            ks: i64,
            first_kiter: bool,
            last_kiter: bool,
            post_op: F,
            post_op_vec: F2,
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
                if last_kiter {
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
                                unsafe {res_ptr.write_unaligned(post_op_vec(res~NR)) };
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
                if last_kiter {
                    for ii in 0..$mr as i64 {
                        for jj in 0..jb as i64 {
                            let res_ptr = unsafe { c.ptr.offset((ii * ldc + jj) as isize) } as *mut T;
                            unsafe {
                                *res_ptr = post_op(res_ptr.read());
                            };
                        }
                    }
                }
            }
        }
    };
}

/// Define Matmul Micro Kernel function for NEON
///
/// # Arguments
///
/// * `$name`: The name of the function
/// * `$nr`: The value must less than or equal to 10
/// * `$mr`: must equal to vector size
///
/// Total registers = ($mr + 1) * $nr + 1
///
/// # Example
///
/// ```rust
/// define_neon_matmul_micro_kernel!(f32x2x1, f32, 2, 1);
/// ```
///
#[cfg(target_feature = "neon")]
#[macro_export]
macro_rules! define_neon_matmul_micro_kernel {
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
            use $crate::types::TypeCommon;
            fn load_vec_neon<const NR: usize>(data: *const f32, mut to_write: *mut f32) {
                use std::{arch::aarch64::*, mem::transmute};
                unsafe {
                    match NR {
                        1 => {
                            let res = vld1q_f32(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 1]) = transmute(res);
                        }
                        2 => {
                            let res = vld1q_f32_x2(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 2]) = transmute(res);
                        }
                        3 => {
                            let res = vld1q_f32_x3(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 3]) = transmute(res);
                        }
                        4 => {
                            let res = vld1q_f32_x4(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                        }
                        5 => {
                            let res = vld1q_f32(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 1]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE);
                            let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                        }
                        6 => {
                            let res = vld1q_f32_x2(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 2]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 2);
                            let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE * 2));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                        }
                        7 => {
                            let res = vld1q_f32_x3(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 3]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 3);
                            let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE * 3));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                        }
                        8 => {
                            let res = vld1q_f32_x4(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 4);
                            let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE * 4));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                        }
                        9 => {
                            let res = vld1q_f32_x4(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 4);
                            let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE * 4));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 4);
                            let res = vld1q_f32(data.add(<f32 as TypeCommon>::Vec::SIZE * 8));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 1]) = transmute(res);
                        }
                        10 => {
                            let res = vld1q_f32_x4(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 4);
                            let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE * 4));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 4);
                            let res = vld1q_f32_x2(data.add(<f32 as TypeCommon>::Vec::SIZE * 8));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 2]) = transmute(res);
                        }
                        _ => { unreachable!() }
                    }
                }
            }
            #[inline(always)]
            fn mma<T: $crate::common::CommonBounds>(mut a: Pointer<T>, mut b: Pointer<T>, _: i64, kc: usize, ks: i64) -> [[<T as TypeCommon>::Vec; $nr]; $mr] {
                let mut c_local = [[<T as TypeCommon>::Vec::splat(<T>::ZERO); $nr]; $mr];
                for _ in 0..kc {
                    let mut b_vec = [<T as TypeCommon>::Vec::splat(<T>::ZERO); $nr];
                    load_vec_neon::<$nr>(b.ptr as *const f32, b_vec.as_mut_ptr() as *mut f32);
                    #[allow(unused_mut)]
                    let mut a_vec;
                    unsafe {
                        a_vec = <T as TypeCommon>::Vec::from_ptr(a.ptr);
                        $crate::re_exports::seq_macro::seq!(MR in 0..$mr {
                            $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                                    c_local[MR][NR] = b_vec[NR].mul_add_lane::<MR>(a_vec, c_local[MR][NR]);
                                });
                            }
                        );
                    }
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

/// Define Matmul Micro Kernel function for NEON with post-operation
///
/// # Arguments
///
/// * `$name`: The name of the function
/// * `$nr`: The value must less than or equal to 10
/// * `$mr`: must equal to vector size
///
/// Total registers = ($mr + 1) * $nr + 1
///
/// # Example
///
/// ```rust
/// define_neon_post_op_matmul_micro_kernel!(f32x2x1, f32, 2, 1);
/// ```
///
#[cfg(target_feature = "neon")]
#[macro_export]
macro_rules! define_neon_post_op_matmul_micro_kernel {
    ($name:ident, $nr:expr, $mr:expr) => {
        fn $name<T: $crate::common::CommonBounds, F: Fn(T) -> T, G: Fn(<T as $crate::types::TypeCommon>::Vec) -> <T as $crate::types::TypeCommon>::Vec>(
            a: $crate::common::Pointer<T>,
            b: $crate::common::Pointer<T>,
            c: $crate::common::Pointer<T>,
            ldc: i64,
            lda: i64,
            kc: usize,
            jb: usize,
            ks: i64,
            first_kiter: bool,
            last_kiter: bool,
            post_op: F,
            post_op_vec: G,
        ) {
            use $crate::types::vectors::traits::VecTrait;
            use $crate::common::Pointer;
            use $crate::types::math::NormalOut;
            use $crate::types::TypeCommon;
            fn load_vec_neon<const NR: usize>(data: *const f32, mut to_write: *mut f32) {
                use std::{arch::aarch64::*, mem::transmute};
                unsafe {
                    match NR {
                        1 => {
                            let res = vld1q_f32(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 1]) = transmute(res);
                        }
                        2 => {
                            let res = vld1q_f32_x2(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 2]) = transmute(res);
                        }
                        3 => {
                            let res = vld1q_f32_x3(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 3]) = transmute(res);
                        }
                        4 => {
                            let res = vld1q_f32_x4(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                        }
                        5 => {
                            let res = vld1q_f32(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 1]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE);
                            let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                        }
                        6 => {
                            let res = vld1q_f32_x2(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 2]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 2);
                            let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE * 2));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                        }
                        7 => {
                            let res = vld1q_f32_x3(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 3]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 3);
                            let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE * 3));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                        }
                        8 => {
                            let res = vld1q_f32_x4(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 4);
                            let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE * 4));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                        }
                        9 => {
                            let res = vld1q_f32_x4(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 4);
                            let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE * 4));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 4);
                            let res = vld1q_f32(data.add(<f32 as TypeCommon>::Vec::SIZE * 8));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 1]) = transmute(res);
                        }
                        10 => {
                            let res = vld1q_f32_x4(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 4);
                            let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE * 4));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 4);
                            let res = vld1q_f32_x2(data.add(<f32 as TypeCommon>::Vec::SIZE * 8));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 2]) = transmute(res);
                        }
                        _ => { unreachable!() }
                    }
                }
            }
            #[inline(always)]
            fn mma<T: $crate::common::CommonBounds>(mut a: Pointer<T>, mut b: Pointer<T>, _: i64, kc: usize, ks: i64) -> [[<T as TypeCommon>::Vec; $nr]; $mr] {
                let mut c_local = [[<T as TypeCommon>::Vec::splat(<T>::ZERO); $nr]; $mr];
                for _ in 0..kc {
                    let mut b_vec = [<T as TypeCommon>::Vec::splat(<T>::ZERO); $nr];
                    load_vec_neon::<$nr>(b.ptr as *const f32, b_vec.as_mut_ptr() as *mut f32);
                    #[allow(unused_mut)]
                    let mut a_vec;
                    unsafe {
                        a_vec = <T as TypeCommon>::Vec::from_ptr(a.ptr);
                        $crate::re_exports::seq_macro::seq!(MR in 0..$mr {
                            $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                                    c_local[MR][NR] = b_vec[NR].mul_add_lane::<MR>(a_vec, c_local[MR][NR]);
                                });
                            }
                        );
                    }
                    b += $nr * <T as $crate::types::TypeCommon>::Vec::SIZE as i64;
                    a += ks;
                }
                c_local
            }

            let c_local = mma(a, b, lda, kc, ks);
            if jb == $nr * <T as $crate::types::TypeCommon>::Vec::SIZE {
                if first_kiter {
                    if last_kiter {
                        $crate::re_exports::seq_macro::seq!(MR in 0..$mr {
                            $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                                    let res_ptr = unsafe {
                                        c.ptr.offset(
                                            (MR * ldc + NR * <T as $crate::types::TypeCommon>::Vec::SIZE as i64) as isize,
                                        )
                                    } as *mut <T as $crate::types::TypeCommon>::Vec;
                                    unsafe {res_ptr.write_unaligned(post_op_vec(c_local[MR as usize][NR as usize])) };
                                    }
                                );
                            }
                        );
                    } else {
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
                    }
                } else {
                    $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                        #[allow(unused_mut)]
                        let mut res~NR;
                        }
                    );
                    if last_kiter {
                        $crate::re_exports::seq_macro::seq!(MR in 0..$mr {
                            $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                                    let res_ptr = unsafe {
                                        c.ptr.offset(
                                            (MR * ldc + NR * <T as $crate::types::TypeCommon>::Vec::SIZE as i64) as isize,
                                        )
                                    } as *mut <T as $crate::types::TypeCommon>::Vec;
                                    res~NR = unsafe {res_ptr.read_unaligned()};
                                    unsafe {res_ptr.write_unaligned(post_op_vec(res~NR._add(c_local[MR as usize][NR as usize]))) };
                                    }
                                );
                            }
                        );
                    } else {
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
                }
            } else {
                if first_kiter {
                    if last_kiter {
                        for ii in 0..$mr as i64 {
                            let c_local_ptr = c_local[ii as usize].as_ptr() as *const T;
                            for jj in 0..jb as i64 {
                                let res_ptr = unsafe { c.ptr.offset((ii * ldc + jj) as isize) } as *mut T;
                                unsafe {
                                    *res_ptr = post_op(c_local_ptr.offset(jj as isize).read());
                                };
                            }
                        }
                    } else {
                        for ii in 0..$mr as i64 {
                            let c_local_ptr = c_local[ii as usize].as_ptr() as *const T;
                            for jj in 0..jb as i64 {
                                let res_ptr = unsafe { c.ptr.offset((ii * ldc + jj) as isize) } as *mut T;
                                unsafe {
                                    *res_ptr = c_local_ptr.offset(jj as isize).read();
                                };
                            }
                        }
                    }
                } else {
                    if last_kiter {
                        for ii in 0..$mr as i64 {
                            let c_local_ptr = c_local[ii as usize].as_ptr() as *const T;
                            for jj in 0..jb as i64 {
                                let res_ptr = unsafe { c.ptr.offset((ii * ldc + jj) as isize) } as *mut T;
                                unsafe {
                                    *res_ptr = post_op((*res_ptr)._add(c_local_ptr.offset(jj as isize).read()));
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
        }
    };
}

/// Define Matmul Micro Kernel function
///
/// # Arguments
///
/// * `$name`: The name of the function
/// * `$nr`: The number of registers to use for the columns of B
/// * `$mr`: The number of rows of C to accumulate in registers
/// * `$nr2`: The number of registers to use for result type
///
/// Total registers = $nr2 * ($mr + 1) + 1
///
/// # Example
///
/// ```rust
/// define_mixed_precision_matmul_micro_kernel!(f32x2x1, f32, 2, 1);
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
                    });
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

/// Define Matmul Micro Kernel function with post operation
///
/// # Arguments
///
/// * `$name`: The name of the function
/// * `$nr`: The number of registers to use for the columns of B
/// * `$mr`: The number of rows of C to accumulate in registers
/// * `$nr2`: The number of registers to use for result type
///
/// Total registers = $nr2 * ($mr + 1) + 1
///
/// # Example
///
/// ```rust
/// define_mixed_precision_post_op_matmul_micro_kernel!(f32x2x1, f32, 2, 1);
/// ```
///
#[macro_export]
macro_rules! define_mixed_precision_post_op_matmul_micro_kernel {
    ($name:ident, $nr:expr, $mr:expr, $nr2:expr) => {
        fn $name<T: $crate::common::CommonBounds, IM: $crate::common::CommonBounds, F: Fn(T) -> T, F2: Fn(T::Vec) -> T::Vec>(
            a: $crate::common::Pointer<IM>,
            b: $crate::common::Pointer<IM>,
            c: $crate::common::Pointer<T>,
            ldc: i64,
            lda: i64,
            kc: usize,
            jb: usize,
            ks: i64,
            first_kiter: bool,
            last_kiter: bool,
            vec_cast_back: fn(*const IM::Vec) -> T::Vec,
            cast_back: fn(IM) -> T,
            post_op: F,
            post_op_vec: F2
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
                    if last_kiter {
                        $crate::re_exports::seq_macro::seq!(MR in 0..$mr {
                            $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                                    let res_ptr = unsafe {
                                        c.ptr.offset(
                                            (MR * ldc + NR * <T as $crate::types::TypeCommon>::Vec::SIZE as i64) as isize,
                                        )
                                    } as *mut <T as $crate::types::TypeCommon>::Vec;
                                    unsafe {res_ptr.write_unaligned(post_op_vec(vec_cast_back(c_local[MR as usize].as_ptr()))) };
                                    }
                                );
                            }
                        );
                    } else {
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
                    }
                } else {
                    $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                        #[allow(unused_mut)]
                        let mut res~NR;
                        }
                    );
                    if last_kiter {
                        $crate::re_exports::seq_macro::seq!(MR in 0..$mr {
                            $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                                    let res_ptr = unsafe {
                                        c.ptr.offset(
                                            (MR * ldc + NR * <T as $crate::types::TypeCommon>::Vec::SIZE as i64) as isize,
                                        )
                                    } as *mut <T as $crate::types::TypeCommon>::Vec;
                                    res~NR = unsafe {res_ptr.read_unaligned()};
                                    unsafe {res_ptr.write_unaligned(post_op_vec(res~NR._add(vec_cast_back(c_local[MR as usize].as_ptr())))) };
                                    }
                                );
                            }
                        );
                    } else {
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
                }
            } else {
                if first_kiter {
                    if last_kiter {
                        for ii in 0..$mr as i64 {
                            let c_local_ptr = c_local[ii as usize].as_ptr() as *const IM;
                            for jj in 0..jb as i64 {
                                let res_ptr = unsafe { c.ptr.offset((ii * ldc + jj) as isize) } as *mut T;
                                unsafe {
                                    *res_ptr = post_op(cast_back(c_local_ptr.offset(jj as isize).read()));
                                };
                            }
                        }
                    } else {
                        for ii in 0..$mr as i64 {
                            let c_local_ptr = c_local[ii as usize].as_ptr() as *const IM;
                            for jj in 0..jb as i64 {
                                let res_ptr = unsafe { c.ptr.offset((ii * ldc + jj) as isize) } as *mut T;
                                unsafe {
                                    *res_ptr = cast_back(c_local_ptr.offset(jj as isize).read());
                                };
                            }
                        }
                    }
                } else {
                    if last_kiter {
                        for ii in 0..$mr as i64 {
                            let c_local_ptr = c_local[ii as usize].as_ptr() as *const IM;
                            for jj in 0..jb as i64 {
                                let res_ptr = unsafe { c.ptr.offset((ii * ldc + jj) as isize) } as *mut T;
                                unsafe {
                                    *res_ptr = post_op((*res_ptr)._add(cast_back(c_local_ptr.offset(jj as isize).read())));
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
        }
    };
}

/// Define Matmul Micro Kernel function for NEON
///
/// # Arguments
///
/// * `$name`: The name of the function
/// * `$nr`: The number of registers to use for the columns of B
/// * `$mr`: must equal to vector size
/// * `$nr2`: must less than or equal to 10
///
/// Total registers = $nr2 * ($mr + 1) + 1
///
/// # Example
///
/// ```rust
/// define_neon_mixed_precision_matmul_micro_kernel!(f32x2x1, f32, 2, 1);
/// ```
///
#[macro_export]
#[cfg(target_feature = "neon")]
macro_rules! define_neon_mixed_precision_matmul_micro_kernel {
    ($name:ident, $nr:expr, $mr:expr, $nr2:expr) => {
        fn $name<T: $crate::common::CommonBounds, IM: $crate::common::CommonBounds>(
            a: $crate::common::Pointer<IM>,
            b: $crate::common::Pointer<IM>,
            c: $crate::common::Pointer<T>,
            ldc: i64,
            _lda: i64,
            kc: usize,
            jb: usize,
            ks: i64,
            first_kiter: bool,
            vec_cast_back: fn(*const IM::Vec) -> T::Vec,
            cast_back: fn(IM) -> T,
        ) {
            fn load_vec_neon<const NR: usize>(data: *const f32, mut to_write: *mut f32) {
                use std::{arch::aarch64::*, mem::transmute};
                unsafe {
                    match NR {
                        1 => {
                            let res = vld1q_f32(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 1]) = transmute(res);
                        }
                        2 => {
                            let res = vld1q_f32_x2(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 2]) = transmute(res);
                        }
                        3 => {
                            let res = vld1q_f32_x3(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 3]) = transmute(res);
                        }
                        4 => {
                            let res = vld1q_f32_x4(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                        }
                        5 => {
                            let res = vld1q_f32(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 1]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE);
                            let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                        }
                        6 => {
                            let res = vld1q_f32_x2(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 2]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 2);
                            let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE * 2));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                        }
                        7 => {
                            let res = vld1q_f32_x3(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 3]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 3);
                            let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE * 3));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                        }
                        8 => {
                            let res = vld1q_f32_x4(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 4);
                            let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE * 4));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                        }
                        9 => {
                            let res = vld1q_f32_x4(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 4);
                            let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE * 4));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 4);
                            let res = vld1q_f32(data.add(<f32 as TypeCommon>::Vec::SIZE * 8));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 1]) = transmute(res);
                        }
                        10 => {
                            let res = vld1q_f32_x4(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 4);
                            let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE * 4));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 4);
                            let res = vld1q_f32_x2(data.add(<f32 as TypeCommon>::Vec::SIZE * 8));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 2]) = transmute(res);
                        }
                        _ => { unreachable!() }
                    }
                }
            }
            use $crate::types::vectors::traits::VecTrait;
            use $crate::common::Pointer;
            use $crate::types::math::NormalOut;
            #[inline(always)]
            fn mma<T: $crate::common::CommonBounds>(mut a: Pointer<T>, mut b: Pointer<T>, kc: usize, ks: i64) -> [[<T as $crate::types::TypeCommon>::Vec; $nr2]; $mr] {
                let mut c_local = [[<T as $crate::types::TypeCommon>::Vec::splat(<T>::ZERO); $nr2]; $mr];
                for _ in 0..kc {
                    let mut b_vec = [<T as $crate::types::TypeCommon>::Vec::splat(<T>::ZERO); $nr2];
                    load_vec_neon::<$nr2>(b.ptr as *const f32, b_vec.as_mut_ptr() as *mut f32);

                    #[allow(unused_mut)]
                    let mut a_vec;
                    unsafe {
                        a_vec = <T as TypeCommon>::Vec::from_ptr(a.ptr);
                        $crate::re_exports::seq_macro::seq!(MR in 0..$mr {
                            $crate::re_exports::seq_macro::seq!(NR in 0..$nr2 {
                                    c_local[MR][NR] = b_vec[NR].mul_add_lane::<MR>(a_vec, c_local[MR][NR]);
                                });
                            }
                        );
                    }
                    b += $nr2 * <T as $crate::types::TypeCommon>::Vec::SIZE as i64;
                    a += ks;
                }
                c_local
            }

            let c_local = mma(a, b, kc, ks);
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

/// Define Matmul Micro Kernel function for NEON with post operation
///
/// # Arguments
///
/// * `$name`: The name of the function
/// * `$nr`: The number of registers to use for the columns of B
/// * `$mr`: must equal to vector size
/// * `$nr2`: must less than or equal to 10
///
/// Total registers = $nr2 * ($mr + 1) + 1
///
/// # Example
///
/// ```rust
/// define_neon_mixed_precision_post_op_matmul_micro_kernel!(f32x2x1, f32, 2, 1);
/// ```
///
#[macro_export]
#[cfg(target_feature = "neon")]
macro_rules! define_neon_mixed_precision_post_op_matmul_micro_kernel {
    ($name:ident, $nr:expr, $mr:expr, $nr2:expr) => {
        fn $name<T: $crate::common::CommonBounds, IM: $crate::common::CommonBounds, F: Fn(T) -> T, G: Fn(T::Vec) -> T::Vec>(
            a: $crate::common::Pointer<IM>,
            b: $crate::common::Pointer<IM>,
            c: $crate::common::Pointer<T>,
            ldc: i64,
            _lda: i64,
            kc: usize,
            jb: usize,
            ks: i64,
            first_kiter: bool,
            last_kiter: bool,
            vec_cast_back: fn(*const IM::Vec) -> T::Vec,
            cast_back: fn(IM) -> T,
            post_op: F,
            post_op_vec: G,
        ) {
            fn load_vec_neon<const NR: usize>(data: *const f32, mut to_write: *mut f32) {
                use std::{arch::aarch64::*, mem::transmute};
                unsafe {
                    match NR {
                        1 => {
                            let res = vld1q_f32(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 1]) = transmute(res);
                        }
                        2 => {
                            let res = vld1q_f32_x2(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 2]) = transmute(res);
                        }
                        3 => {
                            let res = vld1q_f32_x3(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 3]) = transmute(res);
                        }
                        4 => {
                            let res = vld1q_f32_x4(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                        }
                        5 => {
                            let res = vld1q_f32(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 1]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE);
                            let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                        }
                        6 => {
                            let res = vld1q_f32_x2(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 2]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 2);
                            let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE * 2));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                        }
                        7 => {
                            let res = vld1q_f32_x3(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 3]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 3);
                            let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE * 3));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                        }
                        8 => {
                            let res = vld1q_f32_x4(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 4);
                            let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE * 4));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                        }
                        9 => {
                            let res = vld1q_f32_x4(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 4);
                            let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE * 4));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 4);
                            let res = vld1q_f32(data.add(<f32 as TypeCommon>::Vec::SIZE * 8));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 1]) = transmute(res);
                        }
                        10 => {
                            let res = vld1q_f32_x4(data);
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 4);
                            let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE * 4));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
                            to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 4);
                            let res = vld1q_f32_x2(data.add(<f32 as TypeCommon>::Vec::SIZE * 8));
                            *(to_write as *mut [<f32 as TypeCommon>::Vec; 2]) = transmute(res);
                        }
                        _ => { unreachable!() }
                    }
                }
            }
            use $crate::types::vectors::traits::VecTrait;
            use $crate::common::Pointer;
            use $crate::types::math::NormalOut;
            #[inline(always)]
            fn mma<T: $crate::common::CommonBounds>(mut a: Pointer<T>, mut b: Pointer<T>, kc: usize, ks: i64) -> [[<T as $crate::types::TypeCommon>::Vec; $nr2]; $mr] {
                let mut c_local = [[<T as $crate::types::TypeCommon>::Vec::splat(<T>::ZERO); $nr2]; $mr];
                for _ in 0..kc {
                    let mut b_vec = [<T as $crate::types::TypeCommon>::Vec::splat(<T>::ZERO); $nr2];
                    load_vec_neon::<$nr2>(b.ptr as *const f32, b_vec.as_mut_ptr() as *mut f32);

                    #[allow(unused_mut)]
                    let mut a_vec;
                    unsafe {
                        a_vec = <T as TypeCommon>::Vec::from_ptr(a.ptr);
                        $crate::re_exports::seq_macro::seq!(MR in 0..$mr {
                            $crate::re_exports::seq_macro::seq!(NR in 0..$nr2 {
                                    c_local[MR][NR] = b_vec[NR].mul_add_lane::<MR>(a_vec, c_local[MR][NR]);
                                });
                            }
                        );
                    }
                    b += $nr2 * <T as $crate::types::TypeCommon>::Vec::SIZE as i64;
                    a += ks;
                }
                c_local
            }

            let c_local = mma(a, b, kc, ks);
            if jb == $nr * <T as $crate::types::TypeCommon>::Vec::SIZE {
                if first_kiter {
                    if last_kiter {
                        $crate::re_exports::seq_macro::seq!(MR in 0..$mr {
                            $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                                    let res_ptr = unsafe {
                                        c.ptr.offset(
                                            (MR * ldc + NR * <T as $crate::types::TypeCommon>::Vec::SIZE as i64) as isize,
                                        )
                                    } as *mut <T as $crate::types::TypeCommon>::Vec;
                                    unsafe {res_ptr.write_unaligned(post_op_vec(vec_cast_back(c_local[MR as usize].as_ptr()))) };
                                    }
                                );
                            }
                        );
                    } else {
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
                    }
                } else {
                    $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                        #[allow(unused_mut)]
                        let mut res~NR;
                        }
                    );
                    if last_kiter {
                        $crate::re_exports::seq_macro::seq!(MR in 0..$mr {
                            $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                                    let res_ptr = unsafe {
                                        c.ptr.offset(
                                            (MR * ldc + NR * <T as $crate::types::TypeCommon>::Vec::SIZE as i64) as isize,
                                        )
                                    } as *mut <T as $crate::types::TypeCommon>::Vec;
                                    res~NR = unsafe {res_ptr.read_unaligned()};
                                    unsafe {res_ptr.write_unaligned(post_op_vec(res~NR._add(vec_cast_back(c_local[MR as usize].as_ptr())))) };
                                    }
                                );
                            }
                        );
                    } else {
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
                }
            } else {
                if first_kiter {
                    if last_kiter {
                        for ii in 0..$mr as i64 {
                            let c_local_ptr = c_local[ii as usize].as_ptr() as *const IM;
                            for jj in 0..jb as i64 {
                                let res_ptr = unsafe { c.ptr.offset((ii * ldc + jj) as isize) } as *mut T;
                                unsafe {
                                    *res_ptr = post_op(cast_back(c_local_ptr.offset(jj as isize).read()));
                                };
                            }
                        }
                    } else {
                        for ii in 0..$mr as i64 {
                            let c_local_ptr = c_local[ii as usize].as_ptr() as *const IM;
                            for jj in 0..jb as i64 {
                                let res_ptr = unsafe { c.ptr.offset((ii * ldc + jj) as isize) } as *mut T;
                                unsafe {
                                    *res_ptr = cast_back(c_local_ptr.offset(jj as isize).read());
                                };
                            }
                        }
                    }
                } else {
                    if last_kiter {
                        for ii in 0..$mr as i64 {
                            let c_local_ptr = c_local[ii as usize].as_ptr() as *const IM;
                            for jj in 0..jb as i64 {
                                let res_ptr = unsafe { c.ptr.offset((ii * ldc + jj) as isize) } as *mut T;
                                unsafe {
                                    *res_ptr = post_op((*res_ptr)._add(cast_back(c_local_ptr.offset(jj as isize).read())));
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
        }
    };
}
