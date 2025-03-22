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
