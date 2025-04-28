/// define mma macro
macro_rules! define_mma {
    ($nr:expr, $mr:expr, $unroll:expr) => {
        #[inline(always)]
        fn mma<T: CommonBounds>(a: Pointer<T>, b: Pointer<T>, lda: i64, kc: usize, ks: i64) -> [[<T as TypeCommon>::Vec; $nr]; $mr] {
            use crate::utils::prefetch::prefetch_a;
            use crate::utils::prefetch::prefetch_b;
            use hpt_types::type_promote::NormalOut;
            let mut c_local = [[<T as TypeCommon>::Vec::splat(<T>::ZERO); $nr]; $mr];
            let a_ptr = a.ptr as *const T;
            let b_ptr = b.ptr as *const <T as TypeCommon>::Vec;
            let rem = kc % $unroll;
            for k in 0..(kc / $unroll) as i64 {
                seq_macro::seq!(MR in 0..$mr {
                    prefetch_a::<T>(a_ptr, ((k + 1) * $unroll * ks + MR as i64 * lda) as usize);
                });
                seq_macro::seq!(UNROLL in 0..$unroll {
                    seq_macro::seq!(NR in 0..$nr {
                        prefetch_b::<T>(b_ptr, (((k + 1) * $unroll + UNROLL) * $nr + NR) as usize);
                    });
                    seq_macro::seq!(NR in 0..$nr {
                        let b_vec~NR = unsafe {*b_ptr.add(((k * $unroll + UNROLL) * $nr + NR) as usize)};
                    });
                    #[allow(unused_mut)]
                    let mut a_vec;
                    seq_macro::seq!(MR in 0..$mr {
                        a_vec = <T as TypeCommon>::Vec::splat(a[(k * $unroll + UNROLL) * ks + MR as i64 * lda]);
                        seq_macro::seq!(NR in 0..$nr {
                            c_local[MR][NR] = a_vec._mul_add(b_vec~NR, c_local[MR][NR]);
                        });
                    });
                });
            }

            for k in (kc - rem) as i64..kc as i64 {
                seq_macro::seq!(NR in 0..$nr {
                    let b_vec~NR = unsafe {*b_ptr.add((k * $nr + NR) as usize)};
                });
                #[allow(unused_mut)]
                let mut a_vec;
                seq_macro::seq!(MR in 0..$mr {
                    a_vec = <T as TypeCommon>::Vec::splat(a[k * ks + MR as i64 * lda]);
                    seq_macro::seq!(NR in 0..$nr {
                        c_local[MR][NR] = a_vec._mul_add(b_vec~NR, c_local[MR][NR]);
                    });
                });
            }
            c_local
        }
    };
}

pub(crate) use define_mma;


#[cfg(all(target_feature = "neon", target_arch = "aarch64"))]
macro_rules! define_neon_mma {
    ($nr:expr, $mr:expr) => {
        #[inline(always)]
        fn mma<T: CommonBounds>(mut a: Pointer<T>, mut b: Pointer<T>, _: i64, kc: usize, ks: i64) -> [[<T as TypeCommon>::Vec; $nr]; $mr] {
            let mut c_local = [[<T as TypeCommon>::Vec::splat(<T>::ZERO); $nr]; $mr];
            for _ in 0..kc {
                let mut b_vec = [<T as TypeCommon>::Vec::splat(<T>::ZERO); $nr];
                load_vec_neon::<$nr>(b.ptr as *const f32, b_vec.as_mut_ptr() as *mut f32);
                #[allow(unused_mut)]
                let mut a_vec;
                unsafe {
                    a_vec = <T as TypeCommon>::Vec::from_ptr(a.ptr);
                    seq_macro::seq!(MR in 0..$mr {
                        seq_macro::seq!(NR in 0..$nr {
                                c_local[MR][NR] = b_vec[NR].mul_add_lane::<MR>(a_vec, c_local[MR][NR]);
                            });
                        }
                    );
                }
                b += $nr * <T as TypeCommon>::Vec::SIZE as i64;
                a += ks;
            }
            c_local
        }
    };
}

#[cfg(all(target_feature = "neon", target_arch = "aarch64"))]
pub(crate) use define_neon_mma;

/// define mma packed a macro
macro_rules! define_mma_packed_a {
    ($nr:expr, $mr:expr, $unroll:expr) => {
        #[inline(always)]
        fn mma_packed_a<T: CommonBounds>(a: Pointer<T>, b: Pointer<T>, kc: usize) -> [[<T as TypeCommon>::Vec; $nr]; $mr] {
            use crate::utils::prefetch::prefetch_a;
            use crate::utils::prefetch::prefetch_b;
            use hpt_types::type_promote::NormalOut;
            let mut c_local = [[<T as TypeCommon>::Vec::splat(<T>::ZERO); $nr]; $mr];
            let a_ptr = a.ptr as *const T;
            let b_ptr = b.ptr as *const <T as TypeCommon>::Vec;
            let rem = kc % $unroll;
            for k in 0..(kc / $unroll) as i64 {
                prefetch_a::<T>(a_ptr, ((k + 1) * $unroll * $mr) as usize);
                seq_macro::seq!(UNROLL in 0..$unroll {
                    seq_macro::seq!(NR in 0..$nr {
                        prefetch_b::<T>(b_ptr, (((k + 1) * $unroll + UNROLL) * $nr + NR) as usize);
                    });
                    seq_macro::seq!(NR in 0..$nr {
                        let b_vec~NR = unsafe {*b_ptr.add(((k * $unroll + UNROLL) * $nr + NR) as usize)};
                    });
                    #[allow(unused_mut)]
                    let mut a_vec;
                    seq_macro::seq!(MR in 0..$mr {
                        a_vec = <T as TypeCommon>::Vec::splat(a[(k * $unroll + UNROLL) * $mr + MR]);
                        seq_macro::seq!(NR in 0..$nr {
                            c_local[MR][NR] = a_vec._mul_add(b_vec~NR, c_local[MR][NR]);
                        });
                    });
                });
            }

            for k in (kc - rem) as i64..kc as i64 {
                seq_macro::seq!(NR in 0..$nr {
                    let b_vec~NR = unsafe {*b_ptr.add((k * $nr + NR) as usize)};
                });
                #[allow(unused_mut)]
                let mut a_vec;
                seq_macro::seq!(MR in 0..$mr {
                    a_vec = <T as TypeCommon>::Vec::splat(a[k * $mr + MR]);
                    seq_macro::seq!(NR in 0..$nr {
                        c_local[MR][NR] = a_vec._mul_add(b_vec~NR, c_local[MR][NR]);
                    });
                });
            }
            c_local
        }
    };
}

pub(crate) use define_mma_packed_a;

macro_rules! store_res_vec {
    (
        $mr:expr,
        $nr:expr,
        $c_ptr:expr,
        $ldc:expr,
        $c_local:expr,
        [$res_ptr:ident <= $to_store:ident],
        $($stores:tt)*
    ) => {
        seq_macro::seq!(MR in 0..$mr {
            let c_ptr = unsafe { $c_ptr.ptr.offset(MR * $ldc as isize) };
            seq_macro::seq!(NR in 0..$nr {
                    let $res_ptr = unsafe {
                        c_ptr.offset(
                            (NR * <T as TypeCommon>::Vec::SIZE as i64) as isize,
                        )
                    } as *mut <T as TypeCommon>::Vec;
                    let $to_store = $c_local[MR as usize][NR as usize];
                    $($stores)*;
                });
            }
        );
    };
}

pub(crate) use store_res_vec;

macro_rules! store_res_vec_mp {
    (
        $mr:expr,
        $nr:expr,
        $c_ptr:expr,
        $ldc:expr,
        $c_local:expr,
        [$res_ptr:ident <= $to_store:ident],
        $($stores:tt)*
    ) => {
        seq_macro::seq!(MR in 0..$mr {
            let c_ptr = unsafe { $c_ptr.ptr.offset(MR * $ldc as isize) };
            seq_macro::seq!(NR in 0..$nr {
                    let $res_ptr = unsafe {
                        c_ptr.offset(
                            (NR * <T as TypeCommon>::Vec::SIZE as i64) as isize,
                        )
                    } as *mut <T as TypeCommon>::Vec;
                    let $to_store = $c_local[MR as usize].as_ptr();
                    $($stores)*;
                });
            }
        );
    };
}

pub(crate) use store_res_vec_mp;
#[inline(always)]
#[cfg(all(target_feature = "neon", target_arch = "aarch64"))]
pub(crate) fn load_vec_neon<const NR: usize>(data: *const f32, mut to_write: *mut f32) {
    use std::{arch::aarch64::*, mem::transmute};
    use TypeCommon;
    use hpt_types::traits::VecTrait;
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

#[cfg(all(target_feature = "neon", target_arch = "aarch64"))]
pub(crate) use load_vec_neon;

macro_rules! store_res_scalar {
    (
        $mr:expr,
        $nr:expr,
        $c_ptr:expr => $res_ty: ty,
        $ldc:expr,
        $c_local:expr => $c_ty: ty,
        $jb:expr,
        $nn: ident,
        [$res_ptr:ident <= $to_store:ident],
        $($stores:tt)*
    ) => {
        seq_macro::seq!(MR in 0..$mr {
            let c_local_ptr = $c_local[MR].as_ptr() as *const $c_ty;
            for $nn in 0..$jb as i64 {
                let $res_ptr = unsafe { $c_ptr.ptr.offset((MR * $ldc + $nn) as isize) } as *mut $res_ty;
                let $to_store = unsafe { c_local_ptr.offset($nn as isize).read() };
                $($stores)*;
            }
        });
    };
}

pub(crate) use store_res_scalar;
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

macro_rules! define_matmul_micro_kernel {
    ($name:ident, $nr:expr, $mr:expr) => {
        fn $name<T: CommonBounds>(
            a: Pointer<T>,
            b: Pointer<T>,
            c: Pointer<T>,
            ldc: i64,
            lda: i64,
            kc: usize,
            jb: usize,
            ks: i64,
            first_kiter: bool,
        ) {
            use Pointer;
            use crate::ops::tensor::matmul::microkernels::define_mma;
            use crate::ops::tensor::matmul::microkernels::define_mma_packed_a;
            use crate::ops::tensor::matmul::microkernels::store_res_vec;
            use crate::ops::tensor::matmul::microkernels::store_res_scalar;
            define_mma!($nr, $mr, 4);
            define_mma_packed_a!($nr, $mr, 4);

            let c_local = if lda == 1 && ks == $mr {
                mma_packed_a::<T>(a, b, kc)
            }else {
                mma(a, b, lda, kc, ks)
            };
            if jb == $nr * <T as TypeCommon>::Vec::SIZE {
                if first_kiter {
                    store_res_vec!(
                        $mr,
                        $nr,
                        c,
                        ldc,
                        c_local,
                        [res <= c_vec],
                        unsafe { res.write_unaligned(c_vec) }
                    );
                } else {
                    store_res_vec!(
                        $mr,
                        $nr,
                        c,
                        ldc,
                        c_local,
                        [res_ptr <= c_vec],
                        unsafe { res_ptr.write_unaligned(res_ptr.read_unaligned()._add(c_vec)) }
                    );
                }
            } else {
                if first_kiter {
                    store_res_scalar!(
                        $mr, 
                        $nr, 
                        c => T, 
                        ldc, 
                        c_local => T, 
                        jb,
                        nn,
                        [res_ptr <= c_val], 
                        unsafe { *res_ptr = c_val }
                    );
                } else {
                    store_res_scalar!(
                        $mr, 
                        $nr, 
                        c => T, 
                        ldc, 
                        c_local => T, 
                        jb,
                        nn,
                        [res_ptr <= c_val], 
                        unsafe { *res_ptr = (*res_ptr)._add(c_val) }
                    );
                }
            }
        }
    };
}

pub(crate) use define_matmul_micro_kernel;

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
        fn $name<T: CommonBounds>(
            a: Pointer<T>,
            b: Pointer<T>,
            c: Pointer<T>,
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

pub(crate) use define_matmul_micro_kernel_inline_asm;

/// Define Matmul Micro Kernel function with inline asm
///
/// # Arguments
///
/// * `$name`: The name of the function
/// * `$nr`: The number of registers to use for the columns of B

macro_rules! define_matmul_micro_kernel_inline_asm_rem {
    ($name:ident, $nr:expr, $mr:expr) => {
        #[allow(unused_variables)]
        fn $name<T: CommonBounds>(
            a: Pointer<T>,
            b: Pointer<T>,
            c: Pointer<T>,
            ldc: i64,
            lda: i64,
            kc: usize,
            jb: usize,
            ks: i64,
            first_kiter: bool,
        ) {
            use hpt_types::type_promote::NormalOut;
            #[inline(always)]
            fn mma<T: CommonBounds>(mut a: Pointer<T>, mut b: Pointer<T>, lda: i64, kc: usize, ks: i64) -> [[<T as TypeCommon>::Vec; $nr]; $mr] {
                let mut c_local = [[<T as TypeCommon>::Vec::splat(<T>::ZERO); $nr]; $mr];
                for _ in 0..kc {
                    seq_macro::seq!(NR in 0..$nr {
                            let b_vec~NR = unsafe {
                                *(b.ptr.add(NR * <T as TypeCommon>::Vec::SIZE)
                                    as *const <T as TypeCommon>::Vec)
                            };
                        }
                    );
                    #[allow(unused_mut)]
                    let mut a_vec;
                    seq_macro::seq!(MR in 0..$mr {
                            a_vec = <T as TypeCommon>::Vec::splat(a[MR as i64 * lda]);
                            seq_macro::seq!(NR in 0..$nr {
                                c_local[MR][NR] = a_vec._mul_add(b_vec~NR, c_local[MR][NR]);
                            });
                        }
                    );
                    b += $nr * <T as TypeCommon>::Vec::SIZE as i64;
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

pub(crate) use define_matmul_micro_kernel_inline_asm_rem;

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

macro_rules! define_post_op_matmul_micro_kernel {
    ($name:ident, $nr:expr, $mr:expr) => {
        fn $name<T: CommonBounds, F: Fn(T, usize, usize) -> T, G: Fn(T::Vec, usize, usize) -> T::Vec>(
            a: Pointer<T>,
            b: Pointer<T>,
            c: Pointer<T>,
            ldc: i64,
            lda: i64,
            kc: usize,
            jb: usize,
            ks: i64,
            first_kiter: bool,
            last_kiter: bool,
            m_idx: usize,
            n_idx: usize,
            _post_op: F,
            post_op_vec: G,
        ) {
            use Pointer;
            use crate::ops::tensor::matmul::microkernels::define_mma;
            use crate::ops::tensor::matmul::microkernels::define_mma_packed_a;
            use crate::ops::tensor::matmul::microkernels::store_res_vec;
            use crate::ops::tensor::matmul::microkernels::store_res_scalar;
            define_mma!($nr, $mr, 4);
            define_mma_packed_a!($nr, $mr, 4);

            let c_local = if lda == 1 && ks == $mr {
                mma_packed_a::<T>(a, b, kc)
            }else {
                mma(a, b, lda, kc, ks)
            };
            if jb == $nr * <T as TypeCommon>::Vec::SIZE {
                match (first_kiter, last_kiter) {
                    (true, true) => {
                        store_res_vec!(
                            $mr,
                            $nr,
                            c,
                            ldc,
                            c_local,
                            [res_ptr <= c_vec],
                            unsafe { res_ptr.write_unaligned(post_op_vec(c_vec, m_idx + MR, n_idx + NR)) }
                        );
                    }
                    (true, false) => {
                        store_res_vec!(
                            $mr,
                            $nr,
                            c,
                            ldc,
                            c_local,
                            [res <= c_vec],
                            unsafe { res.write_unaligned(c_vec) }
                        );
                    }
                    (false, true) => {
                        store_res_vec!(
                            $mr,
                            $nr,
                            c,
                            ldc,
                            c_local,
                            [res_ptr <= c_vec],
                            unsafe { res_ptr.write_unaligned(post_op_vec(res_ptr.read_unaligned()._add(c_vec), m_idx + MR, n_idx + NR)) }
                        );
                    }
                    (false, false) => {
                        store_res_vec!(
                            $mr,
                            $nr,
                            c,
                            ldc,
                            c_local,
                            [res_ptr <= c_vec],
                            unsafe { res_ptr.write_unaligned(res_ptr.read_unaligned()._add(c_vec)) }
                        );
                    },
                }
            } else {
                match (first_kiter, last_kiter) {
                    (true, true) => {
                        store_res_scalar!($mr, $nr, c => T, ldc, c_local => T, jb, nn, [res_ptr <= c_val], unsafe { *res_ptr = _post_op(c_val, m_idx + MR, n_idx + nn as usize) });
                    }
                    (true, false) => {
                        store_res_scalar!($mr, $nr, c => T, ldc, c_local => T, jb, nn, [res_ptr <= c_val], unsafe { *res_ptr = c_val });
                    }
                    (false, true) => {
                        store_res_scalar!($mr, $nr, c => T, ldc, c_local => T, jb, nn, [res_ptr <= c_val], unsafe { *res_ptr = _post_op((*res_ptr)._add(c_val), m_idx + MR, n_idx + nn as usize) });
                    }
                    (false, false) => {
                        store_res_scalar!($mr, $nr, c => T, ldc, c_local => T, jb, nn, [res_ptr <= c_val], unsafe { *res_ptr = c_val._add(c_val) });
                    },
                }
            }
        }
    };
}

pub(crate) use define_post_op_matmul_micro_kernel;

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
macro_rules! define_neon_matmul_micro_kernel {
    ($name:ident, $nr:expr, $mr:expr) => {
        fn $name<T: CommonBounds>(
            a: Pointer<T>,
            b: Pointer<T>,
            c: Pointer<T>,
            ldc: i64,
            lda: i64,
            kc: usize,
            jb: usize,
            ks: i64,
            first_kiter: bool,
        ) {
            use vectors::traits::VecTrait;
            use Pointer;
            use math::NormalOut;
            use $crate::backends::cpu::kernels::matmul::microkernels::load_vec_neon;
            use $crate::define_neon_mma;
            use $crate::store_res_vec;
            use $crate::store_res_scalar;
            define_neon_mma!($nr, $mr);

            let c_local = mma(a, b, lda, kc, ks);
            if jb == $nr * <T as TypeCommon>::Vec::SIZE {
                if first_kiter {
                    store_res_vec!(
                        $mr,
                        $nr,
                        c,
                        ldc,
                        c_local,
                        [res <= c_vec],
                        res.write_unaligned(c_vec)
                    );
                } else {
                    store_res_vec!(
                        $mr,
                        $nr,
                        c,
                        ldc,
                        c_local,
                        [res_ptr <= c_vec],
                        res_ptr.write_unaligned(res_ptr.read_unaligned()._add(c_vec))
                    );
                }
            } else {
                if first_kiter {
                    store_res_scalar!($mr, $nr, c => T, ldc, c_local => T, jb, [res_ptr <= c_val], unsafe { *res_ptr = c_val });
                } else {
                    store_res_scalar!($mr, $nr, c => T, ldc, c_local => T, jb, [res_ptr <= c_val], unsafe { *res_ptr = (*res_ptr)._add(c_val) });
                }
            }
        }
    };
}

#[cfg(target_feature = "neon")]
pub(crate) use define_neon_matmul_micro_kernel;

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
macro_rules! define_neon_post_op_matmul_micro_kernel {
    ($name:ident, $nr:expr, $mr:expr) => {
        fn $name<T: CommonBounds, F: Fn(T, usize, usize) -> T, G: Fn(<T as TypeCommon>::Vec, usize, usize) -> <T as TypeCommon>::Vec>(
            a: Pointer<T>,
            b: Pointer<T>,
            c: Pointer<T>,
            ldc: i64,
            lda: i64,
            kc: usize,
            jb: usize,
            ks: i64,
            first_kiter: bool,
            last_kiter: bool,
            m_idx: usize,
            n_idx: usize,
            _post_op: F,
            post_op_vec: G,
        ) {
            use vectors::traits::VecTrait;
            use Pointer;
            use math::NormalOut;
            use $crate::backends::cpu::kernels::matmul::microkernels::load_vec_neon;
            use $crate::define_neon_mma;
            use $crate::store_res_vec;
            use $crate::store_res_scalar;
            define_neon_mma!($nr, $mr);

            let c_local = mma(a, b, lda, kc, ks);
            if jb == $nr * <T as TypeCommon>::Vec::SIZE {
                match (first_kiter, last_kiter) {
                    (true, true) => {
                        store_res_vec!(
                            $mr,
                            $nr,
                            c,
                            ldc,
                            c_local,
                            [res_ptr <= c_vec],
                            res_ptr.write_unaligned(post_op_vec(c_vec, m_idx + MR, n_idx + NR))
                        );
                    }
                    (true, false) => {
                        store_res_vec!(
                            $mr,
                            $nr,
                            c,
                            ldc,
                            c_local,
                            [res <= c_vec],
                            res.write_unaligned(c_vec)
                        );
                    }
                    (false, true) => {
                        store_res_vec!(
                            $mr,
                            $nr,
                            c,
                            ldc,
                            c_local,
                            [res_ptr <= c_vec],
                            res_ptr.write_unaligned(post_op_vec(res_ptr.read_unaligned()._add(c_vec), m_idx + MR, n_idx + NR))
                        );
                    }
                    (false, false) => {
                        store_res_vec!(
                            $mr,
                            $nr,
                            c,
                            ldc,
                            c_local,
                            [res_ptr <= c_vec],
                            res_ptr.write_unaligned(res_ptr.read_unaligned()._add(c_vec))
                        );
                    },
                }
            } else {
                match (first_kiter, last_kiter) {
                    (true, true) => {
                        store_res_scalar!($mr, $nr, c => T, ldc, c_local => T, jb, [res_ptr <= c_val], unsafe { *res_ptr = _post_op(c_val, m_idx + MR, n_idx + NR) });
                    }
                    (true, false) => {
                        store_res_scalar!($mr, $nr, c => T, ldc, c_local => T, jb, [res_ptr <= c_val], unsafe { *res_ptr = c_val });
                    }
                    (false, true) => {
                        store_res_scalar!($mr, $nr, c => T, ldc, c_local => T, jb, [res_ptr <= c_val], unsafe { *res_ptr = _post_op((*res_ptr)._add(c_val), m_idx + MR, n_idx + NR) });
                    }
                    (false, false) => {
                        store_res_scalar!($mr, $nr, c => T, ldc, c_local => T, jb, [res_ptr <= c_val], unsafe { *res_ptr = c_val._add(c_val) });
                    },
                }
            }
        }
    };
}

#[cfg(target_feature = "neon")]
pub(crate) use define_neon_post_op_matmul_micro_kernel;

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

macro_rules! define_mixed_precision_matmul_micro_kernel {
    ($name:ident, $nr:expr, $mr:expr, $nr2:expr) => {
        fn $name<T: CommonBounds, IM: CommonBounds>(
            a: Pointer<IM>,
            b: Pointer<IM>,
            c: Pointer<T>,
            ldc: i64,
            lda: i64,
            kc: usize,
            jb: usize,
            ks: i64,
            first_kiter: bool,
            vec_cast_back: fn(*mut T::Vec, *const IM::Vec),
            cast_back: fn(&mut T, &IM),
        ) {
            use crate::ops::tensor::matmul::microkernels::define_mma;
            use crate::ops::tensor::matmul::microkernels::define_mma_packed_a;
            use crate::ops::tensor::matmul::microkernels::store_res_vec_mp;
            use crate::ops::tensor::matmul::microkernels::store_res_scalar;
            use hpt_types::traits::VecTrait;
            use hpt_types::type_promote::NormalOut;

            define_mma!($nr2, $mr, 4);
            define_mma_packed_a!($nr2, $mr, 4);

            let c_local = if lda == 1 && ks == $mr {
                mma_packed_a::<IM>(a, b, kc)
            }else {
                mma(a, b, lda, kc, ks)
            };
            if jb == $nr * <T as TypeCommon>::Vec::SIZE {
                if first_kiter {
                    store_res_vec_mp!(
                        $mr,
                        $nr,
                        c,
                        ldc,
                        c_local,
                        [res_ptr <= c_vec],
                        vec_cast_back(res_ptr, c_vec)
                    );
                } else {
                    store_res_vec_mp!(
                        $mr,
                        $nr,
                        c,
                        ldc,
                        c_local,
                        [res_ptr <= c_vec],
                        let mut res_vec = <T as TypeCommon>::Vec::splat(T::ZERO);
                        vec_cast_back(&mut res_vec, c_vec);
                        unsafe { res_ptr.write_unaligned(res_vec._add(res_ptr.read_unaligned())) }
                    );
                }
            } else {
                if first_kiter {
                    store_res_scalar!(
                        $mr,
                        $nr,
                        c => T,
                        ldc,
                        c_local => IM,
                        jb,
                        nn,
                        [res_ptr <= c_val],
                        unsafe { cast_back(&mut *res_ptr, &c_val) }
                    );
                } else {
                    store_res_scalar!(
                        $mr,
                        $nr,
                        c => T,
                        ldc,
                        c_local => IM,
                        jb,
                        nn,
                        [res_ptr <= c_val],
                        let mut res = T::ZERO;
                        cast_back(&mut res, &c_val);
                        unsafe { *res_ptr = (*res_ptr)._add(res) }
                    );
                }
            }
        }
    };
}

pub(crate) use define_mixed_precision_matmul_micro_kernel;

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

macro_rules! define_mixed_precision_post_op_matmul_micro_kernel {
    ($name:ident, $nr:expr, $mr:expr, $nr2:expr) => {
        fn $name<T: CommonBounds, IM: CommonBounds, F: Fn(T, usize, usize) -> T, F2: Fn(T::Vec, usize, usize) -> T::Vec>(
            a: Pointer<IM>,
            b: Pointer<IM>,
            c: Pointer<T>,
            ldc: i64,
            lda: i64,
            kc: usize,
            jb: usize,
            ks: i64,
            first_kiter: bool,
            last_kiter: bool,
            m_idx: usize,
            n_idx: usize,
            vec_cast_back: fn(*mut T::Vec, *const IM::Vec),
            cast_back: fn(&mut T, &IM),
            post_op: F,
            post_op_vec: F2
        ) {
            use crate::ops::tensor::matmul::microkernels::define_mma;
            use crate::ops::tensor::matmul::microkernels::define_mma_packed_a;
            use crate::ops::tensor::matmul::microkernels::store_res_vec_mp;
            use crate::ops::tensor::matmul::microkernels::store_res_scalar;
            use hpt_types::traits::VecTrait;
            use hpt_types::type_promote::NormalOut;

            define_mma!($nr2, $mr, 4);
            define_mma_packed_a!($nr2, $mr, 4);

            let c_local = if lda == 1 && ks == $mr {
                mma_packed_a::<IM>(a, b, kc)
            }else {
                mma(a, b, lda, kc, ks)
            };
            if jb == $nr * <T as TypeCommon>::Vec::SIZE {
                match (first_kiter, last_kiter) {
                    (true, true) => {
                        store_res_vec_mp!(
                            $mr,
                            $nr,
                            c,
                            ldc,
                            c_local,
                            [res_ptr <= c_vec],
                            let mut res = T::Vec::splat(T::ZERO);
                            vec_cast_back(&mut res, c_vec);
                            unsafe { res_ptr.write_unaligned(post_op_vec(res, m_idx + MR, n_idx + NR)) }
                        );
                    }
                    (true, false) => {
                        store_res_vec_mp!(
                            $mr,
                            $nr,
                            c,
                            ldc,
                            c_local,
                            [res_ptr <= c_vec],
                            vec_cast_back(res_ptr, c_vec)
                        );
                    }
                    (false, true) => {
                        store_res_vec_mp!(
                            $mr,
                            $nr,
                            c,
                            ldc,
                            c_local,
                            [res_ptr <= c_vec],
                            let mut res = T::Vec::splat(T::ZERO);
                            vec_cast_back(&mut res, c_vec);
                            unsafe { res_ptr.write_unaligned(post_op_vec(res._add(res_ptr.read_unaligned()), m_idx + MR, n_idx + NR)) }
                        );
                    }
                    (false, false) => {
                        store_res_vec_mp!(
                            $mr,
                            $nr,
                            c,
                            ldc,
                            c_local,
                            [res_ptr <= c_vec],
                            let mut res = T::Vec::splat(T::ZERO);
                            vec_cast_back(&mut res, c_vec);
                            unsafe { res_ptr.write_unaligned(res._add(res_ptr.read_unaligned())) }
                        );
                    },
                }
            } else {
                match (first_kiter, last_kiter) {
                    (true, true) => {
                        store_res_scalar!(
                            $mr,
                            $nr,
                            c => T,
                            ldc,
                            c_local => IM,
                            jb,
                            nn,
                            [res_ptr <= c_val],
                            let mut res = T::ZERO;
                            cast_back(&mut res, &c_val);
                            unsafe { res_ptr.write(post_op(res, m_idx + MR, n_idx + nn as usize)) }
                        );
                    }
                    (true, false) => {
                        store_res_scalar!(
                            $mr,
                            $nr,
                            c => T,
                            ldc,
                            c_local => IM,
                            jb,
                            nn,
                            [res_ptr <= c_val],
                            unsafe { cast_back(&mut *res_ptr, &c_val) }
                        );
                    }
                    (false, true) => {
                        store_res_scalar!(
                            $mr,
                            $nr,
                            c => T,
                            ldc,
                            c_local => IM,
                            jb,
                            nn,
                            [res_ptr <= c_val],
                            let mut res = T::ZERO;
                            cast_back(&mut res, &c_val);
                            unsafe { res_ptr.write(post_op(res._add(*res_ptr), m_idx + MR, n_idx + nn as usize)) }
                        );
                    }
                    (false, false) => {
                        store_res_scalar!(
                            $mr,
                            $nr,
                            c => T,
                            ldc,
                            c_local => IM,
                            jb,
                            nn,
                            [res_ptr <= c_val],
                            let mut res = T::ZERO;
                            cast_back(&mut res, &c_val);
                            unsafe { res_ptr.write(res._add(*res_ptr)) }
                        );
                    },
                }
            }
        }
    };
}

pub(crate) use define_mixed_precision_post_op_matmul_micro_kernel;

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

#[cfg(target_feature = "neon")]
macro_rules! define_neon_mixed_precision_matmul_micro_kernel {
    ($name:ident, $nr:expr, $mr:expr, $nr2:expr) => {
        fn $name<T: CommonBounds, IM: CommonBounds>(
            a: Pointer<IM>,
            b: Pointer<IM>,
            c: Pointer<T>,
            ldc: i64,
            lda: i64,
            kc: usize,
            jb: usize,
            ks: i64,
            first_kiter: bool,
            vec_cast_back: fn(*mut T::Vec, *const IM::Vec),
            cast_back: fn(&mut T, &IM),
        ) {
            use vectors::traits::VecTrait;
            use Pointer;
            use math::NormalOut;
            use $crate::define_mma;
            use $crate::define_mma_packed_a;
            use $crate::store_res_vec_mp;
            use $crate::store_res_scalar;
            define_mma!($nr2, $mr, 4);
            define_mma_packed_a!($nr2, $mr, 4);

            let c_local = if lda == 1 && ks == $mr {
                mma_packed_a::<IM>(a, b, kc)
            }else {
                mma(a, b, lda, kc, ks)
            };
            if jb == $nr * <T as TypeCommon>::Vec::SIZE {
                match first_kiter {
                    true => {
                        store_res_vec_mp!(
                            $mr,
                            $nr,
                            c,
                            ldc,
                            c_local,
                            [res_ptr <= c_vec],
                            let mut res = T::Vec::splat(T::ZERO);
                            vec_cast_back(&mut res, c_vec);
                            unsafe { res_ptr.write_unaligned(res) }
                        );
                    }
                    false => {
                        store_res_vec_mp!(
                            $mr,
                            $nr,
                            c,
                            ldc,
                            c_local,
                            [res_ptr <= c_vec],
                            let mut res = T::Vec::splat(T::ZERO);
                            vec_cast_back(&mut res, c_vec);
                            unsafe { res_ptr.write_unaligned(res._add(res_ptr.read_unaligned())) }
                        );
                    }
                }
            } else {
                match first_kiter {
                    true => {
                        store_res_scalar!(
                            $mr,
                            $nr,
                            c => T,
                            ldc,
                            c_local => IM,
                            jb,
                            [res_ptr <= c_val],
                            unsafe { cast_back(&mut *res_ptr, &c_val) }
                        );
                    }
                    false => {
                        store_res_scalar!(
                            $mr,
                            $nr,
                            c => T,
                            ldc,
                            c_local => IM,
                            jb,
                            [res_ptr <= c_val],
                            let mut res = T::ZERO;
                            cast_back(&mut res, &c_val);
                            unsafe { res_ptr.write(res._add(*res_ptr)) }
                        );
                    },
                }
            }
        }
    };
}

#[cfg(target_feature = "neon")]
pub(crate) use define_neon_mixed_precision_matmul_micro_kernel;

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

#[cfg(target_feature = "neon")]
macro_rules! define_neon_mixed_precision_post_op_matmul_micro_kernel {
    ($name:ident, $nr:expr, $mr:expr, $nr2:expr) => {
        fn $name<T: CommonBounds, IM: CommonBounds, F: Fn(T, usize, usize) -> T, G: Fn(T::Vec, usize, usize) -> T::Vec>(
            a: Pointer<IM>,
            b: Pointer<IM>,
            c: Pointer<T>,
            ldc: i64,
            _lda: i64,
            kc: usize,
            jb: usize,
            ks: i64,
            first_kiter: bool,
            last_kiter: bool,
            m_idx: usize,
            n_idx: usize,
            vec_cast_back: fn(*mut T::Vec, *const IM::Vec),
            cast_back: fn(&mut T, &IM),
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
            use vectors::traits::VecTrait;
            use Pointer;
            use math::NormalOut;
            #[inline(always)]
            fn mma<T: CommonBounds>(mut a: Pointer<T>, mut b: Pointer<T>, kc: usize, ks: i64) -> [[<T as TypeCommon>::Vec; $nr2]; $mr] {
                let mut c_local = [[<T as TypeCommon>::Vec::splat(<T>::ZERO); $nr2]; $mr];
                for _ in 0..kc {
                    let mut b_vec = [<T as TypeCommon>::Vec::splat(<T>::ZERO); $nr2];
                    load_vec_neon::<$nr2>(b.ptr as *const f32, b_vec.as_mut_ptr() as *mut f32);

                    #[allow(unused_mut)]
                    let mut a_vec;
                    unsafe {
                        a_vec = <T as TypeCommon>::Vec::from_ptr(a.ptr);
                        seq_macro::seq!(MR in 0..$mr {
                            seq_macro::seq!(NR in 0..$nr2 {
                                    c_local[MR][NR] = b_vec[NR].mul_add_lane::<MR>(a_vec, c_local[MR][NR]);
                                });
                            }
                        );
                    }
                    b += $nr2 * <T as TypeCommon>::Vec::SIZE as i64;
                    a += ks;
                }
                c_local
            }

            let c_local = mma(a, b, kc, ks);
            if jb == $nr * <T as TypeCommon>::Vec::SIZE {
                if first_kiter {
                    if last_kiter {
                        seq_macro::seq!(MR in 0..$mr {
                            seq_macro::seq!(NR in 0..$nr {
                                    let res_ptr = unsafe {
                                        c.ptr.offset(
                                            (MR * ldc + NR * <T as TypeCommon>::Vec::SIZE as i64) as isize,
                                        )
                                    } as *mut <T as TypeCommon>::Vec;
                                    let mut res = T::Vec::splat(T::ZERO);
                                    vec_cast_back(&mut res, c_local[MR as usize].as_ptr());
                                    unsafe {res_ptr.write_unaligned(post_op_vec(res, m_idx + MR, n_idx + NR)) };
                                    }
                                );
                            }
                        );
                    } else {
                        seq_macro::seq!(MR in 0..$mr {
                            seq_macro::seq!(NR in 0..$nr {
                                    let res_ptr = unsafe {
                                        c.ptr.offset(
                                            (MR * ldc + NR * <T as TypeCommon>::Vec::SIZE as i64) as isize,
                                        )
                                    } as *mut <T as TypeCommon>::Vec;
                                    let mut res = T::Vec::splat(T::ZERO);
                                    vec_cast_back(&mut res, c_local[MR as usize].as_ptr());
                                    unsafe {res_ptr.write_unaligned(res) };
                                    }
                                );
                            }
                        );
                    }
                } else {
                    seq_macro::seq!(NR in 0..$nr {
                        #[allow(unused_mut)]
                        let mut res~NR;
                        }
                    );
                    if last_kiter {
                        seq_macro::seq!(MR in 0..$mr {
                            seq_macro::seq!(NR in 0..$nr {
                                    let res_ptr = unsafe {
                                        c.ptr.offset(
                                            (MR * ldc + NR * <T as TypeCommon>::Vec::SIZE as i64) as isize,
                                        )
                                    } as *mut <T as TypeCommon>::Vec;
                                    res~NR = unsafe {res_ptr.read_unaligned()};
                                    let mut res = T::Vec::splat(T::ZERO);
                                    vec_cast_back(&mut res, c_local[MR as usize].as_ptr());
                                    unsafe {res_ptr.write_unaligned(post_op_vec(res~NR._add(res), m_idx + MR, n_idx + NR)) };
                                    }
                                );
                            }
                        );
                    } else {
                        seq_macro::seq!(MR in 0..$mr {
                            seq_macro::seq!(NR in 0..$nr {
                                    let res_ptr = unsafe {
                                        c.ptr.offset(
                                            (MR * ldc + NR * <T as TypeCommon>::Vec::SIZE as i64) as isize,
                                        )
                                    } as *mut <T as TypeCommon>::Vec;
                                    res~NR = unsafe {res_ptr.read_unaligned()};
                                    let mut res = T::Vec::splat(T::ZERO);
                                    vec_cast_back(&mut res, c_local[MR as usize].as_ptr());
                                    unsafe {res_ptr.write_unaligned(res~NR._add(res)) };
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
                                let mut res = T::ZERO;
                                unsafe {
                                    cast_back(&mut res, &c_local_ptr.offset(jj as isize).read());
                                    *res_ptr = post_op(res, m_idx + ii as usize, n_idx + jj as usize);
                                };
                            }
                        }
                    } else {
                        for ii in 0..$mr as i64 {
                            let c_local_ptr = c_local[ii as usize].as_ptr() as *const IM;
                            for jj in 0..jb as i64 {
                                let res_ptr = unsafe { c.ptr.offset((ii * ldc + jj) as isize) } as *mut T;
                                let mut res = T::ZERO;
                                unsafe {
                                    cast_back(&mut res, &c_local_ptr.offset(jj as isize).read());
                                    *res_ptr = res;
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
                                let mut res = T::ZERO;
                                unsafe {
                                    cast_back(&mut res, &c_local_ptr.offset(jj as isize).read());
                                    *res_ptr = post_op((*res_ptr)._add(res), m_idx + ii as usize, n_idx + jj as usize);
                                };
                            }
                        }
                    } else {
                        for ii in 0..$mr as i64 {
                            let c_local_ptr = c_local[ii as usize].as_ptr() as *const IM;
                            for jj in 0..jb as i64 {
                                let res_ptr = unsafe { c.ptr.offset((ii * ldc + jj) as isize) } as *mut T;
                                let mut res = T::ZERO;
                                unsafe {
                                    cast_back(&mut res, &c_local_ptr.offset(jj as isize).read());
                                    *res_ptr = (*res_ptr)._add(res);
                                };
                            }
                        }
                    }
                }
            }
        }
    };
}

#[cfg(target_feature = "neon")]
pub(crate) use define_neon_mixed_precision_post_op_matmul_micro_kernel;

