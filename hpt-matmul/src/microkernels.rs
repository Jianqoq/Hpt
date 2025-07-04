/// define mma macro
#[macro_export]
macro_rules! define_mma {
    ($dtype:ty, $vec_type:ty, $nr:expr, $mr:expr, $unroll:expr) => {
        #[inline(always)]
        fn mma(a: $crate::Pointer<$dtype>, b: $crate::Pointer<$dtype>, lda: i64, kc: usize, ks: i64) -> [[$vec_type; $nr]; $mr] {
            use crate::VecTrait;
            let mut c_local = [[<$vec_type>::splat(<$dtype>::ZERO); $nr]; $mr];
            let b_ptr = b.ptr as *const $vec_type;
            let rem = kc % $unroll;
            for k in 0..(kc / $unroll) as i64 {
                seq_macro::seq!(UNROLL in 0..$unroll {
                    seq_macro::seq!(NR in 0..$nr {
                        let b_vec~NR = unsafe {*b_ptr.add(((k * $unroll + UNROLL) * $nr + NR) as usize)};
                    });
                    #[allow(unused_mut)]
                    let mut a_vec;
                    seq_macro::seq!(MR in 0..$mr {
                        a_vec = $vec_type::splat(a[(k * $unroll + UNROLL) * ks + MR as i64 * lda]);
                        seq_macro::seq!(NR in 0..$nr {
                            c_local[MR][NR] = a_vec.mul_add(b_vec~NR, c_local[MR][NR]);
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
                    a_vec = $vec_type::splat(a[k * ks + MR as i64 * lda]);
                    seq_macro::seq!(NR in 0..$nr {
                        c_local[MR][NR] = a_vec.mul_add(b_vec~NR, c_local[MR][NR]);
                    });
                });
            }
            c_local
        }
    };
}

/// define mma macro
#[macro_export]
macro_rules! define_neon_mma {
    ($dtype:ty, $vec_type:ty, $nr:expr, $mr:expr) => {
        #[inline(always)]
        fn mma(mut a: $crate::Pointer<$dtype>, mut b: $crate::Pointer<$dtype>, kc: usize, ks: i64) -> [[$vec_type; $nr]; $mr] {
            use crate::microkernels::load_vec_neon;
            use crate::VecTrait;
            let mut c_local = [[<$vec_type>::splat(<$dtype>::ZERO); $nr]; $mr];
            for _ in 0..kc {
                let mut b_vec = [<$vec_type>::splat(<$dtype>::ZERO); $nr];
                load_vec_neon::<$nr>(b.ptr as *const f32, b_vec.as_mut_ptr() as *mut f32);
                #[allow(unused_mut)]
                let mut a_vec;
                unsafe {
                    a_vec = (a.ptr as *const $vec_type).read_unaligned();
                    seq_macro::seq!(MR in 0..$mr {
                        seq_macro::seq!(NR in 0..$nr {
                                c_local[MR][NR] = b_vec[NR].mul_add_lane::<MR>(a_vec, c_local[MR][NR]);
                            });
                        }
                    );
                }
                b += $nr * vec_size::<$dtype>() as i64;
                a += ks;
            }
            c_local
        }
    };
}

/// define mma packed a macro
#[macro_export]
macro_rules! define_mma_packed_a {
    ($dtype:ty, $vec_type:ty, $nr:expr, $mr:expr, $unroll:expr) => {
        #[inline(always)]
        fn mma_packed_a(a: $crate::Pointer<$dtype>, b: $crate::Pointer<$dtype>, kc: usize) -> [[$vec_type; $nr]; $mr] {
            use crate::VecTrait;
            let mut c_local = [[<$vec_type>::splat(<$dtype>::ZERO); $nr]; $mr];
            let b_ptr = b.ptr as *const $vec_type;
            let rem = kc % $unroll;
            for k in 0..(kc / $unroll) as i64 {
                seq_macro::seq!(UNROLL in 0..$unroll {
                    seq_macro::seq!(NR in 0..$nr {
                        let b_vec~NR = unsafe {*b_ptr.add(((k * $unroll + UNROLL) * $nr + NR) as usize)};
                    });
                    #[allow(unused_mut)]
                    let mut a_vec;
                    seq_macro::seq!(MR in 0..$mr {
                        a_vec = $vec_type::splat(a[(k * $unroll + UNROLL) * $mr + MR]);
                        seq_macro::seq!(NR in 0..$nr {
                            c_local[MR][NR] = a_vec.mul_add(b_vec~NR, c_local[MR][NR]);
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
                    a_vec = $vec_type::splat(a[k * $mr + MR]);
                    seq_macro::seq!(NR in 0..$nr {
                        c_local[MR][NR] = a_vec.mul_add(b_vec~NR, c_local[MR][NR]);
                    });
                });
            }
            c_local
        }
    };
}

/// Define store result vector macro
#[macro_export]
macro_rules! store_res_vec {
    (
        $dtype:ty,
        $vec_type:ty,
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
                            (NR * vec_size::<$dtype>() as i64) as isize,
                        )
                    } as *mut $vec_type;
                    let $to_store = $c_local[MR as usize][NR as usize];
                    unsafe {
                        $($stores)*
                    }
                });
            }
        );
    };
}

/// Define store result vector macro
#[macro_export]
macro_rules! store_res_vec_mp {
    (
        $dtype:ty,
        $vec_type:ty,
        $multiple_of:expr,
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
                            (NR * vec_size::<$dtype>() as i64) as isize,
                        )
                    } as *mut $vec_type;
                    let $to_store = unsafe { $c_local[MR as usize].as_ptr().offset(NR as isize * $multiple_of) };
                    $($stores)*;
                });
            }
        );
    };
}

/// Define load vector macro for neon
#[inline(always)]
#[cfg(all(target_feature = "neon", target_arch = "aarch64"))]
pub(crate) fn load_vec_neon<const NR: usize>(data: *const f32, mut to_write: *mut f32) {
    use std::{ arch::aarch64::*, mem::transmute };
    use crate::vec_size;
    use crate::F32Vec;
    unsafe {
        match NR {
            1 => {
                let res = vld1q_f32(data);
                *(to_write as *mut [F32Vec; 1]) = transmute(res);
            }
            2 => {
                let res = vld1q_f32_x2(data);
                *(to_write as *mut [F32Vec; 2]) = transmute(res);
            }
            3 => {
                let res = vld1q_f32_x3(data);
                *(to_write as *mut [F32Vec; 3]) = transmute(res);
            }
            4 => {
                let res = vld1q_f32_x4(data);
                *(to_write as *mut [F32Vec; 4]) = transmute(res);
            }
            5 => {
                let res = vld1q_f32(data);
                *(to_write as *mut [F32Vec; 1]) = transmute(res);
                to_write = to_write.add(vec_size::<f32>());
                let res = vld1q_f32_x4(data.add(vec_size::<f32>()));
                *(to_write as *mut [F32Vec; 4]) = transmute(res);
            }
            6 => {
                let res = vld1q_f32_x2(data);
                *(to_write as *mut [F32Vec; 2]) = transmute(res);
                to_write = to_write.add(vec_size::<f32>() * 2);
                let res = vld1q_f32_x4(data.add(vec_size::<f32>() * 2));
                *(to_write as *mut [F32Vec; 4]) = transmute(res);
            }
            7 => {
                let res = vld1q_f32_x3(data);
                *(to_write as *mut [F32Vec; 3]) = transmute(res);
                to_write = to_write.add(vec_size::<f32>() * 3);
                let res = vld1q_f32_x4(data.add(vec_size::<f32>() * 3));
                *(to_write as *mut [F32Vec; 4]) = transmute(res);
            }
            8 => {
                let res = vld1q_f32_x4(data);
                *(to_write as *mut [F32Vec; 4]) = transmute(res);
                to_write = to_write.add(vec_size::<f32>() * 4);
                let res = vld1q_f32_x4(data.add(vec_size::<f32>() * 4));
                *(to_write as *mut [F32Vec; 4]) = transmute(res);
            }
            9 => {
                let res = vld1q_f32_x4(data);
                *(to_write as *mut [F32Vec; 4]) = transmute(res);
                to_write = to_write.add(vec_size::<f32>() * 4);
                let res = vld1q_f32_x4(data.add(vec_size::<f32>() * 4));
                *(to_write as *mut [F32Vec; 4]) = transmute(res);
                to_write = to_write.add(vec_size::<f32>() * 4);
                let res = vld1q_f32(data.add(vec_size::<f32>() * 8));
                *(to_write as *mut [F32Vec; 1]) = transmute(res);
            }
            10 => {
                let res = vld1q_f32_x4(data);
                *(to_write as *mut [F32Vec; 4]) = transmute(res);
                to_write = to_write.add(vec_size::<f32>() * 4);
                let res = vld1q_f32_x4(data.add(vec_size::<f32>() * 4));
                *(to_write as *mut [F32Vec; 4]) = transmute(res);
                to_write = to_write.add(vec_size::<f32>() * 4);
                let res = vld1q_f32_x2(data.add(vec_size::<f32>() * 8));
                *(to_write as *mut [F32Vec; 2]) = transmute(res);
            }
            _ => { unreachable!() }
        }
    }
}

/// Define store result scalar macro
#[macro_export]
macro_rules! store_res_scalar {
    (
        $mr:expr,
        $nr:expr,
        $c_ptr:expr => $res_ty:ty,
        $ldc:expr,
        $c_local:expr => $c_ty:ty,
        $jb:expr,
        $nn:ident,
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

/// Define Matmul Micro Kernel function
///
/// # Arguments
///
/// * `$name`: The name of the function
/// * `$nr`: The number of registers to use for the columns of B
/// * `$mr`: The number of rows of C to accumulate in registers
///
/// Total registers = ($mr + 1) * $nr + 1
#[macro_export]
macro_rules! define_matmul_micro_kernel {
    ($dtype:ty, $vec_type:ty, $name:ident, $nr:expr, $mr:expr) => {
        fn $name(
            a: $crate::Pointer<$dtype>,
            b: $crate::Pointer<$dtype>,
            c: $crate::Pointer<$dtype>,
            ldc: i64,
            lda: i64,
            kc: usize,
            jb: usize,
            ks: i64,
            first_kiter: bool,
        ) {
            use crate::define_mma;
            use crate::define_mma_packed_a;
            use crate::store_res_vec;
            use crate::store_res_scalar;
            use crate::vec_size;
            define_mma!($dtype, $vec_type, $nr, $mr, 4);
            define_mma_packed_a!($dtype, $vec_type, $nr, $mr, 4);

            let c_local = if lda == 1 && ks == $mr {
                mma_packed_a(a, b, kc)
            }else {
                mma(a, b, lda, kc, ks)
            };
            if jb == $nr * vec_size::<$dtype>() {
                if first_kiter {
                    store_res_vec!(
                        $dtype,
                        $vec_type,
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
                        $dtype,
                        $vec_type,
                        $mr,
                        $nr,
                        c,
                        ldc,
                        c_local,
                        [res_ptr <= c_vec],
                        res_ptr.write_unaligned(res_ptr.read_unaligned() + c_vec)
                    );
                }
            } else {
                if first_kiter {
                    store_res_scalar!(
                        $mr, 
                        $nr, 
                        c => $dtype, 
                        ldc, 
                        c_local => $dtype, 
                        jb,
                        nn,
                        [res_ptr <= c_val], 
                        unsafe { *res_ptr = c_val }
                    );
                } else {
                    store_res_scalar!(
                        $mr, 
                        $nr, 
                        c => $dtype, 
                        ldc, 
                        c_local => $dtype, 
                        jb,
                        nn,
                        [res_ptr <= c_val], 
                        unsafe { *res_ptr = (*res_ptr).add(c_val) }
                    );
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
#[macro_export]
macro_rules! define_post_op_matmul_micro_kernel {
    ($dtype:ty, $vec_type:ty, $name:ident, $nr:expr, $mr:expr) => {
        fn $name<F: Fn($dtype, usize, usize) -> $dtype, G: Fn($vec_type, usize, usize) -> $vec_type>(
            a: $crate::Pointer<$dtype>,
            b: $crate::Pointer<$dtype>,
            c: $crate::Pointer<$dtype>,
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
            use crate::define_mma;
            use crate::define_mma_packed_a;
            use crate::store_res_vec;
            use crate::store_res_scalar;
            use crate::vec_size;
            define_mma!($dtype, $vec_type, $nr, $mr, 4);
            define_mma_packed_a!($dtype, $vec_type, $nr, $mr, 4);

            let c_local = if lda == 1 && ks == $mr {
                mma_packed_a(a, b, kc)
            }else {
                mma(a, b, lda, kc, ks)
            };
            if jb == $nr * vec_size::<$dtype>() {
                match (first_kiter, last_kiter) {
                    (true, true) => {
                        store_res_vec!(
                            $dtype,
                            $vec_type,
                            $mr,
                            $nr,
                            c,
                            ldc,
                            c_local,
                            [res_ptr <= c_vec],
                            res_ptr.write_unaligned(post_op_vec(c_vec, m_idx + MR, n_idx + NR * vec_size::<$dtype>()))
                        );
                    }
                    (true, false) => {
                        store_res_vec!(
                            $dtype,
                            $vec_type,
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
                            $dtype,
                            $vec_type,
                            $mr,
                            $nr,
                            c,
                            ldc,
                            c_local,
                            [res_ptr <= c_vec],
                            res_ptr.write_unaligned(post_op_vec(res_ptr.read_unaligned() + c_vec, m_idx + MR, n_idx + NR * vec_size::<$dtype>()))
                        );
                    }
                    (false, false) => {
                        store_res_vec!(
                            $dtype,
                            $vec_type,
                            $mr,
                            $nr,
                            c,
                            ldc,
                            c_local,
                            [res_ptr <= c_vec],
                            res_ptr.write_unaligned(res_ptr.read_unaligned() + c_vec)
                        );
                    },
                }
            } else {
                match (first_kiter, last_kiter) {
                    (true, true) => {
                        store_res_scalar!($mr, $nr, c => $dtype, ldc, c_local => $dtype, jb, nn, [res_ptr <= c_val], unsafe { *res_ptr = _post_op(c_val, m_idx + MR, n_idx + nn as usize) });
                    }
                    (true, false) => {
                        store_res_scalar!($mr, $nr, c => $dtype, ldc, c_local => $dtype, jb, nn, [res_ptr <= c_val], unsafe { *res_ptr = c_val });
                    }
                    (false, true) => {
                        store_res_scalar!($mr, $nr, c => $dtype, ldc, c_local => $dtype, jb, nn, [res_ptr <= c_val], unsafe { *res_ptr = _post_op((*res_ptr).add(c_val), m_idx + MR, n_idx + nn as usize) });
                    }
                    (false, false) => {
                        store_res_scalar!($mr, $nr, c => $dtype, ldc, c_local => $dtype, jb, nn, [res_ptr <= c_val], unsafe { *res_ptr = c_val.add(c_val) });
                    },
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
#[cfg(target_feature = "neon")]
#[macro_export]
macro_rules! define_neon_matmul_micro_kernel {
    ($dtype:ty, $vec_type:ty, $name:ident, $nr:expr, $mr:expr) => {
        fn $name(
            a: $crate::Pointer<$dtype>,
            b: $crate::Pointer<$dtype>,
            c: $crate::Pointer<$dtype>,
            ldc: i64,
            _: i64,
            kc: usize,
            jb: usize,
            ks: i64,
            first_kiter: bool,
        ) {
            use $crate::define_neon_mma;
            use $crate::store_res_vec;
            use $crate::store_res_scalar;
            use crate::vec_size;
            define_neon_mma!($dtype, $vec_type, $nr, $mr);
            
            let c_local = mma(a, b, kc, ks);
            if jb == $nr * vec_size::<$dtype>() {
                if first_kiter {
                    store_res_vec!(
                        $dtype,
                        $vec_type,
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
                        $dtype,
                        $vec_type,
                        $mr,
                        $nr,
                        c,
                        ldc,
                        c_local,
                        [res_ptr <= c_vec],
                        res_ptr.write_unaligned(res_ptr.read_unaligned() + c_vec)
                    );
                }
            } else {
                if first_kiter {
                    store_res_scalar!($mr, $nr, c => $dtype, ldc, c_local => $dtype, jb, nn, [res_ptr <= c_val], unsafe { *res_ptr = c_val });
                } else {
                    store_res_scalar!($mr, $nr, c => $dtype, ldc, c_local => $dtype, jb, nn, [res_ptr <= c_val], unsafe { *res_ptr = (*res_ptr).add(c_val) });
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
#[cfg(target_feature = "neon")]
#[macro_export]
macro_rules! define_neon_post_op_matmul_micro_kernel {
    ($dtype:ty, $vec_type:ty, $name:ident, $nr:expr, $mr:expr) => {
        fn $name<F: Fn($dtype, usize, usize) -> $dtype, G: Fn($vec_type, usize, usize) -> $vec_type>(
            a: $crate::Pointer<$dtype>,
            b: $crate::Pointer<$dtype>,
            c: $crate::Pointer<$dtype>,
            ldc: i64,
            _: i64,
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
            use $crate::define_neon_mma;
            use $crate::store_res_vec;
            use $crate::store_res_scalar;
            use crate::vec_size;
            define_neon_mma!($dtype, $vec_type, $nr, $mr);

            let c_local = mma(a, b, kc, ks);
            if jb == $nr * vec_size::<$dtype>() {
                match (first_kiter, last_kiter) {
                    (true, true) => {
                        store_res_vec!(
                            $dtype,
                            $vec_type,
                            $mr,
                            $nr,
                            c,
                            ldc,
                            c_local,
                            [res_ptr <= c_vec],
                            res_ptr.write_unaligned(post_op_vec(c_vec, m_idx + MR, n_idx + NR * vec_size::<$dtype>()))
                        );
                    }
                    (true, false) => {
                        store_res_vec!(
                            $dtype,
                            $vec_type,
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
                            $dtype,
                            $vec_type,
                            $mr,
                            $nr,
                            c,
                            ldc,
                            c_local,
                            [res_ptr <= c_vec],
                            res_ptr.write_unaligned(post_op_vec(res_ptr.read_unaligned() + c_vec, m_idx + MR, n_idx + NR * vec_size::<$dtype>()))
                        );
                    }
                    (false, false) => {
                        store_res_vec!(
                            $dtype,
                            $vec_type,
                            $mr,
                            $nr,
                            c,
                            ldc,
                            c_local,
                            [res_ptr <= c_vec],
                            res_ptr.write_unaligned(res_ptr.read_unaligned() + c_vec)
                        );
                    },
                }
            } else {
                match (first_kiter, last_kiter) {
                    (true, true) => {
                        store_res_scalar!($mr, $nr, c => $dtype, ldc, c_local => $dtype, jb, nn, [res_ptr <= c_val], unsafe { *res_ptr = _post_op(c_val, m_idx + MR, n_idx + nn as usize) });
                    }
                    (true, false) => {
                        store_res_scalar!($mr, $nr, c => $dtype, ldc, c_local => $dtype, jb, nn, [res_ptr <= c_val], unsafe { *res_ptr = c_val });
                    }
                    (false, true) => {
                        store_res_scalar!($mr, $nr, c => $dtype, ldc, c_local => $dtype, jb, nn, [res_ptr <= c_val], unsafe { *res_ptr = _post_op((*res_ptr).add(c_val), m_idx + MR, n_idx + nn as usize) });
                    }
                    (false, false) => {
                        store_res_scalar!($mr, $nr, c => $dtype, ldc, c_local => $dtype, jb, nn, [res_ptr <= c_val], unsafe { *res_ptr = c_val.add(c_val) });
                    },
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
#[macro_export]
macro_rules! define_mixed_precision_matmul_micro_kernel {
    (
        $dtype:ty,
        $vec_type:ty,
        $im:ty,
        $im_vec:ty,
        $name:ident,
        $nr:expr,
        $mr:expr,
        $nr2:expr,
        $multiple_of:expr
    ) => {
        fn $name(
            a: $crate::Pointer<$im>,
            b: $crate::Pointer<$im>,
            c: $crate::Pointer<$dtype>,
            ldc: i64,
            lda: i64,
            kc: usize,
            jb: usize,
            ks: i64,
            first_kiter: bool,
            vec_cast_back: fn(*mut $vec_type, *const $im_vec),
            cast_back: fn(&mut $dtype, &$im),
        ) {
            use $crate::VecTrait;
            use $crate::define_mma;
            use $crate::define_mma_packed_a;
            use $crate::store_res_vec_mp;
            use $crate::store_res_scalar;
            use crate::vec_size;
            define_mma!($im, $im_vec, $nr2, $mr, 4);
            define_mma_packed_a!($im, $im_vec, $nr2, $mr, 4);

            let c_local = if lda == 1 && ks == $mr {
                mma_packed_a(a, b, kc)
            }else {
                mma(a, b, lda, kc, ks)
            };
            if jb == $nr * vec_size::<$dtype>() {
                if first_kiter {
                    store_res_vec_mp!(
                        $dtype,
                        $vec_type,
                        $multiple_of,
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
                        $dtype,
                        $vec_type,
                        $multiple_of,
                        $mr,
                        $nr,
                        c,
                        ldc,
                        c_local,
                        [res_ptr <= c_vec],
                        let mut res_vec = $vec_type::splat(<$dtype>::ZERO);
                        vec_cast_back(&mut res_vec, c_vec);
                        unsafe { res_ptr.write_unaligned(res_vec + res_ptr.read_unaligned()) }
                    );
                }
            } else {
                if first_kiter {
                    store_res_scalar!(
                        $mr,
                        $nr,
                        c => $dtype,
                        ldc,
                        c_local => $im,
                        jb,
                        nn,
                        [res_ptr <= c_val],
                        unsafe { cast_back(&mut *res_ptr, &c_val) }
                    );
                } else {
                    store_res_scalar!(
                        $mr,
                        $nr,
                        c => $dtype,
                        ldc,
                        c_local => $im,
                        jb,
                        nn,
                        [res_ptr <= c_val],
                        let mut res = <$dtype>::ZERO;
                        cast_back(&mut res, &c_val);
                        unsafe { *res_ptr = (*res_ptr) + res }
                    );
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
#[macro_export]
macro_rules! define_mixed_precision_post_op_matmul_micro_kernel {
    (
        $dtype:ty,
        $vec_type:ty,
        $im:ty,
        $im_vec:ty,
        $name:ident,
        $nr:expr,
        $mr:expr,
        $nr2:expr,
        $multiple_of:expr
    ) => {
        fn $name<F: Fn($dtype, usize, usize) -> $dtype, F2: Fn($vec_type, usize, usize) -> $vec_type>(
            a: $crate::Pointer<$im>,
            b: $crate::Pointer<$im>,
            c: $crate::Pointer<$dtype>,
            ldc: i64,
            lda: i64,
            kc: usize,
            jb: usize,
            ks: i64,
            first_kiter: bool,
            last_kiter: bool,
            m_idx: usize,
            n_idx: usize,
            vec_cast_back: fn(*mut $vec_type, *const $im_vec),
            cast_back: fn(&mut $dtype, &$im),
            post_op: F,
            post_op_vec: F2
        ) {
            use $crate::VecTrait;
            use $crate::define_mma;
            use $crate::define_mma_packed_a;
            use $crate::store_res_vec_mp;
            use $crate::store_res_scalar;
            use crate::vec_size;
            define_mma!($im, $im_vec, $nr2, $mr, 4);
            define_mma_packed_a!($im, $im_vec, $nr2, $mr, 4);

            let c_local = if lda == 1 && ks == $mr {
                mma_packed_a(a, b, kc)
            }else {
                mma(a, b, lda, kc, ks)
            };
            if jb == $nr * vec_size::<$dtype>() {
                match (first_kiter, last_kiter) {
                    (true, true) => {
                        store_res_vec_mp!(
                            $dtype,
                            $vec_type,
                            $multiple_of,
                            $mr,
                            $nr,
                            c,
                            ldc,
                            c_local,
                            [res_ptr <= c_vec],
                            let mut res = $vec_type::splat(<$dtype>::ZERO);
                            vec_cast_back(&mut res, c_vec);
                            unsafe { res_ptr.write_unaligned(post_op_vec(res, m_idx + MR, n_idx + NR * vec_size::<$dtype>())) }
                        );
                    }
                    (true, false) => {
                        store_res_vec_mp!(
                            $dtype,
                            $vec_type,
                            $multiple_of,
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
                            $dtype,
                            $vec_type,
                            $multiple_of,
                            $mr,
                            $nr,
                            c,
                            ldc,
                            c_local,
                            [res_ptr <= c_vec],
                            let mut res = $vec_type::splat(<$dtype>::ZERO);
                            vec_cast_back(&mut res, c_vec);
                            unsafe { res_ptr.write_unaligned(post_op_vec(res + res_ptr.read_unaligned(), m_idx + MR, n_idx + NR * vec_size::<$dtype>())) }
                        );
                    }
                    (false, false) => {
                        store_res_vec_mp!(
                            $dtype,
                            $vec_type,
                            $multiple_of,
                            $mr,
                            $nr,
                            c,
                            ldc,
                            c_local,
                            [res_ptr <= c_vec],
                            let mut res = $vec_type::splat(<$dtype>::ZERO);
                            vec_cast_back(&mut res, c_vec);
                            unsafe { res_ptr.write_unaligned(res + res_ptr.read_unaligned()) }
                        );
                    },
                }
            } else {
                match (first_kiter, last_kiter) {
                    (true, true) => {
                        store_res_scalar!(
                            $mr,
                            $nr,
                            c => $dtype,
                            ldc,
                            c_local => $im,
                            jb,
                            nn,
                            [res_ptr <= c_val],
                            let mut res = <$dtype>::ZERO;
                            cast_back(&mut res, &c_val);
                            unsafe { res_ptr.write(post_op(res, m_idx + MR, n_idx + nn as usize)) }
                        );
                    }
                    (true, false) => {
                        store_res_scalar!(
                            $mr,
                            $nr,
                            c => $dtype,
                            ldc,
                            c_local => $im,
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
                            c => $dtype,
                            ldc,
                            c_local => $im,
                            jb,
                            nn,
                            [res_ptr <= c_val],
                            let mut res = <$dtype>::ZERO;
                            cast_back(&mut res, &c_val);
                            unsafe { *res_ptr = post_op(res + *res_ptr, m_idx + MR, n_idx + nn as usize) }
                        );
                    }
                    (false, false) => {
                        store_res_scalar!(
                            $mr,
                            $nr,
                            c => $dtype,
                            ldc,
                            c_local => $im,
                            jb,
                            nn,
                            [res_ptr <= c_val],
                            let mut res = <$dtype>::ZERO;
                            cast_back(&mut res, &c_val);
                            unsafe { *res_ptr = (*res_ptr) + res }
                        );
                    },
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
#[macro_export]
#[cfg(target_feature = "neon")]
macro_rules! define_neon_mixed_precision_matmul_micro_kernel {
    (
        $dtype:ty,
        $vec_type:ty,
        $im:ty,
        $im_vec:ty,
        $name:ident,
        $nr:expr,
        $mr:expr,
        $nr2:expr,
        $multiple_of:expr
    ) => {
        fn $name(
            a: $crate::Pointer<$im>,
            b: $crate::Pointer<$im>,
            c: $crate::Pointer<$dtype>,
            ldc: i64,
            _: i64,
            kc: usize,
            jb: usize,
            ks: i64,
            first_kiter: bool,
            vec_cast_back: fn(*mut $vec_type, *const $im_vec),
            cast_back: fn(&mut $dtype, &$im),
        ) {
            use $crate::define_neon_mma;
            use $crate::store_res_vec_mp;
            use $crate::store_res_scalar;
            use crate::vec_size;
            use crate::VecTrait;
            define_neon_mma!($im, $im_vec, $nr2, $mr);

            let c_local = mma(a, b, kc, ks);
            if jb == $nr * vec_size::<$dtype>() {
                match first_kiter {
                    true => {
                        store_res_vec_mp!(
                            $dtype,
                            $vec_type,
                            $multiple_of,
                            $mr,
                            $nr,
                            c,
                            ldc,
                            c_local,
                            [res_ptr <= c_vec],
                            let mut res = $vec_type::splat(<$dtype>::ZERO);
                            vec_cast_back(&mut res, c_vec);
                            unsafe { res_ptr.write_unaligned(res) }
                        );
                    }
                    false => {
                        store_res_vec_mp!(
                            $dtype,
                            $vec_type,
                            $multiple_of,
                            $mr,
                            $nr,
                            c,
                            ldc,
                            c_local,
                            [res_ptr <= c_vec],
                            let mut res = $vec_type::splat(<$dtype>::ZERO);
                            vec_cast_back(&mut res, c_vec);
                            unsafe { res_ptr.write_unaligned(res + res_ptr.read_unaligned()) }
                        );
                    }
                }
            } else {
                match first_kiter {
                    true => {
                        store_res_scalar!(
                            $mr,
                            $nr,
                            c => $dtype,
                            ldc,
                            c_local => $im,
                            jb,
                            nn,
                            [res_ptr <= c_val],
                            unsafe { cast_back(&mut *res_ptr, &c_val) }
                        );
                    }
                    false => {
                        store_res_scalar!(
                            $mr,
                            $nr,
                            c => $dtype,
                            ldc,
                            c_local => $im,
                            jb,
                            nn,
                            [res_ptr <= c_val],
                            let mut res = <$dtype>::ZERO;
                            cast_back(&mut res, &c_val);
                            unsafe { res_ptr.write(res + *res_ptr) }
                        );
                    },
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
#[macro_export]
#[cfg(target_feature = "neon")]
macro_rules! define_neon_mixed_precision_post_op_matmul_micro_kernel {
    (
        $dtype:ty,
        $vec_type:ty,
        $im:ty,
        $im_vec:ty,
        $name:ident,
        $nr:expr,
        $mr:expr,
        $nr2:expr,
        $multiple_of:expr
    ) => {
        fn $name<F: Fn($dtype, usize, usize) -> $dtype, F2: Fn($vec_type, usize, usize) -> $vec_type>(
            a: $crate::Pointer<$im>,
            b: $crate::Pointer<$im>,
            c: $crate::Pointer<$dtype>,
            ldc: i64,
            _: i64,
            kc: usize,
            jb: usize,
            ks: i64,
            first_kiter: bool,
            last_kiter: bool,
            m_idx: usize,
            n_idx: usize,
            vec_cast_back: fn(*mut $vec_type, *const $im_vec),
            cast_back: fn(&mut $dtype, &$im),
            post_op: F,
            post_op_vec: F2,
        ) {
            use $crate::define_neon_mma;
            use crate::vec_size;
            use crate::VecTrait;
            define_neon_mma!($im, $im_vec, $nr2, $mr);

            let c_local = mma(a, b, kc, ks);
            if jb == $nr * vec_size::<$dtype>() {
                if first_kiter {
                    if last_kiter {
                        seq_macro::seq!(MR in 0..$mr {
                            seq_macro::seq!(NR in 0..$nr {
                                    let res_ptr = unsafe {
                                        c.ptr.offset(
                                            (MR * ldc + NR * vec_size::<$dtype>() as i64) as isize,
                                        )
                                    } as *mut $vec_type;
                                    let mut res = $vec_type::splat(<$dtype>::ZERO);
                                    vec_cast_back(&mut res, unsafe { c_local[MR as usize].as_ptr().offset(NR as isize * $multiple_of) });
                                    unsafe {res_ptr.write_unaligned(post_op_vec(res, m_idx + MR, n_idx + NR * vec_size::<$dtype>())) };
                                    }
                                );
                            }
                        );
                    } else {
                        seq_macro::seq!(MR in 0..$mr {
                            seq_macro::seq!(NR in 0..$nr {
                                    let res_ptr = unsafe {
                                        c.ptr.offset(
                                            (MR * ldc + NR * vec_size::<$dtype>() as i64) as isize,
                                        )
                                    } as *mut $vec_type;
                                    let mut res = $vec_type::splat(<$dtype>::ZERO);
                                    vec_cast_back(&mut res, unsafe { c_local[MR as usize].as_ptr().offset(NR as isize * $multiple_of) });
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
                                            (MR * ldc + NR * vec_size::<$dtype>() as i64) as isize,
                                        )
                                    } as *mut $vec_type;
                                    res~NR = unsafe {res_ptr.read_unaligned()};
                                    let mut res = $vec_type::splat(<$dtype>::ZERO);
                                    vec_cast_back(&mut res, unsafe { c_local[MR as usize].as_ptr().offset(NR as isize * $multiple_of) });
                                    unsafe {res_ptr.write_unaligned(post_op_vec(res~NR + res, m_idx + MR, n_idx + NR * vec_size::<$dtype>())) };
                                    }
                                );
                            }
                        );
                    } else {
                        seq_macro::seq!(MR in 0..$mr {
                            seq_macro::seq!(NR in 0..$nr {
                                    let res_ptr = unsafe {
                                        c.ptr.offset(
                                            (MR * ldc + NR * vec_size::<$dtype>() as i64) as isize,
                                        )
                                    } as *mut $vec_type;
                                    res~NR = unsafe {res_ptr.read_unaligned()};
                                    let mut res = $vec_type::splat(<$dtype>::ZERO);
                                    vec_cast_back(&mut res, unsafe { c_local[MR as usize].as_ptr().offset(NR as isize * $multiple_of) });
                                    unsafe {res_ptr.write_unaligned(res~NR + res) };
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
                            let c_local_ptr = c_local[ii as usize].as_ptr() as *const $im;
                            for jj in 0..jb as i64 {
                                let res_ptr = unsafe { c.ptr.offset((ii * ldc + jj) as isize) } as *mut $dtype;
                                let mut res = <$dtype>::ZERO;
                                unsafe {
                                    cast_back(&mut res, &c_local_ptr.offset(jj as isize).read());
                                    *res_ptr = post_op(res, m_idx + ii as usize, n_idx + jj as usize);
                                };
                            }
                        }
                    } else {
                        for ii in 0..$mr as i64 {
                            let c_local_ptr = c_local[ii as usize].as_ptr() as *const $im;
                            for jj in 0..jb as i64 {
                                let res_ptr = unsafe { c.ptr.offset((ii * ldc + jj) as isize) } as *mut $dtype;
                                let mut res = <$dtype>::ZERO;
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
                            let c_local_ptr = c_local[ii as usize].as_ptr() as *const $im;
                            for jj in 0..jb as i64 {
                                let res_ptr = unsafe { c.ptr.offset((ii * ldc + jj) as isize) } as *mut $dtype;
                                let mut res = <$dtype>::ZERO;
                                unsafe {
                                    cast_back(&mut res, &c_local_ptr.offset(jj as isize).read());
                                    *res_ptr = post_op((*res_ptr) + res, m_idx + ii as usize, n_idx + jj as usize);
                                };
                            }
                        }
                    } else {
                        for ii in 0..$mr as i64 {
                            let c_local_ptr = c_local[ii as usize].as_ptr() as *const $im;
                            for jj in 0..jb as i64 {
                                let res_ptr = unsafe { c.ptr.offset((ii * ldc + jj) as isize) } as *mut $dtype;
                                let mut res = <$dtype>::ZERO;
                                unsafe {
                                    cast_back(&mut res, &c_local_ptr.offset(jj as isize).read());
                                    *res_ptr = (*res_ptr) + res;
                                };
                            }
                        }
                    }
                }
            }
        }
    };
}

macro_rules! gemv_microkernel_impl {
    ($nr:expr) => {
        pub(crate) fn gemv_microkernel<T, SelfVec>(
            a: crate::Pointer<T>,
            b: crate::Pointer<T>,
            c: crate::Pointer<T>,
            n: usize,
            k: usize,
            ldb: i64,
            lhs_col_stride: i64
        )
            where SelfVec: crate::VecTrait<T> + std::ops::Mul<Output = SelfVec> + Copy, T: Copy + crate::Zero
        {
            use crate::vec_size;
            let nc = $nr * vec_size::<T>();
            let rem = n % nc;
            for j in (0..n).step_by(nc) {
                let jb = nc.min(n - j);
                let b = b + j as i64;
                if jb == nc {
                    let mut local_c = [SelfVec::splat(<T>::ZERO); $nr];
                    let c_vec_ptr = (c + j as i64).cast::<SelfVec>();
                    for p in 0..k {
                        let a_vec = <SelfVec>::splat(a[(p as i64) * lhs_col_stride]);
                        let b = (b + (p as i64) * ldb).cast::<SelfVec>();
                        seq_macro::seq!(NR in 0..$nr {
                            let b_vec~NR = b.offset(NR).read_unaligned();
                        });
                        seq_macro::seq!(NR in 0..$nr {
                            local_c[NR] = a_vec.mul_add(b_vec~NR, local_c[NR]);
                        });
                    }
                    seq_macro::seq!(NR in 0..$nr {
                        c_vec_ptr.offset(NR).write_unaligned(local_c[NR]);
                    });
                } else {
                    let mut local_c = [SelfVec::splat(<T>::ZERO); $nr];
                    let mut b_vec = [SelfVec::splat(<T>::ZERO); $nr];
                    let c_ptr = c + j as i64;
                    for p in 0..k {
                        let a_vec = <SelfVec>::splat(a[(p as i64) * lhs_col_stride]);
                        let b = b + (p as i64) * ldb;
                        unsafe {
                            std::ptr::copy_nonoverlapping(b.ptr, b_vec.as_mut_ptr() as *mut _ as *mut T, rem);
                        }
                        seq_macro::seq!(NR in 0..$nr {
                            local_c[NR] = a_vec.mul_add(b_vec[NR], local_c[NR]);
                        });
                    }
                    unsafe {
                        std::ptr::copy_nonoverlapping(local_c.as_ptr() as *const _ as *const T, c_ptr.ptr as *mut T, rem);
                    }
                }
            }
        }
    };
}

pub(crate) use gemv_microkernel_impl;

macro_rules! gemv_microkernel_mp_impl {
    ($nr:expr, $nr2:expr, $multiple_of:expr) => {
        pub(crate) fn gemv_microkernel_mp<T, IM, SelfVec, IMVec>(
            a: crate::Pointer<IM>,
            b: crate::Pointer<IM>,
            c: crate::Pointer<T>,
            n: usize,
            k: usize,
            ldb: i64,
            lhs_col_stride: i64,
            vec_cast_back: fn(*mut SelfVec, *const IMVec),
            cast_back: fn(&mut T, &IM),
        )
            where SelfVec: crate::VecTrait<T> + std::ops::Mul<Output = SelfVec> + Copy, T: Copy + crate::Zero,
            IMVec: crate::VecTrait<IM> + std::ops::Mul<Output = IMVec> + Copy, IM: Copy + crate::Zero
        {
            use crate::vec_size;
            let nc = $nr2 * vec_size::<IM>();
            let rem = n % nc;
            for j in (0..n).step_by(nc) {
                let jb = nc.min(n - j);
                let b = b + j as i64;
                if jb == nc {
                    let mut local_c = [IMVec::splat(<IM>::ZERO); $nr2];
                    let c_vec_ptr = (c + j as i64).cast::<SelfVec>();
                    for p in 0..k {
                        let a_vec = <IMVec>::splat(a[(p as i64) * lhs_col_stride]);
                        let b = (b + (p as i64) * ldb).cast::<IMVec>();
                        seq_macro::seq!(NR in 0..$nr2 {
                            let b_vec~NR = b.offset(NR).read_unaligned();
                        });
                        seq_macro::seq!(NR in 0..$nr2 {
                            local_c[NR] = a_vec.mul_add(b_vec~NR, local_c[NR]);
                        });
                    }
                    let local_c_ptr = local_c.as_ptr() as *const _ as *const IMVec;
                    seq_macro::seq!(NR in 0..$nr {
                        vec_cast_back(c_vec_ptr.offset(NR).ptr, unsafe { local_c_ptr.offset(NR * $multiple_of) });
                    });
                } else {
                    let mut local_c = [IMVec::splat(<IM>::ZERO); $nr2];
                    let mut b_vec = [IMVec::splat(<IM>::ZERO); $nr2];
                    let c_ptr = c + j as i64;
                    for p in 0..k {
                        let a_vec = <IMVec>::splat(a[(p as i64) * lhs_col_stride]);
                        let b = b + (p as i64) * ldb;
                        unsafe {
                            std::ptr::copy_nonoverlapping(b.ptr, b_vec.as_mut_ptr() as *mut _ as *mut IM, rem);
                        }
                        seq_macro::seq!(NR in 0..$nr2 {
                            local_c[NR] = a_vec.mul_add(b_vec[NR], local_c[NR]);
                        });
                    }
                    let local_c_ptr = local_c.as_ptr() as *const _ as *const IM;
                    for ii in 0..rem {
                        cast_back(&mut *c_ptr.offset(ii as i64), &unsafe { local_c_ptr.offset(ii as isize).read() });
                    }
                }
            }
        }
    };
}

pub(crate) use gemv_microkernel_mp_impl;

macro_rules! gemv_microkernel_post_op_impl {
    ($nr:expr) => {
        pub(crate) fn gemv_microkernel_post_op<T, SelfVec, F, F2>(
            a: crate::Pointer<T>,
            b: crate::Pointer<T>,
            c: crate::Pointer<T>,
            n: usize,
            k: usize,
            ldb: i64,
            lhs_col_stride: i64,
            m_offset: usize,
            n_offset: usize,
            post_op: F,
            post_op_vec: F2,
        )
            where SelfVec: crate::VecTrait<T> + std::ops::Mul<Output = SelfVec> + Copy, T: Copy + crate::Zero,
            F: Fn(T, usize, usize) -> T,
            F2: Fn(SelfVec, usize, usize) -> SelfVec
        {
            use crate::vec_size;
            let nc = $nr * vec_size::<T>();
            let rem = n % nc;
            for j in (0..n).step_by(nc) {
                let jb = nc.min(n - j);
                let b = b + j as i64;
                if jb == nc {
                    let mut local_c = [SelfVec::splat(<T>::ZERO); $nr];
                    let c_vec_ptr = (c + j as i64).cast::<SelfVec>();
                    for p in 0..k {
                        let a_vec = <SelfVec>::splat(a[(p as i64) * lhs_col_stride]);
                        let b = (b + (p as i64) * ldb).cast::<SelfVec>();
                        seq_macro::seq!(NR in 0..$nr {
                            let b_vec~NR = b.offset(NR).read_unaligned();
                        });
                        seq_macro::seq!(NR in 0..$nr {
                            local_c[NR] = a_vec.mul_add(b_vec~NR, local_c[NR]);
                        });
                    }
                    seq_macro::seq!(NR in 0..$nr {
                        c_vec_ptr.offset(NR).write_unaligned(post_op_vec(local_c[NR], m_offset, n_offset + j + NR * vec_size::<T>()));
                    });
                } else {
                    let mut local_c = [SelfVec::splat(<T>::ZERO); $nr];
                    let mut b_vec = [SelfVec::splat(<T>::ZERO); $nr];
                    let c_ptr = c + j as i64;
                    for p in 0..k {
                        let a_vec = <SelfVec>::splat(a[(p as i64) * lhs_col_stride]);
                        let b = b + (p as i64) * ldb;
                        unsafe {
                            std::ptr::copy_nonoverlapping(b.ptr, b_vec.as_mut_ptr() as *mut _ as *mut T, rem);
                        }
                        seq_macro::seq!(NR in 0..$nr {
                            local_c[NR] = a_vec.mul_add(b_vec[NR], local_c[NR]);
                        });
                    }
                    for jj in 0..rem {
                        *c_ptr.offset(jj as i64) = post_op(*c_ptr.offset(jj as i64), m_offset, n_offset + j + jj);
                    }
                }
            }
        }
    };
}

pub(crate) use gemv_microkernel_post_op_impl;

macro_rules! gemv_microkernel_mp_post_op_impl {
    ($nr:expr, $nr2:expr, $multiple_of:expr) => {
        pub(crate) fn gemv_microkernel_mp_post_op<T, IM, SelfVec, IMVec, F, F2>(
            a: crate::Pointer<IM>,
            b: crate::Pointer<IM>,
            c: crate::Pointer<T>,
            n: usize,
            k: usize,
            ldb: i64,
            lhs_col_stride: i64,
            m_offset: usize,
            n_offset: usize,
            vec_cast_back: fn(*mut SelfVec, *const IMVec),
            cast_back: fn(&mut T, &IM),
            post_op: F,
            post_op_vec: F2,
        )
            where SelfVec: crate::VecTrait<T> + std::ops::Mul<Output = SelfVec> + Copy, T: Copy + crate::Zero,
            IMVec: crate::VecTrait<IM> + std::ops::Mul<Output = IMVec> + Copy, IM: Copy + crate::Zero,
            F: Fn(T, usize, usize) -> T,
            F2: Fn(SelfVec, usize, usize) -> SelfVec
        {
            use crate::vec_size;
            let nc = $nr2 * vec_size::<IM>();
            let rem = n % nc;
            for j in (0..n).step_by(nc) {
                let jb = nc.min(n - j);
                let b = b + j as i64;
                if jb == nc {
                    let mut local_c = [IMVec::splat(<IM>::ZERO); $nr2];
                    let c_vec_ptr = (c + j as i64).cast::<SelfVec>();
                    for p in 0..k {
                        let a_vec = <IMVec>::splat(a[(p as i64) * lhs_col_stride]);
                        let b = (b + (p as i64) * ldb).cast::<IMVec>();
                        seq_macro::seq!(NR in 0..$nr2 {
                            let b_vec~NR = b.offset(NR).read_unaligned();
                        });
                        seq_macro::seq!(NR in 0..$nr2 {
                            local_c[NR] = a_vec.mul_add(b_vec~NR, local_c[NR]);
                        });
                    }
                    let local_c_ptr = local_c.as_ptr() as *const _ as *const IMVec;
                    seq_macro::seq!(NR in 0..$nr {
                        let ptr = c_vec_ptr.offset(NR).ptr;
                        vec_cast_back(ptr, unsafe { local_c_ptr.offset(NR * $multiple_of) });
                        unsafe {
                            ptr.write_unaligned(post_op_vec(ptr.read_unaligned(), m_offset, n_offset + j + NR * vec_size::<T>()));
                        }
                    });
                } else {
                    let mut local_c = [IMVec::splat(<IM>::ZERO); $nr2];
                    let mut b_vec = [IMVec::splat(<IM>::ZERO); $nr2];
                    let c_ptr = c + j as i64;
                    for p in 0..k {
                        let a_vec = <IMVec>::splat(a[(p as i64) * lhs_col_stride]);
                        let b = b + (p as i64) * ldb;
                        unsafe {
                            std::ptr::copy_nonoverlapping(b.ptr, b_vec.as_mut_ptr() as *mut _ as *mut IM, rem);
                        }
                        seq_macro::seq!(NR in 0..$nr2 {
                            local_c[NR] = a_vec.mul_add(b_vec[NR], local_c[NR]);
                        });
                    }
                    let local_c_ptr = local_c.as_ptr() as *const _ as *const IM;
                    for ii in 0..rem {
                        let mut ptr = c_ptr.offset(ii as i64);
                        cast_back(&mut *ptr, &unsafe { local_c_ptr.offset(ii as isize).read() });
                        ptr.write_unaligned(post_op(*ptr, m_offset, n_offset + j + ii));
                    }
                }
            }
        }
    };
}

pub(crate) use gemv_microkernel_mp_post_op_impl;
