use super::microkernel_trait::MicroKernel;
use hpt_common::Pointer;
use hpt_types::{dtype::TypeCommon, traits::VecTrait, type_promote::NormalOut};
use seq_macro::seq;

macro_rules! define_micro_kernel {
    ($name:ident, $T:ty, $nr:expr, $mr:expr) => {
        fn $name(
            a: Pointer<$T>,
            b: Pointer<$T>,
            c: Pointer<$T>,
            ldc: i64,
            lda: i64,
            kc: usize,
            jb: usize,
            ks: i64,
            first_kiter: bool,
        ) {
            #[inline(always)]
            fn mma(mut a: Pointer<f32>, mut b: Pointer<f32>, lda: i64, kc: usize, ks: i64) -> [[<f32 as TypeCommon>::Vec; 2]; 6] {
                let mut c_local = [[<f32 as TypeCommon>::Vec::splat(<f32>::ZERO); 2]; 6];
                for _ in 0..kc {
                    seq!(NR in 0..$nr {
                        let b_vec~NR = unsafe {
                            *(b.ptr.add(NR * <f32 as TypeCommon>::Vec::SIZE)
                                as *const <f32 as TypeCommon>::Vec)
                        };

                        }
                    );
                    #[allow(unused_mut)]
                    let mut a_vec;
                    seq!(MR in 0..$mr {
                            a_vec = <f32 as TypeCommon>::Vec::splat(a[MR as i64 * lda]);
                            seq!(NR in 0..$nr {
                                c_local[MR][NR] = a_vec._mul_add(b_vec~NR, c_local[MR][NR]);
                            });
                        }
                    );
                    b += $nr * <f32 as TypeCommon>::Vec::SIZE as i64;
                    a += ks;
                }
                c_local
            }

            let c_local = mma(a, b, lda, kc, ks);
            if jb == $nr * <$T as TypeCommon>::Vec::SIZE {
                if first_kiter {
                    seq!(MR in 0..$mr {
                            seq!(NR in 0..$nr {
                                let res_ptr = unsafe {
                                    c.ptr.offset(
                                        (MR * ldc + NR * <$T as TypeCommon>::Vec::SIZE as i64) as isize,
                                    )
                                } as *mut <$T as TypeCommon>::Vec;
                                unsafe {res_ptr.write_unaligned(c_local[MR as usize][NR as usize]) };
                                }
                            );
                        }
                    );
                } else {
                    seq!(NR in 0..$nr {
                        #[allow(unused_mut)]
                        let mut res~NR;
                        }
                    );
                    seq!(MR in 0..$mr {
                            seq!(NR in 0..$nr {
                                let res_ptr = unsafe {
                                    c.ptr.offset(
                                        (MR * ldc + NR * <$T as TypeCommon>::Vec::SIZE as i64) as isize,
                                    )
                                } as *mut <$T as TypeCommon>::Vec;
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
                        let c_local_ptr = c_local[ii as usize].as_ptr() as *const $T;
                        for jj in 0..jb as i64 {
                            let res_ptr = unsafe { c.ptr.offset((ii * ldc + jj) as isize) } as *mut $T;
                            unsafe {
                                *res_ptr = c_local_ptr.offset(jj as isize).read();
                            };
                        }
                    }
                } else {
                    for ii in 0..$mr as i64 {
                        let c_local_ptr = c_local[ii as usize].as_ptr() as *const $T;
                        for jj in 0..jb as i64 {
                            let res_ptr = unsafe { c.ptr.offset((ii * ldc + jj) as isize) } as *mut $T;
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

impl MicroKernel for f32 {
    #[cfg(target_feature = "avx2")]
    fn get_kernel(
        _: usize,
        mr: usize,
    ) -> fn(Pointer<Self>, Pointer<Self>, Pointer<Self>, i64, i64, usize, usize, i64, bool) {
        define_micro_kernel!(f32x2x1, f32, 2, 1);
        define_micro_kernel!(f32x2x2, f32, 2, 2);
        define_micro_kernel!(f32x2x3, f32, 2, 3);
        define_micro_kernel!(f32x2x4, f32, 2, 4);
        define_micro_kernel!(f32x2x5, f32, 2, 5);
        define_micro_kernel!(f32x2x6, f32, 2, 6);

        [f32x2x1, f32x2x2, f32x2x3, f32x2x4, f32x2x5, f32x2x6][mr - 1]
    }
    #[cfg(all(not(target_feature = "avx2"), target_feature = "sse"))]
    fn get_kernel(
        _: usize,
        _: usize,
    ) -> fn(Pointer<Self>, Pointer<Self>, Pointer<Self>, i64, i64, usize, usize, i64, bool) {
        define_micro_kernel!(f32x4x1, f32, 4, 1);
        f32x4x1
    }
    #[cfg(all(not(target_feature = "avx2"), target_feature = "sse"))]
    fn get_max_mr() -> usize {
        1
    }
    #[cfg(target_feature = "avx2")]
    fn get_max_mr() -> usize {
        6
    }
}
