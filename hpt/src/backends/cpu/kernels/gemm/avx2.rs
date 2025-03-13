use hpt_common::Pointer;
use hpt_types::{dtype::TypeCommon, traits::VecTrait, type_promote::NormalOut};
use unroll::unroll_for_loops;

use super::microkernel_trait::MicroKernel;

macro_rules! define_micro_kernel {
    ($name:ident, $T:ty, $nr:expr, $mr:expr) => {
        #[unroll_for_loops]
        pub(crate) fn $name(
            a: Pointer<$T>,
            mut b: Pointer<$T>,
            c: Pointer<$T>,
            ldc: i64,
            lda: i64,
            kc: usize,
            jb: usize,
        ) {
            let mut c_local = [[<$T as TypeCommon>::Vec::splat(<$T>::ZERO); $nr]; $mr];
            for k in 0..kc {
                for ii in 0..$mr {
                    let a_vec = <$T as TypeCommon>::Vec::splat(unsafe {
                        *(a.ptr as *const $T).offset(k as isize + ii as isize * lda as isize)
                    });
                    for jj in 0..$nr {
                        let b_vec = unsafe {
                            *(b.ptr.add(jj * <$T as TypeCommon>::Vec::SIZE)
                                as *const <$T as TypeCommon>::Vec)
                        };
                        c_local[ii][jj] = a_vec._mul_add(b_vec, c_local[ii][jj]);
                    }
                }
                b += $nr * <$T as TypeCommon>::Vec::SIZE as i64;
            }
            if jb == $nr * <$T as TypeCommon>::Vec::SIZE {
                for ii in 0..$mr as i64 {
                    for jj in 0..$nr as i64 {
                        let res_ptr = unsafe {
                            c.ptr.offset(
                                (ii * ldc + jj * <$T as TypeCommon>::Vec::SIZE as i64) as isize,
                            )
                        } as *mut <$T as TypeCommon>::Vec;
                        unsafe {
                            res_ptr.write_unaligned(c_local[ii as usize][jj as usize]);
                        };
                    }
                }
            } else {
                for ii in 0..$mr as i64 {
                    let c_local_ptr = c_local[ii as usize].as_ptr() as *const $T;
                    for jj in 0..jb as i64 {
                        let res_ptr = unsafe { c.ptr.offset((ii * ldc + jj) as isize) } as *mut $T;
                        unsafe {
                            *res_ptr = c_local_ptr.offset(jj as isize).read();
                        };
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
    ) -> fn(Pointer<Self>, Pointer<Self>, Pointer<Self>, i64, i64, usize, usize) {
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
    ) -> fn(Pointer<Self>, Pointer<Self>, Pointer<Self>, i64, i64, usize, usize) {
        define_micro_kernel!(f32x4x1, f32, 4, 1);
        f32x4x1
    }
}
