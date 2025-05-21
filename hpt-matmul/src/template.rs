use std::cmp::min;

use dyn_stack::DynStack;
use gemm_common::gemm::CACHELINE_ALIGN;

use crate::{
    Pointer, Zero,
    microkernel_trait::MatmulMicroKernel,
    utils::{
        L2_SLAB, L3_SLAB, PrePackedRhs, PrePackedLhs, pack_a_mixed_precision_single_thread,
        pack_a_single_thread, pack_b_mixed_precision, pack_b_single_thread,
    },
    vec_size,
};

#[duplicate::duplicate_item(
    func_name       mv_method       get_kernel                 extra_args;
    [matmul]        [mv]            [get_kernel]               [first_kiter];
    [matmul_post]   [mv_post]       [get_kernel_with_post_op]  [first_kiter,p + pb == k, m_offset + i + ii, n_offset + j + jj, post_op.clone(), post_op_vec.clone()];
)]
pub(crate) fn func_name<T, F1, F2>(
    a: Pointer<T>,
    b: Pointer<T>,
    out: Pointer<T>,
    #[allow(unused_variables)] m_offset: usize,
    #[allow(unused_variables)] n_offset: usize,
    m: usize,
    n: usize,
    k: usize,
    lda: i64,
    ldb: i64,
    ldc: i64,
    lhs_col_stride: i64,
    rhs_col_stride: i64,
    kc: usize,
    mc: usize,
    nc: usize,
    nr: usize,
    mr: usize,
    do_lhs_pack: bool,
    prepacked_lhs: Option<&PrePackedLhs>,
    prepack_rhs: Option<&PrePackedRhs>,
    #[allow(unused_variables)] post_op: F1,
    #[allow(unused_variables)] post_op_vec: F2,
) where
    T: MatmulMicroKernel + Send + Sync + Copy + Zero,
    F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
    F2: Fn(<T as MatmulMicroKernel>::SelfVec, usize, usize) -> <T as MatmulMicroKernel>::SelfVec
        + Clone
        + Send
        + Sync
        + 'static,
{
    assert_eq!(nr % vec_size::<T>(), 0);

    if m == 1 && kc == k && nc == nr && mc == 1 && (rhs_col_stride == 1 || prepack_rhs.is_some()) {
        mv_method(
            a,
            b,
            out,
            m_offset,
            n_offset,
            n,
            k,
            ldb,
            lhs_col_stride,
            kc,
            nc,
            nr,
            prepack_rhs,
            post_op,
            post_op_vec,
        );
        return;
    }

    let num_mr_blocks = (mc + mr - 1) / mr;
    let num_nr_blocks = (nc + nr - 1) / nr;

    let mut do_rhs_pack = true;
    if let Some(prepacked) = &prepack_rhs {
        assert_eq!(prepacked.kc, kc);
        assert_eq!(prepacked.nc, nc);
        assert_eq!(prepacked.nr, nr);
        do_rhs_pack = false;
    }

    let panel_size = num_nr_blocks * nr * kc;

    let has_prepacked_lhs = prepacked_lhs.is_some();

    let (param1, param2) = if has_prepacked_lhs || do_lhs_pack {
        (kc as i64, 1)
    } else {
        (lda, lda)
    };

    let packed_a = if do_lhs_pack && !has_prepacked_lhs {
        L3_SLAB.with(|mem| {
            if do_lhs_pack {
                let mut mem = mem.borrow_mut();
                let stack = DynStack::new(&mut mem);
                let (packed_a_storage, _) =
                    stack.make_aligned_uninit::<T>(num_mr_blocks * mr * kc, 64);
                let packed_a = Pointer::new(
                    packed_a_storage.as_mut_ptr() as *mut T,
                    (num_mr_blocks * mr * kc) as i64,
                );
                packed_a
            } else {
                a
            }
        })
    } else {
        Pointer::new(std::ptr::null_mut(), 0)
    };

    for (i_idx, i) in (0..m).step_by(mc).enumerate() {
        let ib = min(mc, m - i);
        for (p_idx, p) in (0..k).step_by(kc).enumerate() {
            let first_kiter = p == 0;
            let pb = min(kc, k - p);

            let packed_a = if let Some(prepacked_lhs) = &prepacked_lhs {
                prepacked_lhs.buffers[i_idx][p_idx].cast::<T>()
            } else {
                if do_lhs_pack {
                    pack_a_single_thread::<T>(
                        a + (i as i64) * lda + (p as i64) * lhs_col_stride,
                        packed_a,
                        lda,
                        lhs_col_stride,
                        ib,
                        pb,
                        kc,
                        mr,
                    );
                    packed_a
                } else {
                    a + ((i as i64) * lda + (p as i64) * lhs_col_stride)
                }
            };

            let mut packed_b = if let Some(prepacked) = &prepack_rhs {
                prepacked.buffers[p_idx].cast::<T>()
            } else {
                L2_SLAB.with_borrow_mut(|mem| {
                    let stack = DynStack::new(mem);
                    let (packed_b_storage, _) =
                        stack.make_aligned_uninit::<T>(panel_size, CACHELINE_ALIGN);
                    Pointer::new(
                        packed_b_storage.as_mut_ptr() as *mut T,
                        (num_nr_blocks * nr * kc) as i64,
                    )
                })
            };
            let packed_b_cpy = packed_b;
            for (j_idx, j) in (0..n).step_by(nc).enumerate() {
                let jb = min(nc, n - j);
                let c = out + (i as i64) * ldc + (j as i64);
                if do_rhs_pack {
                    pack_b_single_thread::<T, <T as MatmulMicroKernel>::SelfVec>(
                        b + ((p as i64) * ldb + (j as i64) * rhs_col_stride),
                        packed_b,
                        ldb,
                        rhs_col_stride,
                        jb,
                        pb,
                        kc,
                        nr,
                    );
                } else {
                    packed_b = packed_b_cpy + ((j_idx * panel_size) as i64);
                }
                for ii in (0..ib).step_by(mr) {
                    let mb = min(mr, ib - ii);
                    let micro_kernel = <T>::get_kernel(nr / vec_size::<T>(), mb);
                    for jj in (0..jb).step_by(nr) {
                        let jjb = min(nr, jb - jj);
                        micro_kernel(
                            packed_a + param1 * (ii as i64),
                            packed_b + (jj as i64) * (kc as i64),
                            c + (ii as i64) * ldc + (jj as i64),
                            ldc,
                            param2,
                            pb,
                            jjb,
                            if do_lhs_pack || has_prepacked_lhs {
                                mb as i64
                            } else {
                                lhs_col_stride
                            },
                            extra_args,
                        );
                    }
                }
            }
        }
    }
}

#[duplicate::duplicate_item(
    func_name               mv_method  get_kernel                                 extra_args;
    [matmul_mp_sg]          [mv_mp]  [get_mixed_precision_kernel]               [vec_cast_back, cast_back];
    [matmul_post_mp_sg]     [mv_post_mp]  [get_mixed_precision_kernel_with_post_op]  [p + pb == k, m_offset + i + ii, n_offset + j + jj, vec_cast_back, cast_back, post_op.clone(), post_op_vec.clone()];
)]
pub(crate) fn func_name<T, F1, F2>(
    a: Pointer<T>,
    b: Pointer<T>,
    out: Pointer<T>,
    #[allow(unused_variables)] m_offset: usize,
    #[allow(unused_variables)] n_offset: usize,
    m: usize,
    n: usize,
    k: usize,
    lda: i64,
    ldb: i64,
    ldc: i64,
    lhs_col_stride: i64,
    rhs_col_stride: i64,
    kc: usize,
    mc: usize,
    nc: usize,
    nr: usize,
    mr: usize,
    do_lhs_pack: bool,
    prepacked_lhs: Option<&PrePackedLhs>,
    prepack_rhs: Option<&PrePackedRhs>,
    #[allow(unused_variables)] post_op: F1,
    #[allow(unused_variables)] post_op_vec: F2,
    pack_vec: fn(
        *mut <T as MatmulMicroKernel>::MixedVec,
        *const <T as MatmulMicroKernel>::SelfVec,
        usize,
    ),
    pack_vec_exceed: fn(*mut <T as MatmulMicroKernel>::MixedVec, usize),
    cast: fn(&mut <T as MatmulMicroKernel>::MixedType, &T),
    vec_cast_back: fn(
        *mut <T as MatmulMicroKernel>::SelfVec,
        *const <T as MatmulMicroKernel>::MixedVec,
    ),
    cast_back: fn(&mut T, &<T as MatmulMicroKernel>::MixedType),
) where
    T: MatmulMicroKernel + Copy + Zero,
    <T as MatmulMicroKernel>::MixedType: Copy + Zero,
    F1: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
    F2: Fn(<T as MatmulMicroKernel>::SelfVec, usize, usize) -> <T as MatmulMicroKernel>::SelfVec
        + Clone
        + Send
        + Sync
        + 'static,
{
    if m == 1 && kc == k && nc == nr && mc == 1  {
        if let Some(prepacked_lhs) = &prepacked_lhs {
            mv_method(
                prepacked_lhs.buffers[0][0].cast::<<T as MatmulMicroKernel>::MixedType>(),
                b.cast::<<T as MatmulMicroKernel>::MixedType>(),
                out,
                m_offset,
                n_offset,
                n,
                k,
                ldb,
                lhs_col_stride,
                kc,
                nc,
                nr,
                prepack_rhs,
                post_op,
                post_op_vec,
                vec_cast_back,
                cast_back,
            );
        } else {
            panic!("prepacked_lhs is None");
        }
        return;
    }

    let num_mr_blocks = (mc + mr - 1) / mr;
    let num_nr_blocks = (nc + nr - 1) / nr;

    let mut do_rhs_pack = true;
    if let Some(prepacked) = &prepack_rhs {
        assert_eq!(prepacked.kc, kc);
        assert_eq!(prepacked.nc, nc);
        assert_eq!(prepacked.nr, nr);
        do_rhs_pack = false;
    }

    let panel_size = num_nr_blocks * nr * kc;

    let has_prepacked_lhs = prepacked_lhs.is_some();

    let (param1, param2) = if has_prepacked_lhs || do_lhs_pack {
        (kc as i64, 1)
    } else {
        (lda, lda)
    };

    let packed_a = if do_lhs_pack && !has_prepacked_lhs {
        L3_SLAB.with(|mem| {
            let mut mem = mem.borrow_mut();
            let stack = DynStack::new(&mut mem);
            let (packed_a_storage, _) = stack
                .make_aligned_uninit::<<T as MatmulMicroKernel>::MixedType>(
                    num_mr_blocks * mr * kc,
                    64,
                );
            let packed_a = Pointer::new(
                packed_a_storage.as_mut_ptr() as *mut <T as MatmulMicroKernel>::MixedType,
                (num_mr_blocks * mr * kc) as i64,
            );
            packed_a
        })
    } else {
        Pointer::new(std::ptr::null_mut(), 0)
    };

    for (i_idx, i) in (0..m).step_by(mc).enumerate() {
        let ib = min(mc, m - i);
        for (p_idx, p) in (0..k).step_by(kc).enumerate() {
            let first_kiter = p == 0;
            let pb = min(kc, k - p);

            let packed_a = if let Some(prepacked_lhs) = &prepacked_lhs {
                prepacked_lhs.buffers[i_idx][p_idx].cast::<<T as MatmulMicroKernel>::MixedType>()
            } else {
                pack_a_mixed_precision_single_thread::<T, <T as MatmulMicroKernel>::MixedType>(
                    a + (i as i64) * lda + (p as i64) * lhs_col_stride,
                    packed_a,
                    lda,
                    lhs_col_stride,
                    ib,
                    pb,
                    kc,
                    mr,
                    cast,
                );
                packed_a
            };

            let mut packed_b = if let Some(prepacked) = &prepack_rhs {
                prepacked.buffers[p_idx].cast::<<T as MatmulMicroKernel>::MixedType>()
            } else {
                L2_SLAB.with(|mem| {
                    let mut mem = mem.borrow_mut();
                    let stack = DynStack::new(&mut mem);
                    let (packed_b_storage, _) = stack
                        .make_aligned_uninit::<<T as MatmulMicroKernel>::MixedType>(
                            num_nr_blocks * nr * kc,
                            64,
                        );
                    Pointer::new(
                        packed_b_storage.as_mut_ptr() as *mut <T as MatmulMicroKernel>::MixedType,
                        (num_nr_blocks * nr * kc) as i64,
                    )
                })
            };
            let packed_b_cpy = packed_b;
            for (j_idx, j) in (0..n).step_by(nc).enumerate() {
                let jb = min(nc, n - j);
                let c = out + (i as i64) * ldc + (j as i64);
                if do_rhs_pack {
                    pack_b_mixed_precision::<
                        T,
                        <T as MatmulMicroKernel>::MixedType,
                        <T as MatmulMicroKernel>::SelfVec,
                        <T as MatmulMicroKernel>::MixedVec,
                    >(
                        b + ((p as i64) * ldb + (j as i64) * rhs_col_stride),
                        packed_b,
                        ldb,
                        rhs_col_stride,
                        jb,
                        pb,
                        kc,
                        nr,
                        pack_vec,
                        pack_vec_exceed,
                        cast,
                    );
                } else {
                    packed_b = packed_b_cpy + ((j_idx * panel_size) as i64);
                }
                for ii in (0..ib).step_by(mr) {
                    let mb = min(mr, ib - ii);
                    let micro_kernel = <T>::get_kernel(nr / vec_size::<T>(), mb);
                    for jj in (0..jb).step_by(nr) {
                        let jjb = min(nr, jb - jj);
                        micro_kernel(
                            packed_a + param1 * (ii as i64),
                            packed_b + (jj as i64) * (kc as i64),
                            c + (ii as i64) * ldc + (jj as i64),
                            ldc,
                            param2,
                            pb,
                            jjb,
                            if do_lhs_pack || has_prepacked_lhs {
                                mb as i64
                            } else {
                                lhs_col_stride
                            },
                            first_kiter,
                            extra_args,
                        );
                    }
                }
            }
        }
    }
}

#[duplicate::duplicate_item(
    func_name   get_kernel                      extra_args;
    [mv]        [get_gemv_kernel]               [];
    [mv_post]   [get_gemv_kernel_with_post_op]  [m_offset, n_offset, post_op, post_op_vec];
)]
pub(crate) fn func_name<T, F, F2>(
    a: Pointer<T>,
    b: Pointer<T>,
    out: Pointer<T>,
    #[allow(unused_variables)] m_offset: usize,
    #[allow(unused_variables)] n_offset: usize,
    n: usize,
    k: usize,
    ldb: i64,
    lhs_col_stride: i64,
    kc: usize,
    nc: usize,
    nr: usize,
    prepack_rhs: Option<&PrePackedRhs>,
    #[allow(unused_variables)] post_op: F,
    #[allow(unused_variables)] post_op_vec: F2,
) where
    T: MatmulMicroKernel + Send + Sync + Copy + Zero,
    F: Fn(T, usize, usize) -> T + Clone + Send + Sync + 'static,
    F2: Fn(<T as MatmulMicroKernel>::SelfVec, usize, usize) -> <T as MatmulMicroKernel>::SelfVec
        + Clone
        + Send
        + Sync
        + 'static,
{
    let micro_kernel = <T>::get_kernel();

    if let Some(prepacked_rhs) = &prepack_rhs {
        assert_eq!(prepacked_rhs.kc, kc);
        assert_eq!(prepacked_rhs.nc, nc);
        assert_eq!(prepacked_rhs.nr, nr);
        assert_eq!(prepacked_rhs.buffers.len(), 1);
        let prepacked_b = prepacked_rhs.buffers[0].cast::<T>();
        micro_kernel(a, prepacked_b, out, n, k, ldb, lhs_col_stride, extra_args)
    } else {
        micro_kernel(a, b, out, n, k, ldb, lhs_col_stride, extra_args)
    }
}

#[duplicate::duplicate_item(
    func_name       get_kernel                      extra_args;
    [mv_mp]         [get_gemv_kernel_mp]               [vec_cast_back, cast_back];
    [mv_post_mp]    [get_gemv_kernel_mp_with_post_op]  [m_offset, n_offset, vec_cast_back, cast_back, post_op, post_op_vec];
)]
pub(crate) fn func_name<T, F, F2>(
    a: Pointer<<T as MatmulMicroKernel>::MixedType>,
    b: Pointer<<T as MatmulMicroKernel>::MixedType>,
    out: Pointer<T>,
    #[allow(unused_variables)] m_offset: usize,
    #[allow(unused_variables)] n_offset: usize,
    n: usize,
    k: usize,
    ldb: i64,
    lhs_col_stride: i64,
    kc: usize,
    nc: usize,
    nr: usize,
    prepack_rhs: Option<&PrePackedRhs>,
    #[allow(unused_variables)] post_op: F,
    #[allow(unused_variables)] post_op_vec: F2,
    vec_cast_back: fn(
        *mut <T as MatmulMicroKernel>::SelfVec,
        *const <T as MatmulMicroKernel>::MixedVec,
    ),
    cast_back: fn(&mut T, &<T as MatmulMicroKernel>::MixedType),
) where
    T: MatmulMicroKernel + Copy + Zero,
    F: Fn(T, usize, usize) -> T + Clone + 'static,
    F2: Fn(<T as MatmulMicroKernel>::SelfVec, usize, usize) -> <T as MatmulMicroKernel>::SelfVec
        + Clone
        + 'static,
{
    let micro_kernel = <T>::get_kernel();

    if let Some(prepacked_rhs) = &prepack_rhs {
        assert_eq!(prepacked_rhs.kc, kc);
        assert_eq!(prepacked_rhs.nc, nc);
        assert_eq!(prepacked_rhs.nr, nr);
        assert_eq!(prepacked_rhs.buffers.len(), 1);
        let prepacked_b = prepacked_rhs.buffers[0].cast::<<T as MatmulMicroKernel>::MixedType>();
        micro_kernel(
            a,
            prepacked_b,
            out,
            n,
            k,
            nc as i64,
            lhs_col_stride,
            extra_args,
        )
    } else {
        micro_kernel(a, b, out, n, k, ldb, lhs_col_stride, extra_args)
    }
}
