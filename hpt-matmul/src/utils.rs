use std::cmp::min;

use gemm_common::{ cache::CACHE_INFO, gemm::CACHELINE_ALIGN };

use crate::{vec_size, Pointer, Zero};

thread_local! {
    pub(crate) static L2_SLAB: core::cell::RefCell<dyn_stack::MemBuffer> = core::cell::RefCell::new(
        dyn_stack::MemBuffer::new(
            dyn_stack::StackReq::new_aligned::<u8>(CACHE_INFO[1].cache_bytes * 8, CACHELINE_ALIGN)
        )
    );
}

thread_local! {
    pub(crate) static L3_SLAB: core::cell::RefCell<dyn_stack::MemBuffer> = core::cell::RefCell::new(
        dyn_stack::MemBuffer::new(
            dyn_stack::StackReq::new_aligned::<u8>(
                CACHE_INFO[2].cache_bytes.max(1024 * 1024 * 8) * 8,
                CACHELINE_ALIGN
            )
        )
    );
}

pub(crate) struct KernelParams {
    pub(crate) kc: usize,
    pub(crate) mc: usize,
    pub(crate) nc: usize,
}

pub(crate) fn kernel_params(
    n: usize,
    m: usize,
    k: usize,
    nr: usize,
    mr: usize,
    sizeof: usize,
    packed_a: bool,
) -> KernelParams {
    fn round_down(a: usize, b: usize) -> usize {
        a / b * b
    }
    if n == 0 || m == 0 || k == 0 {
        return KernelParams {
            kc: k,
            mc: n,
            nc: m,
        };
    }

    let info = *CACHE_INFO;

    let l1_cache_bytes = info[0].cache_bytes.max(32 * 1024);
    let l2_cache_bytes = info[1].cache_bytes;
    let l3_cache_bytes = info[2].cache_bytes;

    let l1_line_bytes = info[0].cache_line_bytes.max(64);

    let l1_assoc = info[0].associativity.max(2);
    let l2_assoc = info[1].associativity.max(2);
    let l3_assoc = info[2].associativity.max(2);

    let l1_n_sets = l1_cache_bytes / (l1_line_bytes * l1_assoc);

    let gcd = num_integer::gcd(nr * sizeof, l1_line_bytes * l1_n_sets);
    let kc_0 = (l1_line_bytes * l1_n_sets) / gcd; // maximum # of nr * sizeof access that has no conflicts
    let c_rhs = (nr * kc_0 * sizeof).next_multiple_of(l1_line_bytes) / (l1_line_bytes * l1_n_sets);
    let c_lhs = if packed_a {
        (mr * sizeof).next_multiple_of(l1_line_bytes) / (l1_line_bytes * l1_n_sets)
    } else {
        (mr * (kc_0 * sizeof).next_multiple_of(l1_line_bytes)) / (l1_line_bytes * l1_n_sets)
    };
    let kc_multiplier = l1_assoc / (c_rhs + c_lhs);
    let auto_kc = (kc_0 * kc_multiplier.max(1))
        .next_power_of_two()
        .max(512)
        .min(k);
    let k_iter = k.div_ceil(auto_kc);
    let auto_kc = k.div_ceil(k_iter);

    let auto_nc = if l2_cache_bytes == 0 {
        panic!();
    } else {
        let lhs_micropanel_bytes = if packed_a {
            (mr * auto_kc * sizeof).next_multiple_of(l1_line_bytes)
        } else {
            mr * (auto_kc * sizeof).next_multiple_of(l1_line_bytes)
        };
        let lhs_l2_assoc = lhs_micropanel_bytes.div_ceil(l2_cache_bytes / l2_assoc);

        let rhs_l2_assoc = (l2_assoc - lhs_l2_assoc).max(1);

        let nc_from_rhs_l2_assoc = (rhs_l2_assoc * l2_cache_bytes) / (l2_assoc * sizeof * auto_kc);

        let auto_nc = round_down(nc_from_rhs_l2_assoc, nr);
        let n_iter = n.div_ceil(auto_nc);
        n.div_ceil(n_iter * nr) * nr
    };
    let auto_nc = Ord::min(auto_nc, 2 * nr);

    let auto_mc = if l3_cache_bytes == 0 {
        0
    } else {
        let rhs_l3_assoc = l3_assoc - 1;
        let rhs_macropanel_max_bytes = (rhs_l3_assoc * l3_cache_bytes) / l3_assoc;

        let auto_nc = round_down(rhs_macropanel_max_bytes / (sizeof * auto_kc), mr);
        let n_iter = m.div_ceil(auto_nc);
        m.div_ceil(n_iter * mr) * mr
    };

    KernelParams {
        kc: auto_kc,
        mc: auto_mc,
        nc: auto_nc,
    }
}

pub(crate) fn calculate_jobs(n: usize, nc: usize, mr: usize, nr: usize, ib: usize) -> usize {
    let mut jobs = 0;
    for j in (0..n).step_by(nc) {
        let jb = min(nc, n - j);
        for _ in (0..ib).step_by(mr) {
            for _ in (0..jb).step_by(nr) {
                jobs += 1;
            }
        }
    }
    jobs
}

pub(crate) fn calculate_prg(
    n: usize,
    nc: usize,
    mr: usize,
    nr: usize,
    ib: usize,
    prg: [usize; 3],
    mut start: usize,
    end: usize
) -> [usize; 3] {
    let mut ret = prg;
    let j_start = prg[0] * nc;
    let mut i_start = prg[1] * mr;
    let mut jj_start = prg[2] * nr;
    for j in (j_start..n).step_by(nc) {
        let jb = min(nc, n - j);
        // pack b
        for _ in (i_start..ib).step_by(mr) {
            for _ in (jj_start..jb).step_by(nr) {
                ret[2] += 1;
                start += 1;
                if start >= end {
                    return ret;
                }
            }
            ret[1] += 1;
            ret[2] = 0;
            jj_start = 0;
        }
        ret[0] += 1;
        ret[1] = 0;
        ret[2] = 0;
        i_start = 0;
    }
    ret
}

pub(crate) fn calculate_prgs(
    n: usize,
    nc: usize,
    mr: usize,
    nr: usize,
    ib: usize,
    intervals: &[(usize, usize)]
) -> Vec<[usize; 3]> {
    let mut prgs = vec![[0, 0, 0]; intervals.len()];
    let mut prg = [0, 0, 0];
    for (tid, (start, end)) in intervals.iter().enumerate() {
        prgs[tid] = prg;
        prg = calculate_prg(n, nc, mr, nr, ib, prg, *start, *end);
    }
    prgs
}

pub(crate) fn pack_a_mixed_precision<T, I: Zero>(
    a: Pointer<T>,
    mut packed_a: Pointer<I>,
    lda: i64,
    stride: i64,
    mc: usize,
    kb: usize,
    kc: usize,
    mr: usize,
    tid: usize,
    mb_per_thread: usize,
    num_mr_blocks: usize,
    cast: fn(&mut I, &T),
)
{
    let start_block = tid * mb_per_thread;
    let end_block = std::cmp::min((tid + 1) * mb_per_thread, num_mr_blocks);
    if start_block >= num_mr_blocks {
        return;
    }
    let start_i = start_block * mr;
    let end_i = std::cmp::min(end_block * mr, mc);
    let offset = start_block * mr * kc;
    packed_a += offset as i64;
    for i in (start_i..end_i).step_by(mr) {
        let mb = mr.min(mc - i);
        for p in 0..kb as i64 {
            for ii in 0..mb as i64 {
                let row = (i as i64) + ii;
                cast(&mut *packed_a, &a[row * lda + p * stride]);
                packed_a += 1i64;
            }
        }
        for _ in kb..kc {
            for _ in 0..mb as i64 {
                *packed_a = I::ZERO;
                packed_a += 1i64;
            }
        }
    }
}

pub(crate) fn pack_b_mixed_precision<T, IM: Zero, TVec, IMVec>(
    b: Pointer<T>,
    mut packed_b: Pointer<IM>,
    ldb: i64,
    stride: i64,
    nc: usize,
    kb: usize,
    kc: usize,
    nr: usize,
    pack_vec: fn(*mut IMVec, *const TVec, usize),
    pack_vec_exceed: fn(*mut IMVec, usize),
    pack_zero: fn(&mut IM, &T)
)
{
    let nr_div_lane = nr / vec_size::<T>();

    for j in (0..nc).step_by(nr) {
        let nb = nr.min(nc - j);
        if nb == nr && stride == 1 {
            for p in 0..kb as i64 {
                for i in 0..nr_div_lane {
                    pack_vec(
                        packed_b.ptr as *mut IMVec,
                        (unsafe {
                            b.ptr.offset(((p * ldb) as isize) + (j as isize))
                        }) as *const TVec,
                        i
                    );
                }
                packed_b += nr as i64;
            }
            for _ in kb..kc {
                for i in 0..nr_div_lane {
                    pack_vec_exceed(packed_b.ptr as *mut IMVec, i);
                }
                packed_b += nr as i64;
            }
        } else {
            for p in 0..kb as i64 {
                for jj in 0..nb as i64 {
                    let j = (j as i64) + jj;
                    let mut res = IM::ZERO;
                    pack_zero(&mut res, &b[p * ldb + j * stride]);
                    *packed_b = res;
                    packed_b += 1i64;
                }
                for _ in nb..nr {
                    *packed_b = IM::ZERO;
                    packed_b += 1i64;
                }
            }
            for _ in kb..kc {
                for _ in 0..nb as i64 {
                    *packed_b = IM::ZERO;
                    packed_b += 1i64;
                }
                for _ in nb..nr {
                    *packed_b = IM::ZERO;
                    packed_b += 1i64;
                }
            }
        }
    }
}

pub(crate) fn mt_intervals(size: usize, num_threads: usize) -> Vec<(usize, usize)> {
    let mut intervals = Vec::with_capacity(num_threads);
    for i in 0..num_threads {
        let start_index =
            i * (size / num_threads) + std::cmp::min(i, size % num_threads);
        let end_index = start_index
            + size / num_threads
            + ((i < size % num_threads) as usize);
        intervals.push((start_index, end_index));
    }
    intervals
}

pub(crate) fn pack_a<T: Zero + Copy>(
    a: Pointer<T>,
    mut packed_a: Pointer<T>,
    lda: i64,
    stride: i64,
    mc: usize,
    kb: usize,
    kc: usize,
    mr: usize,
    tid: usize,
    mb_per_thread: usize,
    num_mr_blocks: usize,
)
{
    let start_block = tid * mb_per_thread;
    let end_block = std::cmp::min((tid + 1) * mb_per_thread, num_mr_blocks);
    if start_block >= num_mr_blocks {
        return;
    }
    let start_i = start_block * mr;
    let end_i = std::cmp::min(end_block * mr, mc);
    let offset = start_block * mr * kc;
    packed_a += offset as i64;
    for i in (start_i..end_i).step_by(mr) {
        let mb = mr.min(mc - i);
        for p in 0..kb as i64 {
            for ii in 0..mb as i64 {
                let row = (i as i64) + ii;
                *packed_a = a[row * lda + p * stride];
                packed_a += 1i64;
            }
        }
        for _ in kb..kc {
            for _ in 0..mb as i64 {
                *packed_a = T::ZERO;
                packed_a += 1i64;
            }
        }
    }
}

pub(crate) fn pack_b<T: Copy, TVec>(
    b: Pointer<T>,
    mut packed_b: Pointer<T>,
    ldb: i64,
    stride: i64,
    nc: usize,
    kb: usize,
    kc: usize,
    nr: usize,
    jj_start: usize,
    need_full_pack: bool,
)
{
    let start = if need_full_pack {
        0
    } else {
        for j in (0..jj_start).step_by(nr) {
            let nb = nr.min(nc - j);
            if nb == nr {
                packed_b += (nr as i64) * (kb as i64);
                packed_b += ((kc - kb) as i64) * (nr as i64);
            } else {
                packed_b += ((nr - nb) as i64) * (kb as i64);
                packed_b += ((kc - kb) as i64) * (nr as i64);
            }
        }
        jj_start
    };
    let nr_div_lane = nr / vec_size::<T>();
    for j in (start..nc).step_by(nr) {
        let nb = nr.min(nc - j);
        if nb == nr && stride == 1 {
            for p in 0..kb as i64 {
                unsafe {
                    let b_ptr = b.ptr.offset((p * ldb) as isize + j as isize) as *const TVec;
                    for i in 0..nr_div_lane {
                        let packed_b_vec = packed_b.ptr.add(i * vec_size::<T>()) as *mut TVec;
                        packed_b_vec.write(b_ptr.add(i).read_unaligned());
                    }
                }
                packed_b += nr as i64;
            }
            packed_b += ((kc - kb) as i64) * (nr as i64);
        } else {
            for p in 0..kb as i64 {
                for jj in 0..nb as i64 {
                    let j = (j as i64) + jj;
                    *packed_b = b[p * ldb + j * stride];
                    packed_b += 1i64;
                }
                packed_b += (nr - nb) as i64;
            }
            packed_b += ((kc - kb) as i64) * (nr as i64);
        }
    }
}

#[allow(dead_code)]
#[derive(Clone)]
pub(crate) struct PrePackedRhs {
    pub(crate) buffers: Vec<Vec<Pointer<u8>>>,
    pub(crate) buffer_rems: Vec<Vec<Pointer<u8>>>,
    pub(crate) buffer: (Pointer<u8>, std::alloc::Layout),
    pub(crate) buffer_rem: (Pointer<u8>, std::alloc::Layout),
    pub(crate) mr: usize,
    pub(crate) mc: usize,
    pub(crate) kc: usize,
    pub(crate) nr: usize,
    pub(crate) nc: usize,
    pub(crate) num_threads: usize,
    pub(crate) prgs: Vec<[usize; 3]>,
    pub(crate) rem_prgs: Vec<[usize; 3]>,
    pub(crate) intervals: Vec<(usize, usize)>,
    pub(crate) rem_intervals: Vec<(usize, usize)>,
}
