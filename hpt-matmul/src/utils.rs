use num_integer::gcd;
use std::{ cell::OnceCell, cmp::min };

use crate::{ Pointer, Zero, vec_size };
use gemm_common::gemm::CACHELINE_ALIGN;

thread_local! {
    pub(crate) static L2_SLAB: core::cell::RefCell<dyn_stack::MemBuffer> = core::cell::RefCell::new(
        dyn_stack::MemBuffer::new(
            dyn_stack::StackReq::new_aligned::<u8>(get_cache_info()[1].bytes * 8, CACHELINE_ALIGN)
        )
    );
}

thread_local! {
    pub(crate) static L3_SLAB: core::cell::RefCell<dyn_stack::MemBuffer> = core::cell::RefCell::new(
        dyn_stack::MemBuffer::new(
            dyn_stack::StackReq::new_aligned::<u8>(
                get_cache_info()[2].bytes.max(1024 * 1024 * 8) * 8,
                CACHELINE_ALIGN
            )
        )
    );
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
    if intervals.len() == 1 {
        return vec![[0, 0, 0]];
    }
    let mut prgs = vec![[0, 0, 0]; intervals.len()];
    let mut prg = [0, 0, 0];
    for (tid, (start, end)) in intervals.iter().enumerate() {
        prgs[tid] = prg;
        prg = calculate_prg(n, nc, mr, nr, ib, prg, *start, *end);
    }
    prgs
}

pub(crate) fn mt_intervals(size: usize, num_threads: usize) -> Vec<(usize, usize)> {
    let mut intervals = Vec::with_capacity(num_threads);
    if num_threads == 1 {
        return vec![(0, size)];
    }
    for i in 0..num_threads {
        let start_index = i * (size / num_threads) + std::cmp::min(i, size % num_threads);
        let end_index = start_index + size / num_threads + ((i < size % num_threads) as usize);
        intervals.push((start_index, end_index));
    }
    intervals
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CacheInfo {
    pub(crate) bytes: usize,
    pub(crate) associativity: usize,
    pub(crate) cache_line_bytes: usize,
}

const CACHE_INFO: OnceCell<[CacheInfo; 3]> = OnceCell::new();

pub(crate) fn get_cache_info() -> [CacheInfo; 3] {
    #[cfg(target_arch = "x86_64")]
    {
        use raw_cpuid::{ CacheType, CpuId };
        *CACHE_INFO.get_or_init(|| {
            let cpuid = CpuId::new();
            let mut cache_info = [
                CacheInfo {
                    bytes: 0,
                    associativity: 0,
                    cache_line_bytes: 0,
                };
                3
            ];
            if let Some(cparams) = cpuid.get_cache_parameters() {
                for cache in cparams {
                    let size =
                        cache.associativity() *
                        cache.physical_line_partitions() *
                        cache.coherency_line_size() *
                        cache.sets();
                    let valid_cache =
                        (cache.cache_type() == CacheType::Data ||
                            cache.cache_type() == CacheType::Unified) && cache.level() <= 3;
                    if valid_cache {
                        let info = CacheInfo {
                            bytes: size,
                            associativity: cache.associativity(),
                            cache_line_bytes: cache.coherency_line_size(),
                        };
                        cache_info[(cache.level() as usize) - 1] = info;
                    }
                }
            } else {
                panic!("No cache parameter information available");
            }
            cache_info
        })
    }
    #[cfg(target_os = "macos")]
    {
        use std::ffi::CString;
        *CACHE_INFO.get_or_init(|| {
            let mut cache_info = [
                CacheInfo {
                    bytes: 0,
                    associativity: 0,
                    cache_line_bytes: 0,
                };
                3
            ];
            for level in 1..=3 {
                let mut size: u64 = 0;
                let mut line_size: u64 = 0;

                let mut size_len = std::mem::size_of::<u64>();
                let mut line_size_len = std::mem::size_of::<u64>();

                let name = if level == 1 {
                    "hw.l1dcachesize"
                } else if level == 2 {
                    "hw.l2cachesize"
                } else {
                    "hw.l3cachesize"
                };
                unsafe {
                    libc::sysctlbyname(
                        CString::new(name).unwrap().as_ptr(),
                        &mut size as *mut _ as *mut libc::c_void,
                        &mut size_len,
                        std::ptr::null_mut(),
                        0
                    );
                }

                unsafe {
                    libc::sysctlbyname(
                        CString::new("hw.cachelinesize").unwrap().as_ptr(),
                        &mut line_size as *mut _ as *mut libc::c_void,
                        &mut line_size_len,
                        std::ptr::null_mut(),
                        0
                    );
                }
                cache_info[level - 1] = CacheInfo {
                    bytes: size as usize,
                    cache_line_bytes: line_size as usize,
                    associativity: 8,
                };
            }
            cache_info
        })
    }
}

// code is from gemm-common
pub fn _kernel_params(
    n: usize,
    m: usize,
    k: usize,
    nr: usize,
    mr: usize,
    sizeof: usize
) -> KernelParams {
    #[inline]
    fn round_down(a: usize, b: usize) -> usize {
        (a / b) * b
    }
    if n == 0 || m == 0 || k == 0 {
        return KernelParams {
            kc: k,
            mc: n,
            nc: m,
        };
    }

    let info = get_cache_info();

    let l1_cache_bytes = info[0].bytes.max(32 * 1024);
    let l2_cache_bytes = info[1].bytes;
    let l3_cache_bytes = info[2].bytes;

    let l1_line_bytes = info[0].cache_line_bytes.max(64);

    let l1_assoc = info[0].associativity.max(3);
    let l2_assoc = info[1].associativity.max(3);
    let l3_assoc = info[2].associativity.max(3);

    let l1_n_sets = l1_cache_bytes / (l1_line_bytes * l1_assoc);

    // requires
    // A micropanels must occupy different cache sets
    // so that loading a micropanel evicts the previous one
    // => byte stride must be multiple of n_sets×line_bytes
    //
    // => mr×kc×scalar_bytes == C_A × l1_line_bytes × l1_n_sets
    //
    // l1 must be able to hold A micropanel, B micropanel
    //
    // => C_A + C_B <= l1_assoc

    // a×n = b×m
    // find lcm of a, b
    // n = lcm / a = b/gcd(a,b)
    // m = lcm / b = a/gcd(a,b)

    let gcd = gcd(nr * sizeof, l1_line_bytes * l1_n_sets);
    let kc_0 = (l1_line_bytes * l1_n_sets) / gcd;
    let c_lhs = (nr * sizeof) / gcd;
    let c_rhs = (mr * kc_0 * sizeof) / (l1_line_bytes * l1_n_sets);
    let kc_multiplier = l1_assoc / (c_lhs + c_rhs);
    // let auto_kc = kc_0 * kc_multiplier;
    let auto_kc = (kc_0 * kc_multiplier.next_power_of_two()).max(512).min(k);
    let k_iter = k.div_ceil(auto_kc);
    let auto_kc = k.div_ceil(k_iter);

    // l2 cache must hold
    //  - B micropanel: nr×kc: assume 1 assoc degree
    //  - A macropanel: mc×kc
    // mc×kc×scalar_bytes
    let auto_nc = if l2_cache_bytes == 0 {
        panic!();
    } else {
        let rhs_micropanel_bytes = mr * auto_kc * sizeof;
        let rhs_l2_assoc = rhs_micropanel_bytes.div_ceil(l2_cache_bytes / l2_assoc);
        let lhs_l2_assoc = (l2_assoc - 1 - rhs_l2_assoc).max(1);

        let mc_from_lhs_l2_assoc = |lhs_l2_assoc: usize| -> usize {
            (lhs_l2_assoc * l2_cache_bytes) / (l2_assoc * sizeof * auto_kc)
        };

        let auto_nc = round_down(mc_from_lhs_l2_assoc(lhs_l2_assoc), nr);
        let m_iter = n.div_ceil(auto_nc);
        n.div_ceil(m_iter * nr) * nr
    };
    let auto_nc = Ord::min(auto_nc, 8 * nr);

    // l3 cache must hold
    //  - A macropanel: mc×kc: assume 1 assoc degree
    //  - B macropanel: nc×kc
    let auto_mc = if l3_cache_bytes == 0 {
        0
    } else {
        // let lhs_macropanel_bytes = auto_mc * auto_kc * sizeof;
        // let lhs_l3_assoc = msrv_div_ceil(lhs_macropanel_bytes, l3_cache_bytes / l3_assoc);
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

pub struct KernelParams {
    pub kc: usize,
    pub mc: usize,
    pub nc: usize,
}

pub fn kernel_params(
    n: usize,
    m: usize,
    k: usize,
    nr: usize,
    mr: usize,
    sizeof: usize
) -> KernelParams {
    let params = _kernel_params(n, m, k, nr, mr, sizeof);
    KernelParams {
        kc: params.kc,
        mc: params.mc,
        nc: params.nc,
    }
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
    cast: fn(&mut I, &T)
) {
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
    cast: fn(&mut IM, &T)
) {
    let nr_div_lane = nr / vec_size::<T>();

    for j in (0..nc).step_by(nr) {
        let nb = nr.min(nc - j);
        if nb == nr && stride == 1 {
            for p in 0..kb as i64 {
                let b_ptr = b.offset(p * ldb + (j as i64)).cast::<TVec>();
                for i in 0..nr_div_lane {
                    pack_vec(packed_b.ptr as *mut IMVec, b_ptr.ptr as *const TVec, i);
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
                    cast(&mut res, &b[p * ldb + j * stride]);
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

#[cfg(not(target_feature = "neon"))]
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
    num_mr_blocks: usize
) {
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

#[cfg(target_feature = "neon")]
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
    num_mr_blocks: usize
) {
    let start_block = tid * mb_per_thread;
    let end_block = std::cmp::min((tid + 1) * mb_per_thread, num_mr_blocks);
    if start_block >= num_mr_blocks {
        return;
    }
    let start_i = start_block * mr;
    let end_i = std::cmp::min(end_block * mr, mc);
    let offset = start_block * mr * kc;
    packed_a += offset as i64;
    if stride == 1 {
        #[inline(always)]
        fn transpose_4x4<T: Copy>(ptr: Pointer<T>, mut ptr_out: Pointer<T>) {
            let val0 = ptr[0];
            let val1 = ptr[1];
            let val2 = ptr[2];
            let val3 = ptr[3];
            ptr_out[0] = val0;
            ptr_out[4] = val1;
            ptr_out[8] = val2;
            ptr_out[12] = val3;
        }
        for i in (start_i..end_i).step_by(mr) {
            let mb = mr.min(mc - i);
            let i_lda = (i as i64) * lda;
            if mb == 4 {
                for p in (0..kb).step_by(4) {
                    transpose_4x4(a.offset(i_lda + (p as i64)), packed_a);
                    transpose_4x4(a.offset(i_lda + lda + (p as i64)), packed_a.offset(1));
                    transpose_4x4(a.offset(i_lda + 2 * lda + (p as i64)), packed_a.offset(2));
                    transpose_4x4(a.offset(i_lda + 3 * lda + (p as i64)), packed_a.offset(3));
                    packed_a += 16i64;
                }
                for p in kb / 4 * 4..kb {
                    for ii in 0..mb as i64 {
                        let row = (i as i64) + ii;
                        *packed_a = a[row * lda + (p as i64)];
                        packed_a += 1i64;
                    }
                }
            } else {
                for p in 0..kb {
                    for ii in 0..mb as i64 {
                        let row = (i as i64) + ii;
                        *packed_a = a[row * lda + (p as i64)];
                        packed_a += 1i64;
                    }
                }
            }
            for _ in kb..kc {
                for _ in 0..mb as i64 {
                    *packed_a = T::ZERO;
                    packed_a += 1i64;
                }
            }
        }
    } else {
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
}

pub(crate) fn pack_b<T: Copy + Zero, TVec>(
    b: Pointer<T>,
    mut packed_b: Pointer<T>,
    ldb: i64,
    stride: i64,
    nc: usize,
    kb: usize,
    kc: usize,
    nr: usize,
    jj_start: usize,
    need_full_pack: bool
) {
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
                let b_ptr = b.offset(p * ldb + (j as i64)).cast::<TVec>();
                let packed_b_ptr = packed_b.cast::<TVec>();
                for i in 0..nr_div_lane {
                    let packed_b_vec = packed_b_ptr.offset(i as i64);
                    packed_b_vec.write_unaligned(b_ptr.offset(i as i64).read_unaligned());
                }
                packed_b += nr as i64;
            }
            for _ in kb..kc {
                for _ in 0..nr {
                    *packed_b = T::ZERO;
                    packed_b += 1i64;
                }
            }
        } else {
            for p in 0..kb as i64 {
                for jj in 0..nb as i64 {
                    let j = (j as i64) + jj;
                    *packed_b = b[p * ldb + j * stride];
                    packed_b += 1i64;
                }
                for _ in nb..nr {
                    *packed_b = T::ZERO;
                    packed_b += 1i64;
                }
            }
            for _ in kb..kc {
                for _ in 0..nr as i64 {
                    *packed_b = T::ZERO;
                    packed_b += 1i64;
                }
            }
        }
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct PrePackedRhs {
    pub buffer: (Pointer<u8>, std::alloc::Layout),
    pub kc: usize,
    pub nr: usize,
    pub nc: usize,
}

impl Drop for PrePackedRhs {
    fn drop(&mut self) {
        if self.buffer.0.ptr != std::ptr::null_mut() {
            unsafe {
                std::alloc::dealloc(self.buffer.0.ptr as *mut u8, self.buffer.1);
            }
        }
    }
}
