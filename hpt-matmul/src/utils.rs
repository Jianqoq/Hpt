use num_integer::gcd;
use rayon::iter::ParallelIterator;
use std::{ cell::OnceCell, cmp::min };

use crate::{ vec_size, Pointer, VecTrait, Zero, ALIGN };
use gemm_common::gemm::CACHELINE_ALIGN;
use rayon::iter::IntoParallelIterator;

type IM<T> = <T as crate::MatmulMicroKernel>::MixedType;
type IMVec<T> = <T as crate::MatmulMicroKernel>::MixedVec;

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
    m: usize,
    n: usize,
    k: usize,
    mr: usize,
    nr: usize,
    sizeof: usize
) -> KernelParams {
    #[inline]
    fn round_down(a: usize, b: usize) -> usize {
        (a / b) * b
    }
    if m == 0 || n == 0 || k == 0 {
        return KernelParams {
            kc: k,
            mc: m,
            nc: n,
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

    let gcd = gcd(mr * sizeof, l1_line_bytes * l1_n_sets);
    let kc_0 = (l1_line_bytes * l1_n_sets) / gcd;
    let c_lhs = (mr * sizeof) / gcd;
    let c_rhs = (nr * kc_0 * sizeof) / (l1_line_bytes * l1_n_sets);
    let kc_multiplier = l1_assoc / (c_lhs + c_rhs);
    // let auto_kc = kc_0 * kc_multiplier;
    let auto_kc = (kc_0 * kc_multiplier.next_power_of_two()).max(512).min(k);
    let k_iter = k.div_ceil(auto_kc);
    let auto_kc = k.div_ceil(k_iter);

    // l2 cache must hold
    //  - B micropanel: nr×kc: assume 1 assoc degree
    //  - A macropanel: mc×kc
    // mc×kc×scalar_bytes
    let auto_mc = if l2_cache_bytes == 0 {
        panic!();
    } else {
        let rhs_micropanel_bytes = nr * auto_kc * sizeof;
        let rhs_l2_assoc = rhs_micropanel_bytes.div_ceil(l2_cache_bytes / l2_assoc);
        let lhs_l2_assoc = (l2_assoc - 1 - rhs_l2_assoc).max(1);

        let mc_from_lhs_l2_assoc = |lhs_l2_assoc: usize| -> usize {
            (lhs_l2_assoc * l2_cache_bytes) / (l2_assoc * sizeof * auto_kc)
        };

        let auto_mc = round_down(mc_from_lhs_l2_assoc(lhs_l2_assoc), mr);
        let m_iter = m.div_ceil(auto_mc);
        m.div_ceil(m_iter * mr) * mr
    };
    let auto_mc = Ord::min(auto_mc, 8 * mr);

    // l3 cache must hold
    //  - A macropanel: mc×kc: assume 1 assoc degree
    //  - B macropanel: nc×kc
    let auto_nc = if l3_cache_bytes == 0 {
        0
    } else {
        // let lhs_macropanel_bytes = auto_mc * auto_kc * sizeof;
        // let lhs_l3_assoc = msrv_div_ceil(lhs_macropanel_bytes, l3_cache_bytes / l3_assoc);
        let rhs_l3_assoc = l3_assoc - 1;
        let rhs_macropanel_max_bytes = (rhs_l3_assoc * l3_cache_bytes) / l3_assoc;

        let auto_nc = round_down(rhs_macropanel_max_bytes / (sizeof * auto_kc), nr);
        let n_iter = n.div_ceil(auto_nc);
        n.div_ceil(n_iter * nr) * nr
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
        mc: params.nc,
        nc: params.mc,
    }
}

pub(crate) fn pack_a_mixed_precision_single_thread<T, I: Zero>(
    a: Pointer<T>,
    mut packed_a: Pointer<I>,
    lda: i64,
    stride: i64,
    mc: usize,
    kb: usize,
    kc: usize,
    mr: usize,
    cast: fn(&mut I, &T)
) {
    for i in (0..mc).step_by(mr) {
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

pub(crate) fn pack_a_single_thread<T: Zero + Copy>(
    a: Pointer<T>,
    mut packed_a: Pointer<T>,
    lda: i64,
    stride: i64,
    mc: usize,
    kb: usize,
    kc: usize,
    mr: usize
) {
    for i in (0..mc).step_by(mr) {
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

pub(crate) fn pack_b_single_thread<T: Copy, TVec: VecTrait<T>>(
    b: Pointer<T>,
    mut packed_b: Pointer<T>,
    ldb: i64,
    stride: i64,
    nc: usize,
    kb: usize,
    kc: usize,
    nr: usize
) {
    let nr_div_lane = nr / vec_size::<T>();
    for j in (0..nc).step_by(nr) {
        let nb = nr.min(nc - j);
        if nb == nr && stride == 1 {
            for p in 0..kb as i64 {
                unsafe {
                    let b_ptr = b.ptr.offset(((p * ldb) as isize) + (j as isize)) as *const TVec;
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

pub(crate) fn prepack_b_single_thread<T>(
    b: Pointer<T>,
    n: usize,
    k: usize,
    ldb: i64,
    rhs_col_stride: i64,
    kc: usize,
    nc: usize,
    nr: usize,
    parallel: bool
) -> (Vec<Pointer<u8>>, Pointer<u8>, std::alloc::Layout)
    where T: crate::MatmulMicroKernel + Copy
{
    let num_nr_blocks = (nc + nr - 1) / nr;
    let panel_size = num_nr_blocks * nr * kc;
    let num_kc = k.div_ceil(kc);
    let num_nc = n.div_ceil(nc);
    let layout = std::alloc::Layout
        ::from_size_align(panel_size * num_kc * num_nc * size_of::<T>(), ALIGN)
        .unwrap();
    let buffer_raw = unsafe { std::alloc::alloc_zeroed(layout) as *mut T };
    if buffer_raw.is_null() {
        panic!("Failed to allocate memory for prepacked B");
    }
    let packed_b = Pointer::new(buffer_raw, (panel_size * num_kc * num_nc) as i64);
    if !parallel {
        let mut ptrs = vec![Pointer::new(std::ptr::null_mut() as *mut u8, 0); num_kc];
        for (p_idx, p) in (0..k).step_by(kc).enumerate() {
            let pb = min(kc, k - p);
            for (j_idx, j) in (0..n).step_by(nc).enumerate() {
                let jb = min(nc, n - j);
                let tmp = packed_b + ((p_idx * num_nc * panel_size + j_idx * panel_size) as i64);
                pack_b_single_thread::<T, <T as crate::MatmulMicroKernel>::SelfVec>(
                    b + ((p as i64) * ldb + (j as i64) * rhs_col_stride),
                    tmp,
                    ldb,
                    rhs_col_stride,
                    jb,
                    pb,
                    kc,
                    nr
                );
            }
            ptrs[p_idx] = (packed_b + ((p_idx * num_nc * panel_size) as i64)).cast::<u8>();
        }
        (ptrs, packed_b.cast::<u8>(), layout)
    } else {
        let mut work_items = Vec::new();
        for p_idx in 0..num_kc {
            let pb = min(kc, k - p_idx * kc);
            for j_idx in 0..num_nc {
                let jb = min(nc, n - j_idx * nc);
                work_items.push((p_idx, j_idx, pb, jb));
            }
        }
        work_items.into_par_iter().for_each(|(p_idx, j_idx, pb, jb)| {
            let p = p_idx * kc;
            let j = j_idx * nc;
            let tmp = packed_b + ((p_idx * num_nc * panel_size + j_idx * panel_size) as i64);
            pack_b_single_thread::<T, <T as crate::MatmulMicroKernel>::SelfVec>(
                b + ((p as i64) * ldb + (j as i64) * rhs_col_stride),
                tmp,
                ldb,
                rhs_col_stride,
                jb,
                pb,
                kc,
                nr
            );
        });
        let mut ptrs = vec![Pointer::new(std::ptr::null_mut() as *mut u8, 0); num_kc];
        for p_idx in 0..num_kc {
            ptrs[p_idx] = (packed_b + ((p_idx * num_nc * panel_size) as i64)).cast::<u8>();
        }
        (ptrs, packed_b.cast::<u8>(), layout)
    }
}

pub(crate) fn prepack_b_mp_single_thread<T>(
    b: Pointer<T>,
    n: usize,
    k: usize,
    ldb: i64,
    rhs_col_stride: i64,
    kc: usize,
    nc: usize,
    nr: usize,
    parallel: bool,
    pack_vec: fn(*mut IMVec<T>, *const <T as crate::MatmulMicroKernel>::SelfVec, usize),
    pack_vec_exceed: fn(*mut IMVec<T>, usize),
    pack_zero: fn(&mut IM<T>, &T)
)
    -> (Vec<Pointer<u8>>, Pointer<u8>, std::alloc::Layout)
    where T: crate::MatmulMicroKernel + Copy, IM<T>: Zero + Copy
{
    let num_nr_blocks = (nc + nr - 1) / nr;
    let panel_size = num_nr_blocks * nr * kc;
    let num_kc = k.div_ceil(kc);
    let num_nc = n.div_ceil(nc);
    let layout = std::alloc::Layout
        ::from_size_align(panel_size * num_kc * num_nc * size_of::<IM<T>>(), ALIGN)
        .unwrap();
    let buffer_raw = unsafe { std::alloc::alloc_zeroed(layout) as *mut IM<T> };
    if buffer_raw.is_null() {
        panic!("Failed to allocate memory for prepacked B");
    }
    let packed_b = Pointer::new(buffer_raw, (panel_size * num_kc * num_nc) as i64);
    if !parallel {
        let mut ptrs = vec![Pointer::new(std::ptr::null_mut() as *mut u8, 0); num_kc];
        for (p_idx, p) in (0..k).step_by(kc).enumerate() {
            let pb = min(kc, k - p);
            for (j_idx, j) in (0..n).step_by(nc).enumerate() {
                let jb = min(nc, n - j);
                let tmp = packed_b + ((p_idx * num_nc * panel_size + j_idx * panel_size) as i64);
                pack_b_mixed_precision::<
                    T,
                    IM<T>,
                    <T as crate::MatmulMicroKernel>::SelfVec,
                    <T as crate::MatmulMicroKernel>::MixedVec
                >(
                    b + ((p as i64) * ldb + (j as i64) * rhs_col_stride),
                    tmp,
                    ldb,
                    rhs_col_stride,
                    jb,
                    pb,
                    kc,
                    nr,
                    pack_vec,
                    pack_vec_exceed,
                    pack_zero
                );
            }
            ptrs[p_idx] = (packed_b + ((p_idx * num_nc * panel_size) as i64)).cast::<u8>();
        }
        (ptrs, packed_b.cast::<u8>(), layout)
    } else {
        let mut work_items = Vec::new();
        for p_idx in 0..num_kc {
            let pb = min(kc, k - p_idx * kc);
            for j_idx in 0..num_nc {
                let jb = min(nc, n - j_idx * nc);
                work_items.push((p_idx, j_idx, pb, jb));
            }
        }
        work_items.into_par_iter().for_each(|(p_idx, j_idx, pb, jb)| {
            let p = p_idx * kc;
            let j = j_idx * nc;
            let tmp = packed_b + ((p_idx * num_nc * panel_size + j_idx * panel_size) as i64);
            pack_b_mixed_precision::<
                T,
                IM<T>,
                <T as crate::MatmulMicroKernel>::SelfVec,
                <T as crate::MatmulMicroKernel>::MixedVec
            >(
                b + ((p as i64) * ldb + (j as i64) * rhs_col_stride),
                tmp,
                ldb,
                rhs_col_stride,
                jb,
                pb,
                kc,
                nr,
                pack_vec,
                pack_vec_exceed,
                pack_zero
            );
        });
        let mut ptrs = vec![Pointer::new(std::ptr::null_mut() as *mut u8, 0); num_kc];
        for p_idx in 0..num_kc {
            ptrs[p_idx] = (packed_b + ((p_idx * num_nc * panel_size) as i64)).cast::<u8>();
        }
        (ptrs, packed_b.cast::<u8>(), layout)
    }
}

#[allow(dead_code)]
pub struct NewPrePackedRhs {
    pub(crate) buffers: Vec<Pointer<u8>>,
    pub(crate) buffer: (Pointer<u8>, std::alloc::Layout),
    pub(crate) nr: usize,
    pub(crate) nc: usize,
    pub(crate) kc: usize,
}

impl Drop for NewPrePackedRhs {
    fn drop(&mut self) {
        unsafe {
            std::alloc::dealloc(self.buffer.0.ptr as *mut u8, self.buffer.1);
        }
    }
}

#[allow(dead_code)]
pub struct PrePackedLhs {
    pub(crate) buffers: Vec<Vec<Pointer<u8>>>,
    pub(crate) buffer: (Pointer<u8>, std::alloc::Layout),
    pub(crate) mr: usize,
    pub(crate) mc: usize,
    pub(crate) kc: usize,
}

impl Drop for PrePackedLhs {
    fn drop(&mut self) {
        unsafe {
            std::alloc::dealloc(self.buffer.0.ptr as *mut u8, self.buffer.1);
        }
    }
}

pub fn prepack_lhs<T: Zero + Copy>(
    #[cfg(feature = "bound_check")] lhs_ptr: (*const T, i64),
    #[cfg(not(feature = "bound_check"))] lhs_ptr: *const T,
    lhs_strides: [i64; 2],
    mr: usize,
    mc: usize,
    kc: usize,
    k: usize,
    m: usize
) -> PrePackedLhs {
    #[cfg(feature = "bound_check")]
    let lhs_ptr = Pointer::new(lhs_ptr.0 as *mut T, lhs_ptr.1);
    #[cfg(not(feature = "bound_check"))]
    let lhs_ptr = Pointer::new(lhs_ptr as *mut T, 0);
    let lda = lhs_strides[0];
    let lhs_col_stride = lhs_strides[1];
    let num_mr_blocks = (mc + mr - 1) / mr;
    let packed_a_panel_size = num_mr_blocks * mr * kc;
    let num_panels = m.div_ceil(mc) * k.div_ceil(kc);
    let layout = std::alloc::Layout
        ::from_size_align(packed_a_panel_size * num_panels * size_of::<T>(), ALIGN)
        .unwrap();
    let packed_a_buffer = unsafe { std::alloc::alloc(layout) as *mut T };
    let packed_a_buffer = Pointer::new(
        packed_a_buffer,
        (packed_a_panel_size * num_panels * size_of::<T>()) as i64
    );
    let num_i = m.div_ceil(mc);
    let num_p = k.div_ceil(kc);
    let mut packed_a_buffers = vec![vec![packed_a_buffer.cast::<u8>(); num_p]; num_i];
    for i_idx in 0..num_i {
        for p_idx in 0..num_p {
            packed_a_buffers[i_idx][p_idx] = (
                packed_a_buffer +
                ((i_idx * num_p * packed_a_panel_size + p_idx * packed_a_panel_size) as i64) *
                    (size_of::<T>() as i64)
            ).cast::<u8>();
        }
    }
    if num_panels == 1 {
        pack_a_single_thread::<T>(
            lhs_ptr,
            packed_a_buffer,
            lda,
            lhs_col_stride,
            mc.min(m),
            kc.min(k),
            kc,
            mr
        );
    } else {
        let mut work_items = Vec::new();
        for (i_idx, i) in (0..m).step_by(mc).enumerate() {
            let ib = mc.min(m - i);
            for (p_idx, p) in (0..k).step_by(kc).enumerate() {
                let pb = kc.min(k - p);
                work_items.push((i, p, i_idx, p_idx, ib, pb));
            }
        }
        work_items.into_par_iter().for_each(|(i, p, i_idx, p_idx, ib, pb)| {
            pack_a_single_thread::<T>(
                lhs_ptr + (i as i64) * lda + (p as i64) * lhs_col_stride,
                packed_a_buffers[i_idx][p_idx].cast::<T>(),
                lda,
                lhs_col_stride,
                ib,
                pb,
                kc,
                mr
            );
        });
    }
    let prepacked_lhs = PrePackedLhs {
        buffers: packed_a_buffers,
        buffer: (packed_a_buffer.cast::<u8>(), layout),
        mr,
        mc,
        kc,
    };
    prepacked_lhs
}

pub fn prepack_lhs_mp<T: Zero + Copy, IM: Zero + Copy>(
    #[cfg(feature = "bound_check")] lhs_ptr: (*const T, i64),
    #[cfg(not(feature = "bound_check"))] lhs_ptr: *const T,
    lhs_strides: [i64; 2],
    mr: usize,
    mc: usize,
    kc: usize,
    k: usize,
    m: usize,
    pack_zero: fn(&mut IM, &T)
) -> PrePackedLhs {
    #[cfg(feature = "bound_check")]
    let lhs_ptr = Pointer::new(lhs_ptr.0 as *mut T, lhs_ptr.1);
    #[cfg(not(feature = "bound_check"))]
    let lhs_ptr = Pointer::new(lhs_ptr as *mut T, 0);
    let lda = lhs_strides[0];
    let lhs_col_stride = lhs_strides[1];
    let num_mr_blocks = (mc + mr - 1) / mr;
    let packed_a_panel_size = num_mr_blocks * mr * kc;
    let num_panels = m.div_ceil(mc) * k.div_ceil(kc);
    let layout = std::alloc::Layout
        ::from_size_align(packed_a_panel_size * num_panels * size_of::<IM>(), ALIGN)
        .unwrap();
    let packed_a_buffer = unsafe { std::alloc::alloc(layout) as *mut IM };
    let packed_a_buffer = Pointer::new(
        packed_a_buffer,
        (packed_a_panel_size * num_panels * size_of::<IM>()) as i64
    );
    let num_i = m.div_ceil(mc);
    let num_p = k.div_ceil(kc);
    let mut packed_a_buffers = vec![vec![packed_a_buffer.cast::<u8>(); num_p]; num_i];
    for i_idx in 0..num_i {
        for p_idx in 0..num_p {
            packed_a_buffers[i_idx][p_idx] = (
                packed_a_buffer +
                ((i_idx * num_p * packed_a_panel_size + p_idx * packed_a_panel_size) as i64) *
                    (size_of::<IM>() as i64)
            ).cast::<u8>();
        }
    }
    if num_panels == 1 {
        pack_a_mixed_precision_single_thread::<T, IM>(
            lhs_ptr,
            packed_a_buffer,
            lda,
            lhs_col_stride,
            mc.min(m),
            kc.min(k),
            kc,
            mr,
            pack_zero
        );
    } else {
        let mut work_items = Vec::new();
        for (i_idx, i) in (0..m).step_by(mc).enumerate() {
            let ib = mc.min(m - i);
            for (p_idx, p) in (0..k).step_by(kc).enumerate() {
                let pb = kc.min(k - p);
                work_items.push((i, p, i_idx, p_idx, ib, pb));
            }
        }
        work_items.into_par_iter().for_each(|(i, p, i_idx, p_idx, ib, pb)| {
            pack_a_mixed_precision_single_thread::<T, IM>(
                lhs_ptr + (i as i64) * lda + (p as i64) * lhs_col_stride,
                packed_a_buffers[i_idx][p_idx].cast::<IM>(),
                lda,
                lhs_col_stride,
                ib,
                pb,
                kc,
                mr,
                pack_zero
            );
        });
    }
    let prepacked_lhs = PrePackedLhs {
        buffers: packed_a_buffers,
        buffer: (packed_a_buffer.cast::<u8>(), layout),
        mr,
        mc,
        kc,
    };
    prepacked_lhs
}
