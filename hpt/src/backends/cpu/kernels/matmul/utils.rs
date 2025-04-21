use gemm_common::cache::{KernelParams, CACHE_INFO};
use num::integer::gcd;

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

    let gcd = gcd(nr * sizeof, l1_line_bytes * l1_n_sets);
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
    let auto_nc = Ord::min(auto_nc, 1 * nr);

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

pub(crate) fn kernel_params_n_k_m(
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

    let gcd = gcd(nr * sizeof, l1_line_bytes * l1_n_sets);
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
    let auto_nc = Ord::min(auto_nc, 1 * nr);

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
