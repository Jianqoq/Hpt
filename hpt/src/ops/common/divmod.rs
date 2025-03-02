#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub(crate) struct FastDivmod {
    pub divisor: i32,
    pub multiplier: u32,
    pub shift_right: u32,
}

// https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/fast_math.h#L247
fn find_log2_i32(x: i32) -> i32 {
    let mut a = (31 - x.leading_zeros()) as i32;
    a += ((x & (x - 1)) != 0) as i32;
    a
}

#[allow(unused)]
fn fast_divmod(quo: &mut i32, rem: &mut i32, src: i32, div: i32, mul: u32, shr: u32) {
    if div != 1 {
        *quo = ((src as i64 * mul as i64) >> 32) as i32 >> shr;
    } else {
        *quo = src;
    }
    *rem = src - *quo * div;
}

impl FastDivmod {
    pub(crate) fn new(divisor: i32) -> Self {
        assert!(divisor >= 0, "Divisor must be non-negative");

        if divisor == 1 {
            return Self {
                divisor: 1,
                multiplier: 0,
                shift_right: 0,
            };
        }
        let p = (31 + find_log2_i32(divisor)) as u32;

        let m = ((1u64 << p) + (divisor as u64) - 1) / (divisor as u64);
        let multiplier = m as u32;

        Self {
            divisor,
            multiplier,
            shift_right: p - 32,
        }
    }

    #[allow(unused)]
    pub(crate) fn divmod(&self, dividend: i32) -> (i32, i32) {
        let mut quo = 0;
        let mut rem = 0;
        fast_divmod(
            &mut quo,
            &mut rem,
            dividend,
            self.divisor,
            self.multiplier,
            self.shift_right,
        );
        (quo, rem)
    }
}

#[cfg(feature = "cuda")]
use cudarc::driver::DeviceRepr;
#[cfg(feature = "cuda")]
unsafe impl DeviceRepr for FastDivmod {}
