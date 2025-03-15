use std::{cell::RefCell, rc::Rc, time::Instant};

pub mod unary {
    pub mod float_cmp;
    pub mod unary_benches;
}
pub mod binary {
    pub mod add_broadcast_f32;
    pub mod add_f32;
    pub mod matmul_f32;
}
pub mod shape_manipulate {
    pub mod concat;
}
pub mod reduction {
    pub mod reduction_benches;
}
pub mod conv {
    #[cfg(any(feature = "f32", feature = "conv2d"))]
    pub mod conv2d;
    #[cfg(any(feature = "f32", feature = "maxpool"))]
    pub mod maxpool;
}
pub mod signals {
    #[cfg(feature = "hamming")]
    pub mod hamming_window;
}
pub mod softmax {
    pub mod softmax;
}
pub mod fft {
    pub mod fft;
}

pub struct GFlops {
    ns: Vec<f64>,
    idx: Rc<RefCell<usize>>,
}
impl criterion::measurement::Measurement for GFlops {
    type Intermediate = Instant;
    type Value = f64;

    fn start(&self) -> Self::Intermediate {
        Instant::now()
    }
    fn end(&self, i: Self::Intermediate) -> Self::Value {
        self.ns[*self.idx.borrow()] / (i.elapsed().as_secs_f64() * 1e9)
    }
    fn add(&self, v1: &Self::Value, v2: &Self::Value) -> Self::Value {
        *v1 + *v2
    }
    fn zero(&self) -> Self::Value {
        0.0
    }
    fn to_f64(&self, val: &Self::Value) -> f64 {
        *val
    }
    fn formatter(&self) -> &dyn criterion::measurement::ValueFormatter {
        &GFlopsFormatter
    }
}

pub struct GFlopsFormatter;
impl criterion::measurement::ValueFormatter for GFlopsFormatter {
    fn format_value(&self, value: f64) -> String {
        format!("{:.20} GFLOPS", value)
    }

    fn scale_values(&self, _: f64, _: &mut [f64]) -> &'static str {
        "GFLOPS"
    }

    fn scale_throughputs(&self, _: f64, _: &criterion::Throughput, _: &mut [f64]) -> &'static str {
        "GFLOPS"
    }

    fn scale_for_machines(&self, _: &mut [f64]) -> &'static str {
        "GFLOPS"
    }
}

pub struct Timer;
impl criterion::measurement::Measurement for Timer {
    type Intermediate = Instant;
    type Value = u128;

    fn start(&self) -> Self::Intermediate {
        Instant::now()
    }
    fn end(&self, i: Self::Intermediate) -> Self::Value {
        i.elapsed().as_millis()
    }
    fn add(&self, v1: &Self::Value, v2: &Self::Value) -> Self::Value {
        *v1 + *v2
    }
    fn zero(&self) -> Self::Value {
        0
    }
    fn to_f64(&self, val: &Self::Value) -> f64 {
        *val as f64
    }
    fn formatter(&self) -> &dyn criterion::measurement::ValueFormatter {
        &TimerFormatter
    }
}

pub struct TimerFormatter;
impl criterion::measurement::ValueFormatter for TimerFormatter {
    fn format_value(&self, value: f64) -> String {
        format!("{:.8} ms", value)
    }

    fn scale_values(&self, _: f64, _: &mut [f64]) -> &'static str {
        "ms"
    }

    fn scale_throughputs(&self, _: f64, _: &criterion::Throughput, _: &mut [f64]) -> &'static str {
        "ms"
    }

    fn scale_for_machines(&self, _: &mut [f64]) -> &'static str {
        "ms"
    }
}
