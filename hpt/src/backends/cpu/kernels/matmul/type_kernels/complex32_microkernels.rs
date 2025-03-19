use crate::backends::cpu::kernels::matmul::microkernel_trait::MatmulMicroKernel;
use num::complex::Complex32;
impl MatmulMicroKernel for Complex32 {}
