use crate::backends::cpu::kernels::matmul::microkernel_trait::MatmulMicroKernel;
use num::complex::Complex64;
impl MatmulMicroKernel for Complex64 {}
