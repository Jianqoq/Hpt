use num::complex::Complex64;

use crate::backends::cpu::kernels::conv2d::microkernel_trait::Conv2dMicroKernel;

impl Conv2dMicroKernel for Complex64 {}
