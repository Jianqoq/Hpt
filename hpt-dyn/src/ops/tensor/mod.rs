pub(crate) mod creation;
pub(crate) mod unary;
pub(crate) mod binary;
pub(crate) mod shape_manipulate;
pub(crate) mod random;
pub(crate) mod cmp;
pub(crate) mod reduce {
    pub(crate) mod reduce_utils;
    pub(crate) mod reduce_template;
    pub(crate) mod reduce;
    pub(crate) mod kernels;
}
pub(crate) mod matmul {
    pub(crate) mod common;
    pub(crate) mod microkernels;
    pub(crate) mod microkernel_trait;
    pub(crate) mod template;
    pub(crate) mod matmul;
    pub(crate) mod matmul_post;
    pub(crate) mod matmul_mp;
    pub(crate) mod matmul_mp_post;
    pub(crate) mod utils;
    pub(crate) mod type_kernels {
        pub(crate) mod i8_microkernels;
        pub(crate) mod u8_microkernels;
        pub(crate) mod f16_microkernels;
        pub(crate) mod f32_microkernels;
        pub(crate) mod bf16_microkernels;
    }
}

pub(crate) mod normalization {
    pub(crate) mod batch_norm;
    pub(crate) mod normalize_utils;
    pub(crate) mod softmax;
    pub(crate) mod kernels;
    pub(crate) mod logsoftmax;
}

pub(crate) mod conv2d {
    pub(crate) mod conv2d;
    pub(crate) mod conv2d_direct;
    pub(crate) mod conv2d_group;
    pub(crate) mod conv2d_img2col;
    pub(crate) mod conv2d_micro_kernels;
    pub(crate) mod conv2d_mp;
    pub(crate) mod batchnorm_conv2d;
    pub(crate) mod microkernel_trait;
    pub(crate) mod utils;
    pub(crate) mod type_kernels {
        pub(crate) mod i8_microkernels;
        pub(crate) mod u8_microkernels;
        pub(crate) mod f16_microkernels;
        pub(crate) mod f32_microkernels;
        pub(crate) mod bf16_microkernels;
    }
}

pub(crate) mod pooling {
    pub(crate) mod template;
    pub(crate) mod maxpool;
    pub(crate) mod avgpool;
}
