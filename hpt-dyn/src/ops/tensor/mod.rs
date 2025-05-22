pub(crate) mod binary;
pub(crate) mod cmp;
pub(crate) mod copy;
pub(crate) mod creation;
pub(crate) mod gather;
pub(crate) mod random;
pub(crate) mod shape_manipulate;
pub(crate) mod unary;
pub(crate) mod reduce {
    pub(crate) mod kernels;
    pub(crate) mod reduce;
    pub(crate) mod reduce_template;
    pub(crate) mod reduce_utils;
}
pub(crate) mod matmul {
    pub(crate) mod common;
    pub(crate) mod matmul;
}

pub(crate) mod normalization {
    pub(crate) mod batch_norm;
    pub(crate) mod kernels;
    pub(crate) mod logsoftmax;
    pub(crate) mod normalize_utils;
    pub(crate) mod softmax;
}

pub(crate) mod conv2d {
    pub(crate) mod batchnorm_conv2d;
    pub(crate) mod conv2d;
    pub(crate) mod conv2d_direct;
    pub(crate) mod conv2d_group;
    pub(crate) mod conv2d_img2col;
    pub(crate) mod conv2d_micro_kernels;
    pub(crate) mod conv2d_mp;
    pub(crate) mod microkernel_trait;
    pub(crate) mod utils;
    pub(crate) mod type_kernels {
        #[cfg(feature = "bf16")]
        pub(crate) mod bf16_microkernels;
        #[cfg(feature = "f16")]
        pub(crate) mod f16_microkernels;
        #[cfg(feature = "f32")]
        pub(crate) mod f32_microkernels;
        #[cfg(feature = "i8")]
        pub(crate) mod i8_microkernels;
        #[cfg(feature = "u8")]
        pub(crate) mod u8_microkernels;
        #[cfg(feature = "i16")]
        pub(crate) mod i16_microkernels;
        #[cfg(feature = "u16")]
        pub(crate) mod u16_microkernels;
        #[cfg(feature = "i32")]
        pub(crate) mod i32_microkernels;
        #[cfg(feature = "u32")]
        pub(crate) mod u32_microkernels;
        #[cfg(feature = "i64")]
        pub(crate) mod i64_microkernels;
        #[cfg(feature = "u64")]
        pub(crate) mod u64_microkernels;
        #[cfg(feature = "f64")]
        pub(crate) mod f64_microkernels;
        #[cfg(feature = "bool")]
        pub(crate) mod bool_microkernels;
    }
}

pub(crate) mod pooling {
    pub(crate) mod avgpool;
    pub(crate) mod maxpool;
    pub(crate) mod template;
}

pub(crate) mod rnn {
    pub(crate) mod lstm;
}
