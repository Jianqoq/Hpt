// pub(crate) type TestTypes = half::f16;
// static TCH_TEST_TYPES: tch::Kind = tch::Kind::Half;
// static TEST_RTOL: TestTypes = half::f16::from_f32_const(1e-1);
// static TEST_ATOL: TestTypes = half::f16::from_f32_const(1e-1);
// static EPSILON: TestTypes = half::f16::from_f32_const(1e-5);

// pub(crate) type TestTypes = half::bf16;
// static TCH_TEST_TYPES: tch::Kind = tch::Kind::BFloat16;
// static TEST_RTOL: TestTypes = half::bf16::from_f32_const(1e-1);
// static TEST_ATOL: TestTypes = half::bf16::from_f32_const(1e-1);
// static EPSILON: TestTypes = half::bf16::from_f32_const(1e-5);

pub(crate) type TestTypes = f32;
static DYN_TEST_TYPES: hpt_dyn::DType = hpt_dyn::DType::F32;
static TCH_TEST_TYPES: tch::Kind = tch::Kind::Float;
static TEST_RTOL: TestTypes = 1e-3;
static TEST_ATOL: TestTypes = 1e-3;
static EPSILON: TestTypes = 1e-5;

pub mod hpt {
    pub mod cpu {
        pub mod adaptive_avg_pool;
        pub mod adaptive_max_pool;
        pub mod assert_utils;
        pub mod avg_pool;
        pub mod binary;
        pub mod binary_out;
        pub mod bn_conv2d;
        pub mod conv2d;
        pub mod conv2d_group;
        // pub mod conv2d_transpose;
        pub mod creation;
        pub mod cumulate;
        pub mod dwconv2d;
        pub mod fft;
        pub mod from_raw;
        pub mod gather;
        pub mod matmul;
        pub mod maxpool;
        pub mod onehot;
        pub mod reduce;
        pub mod scatter;
        pub mod shape_manipulate;
        pub mod slice;
        pub mod softmax;
        pub mod test_lib;
        pub mod topk;
        pub mod unary;
        pub mod utils;
    }
    #[cfg(feature = "cuda")]
    pub mod cuda {
        pub mod assert_utils;
        pub mod binary;
        pub mod bn_conv2d;
        pub mod conv2d;
        pub mod conv2d_group;
        pub mod creation;
        pub mod dwconv2d;
        pub mod from_raw;
        pub mod matmul;
        pub mod normalization;
        pub mod reduce;
        pub mod unary;
    }
}

pub mod hpt_common {
    pub mod axis;
    pub mod err_handler;
    pub mod layout;
    pub mod pointer;
    pub mod shape;
    pub mod shape_utils;
    pub mod slice;
    pub mod strides;
}

pub mod hpt_types {
    pub mod test_display;
    pub mod test_vector_index;
    pub mod tests;
}

pub mod hpt_dataloader {
    pub mod save_load;
}

pub(crate) mod utils {
    pub(crate) mod random_utils;
}

// pub mod macro_tests {
//     pub mod tests;
// }
