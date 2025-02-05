pub mod hpt_core {
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
        pub mod conv2d_transpose;
        pub mod creation;
        pub mod cumulate;
        pub mod dwconv2d;
        pub mod gather;
        pub mod maxpool;
        pub mod onehot;
        pub mod pwconv2d;
        pub mod reduce;
        pub mod scatter;
        pub mod shape_manipulate;
        pub mod slice;
        pub mod softmax;
        pub mod test_lib;
        pub mod topk;
        pub mod unary;
    }
    #[cfg(feature = "cuda")]
    pub mod cuda {
        pub mod creation;
    }
}

pub mod hpt_common {
    pub mod axis;
    pub mod err_handler;
    pub mod layout;
    pub mod pointer;
    pub mod shape;
    pub mod shape_utils;
    pub mod strides;
}

pub mod hpt_types {
    pub mod test_display;
    pub mod test_vector_index;
    pub mod tests;
}

pub mod macro_tests {
    pub mod tests;
}
