pub mod tensor_dyn {
    pub mod cpu {
        pub mod binary;
        pub mod binary_out;
        pub mod bn_conv2d;
        pub mod conv2d;
        pub mod conv2d_group;
        pub mod creation;
        pub mod cumulate;
        pub mod dwconv2d;
        pub mod lp_pool2d;
        pub mod maxpool;
        pub mod pwconv2d;
        pub mod reduce;
        pub mod shape_manipulate;
        pub mod slice;
        pub mod softmax;
        pub mod test_lib;
        pub mod topk;
        pub mod unary;
        pub mod avg_pool;
    }
    #[cfg(feature = "cuda")]
    pub mod cuda {
        pub mod creation;
    }
}

pub mod tensor_common {
    pub mod axis;
    pub mod err_handler;
    pub mod layout;
    pub mod pointer;
    pub mod shape;
    pub mod shape_utils;
    pub mod strides;
}

pub mod tensor_types {
    pub mod test_display;
    pub mod test_vector_index;
    pub mod tests;
}

pub mod macro_tests {
    pub mod tests;
}
