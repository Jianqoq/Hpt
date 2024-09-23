pub mod tensor_dyn {
    pub mod conv2d;
    pub mod max_pool;
    pub mod reduce;
    pub mod unary;
    pub mod creation;
    pub mod shape_manipulate;
    pub mod binary;
    pub mod avg_pool2d;
    pub mod lp_pool2d;
    pub mod slice;
    pub mod test_lib;
}

pub mod tensor_common {
    pub mod shape_utils;
    pub mod axis;
    pub mod err_handler;
    pub mod layout;
    pub mod pointer;
    pub mod shape;
    pub mod strides;
}