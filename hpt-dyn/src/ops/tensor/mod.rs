pub(crate) mod creation;
pub(crate) mod unary;
pub(crate) mod binary;
pub(crate) mod shape_manipulate;
pub(crate) mod random;
pub(crate) mod reduce {
    pub(crate) mod reduce_utils;
    pub(crate) mod reduce_template;
    pub(crate) mod reduce;
    pub(crate) mod kernels;
}