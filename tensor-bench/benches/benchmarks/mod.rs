pub mod unary {
    pub mod float_cmp;
    pub mod unary_benches;
}
pub mod shape_manipulate {
    pub mod concat;
}
pub mod reduction {
    pub mod reduction_benches;
}
pub mod conv {
    #[cfg(any(feature = "f32", feature = "conv2d"))]
    pub mod conv2d;
    #[cfg(any(feature = "f32", feature = "maxpool"))]
    pub mod maxpool;
}
pub mod signals {
    #[cfg(feature = "hamming")]
    pub mod hamming_window;
}
pub mod softmax {
    pub mod softmax;
}
