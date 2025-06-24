pub(crate) mod allocator;
pub(crate) mod backend;
pub(crate) mod device;
pub(crate) mod display;
pub(crate) mod index_cal;
pub(crate) mod prefetch;
pub(crate) mod onnx {
    pub(crate) mod load_model;
    pub(crate) mod proto;
    pub(crate) mod execute;
    pub(crate) mod map_dtype;
    pub(crate) mod init;
    pub(crate) mod operators;
    pub(crate) mod fwd;
    pub(crate) mod layout_sense;
    pub(crate) mod build_graph;
    pub(crate) mod plot;
    pub(crate) mod run_fwd;
    pub(crate) mod run_init;
    pub(crate) mod optimize {
        pub(crate) mod constant_fold;
        pub(crate) mod fuse;
    }
    pub(crate) mod parse_args {
        pub(crate) mod parse;
        pub(crate) mod affine_grid;
        pub(crate) mod squeeze;
    }
}
pub(crate) mod threadpool;