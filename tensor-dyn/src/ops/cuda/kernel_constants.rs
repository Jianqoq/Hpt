pub(crate) const FILL_KERNELS: [&str; 11] = [
    "fill_f16",
    "fill_f32",
    "fill_f64",
    "fill_i8",
    "fill_i16",
    "fill_i32",
    "fill_i64",
    "fill_u8",
    "fill_u16",
    "fill_u32",
    "fill_u64",
];

pub(crate) const ARANGE_KERNELS: [&str; 11] = [
    "arange_f16",
    "arange_f32",
    "arange_f64",
    "arange_i8",
    "arange_i16",
    "arange_i32",
    "arange_i64",
    "arange_u8",
    "arange_u16",
    "arange_u32",
    "arange_u64",
];

pub(crate) const LINSPACE_KERNELS: [&str; 11] = [
    "linspace_f16",
    "linspace_f32",
    "linspace_f64",
    "linspace_i8",
    "linspace_i16",
    "linspace_i32",
    "linspace_i64",
    "linspace_u8",
    "linspace_u16",
    "linspace_u32",
    "linspace_u64",
];

pub(crate) const LOGSPACE_KERNELS: [&str; 11] = [
    "logspace_f16",
    "logspace_f32",
    "logspace_f64",
    "logspace_i8",
    "logspace_i16",
    "logspace_i32",
    "logspace_i64",
    "logspace_u8",
    "logspace_u16",
    "logspace_u32",
    "logspace_u64",
];

pub(crate) const GEOMSPACE_KERNELS: [&str; 11] = [
    "geomspace_f16",
    "geomspace_f32",
    "geomspace_f64",
    "geomspace_i8",
    "geomspace_i16",
    "geomspace_i32",
    "geomspace_i64",
    "geomspace_u8",
    "geomspace_u16",
    "geomspace_u32",
    "geomspace_u64",
];

pub(crate) const EYE_KERNELS: [&str; 11] = [
    "eye_f16", "eye_f32", "eye_f64", "eye_i8", "eye_i16", "eye_i32", "eye_i64", "eye_u8",
    "eye_u16", "eye_u32", "eye_u64",
];

pub(crate) const TRIU_KERNELS: [&str; 11] = [
    "triu_f16", "triu_f32", "triu_f64", "triu_i8", "triu_i16", "triu_i32", "triu_i64", "triu_u8",
    "triu_u16", "triu_u32", "triu_u64",
];

pub(crate) const REDUCE_KERNELS: [&str; 4] = [
    "contiguous_reduce",
    "contiguous_reduce2",
    "contiguous_reduce22",
    "contiguous_reduce3",
];
