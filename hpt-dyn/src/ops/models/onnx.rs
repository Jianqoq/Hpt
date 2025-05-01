use std::collections::HashMap;

use crate::Tensor;
use crate::onnx::ModelProto;

#[derive(Debug)]
pub(crate) struct Meta {
    pub(crate) permute: Option<Vec<i64>>,
}

#[derive(Debug)]
pub struct Initialized {
    pub(crate) model: ModelProto,
    pub(crate) initializer_map: HashMap<String, Tensor>,
    pub(crate) permutes: HashMap<String, Meta>,
}

#[derive(Debug)]
pub enum OnnxModel {
    Model(ModelProto),
    Initialized(Initialized),
}
