use crate::onnx::ModelProto;

#[derive(Debug)]
pub enum OnnxModel {
    Model(ModelProto),
}