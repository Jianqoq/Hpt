use std::{fs::File, io::Read};

use prost::Message;

use crate::{onnx::ModelProto, ops::models::onnx::OnnxModel};

pub fn load_onnx(path: &str) -> Result<OnnxModel, String> {
    let mut file = File::open(path).expect("找不到模型文件");
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();

    let model = ModelProto::decode(&*buf).expect("模型解析失败");

    Ok(OnnxModel::Model(model))
}
