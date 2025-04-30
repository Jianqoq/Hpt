use std::collections::HashMap;

use crate::{Tensor, onnx::ModelProto, ops::models::onnx::OnnxModel};

impl OnnxModel {
    pub fn execute(
        &self,
        inputs: HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>, String> {

        match self {
            OnnxModel::Model(model_proto) => {
                if let Some(graph) = model_proto.graph.as_ref() {
                    for initzer in graph.initializer.iter() {
                        println!("初始化器: {:?}", initzer.name);
                    }
                    for input in graph.input.iter() {
                        println!("输入: {:?}", input);
                    }
                    for node in graph.node.iter() {
                        // println!("算子: {:?}, 输入: {:?}, 输出: {:?}", node.op_type, node.input, node.output);
                        // match node.op_type() {
                        //     "Identity" => {

                        //     }
                        //     _ => {
                        //         return Err(format!("unsupported op: {:?}", node.op_type));
                        //     }
                        // }
                    }
                }
            },
        }
        Ok(HashMap::new())
    }
}
