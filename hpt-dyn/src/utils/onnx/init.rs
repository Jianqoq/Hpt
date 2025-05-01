use crate::{onnx::NodeProto, ops::models::onnx::Meta};
use std::collections::{HashMap, HashSet};

pub(crate) fn conv_init(
    node: &NodeProto,
    permutes: &mut HashMap<String, Meta>,
    node_degree: &mut HashMap<String, u32>,
    all_inputs: &HashSet<String>,
) {
    let input_name = node.input[0].as_str();
    if all_inputs.contains(input_name) {
        permutes
            .entry(input_name.to_string())
            .or_insert(Meta {
                permute: Some(vec![0, 2, 3, 1]),
            })
            .permute = Some(vec![0, 2, 3, 1]);
    }
    let kernel_name = node.input[1].as_str();
    if all_inputs.contains(kernel_name) {
        permutes
            .entry(kernel_name.to_string())
            .or_insert(Meta {
                permute: Some(vec![2, 3, 1, 0]),
            })
            .permute = Some(vec![2, 3, 1, 0]);
    }
    let bias_name = node.input[2].as_str();
    permutes
        .entry(bias_name.to_string())
        .or_insert(Meta { permute: None });
    *node_degree.entry(input_name.to_string()).or_insert(0) += 1;
    *node_degree.entry(kernel_name.to_string()).or_insert(0) += 1;
    *node_degree.entry(bias_name.to_string()).or_insert(0) += 1;
}

pub(crate) fn pooling_init(
    node: &NodeProto,
    permutes: &mut HashMap<String, Meta>,
    node_degree: &mut HashMap<String, u32>,
    all_inputs: &HashSet<String>,
) {
    let input_name = node.input[0].as_str();
    if all_inputs.contains(input_name) {
        permutes
            .entry(input_name.to_string())
            .or_insert(Meta {
                permute: Some(vec![0, 2, 3, 1]),
            })
            .permute = Some(vec![0, 2, 3, 1]);
    }
    *node_degree.entry(input_name.to_string()).or_insert(0) += 1;
}
