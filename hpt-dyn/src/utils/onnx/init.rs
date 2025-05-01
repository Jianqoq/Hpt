use crate::{onnx::NodeProto, ops::models::onnx::Meta, utils::onnx::operators::Conv2d};
use std::collections::{HashMap, HashSet};

use super::operators::{Operator, Unary};

pub(crate) fn conv_init(
    node: &NodeProto,
    permutes: &mut HashMap<String, Meta>,
    node_degree: &mut HashMap<String, u32>,
    all_inputs: &HashSet<String>,
) -> Operator {
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

    assert_eq!(node.attribute[0].name(), "dilations");
    let dilations = &node.attribute[0].ints.as_slice();
    assert_eq!(dilations.len(), 2);
    assert_eq!(node.attribute[3].name(), "pads");
    let pads = node.attribute[3].ints.as_slice();
    assert_eq!(node.attribute[4].name(), "strides");
    let strides = node.attribute[4].ints.as_slice();
    assert_eq!(strides.len(), 2);

    Operator::Conv2d(Conv2d {
        input: input_name.to_string(),
        output: node.output[0].to_string(),
        kernel: kernel_name.to_string(),
        bias: bias_name.to_string(),
        pads: [(pads[0], pads[1]), (pads[2], pads[3])],
        strides: [strides[0], strides[1]],
        dilations: [dilations[0], dilations[1]],
        group: node.attribute[1].i.unwrap_or(1),
    })
}

pub(crate) fn unary_init(
    node: &NodeProto,
    permutes: &mut HashMap<String, Meta>,
    node_degree: &mut HashMap<String, u32>,
    all_inputs: &HashSet<String>,
) -> Operator {
    let input_name = node.input[0].as_str();
    if all_inputs.contains(input_name) {
        permutes
            .entry(input_name.to_string())
            .or_insert(Meta { permute: None });
    }
    *node_degree.entry(input_name.to_string()).or_insert(0) += 1;
    let unary = Unary {
        input: input_name.to_string(),
        output: node.output[0].to_string(),
    };
    match node.op_type() {
        "Abs" => Operator::Abs(unary),
        "Acos" => Operator::Acos(unary),
        "Acosh" => Operator::Acosh(unary),
        "Asin" => Operator::Asin(unary),
        "Asinh" => Operator::Asinh(unary),
        "Atan" => Operator::Atan(unary),
        "Atanh" => Operator::Atanh(unary),
        "BitwiseNot" => Operator::BitwiseNot(unary),
        "Ceil" => Operator::Ceil(unary),
        "Cos" => Operator::Cos(unary),
        "Cosh" => Operator::Cosh(unary),
        "Erf" => Operator::Erf(unary),
        "Exp" => Operator::Exp(unary),
        "Floor" => Operator::Floor(unary),
        "IsInf" => Operator::IsInf(unary),
        "IsNaN" => Operator::IsNaN(unary),
        "Log" => Operator::Log(unary),
        "Neg" => Operator::Neg(unary),
        "Not" => Operator::Not(unary),
        "Reciprocal" => Operator::Reciprocal(unary),
        "Round" => Operator::Round(unary),
        "Sigmoid" => Operator::Sigmoid(unary),
        "Sign" => Operator::Sign(unary),
        "Sin" => Operator::Sin(unary),
        "Sinh" => Operator::Sinh(unary),
        "Sqrt" => Operator::Sqrt(unary),
        "Tan" => Operator::Tan(unary),
        "Tanh" => Operator::Tanh(unary),
        "Gelu" => Operator::Gelu(unary),
        "HardSigmoid" => Operator::HardSigmoid(unary),
        "HardSwish" => Operator::HardSwish(unary),
        "LeakyRelu" => Operator::LeakyRelu(unary),
        "Mish" => Operator::Mish(unary),
        "Shrink" => Operator::Shrink(unary),
        "Relu" => Operator::Relu(unary),
        "Softplus" => Operator::Softplus(unary),
        "Softsign" => Operator::Softsign(unary),
        _ => unimplemented!("unary operator {} not implemented", node.op_type()),
    }
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
