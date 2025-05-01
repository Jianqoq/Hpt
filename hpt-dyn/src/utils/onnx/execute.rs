use core::panic;
use std::collections::{ HashMap, HashSet };

use hpt_common::error::{ base::TensorError, common::CommonError, shape::ShapeError };
use hpt_traits::tensor::TensorInfo;

use crate::{
    Tensor,
    onnx::{ TensorShapeProto, tensor_shape_proto, type_proto },
    ops::models::onnx::{ Initialized, Meta, OnnxModel },
};

use super::{ init::{ conv_init, pooling_init, unary_init }, map_dtype::to_dtype };

fn validate_tensor_shape(
    tensor: &Tensor,
    onnx_shape: &TensorShapeProto,
    tensor_name: &str
) -> Result<(), TensorError> {
    if onnx_shape.dim.is_empty() {
        return Ok(());
    }

    if tensor.shape().len() != onnx_shape.dim.len() {
        return Err(
            (ShapeError::InvalidDimension {
                message: format!(
                    "dimension mismatch: tensor '{}' has {} dimensions, but model expects {}",
                    tensor_name,
                    tensor.shape().len(),
                    onnx_shape.dim.len()
                ),
                location: panic::Location::caller(),
            }).into()
        );
    }

    for (i, (tensor_dim, onnx_dim)) in tensor
        .shape()
        .iter()
        .zip(onnx_shape.dim.iter())
        .enumerate() {
        if let Some(dim_value) = &onnx_dim.value {
            match dim_value {
                tensor_shape_proto::dimension::Value::DimValue(expected_size) => {
                    if *tensor_dim != *expected_size && *expected_size != -1 {
                        return Err(
                            (ShapeError::AnyError {
                                message: format!(
                                    "dim mismatch: tensor '{}' dim({}) is {}, but model expects {}",
                                    tensor_name,
                                    i,
                                    tensor_dim,
                                    expected_size
                                ),
                                location: panic::Location::caller(),
                            }).into()
                        );
                    }
                }
                tensor_shape_proto::dimension::Value::DimParam(_) => {
                    continue;
                }
            }
        }
    }

    Ok(())
}

fn validate_tensor_type(
    tensor: &Tensor,
    onnx_type: &type_proto::Value,
    tensor_name: &str
) -> Result<(), TensorError> {
    match onnx_type {
        type_proto::Value::TensorType(tensor_type) => {
            let dtype = to_dtype(tensor_type.elem_type());
            if dtype != tensor.dtype {
                return Err(
                    (CommonError::DtypeMismatch {
                        message: format!(
                            "dtype mismatch: tensor '{}' has dtype {:?}, but model expects {:?}",
                            tensor_name,
                            tensor.dtype,
                            dtype
                        ),
                        location: panic::Location::caller(),
                    }).into()
                );
            }
            if let Some(shape) = &tensor_type.shape {
                validate_tensor_shape(tensor, shape, tensor_name)?;
            }
        }
        type_proto::Value::SequenceType(_) => todo!(),
        type_proto::Value::MapType(_) => todo!(),
        type_proto::Value::OptionalType(_) => todo!(),
        type_proto::Value::SparseTensorType(_) => todo!(),
    }

    Ok(())
}

impl OnnxModel {
    pub fn execute(
        &self,
        inputs: HashMap<String, Tensor>
    ) -> Result<HashMap<String, Tensor>, TensorError> {
        match self {
            OnnxModel::Model(_) => panic!("model not initialized"),
            OnnxModel::Initialized(Initialized { model, initializer_map, permutes }) => {
                let mut tensors = HashMap::new();
                for (name, input) in inputs.iter() {
                    tensors.insert(name.as_str(), input.clone());
                }
                for (name, tensor) in initializer_map.iter() {
                    tensors.insert(name.as_str(), tensor.clone());
                }
                if let Some(graph) = model.graph.as_ref() {
                    for inp in graph.input.iter() {
                        if let Some(tensor) = tensors.get(inp.name()) {
                            if let Some(ty) = &inp.r#type {
                                if let Some(value) = ty.value.as_ref() {
                                    validate_tensor_type(tensor, value, inp.name())?;
                                }
                            }
                            if let Some(meta) = permutes.get(inp.name()) {
                                if let Some(permute) = &meta.permute {
                                    tensors.insert(
                                        inp.name(),
                                        tensor.permute(&permute)?.contiguous()?
                                    );
                                }
                            }
                        } else {
                            panic!("input {:?} not found", inp.name);
                        }
                    }
                    for node in graph.node.iter() {
                        match node.op_type() {
                            "Identity" => {}
                            "Conv" => {
                                let input = &tensors[node.input[0].as_str()];
                                let weight = &tensors[node.input[1].as_str()];
                                let bias = tensors.get(node.input[2].as_str());
                                assert_eq!(node.attribute[0].name(), "dilations");
                                let dilations = &node.attribute[0].ints.as_slice();
                                assert_eq!(dilations.len(), 2);
                                // assert_eq!(node.attribute[1].name(), "group");
                                // let group = node.attribute[1].i;
                                // assert_eq!(node.attribute[2].name(), "kernel_shape");
                                // let kernel_shape = &node.attribute[2].ints.as_slice();
                                assert_eq!(node.attribute[3].name(), "pads");
                                let pads = node.attribute[3].ints.as_slice();
                                // assert_eq!(pads.len(), 2);
                                assert_eq!(node.attribute[4].name(), "strides");
                                let strides = node.attribute[4].ints.as_slice();
                                assert_eq!(strides.len(), 2);
                                let output = input.conv2d(
                                    &weight,
                                    bias,
                                    [strides[0], strides[1]],
                                    [
                                        (pads[0], pads[1]),
                                        (pads[2], pads[3]),
                                    ],
                                    [dilations[0], dilations[1]]
                                )?;
                                tensors.insert(node.output[0].as_str(), output);
                            }
                            "Relu" => {
                                let input = &tensors[node.input[0].as_str()];
                                let output = input.relu()?;
                                tensors.insert(node.output[0].as_str(), output);
                            }
                            "MaxPool" => {
                                let input = &tensors[node.input[0].as_str()];
                                // let ceil_mode = node.attribute[0].i == Some(1);
                                let dilations = node.attribute[1].ints.as_slice();
                                let kernel_shape = node.attribute[2].ints.as_slice();
                                let pads = node.attribute[3].ints.as_slice();
                                let strides = node.attribute[4].ints.as_slice();

                                let output = input.maxpool2d(
                                    &kernel_shape,
                                    [strides[0], strides[1]],
                                    [
                                        (pads[0], pads[1]),
                                        (pads[2], pads[3]),
                                    ],
                                    [dilations[0], dilations[1]]
                                )?;
                                tensors.insert(node.output[0].as_str(), output);
                            }
                            "Add" => {
                                let input = &tensors[node.input[0].as_str()];
                                let other = &tensors[node.input[1].as_str()];
                                let output = input + other;
                                tensors.insert(node.output[0].as_str(), output);
                            }
                            "GlobalAveragePool" => {
                                let input = &tensors[node.input[0].as_str()];
                                let output = input.adaptive_avgpool2d([1, 1])?;
                                tensors.insert(node.output[0].as_str(), output);
                            }
                            "Flatten" => {
                                let input = &tensors[node.input[0].as_str()];
                                let output = if let Some(axis) = node.attribute[0].i {
                                    input.flatten(axis, axis)?
                                } else {
                                    let axes = node.attribute[0].ints.as_slice();
                                    input.flatten(axes[0], axes[1])?
                                };
                                tensors.insert(node.output[0].as_str(), output);
                            }
                            "Gemm" => {
                                let alpha = node.attribute[0].f.map(|f| f as f64);
                                let beta = node.attribute[1].f.map(|f| f as f64);
                                let input = &tensors[node.input[0].as_str()];
                                let weight = &tensors[node.input[1].as_str()];
                                let trans_b = node.attribute[2].i == Some(1);
                                let bias = tensors.get(node.input[2].as_str());
                                let output = if trans_b {
                                    input.gemm(
                                        &weight.t()?,
                                        bias,
                                        alpha.unwrap_or(1.0),
                                        beta.unwrap_or(0.0)
                                    )?
                                } else {
                                    input.gemm(
                                        &weight,
                                        bias,
                                        alpha.unwrap_or(1.0),
                                        beta.unwrap_or(0.0)
                                    )?
                                };
                                tensors.insert(node.output[0].as_str(), output);
                            }
                            _ => {
                                println!(
                                    "operator: {:?}, input: {:?}, output: {:?}, attribute: {:?}",
                                    node.op_type,
                                    node.input,
                                    node.output,
                                    node.attribute
                                );
                                panic!("unsupported op: {:?}", node.op_type);
                            }
                        }
                    }
                }
            }
        }
        Ok(HashMap::new())
    }

    pub fn initialize(self) -> Result<Self, TensorError> {
        match self {
            OnnxModel::Model(mut model_proto) => {
                let mut initializer_map = HashMap::new();
                let mut permutes = HashMap::new();
                let mut node_degree = HashMap::new();

                let mut all_inputs = HashSet::new();
                let mut operators = Vec::new();
                if let Some(graph) = model_proto.graph.as_mut() {
                    for input in graph.input.iter() {
                        let name = input.name();
                        permutes.insert(name.to_string(), Meta { permute: None });
                        all_inputs.insert(name.to_string());
                    }
                    for initializer in graph.initializer.iter() {
                        let name = initializer.name();
                        permutes.insert(name.to_string(), Meta { permute: None });
                        all_inputs.insert(name.to_string());
                    }
                    for node in graph.node.iter() {
                        match node.op_type() {
                            "Conv" =>
                                operators.push(
                                    conv_init(node, &mut permutes, &mut node_degree, &all_inputs)
                                ),
                            "MaxPool" | "GlobalAveragePool" | "GlobalMaxPool" | "AveragePool" => {
                                pooling_init(node, &mut permutes, &mut node_degree, &all_inputs);
                            }
                            | "Abs"
                            | "Acos"
                            | "Acosh"
                            | "Asin"
                            | "Asinh"
                            | "Atan"
                            | "Atanh"
                            | "BitwiseNot"
                            | "Ceil"
                            | "Cos"
                            | "Cosh"
                            | "Erf"
                            | "Exp"
                            | "Floor"
                            | "IsInf"
                            | "IsNaN"
                            | "Log"
                            | "Neg"
                            | "Not"
                            | "Reciprocal"
                            | "Round"
                            | "Sigmoid"
                            | "Sign"
                            | "Sin"
                            | "Sinh"
                            | "Sqrt"
                            | "Tan"
                            | "Tanh"
                            | "Gelu"
                            | "HardSigmoid"
                            | "HardSwish"
                            | "LeakyRelu"
                            | "Mish"
                            | "Shrink"
                            | "Relu"
                            | "Softplus"
                            | "Softsign" =>
                                operators.push(
                                    unary_init(node, &mut permutes, &mut node_degree, &all_inputs)
                                ),
                            "Identity" | "Add" | "Flatten" | "Gemm" => {}
                            _ =>
                                unimplemented!(
                                    "unsupported op when initializing: {:?}",
                                    node.op_type
                                ),
                        }
                    }

                    for initializer in graph.initializer.iter_mut() {
                        let name = initializer.name().to_string();
                        let tensor = Tensor::from_onnx_tensor(
                            initializer,
                            &permutes[&name].permute
                        )?;
                        initializer_map.insert(name, tensor);
                    }
                }
                Ok(
                    OnnxModel::Initialized(Initialized {
                        model: model_proto,
                        initializer_map,
                        permutes,
                    })
                )
            }
            OnnxModel::Initialized(_) => panic!("model already initialized"),
        }
    }
}
