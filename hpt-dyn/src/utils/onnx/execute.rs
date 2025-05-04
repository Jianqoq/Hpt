#![allow(unused)]

use core::panic;
use std::collections::{HashMap, HashSet};

use hpt_common::error::{base::TensorError, common::CommonError, shape::ShapeError};
use hpt_traits::tensor::TensorInfo;

use crate::{
    Tensor,
    onnx::{TensorShapeProto, tensor_shape_proto, type_proto},
    ops::models::onnx::{Initialized, OnnxModel},
};

use super::{fwd::*, init::*, map_dtype::to_dtype};

fn validate_tensor_shape(
    tensor: &Tensor,
    onnx_shape: &TensorShapeProto,
    tensor_name: &str,
) -> Result<(), TensorError> {
    if onnx_shape.dim.is_empty() {
        return Ok(());
    }

    if tensor.shape().len() != onnx_shape.dim.len() {
        return Err((ShapeError::InvalidDimension {
            message: format!(
                "dimension mismatch: tensor '{}' has shape [{}] but model expects shape [{}]",
                tensor_name,
                tensor
                    .shape()
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
                onnx_shape
                    .dim
                    .iter()
                    .map(|d| match &d.value {
                        Some(value) => match value {
                            tensor_shape_proto::dimension::Value::DimValue(dim) => dim.to_string(),
                            tensor_shape_proto::dimension::Value::DimParam(param) =>
                                param.to_string(),
                        },
                        None => "?".to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join(", "),
            ),
            location: panic::Location::caller(),
        })
        .into());
    }

    for (i, (tensor_dim, onnx_dim)) in tensor.shape().iter().zip(onnx_shape.dim.iter()).enumerate()
    {
        if let Some(dim_value) = &onnx_dim.value {
            match dim_value {
                tensor_shape_proto::dimension::Value::DimValue(expected_size) => {
                    if *tensor_dim != *expected_size && *expected_size != -1 {
                        return Err((ShapeError::AnyError {
                            message: format!(
                                "dim mismatch: tensor '{}' dim({}) is {}, but model expects {}",
                                tensor_name, i, tensor_dim, expected_size
                            ),
                            location: panic::Location::caller(),
                        })
                        .into());
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
    tensor_name: &str,
) -> Result<(), TensorError> {
    match onnx_type {
        type_proto::Value::TensorType(tensor_type) => {
            let dtype = to_dtype(tensor_type.elem_type());
            if dtype != tensor.dtype {
                return Err((CommonError::DtypeMismatch {
                    message: format!(
                        "dtype mismatch: tensor '{}' has dtype {:?}, but model expects {:?}",
                        tensor_name, tensor.dtype, dtype
                    ),
                    location: panic::Location::caller(),
                })
                .into());
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
        inputs: HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>, TensorError> {
        match self {
            OnnxModel::Model(_) => panic!("model not initialized"),
            OnnxModel::Initialized(Initialized {
                model,
                initializer_map,
                permutes,
                operators,
            }) => {
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
                                        tensor.permute(&permute)?.contiguous()?,
                                    );
                                }
                            }
                        } else {
                            panic!("input {:?} not found", inp.name);
                        }
                    }
                    // for node in graph.node.iter() {
                    //     match node.op_type() {
                    //         "Identity" => {}
                    //         "Conv" => {
                    //             let input = &tensors[node.input[0].as_str()];
                    //             let weight = &tensors[node.input[1].as_str()];
                    //             let bias = tensors.get(node.input[2].as_str());
                    //             assert_eq!(node.attribute[0].name(), "dilations");
                    //             let dilations = &node.attribute[0].ints.as_slice();
                    //             assert_eq!(dilations.len(), 2);
                    //             // assert_eq!(node.attribute[1].name(), "group");
                    //             // let group = node.attribute[1].i;
                    //             // assert_eq!(node.attribute[2].name(), "kernel_shape");
                    //             // let kernel_shape = &node.attribute[2].ints.as_slice();
                    //             assert_eq!(node.attribute[3].name(), "pads");
                    //             let pads = node.attribute[3].ints.as_slice();
                    //             // assert_eq!(pads.len(), 2);
                    //             assert_eq!(node.attribute[4].name(), "strides");
                    //             let strides = node.attribute[4].ints.as_slice();
                    //             assert_eq!(strides.len(), 2);
                    //             let output = input.conv2d(
                    //                 &weight,
                    //                 bias,
                    //                 [strides[0], strides[1]],
                    //                 [(pads[0], pads[1]), (pads[2], pads[3])],
                    //                 [dilations[0], dilations[1]],
                    //             )?;
                    //             tensors.insert(node.output[0].as_str(), output);
                    //         }
                    //         "Relu" => {
                    //             let input = &tensors[node.input[0].as_str()];
                    //             let output = input.relu()?;
                    //             tensors.insert(node.output[0].as_str(), output);
                    //         }
                    //         "MaxPool" => {
                    //             let input = &tensors[node.input[0].as_str()];
                    //             // let ceil_mode = node.attribute[0].i == Some(1);
                    //             let dilations = node.attribute[1].ints.as_slice();
                    //             let kernel_shape = node.attribute[2].ints.as_slice();
                    //             let pads = node.attribute[3].ints.as_slice();
                    //             let strides = node.attribute[4].ints.as_slice();

                    //             let output = input.maxpool2d(
                    //                 &kernel_shape,
                    //                 [strides[0], strides[1]],
                    //                 [(pads[0], pads[1]), (pads[2], pads[3])],
                    //                 [dilations[0], dilations[1]],
                    //             )?;
                    //             tensors.insert(node.output[0].as_str(), output);
                    //         }
                    //         "Add" => {
                    //             let input = &tensors[node.input[0].as_str()];
                    //             let other = &tensors[node.input[1].as_str()];
                    //             let output = input + other;
                    //             tensors.insert(node.output[0].as_str(), output);
                    //         }
                    //         "GlobalAveragePool" => {
                    //             let input = &tensors[node.input[0].as_str()];
                    //             let output = input.adaptive_avgpool2d([1, 1])?;
                    //             tensors.insert(node.output[0].as_str(), output);
                    //         }
                    //         "Flatten" => {
                    //             let input = &tensors[node.input[0].as_str()];
                    //             let output = if let Some(axis) = node.attribute[0].i {
                    //                 input.flatten(axis, axis)?
                    //             } else {
                    //                 let axes = node.attribute[0].ints.as_slice();
                    //                 input.flatten(axes[0], axes[1])?
                    //             };
                    //             tensors.insert(node.output[0].as_str(), output);
                    //         }
                    //         "Gemm" => {
                    //             let alpha = node.attribute[0].f.map(|f| f as f64);
                    //             let beta = node.attribute[1].f.map(|f| f as f64);
                    //             let input = &tensors[node.input[0].as_str()];
                    //             let weight = &tensors[node.input[1].as_str()];
                    //             let trans_b = node.attribute[2].i == Some(1);
                    //             let bias = tensors.get(node.input[2].as_str());
                    //             let output = if trans_b {
                    //                 input.gemm(
                    //                     &weight.t()?,
                    //                     bias,
                    //                     alpha.unwrap_or(1.0),
                    //                     beta.unwrap_or(0.0),
                    //                 )?
                    //             } else {
                    //                 input.gemm(
                    //                     &weight,
                    //                     bias,
                    //                     alpha.unwrap_or(1.0),
                    //                     beta.unwrap_or(0.0),
                    //                 )?
                    //             };
                    //             tensors.insert(node.output[0].as_str(), output);
                    //         }
                    //         _ => {
                    //             println!(
                    //                 "operator: {:?}, input: {:?}, output: {:?}, attribute: {:?}",
                    //                 node.op_type, node.input, node.output, node.attribute
                    //             );
                    //             panic!("unsupported op: {:?}", node.op_type);
                    //         }
                    //     }
                    // }
                }
                // for operator in operators.iter() {
                //     println!("operator: {:?}", operator);
                // }
                for operator in operators.iter() {
                    match operator {
                        super::operators::Operator::Constant => {},
                        super::operators::Operator::Abs(unary) => todo!(),
                        super::operators::Operator::Acos(unary) => todo!(),
                        super::operators::Operator::Acosh(unary) => todo!(),
                        super::operators::Operator::Add(binary) => add_fwd(binary, &mut tensors)?,
                        super::operators::Operator::And(binary) => todo!(),
                        super::operators::Operator::ArgMax(arg_reduce) => todo!(),
                        super::operators::Operator::ArgMin(arg_reduce) => todo!(),
                        super::operators::Operator::Asin(unary) => todo!(),
                        super::operators::Operator::Asinh(unary) => todo!(),
                        super::operators::Operator::Atan(unary) => todo!(),
                        super::operators::Operator::Atanh(unary) => todo!(),
                        super::operators::Operator::AveragePool(pooling) => todo!(),
                        super::operators::Operator::BatchNormalization(batch_normalization) => {
                            todo!()
                        }
                        super::operators::Operator::BitShift(binary) => todo!(),
                        super::operators::Operator::BitwiseAnd(binary) => todo!(),
                        super::operators::Operator::BitwiseNot(unary) => todo!(),
                        super::operators::Operator::BitwiseOr(binary) => todo!(),
                        super::operators::Operator::BitwiseXor(binary) => todo!(),
                        super::operators::Operator::Cast(cast) => todo!(),
                        super::operators::Operator::Ceil(unary) => todo!(),
                        super::operators::Operator::Concat(concat) => {
                            concat_fwd(concat, &mut tensors)?
                        }
                        super::operators::Operator::Conv2d(conv2d) => todo!(),
                        super::operators::Operator::Conv2dInteger(conv2d) => todo!(),
                        super::operators::Operator::Cos(unary) => todo!(),
                        super::operators::Operator::Cosh(unary) => todo!(),
                        super::operators::Operator::ConstantOfShape(constant_of_shape) => {
                            constant_of_shape_fwd(constant_of_shape, &mut tensors)?
                        }
                        super::operators::Operator::Div(binary) => todo!(),
                        super::operators::Operator::Dropout(dropout) => todo!(),
                        super::operators::Operator::Equal(binary) => todo!(),
                        super::operators::Operator::Erf(unary) => todo!(),
                        super::operators::Operator::Exp(unary) => todo!(),
                        super::operators::Operator::Expand(expand) => todo!(),
                        super::operators::Operator::EyeLike(eye_like) => todo!(),
                        super::operators::Operator::Flatten(flatten) => todo!(),
                        super::operators::Operator::Floor(unary) => todo!(),
                        super::operators::Operator::Gather(gather) => {
                            gather_fwd(gather, &mut tensors)?
                        }
                        super::operators::Operator::Gemm(gemm) => todo!(),
                        super::operators::Operator::GlobalAveragePool(pooling) => todo!(),
                        super::operators::Operator::GlobalMaxPool(pooling) => todo!(),
                        super::operators::Operator::Greater(binary) => todo!(),
                        super::operators::Operator::Identity(eye_like) => todo!(),
                        super::operators::Operator::If => todo!(),
                        super::operators::Operator::IsInf(unary) => todo!(),
                        super::operators::Operator::IsNaN(unary) => todo!(),
                        super::operators::Operator::Less(binary) => todo!(),
                        super::operators::Operator::Log(unary) => todo!(),
                        super::operators::Operator::Loop => todo!(),
                        super::operators::Operator::Lstm(lstm) => lstm_fwd(lstm, &mut tensors)?,
                        super::operators::Operator::MatMul(matmul) => {
                            matmul_fwd(matmul, &mut tensors)?
                        }
                        super::operators::Operator::MatMulInteger(matmul) => todo!(),
                        super::operators::Operator::Max(binary) => todo!(),
                        super::operators::Operator::MaxPool(pooling) => todo!(),
                        super::operators::Operator::Mean(reduce) => todo!(),
                        super::operators::Operator::Min(binary) => todo!(),
                        super::operators::Operator::Mod(binary) => todo!(),
                        super::operators::Operator::Mul(binary) => todo!(),
                        super::operators::Operator::Neg(unary) => todo!(),
                        super::operators::Operator::Not(unary) => todo!(),
                        super::operators::Operator::OneHot(one_hot) => todo!(),
                        super::operators::Operator::Or(binary) => todo!(),
                        super::operators::Operator::Pad(pad) => todo!(),
                        super::operators::Operator::Pow(binary) => todo!(),
                        super::operators::Operator::RandomNormal(random_normal) => todo!(),
                        super::operators::Operator::RandomNormalLike(random_normal) => todo!(),
                        super::operators::Operator::RandomUniform(random_uniform) => todo!(),
                        super::operators::Operator::RandomUniformLike(random_uniform) => todo!(),
                        super::operators::Operator::Reciprocal(unary) => todo!(),
                        super::operators::Operator::ReduceMax(reduce) => todo!(),
                        super::operators::Operator::ReduceMean(reduce) => todo!(),
                        super::operators::Operator::ReduceMin(reduce) => todo!(),
                        super::operators::Operator::ReduceProd(reduce) => todo!(),
                        super::operators::Operator::ReduceSum(reduce) => todo!(),
                        super::operators::Operator::Reshape(reshape) => todo!(),
                        super::operators::Operator::Round(unary) => todo!(),
                        super::operators::Operator::Sigmoid(unary) => todo!(),
                        super::operators::Operator::Sign(unary) => todo!(),
                        super::operators::Operator::Sin(unary) => todo!(),
                        super::operators::Operator::Sinh(unary) => todo!(),
                        super::operators::Operator::Slice(slice) => slice_fwd(slice, &mut tensors)?,
                        super::operators::Operator::Split(split) => todo!(),
                        super::operators::Operator::Sqrt(unary) => todo!(),
                        super::operators::Operator::Squeeze(squeeze) => {
                            squeeze_fwd(squeeze, &mut tensors)?
                        }
                        super::operators::Operator::Sub(binary) => todo!(),
                        super::operators::Operator::Sum(reduce) => todo!(),
                        super::operators::Operator::Shape(unary) => shape_fwd(unary, &mut tensors)?,
                        super::operators::Operator::Tan(unary) => todo!(),
                        super::operators::Operator::Tanh(unary) => todo!(),
                        super::operators::Operator::Transpose(permute) => {
                            transpose_fwd(permute, &mut tensors)?
                        }
                        super::operators::Operator::Trilu(unary) => todo!(),
                        super::operators::Operator::Unsqueeze(unsqueeze) => {
                            unsqueeze_fwd(unsqueeze, &mut tensors)?
                        }
                        super::operators::Operator::Where(_) => todo!(),
                        super::operators::Operator::Xor(binary) => todo!(),
                        super::operators::Operator::Bernoulli(bernoulli) => todo!(),
                        super::operators::Operator::BlackmanWindow(unary) => todo!(),
                        super::operators::Operator::CastLike(cast) => todo!(),
                        super::operators::Operator::Celu(unary) => todo!(),
                        super::operators::Operator::Clip(clip) => todo!(),
                        super::operators::Operator::Elu(unary) => todo!(),
                        super::operators::Operator::Gelu(unary) => todo!(),
                        super::operators::Operator::GreaterOrEqual(binary) => todo!(),
                        super::operators::Operator::HammingWindow(unary) => todo!(),
                        super::operators::Operator::HannWindow(unary) => todo!(),
                        super::operators::Operator::HardSigmoid(unary) => todo!(),
                        super::operators::Operator::HardSwish(unary) => todo!(),
                        super::operators::Operator::LayerNormalization(layer_normalization) => {
                            todo!()
                        }
                        super::operators::Operator::LeakyRelu(unary) => todo!(),
                        super::operators::Operator::LessOrEqual(binary) => todo!(),
                        super::operators::Operator::LogSoftmax(reduce) => todo!(),
                        super::operators::Operator::Mish(unary) => todo!(),
                        super::operators::Operator::ReduceL1(reduce) => todo!(),
                        super::operators::Operator::ReduceL2(reduce) => todo!(),
                        super::operators::Operator::ReduceLogSum(reduce) => todo!(),
                        super::operators::Operator::ReduceLogSumExp(reduce) => todo!(),
                        super::operators::Operator::ReduceSumSquare(reduce) => todo!(),
                        super::operators::Operator::Relu(unary) => todo!(),
                        super::operators::Operator::Selu(unary) => todo!(),
                        super::operators::Operator::Shrink(unary) => todo!(),
                        super::operators::Operator::Softmax(reduce) => todo!(),
                        super::operators::Operator::SoftmaxCrossEntropyLoss(reduce) => todo!(),
                        super::operators::Operator::Softplus(unary) => todo!(),
                        super::operators::Operator::Softsign(unary) => todo!(),
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
                let mut node_degree = HashMap::new();

                let mut all_inputs = HashSet::new();
                let mut operators = Vec::new();
                if let Some(graph) = model_proto.graph.as_mut() {
                    for input in graph.input.iter() {
                        let name = input.name();
                        all_inputs.insert(name.to_string());
                    }
                    for initializer in graph.initializer.iter() {
                        let name = initializer.name();
                        all_inputs.insert(name.to_string());
                    }
                    for node in graph.node.iter() {
                        match node.op_type() {
                            "Conv" => operators.push(conv_init(node, &mut node_degree)),
                            "MaxPool" | "GlobalAveragePool" | "GlobalMaxPool" | "AveragePool" => {
                                operators.push(pooling_init(node, &mut node_degree));
                            }
                            "Abs" | "Acos" | "Acosh" | "Asin" | "Asinh" | "Atan" | "Atanh"
                            | "BitwiseNot" | "Ceil" | "Cos" | "Cosh" | "Erf" | "Exp" | "Floor"
                            | "IsInf" | "IsNaN" | "Log" | "Neg" | "Not" | "Reciprocal"
                            | "Round" | "Sigmoid" | "Sign" | "Sin" | "Sinh" | "Sqrt" | "Tan"
                            | "Tanh" | "Gelu" | "HardSigmoid" | "HardSwish" | "LeakyRelu"
                            | "Mish" | "Shrink" | "Relu" | "Softplus" | "Softsign" | "Shape" => {
                                operators.push(unary_init(node, &mut node_degree))
                            }
                            "Add" | "Sub" | "Mul" | "Div" | "Max" | "Min" => {
                                operators.push(binary_init(node, &mut node_degree))
                            }
                            "Gather" => operators.push(gather_init(node, &mut node_degree)),
                            "Constant" => {
                                operators.push(constant_init(node, &mut initializer_map)?)
                            }
                            "Gemm" => operators.push(gemm_init(node, &mut node_degree)),
                            "MatMul" => operators.push(matmul_init(node, &mut node_degree)),
                            "Unsqueeze" => operators.push(unsqueeze_init(
                                node,
                                &mut node_degree,
                                &mut initializer_map,
                            )),
                            "Squeeze" => operators.push(squeeze_init(
                                node,
                                &mut node_degree,
                                &mut initializer_map,
                            )),
                            "Concat" => operators.push(concat_init(node, &mut node_degree)),
                            "ConstantOfShape" => {
                                operators.push(const_of_shape_init(node, &mut node_degree)?)
                            }
                            "Transpose" => operators.push(transpose_init(node, &mut node_degree)),
                            "Slice" => operators.push(slice_init(node, &mut node_degree)),
                            "LSTM" => {
                                operators.push(lstm_init(node, &mut node_degree));
                            }
                            _ => unimplemented!(
                                "unsupported op when initializing: {:?}",
                                node.op_type
                            ),
                        }
                    }

                    for initializer in graph.initializer.iter_mut() {
                        let name = initializer.name().to_string();
                        let tensor = Tensor::from_onnx_tensor(initializer, &None)?;
                        initializer_map.insert(name, tensor);
                    }
                }
                Ok(OnnxModel::Initialized(Initialized {
                    model: model_proto,
                    initializer_map,
                    permutes: HashMap::new(),
                    operators,
                }))
            }
            OnnxModel::Initialized(_) => panic!("model already initialized"),
        }
    }
}
