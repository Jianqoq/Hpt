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
        num_threads: usize,
        inputs: HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>, TensorError> {
        let func = || {
            fn execute(
                model: &OnnxModel,
                inputs: &HashMap<String, Tensor>,
            ) -> Result<HashMap<String, Tensor>, TensorError> {
                let mut res = HashMap::new();
                match model {
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
                        }
                        for operator in operators.iter() {
                            match operator {
                                crate::utils::onnx::operators::Operator::Constant => {}
                                crate::utils::onnx::operators::Operator::Abs(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::Acos(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::Acosh(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::Add(binary) => {
                                                                                            add_fwd(binary, &mut tensors)?
                                                                                        }
                                crate::utils::onnx::operators::Operator::And(binary) => todo!(),
                                crate::utils::onnx::operators::Operator::ArgMax(arg_reduce) => todo!(),
                                crate::utils::onnx::operators::Operator::ArgMin(arg_reduce) => todo!(),
                                crate::utils::onnx::operators::Operator::Asin(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::Asinh(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::Atan(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::Atanh(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::AveragePool(pooling) => todo!(),
                                crate::utils::onnx::operators::Operator::BatchNormalization(
                                                                                            batch_normalization,
                                                                                        ) => {
                                                                                            todo!()
                                                                                        }
                                crate::utils::onnx::operators::Operator::BitShift(binary) => todo!(),
                                crate::utils::onnx::operators::Operator::BitwiseAnd(binary) => todo!(),
                                crate::utils::onnx::operators::Operator::BitwiseNot(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::BitwiseOr(binary) => todo!(),
                                crate::utils::onnx::operators::Operator::BitwiseXor(binary) => todo!(),
                                crate::utils::onnx::operators::Operator::Cast(cast) => todo!(),
                                crate::utils::onnx::operators::Operator::Ceil(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::Concat(concat) => {
                                                                                            concat_fwd(concat, &mut tensors)?
                                                                                        }
                                crate::utils::onnx::operators::Operator::Conv2d(conv2d) => todo!(),
                                crate::utils::onnx::operators::Operator::Conv2dInteger(conv2d) => todo!(),
                                crate::utils::onnx::operators::Operator::Cos(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::Cosh(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::ConstantOfShape(constant_of_shape) => {
                                                                                            constant_of_shape_fwd(constant_of_shape, &mut tensors)?
                                                                                        }
                                crate::utils::onnx::operators::Operator::Div(binary) => todo!(),
                                crate::utils::onnx::operators::Operator::Dropout(dropout) => todo!(),
                                crate::utils::onnx::operators::Operator::Equal(binary) => todo!(),
                                crate::utils::onnx::operators::Operator::Erf(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::Exp(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::Expand(expand) => todo!(),
                                crate::utils::onnx::operators::Operator::EyeLike(eye_like) => todo!(),
                                crate::utils::onnx::operators::Operator::Flatten(flatten) => todo!(),
                                crate::utils::onnx::operators::Operator::Floor(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::Gather(gather) => {
                                                                                            gather_fwd(gather, &mut tensors)?
                                                                                        }
                                crate::utils::onnx::operators::Operator::Gemm(gemm) => todo!(),
                                crate::utils::onnx::operators::Operator::GlobalAveragePool(pooling) => todo!(),
                                crate::utils::onnx::operators::Operator::GlobalMaxPool(pooling) => todo!(),
                                crate::utils::onnx::operators::Operator::Greater(binary) => todo!(),
                                crate::utils::onnx::operators::Operator::Identity(eye_like) => todo!(),
                                crate::utils::onnx::operators::Operator::If => todo!(),
                                crate::utils::onnx::operators::Operator::IsInf(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::IsNaN(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::Less(binary) => todo!(),
                                crate::utils::onnx::operators::Operator::Log(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::Loop => todo!(),
                                crate::utils::onnx::operators::Operator::Lstm(lstm) => {
                                                                                            lstm_fwd(lstm, &mut tensors)?
                                                                                        }
                                crate::utils::onnx::operators::Operator::MatMul(matmul) => {
                                                                                            matmul_fwd(matmul, &mut tensors)?
                                                                                        }
                                crate::utils::onnx::operators::Operator::MatMulInteger(matmul) => todo!(),
                                crate::utils::onnx::operators::Operator::Max(binary) => todo!(),
                                crate::utils::onnx::operators::Operator::MaxPool(pooling) => todo!(),
                                crate::utils::onnx::operators::Operator::Mean(reduce) => todo!(),
                                crate::utils::onnx::operators::Operator::Min(binary) => todo!(),
                                crate::utils::onnx::operators::Operator::Mod(binary) => todo!(),
                                crate::utils::onnx::operators::Operator::Mul(binary) => todo!(),
                                crate::utils::onnx::operators::Operator::Neg(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::Not(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::OneHot(one_hot) => todo!(),
                                crate::utils::onnx::operators::Operator::Or(binary) => todo!(),
                                crate::utils::onnx::operators::Operator::Pad(pad) => todo!(),
                                crate::utils::onnx::operators::Operator::Pow(binary) => todo!(),
                                crate::utils::onnx::operators::Operator::RandomNormal(random_normal) => todo!(),
                                crate::utils::onnx::operators::Operator::RandomNormalLike(random_normal) => {
                                                                                            todo!()
                                                                                        }
                                crate::utils::onnx::operators::Operator::RandomUniform(random_uniform) => {
                                                                                            todo!()
                                                                                        }
                                crate::utils::onnx::operators::Operator::RandomUniformLike(random_uniform) => {
                                                                                            todo!()
                                                                                        }
                                crate::utils::onnx::operators::Operator::Reciprocal(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::ReduceMax(reduce) => todo!(),
                                crate::utils::onnx::operators::Operator::ReduceMean(reduce) => todo!(),
                                crate::utils::onnx::operators::Operator::ReduceMin(reduce) => todo!(),
                                crate::utils::onnx::operators::Operator::ReduceProd(reduce) => todo!(),
                                crate::utils::onnx::operators::Operator::ReduceSum(reduce) => todo!(),
                                crate::utils::onnx::operators::Operator::Reshape(reshape) => todo!(),
                                crate::utils::onnx::operators::Operator::Round(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::Sigmoid(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::Sign(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::Sin(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::Sinh(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::Slice(slice) => {
                                                                                            slice_fwd(slice, &mut tensors)?
                                                                                        }
                                crate::utils::onnx::operators::Operator::Split(split) => todo!(),
                                crate::utils::onnx::operators::Operator::Sqrt(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::Squeeze(squeeze) => {
                                                                                            squeeze_fwd(squeeze, &mut tensors)?
                                                                                        }
                                crate::utils::onnx::operators::Operator::Sub(binary) => todo!(),
                                crate::utils::onnx::operators::Operator::Sum(reduce) => todo!(),
                                crate::utils::onnx::operators::Operator::Shape(unary) => {
                                                                                            shape_fwd(unary, &mut tensors)?
                                                                                        }
                                crate::utils::onnx::operators::Operator::Tan(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::Tanh(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::Transpose(permute) => {
                                                                                            transpose_fwd(permute, &mut tensors)?
                                                                                        }
                                crate::utils::onnx::operators::Operator::Trilu(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::Unsqueeze(unsqueeze) => {
                                                                                            unsqueeze_fwd(unsqueeze, &mut tensors)?
                                                                                        }
                                crate::utils::onnx::operators::Operator::Where(_) => todo!(),
                                crate::utils::onnx::operators::Operator::Xor(binary) => todo!(),
                                crate::utils::onnx::operators::Operator::Bernoulli(bernoulli) => todo!(),
                                crate::utils::onnx::operators::Operator::BlackmanWindow(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::CastLike(cast) => todo!(),
                                crate::utils::onnx::operators::Operator::Celu(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::Clip(clip) => todo!(),
                                crate::utils::onnx::operators::Operator::Elu(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::Gelu(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::GreaterOrEqual(binary) => todo!(),
                                crate::utils::onnx::operators::Operator::HammingWindow(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::HannWindow(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::HardSigmoid(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::HardSwish(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::LayerNormalization(
                                                                                            layer_normalization,
                                                                                        ) => {
                                                                                            todo!()
                                                                                        }
                                crate::utils::onnx::operators::Operator::LeakyRelu(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::LessOrEqual(binary) => todo!(),
                                crate::utils::onnx::operators::Operator::LogSoftmax(reduce) => todo!(),
                                crate::utils::onnx::operators::Operator::Mish(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::ReduceL1(reduce) => todo!(),
                                crate::utils::onnx::operators::Operator::ReduceL2(reduce) => todo!(),
                                crate::utils::onnx::operators::Operator::ReduceLogSum(reduce) => todo!(),
                                crate::utils::onnx::operators::Operator::ReduceLogSumExp(reduce) => todo!(),
                                crate::utils::onnx::operators::Operator::ReduceSumSquare(reduce) => todo!(),
                                crate::utils::onnx::operators::Operator::Relu(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::Selu(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::Shrink(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::Softmax(reduce) => todo!(),
                                crate::utils::onnx::operators::Operator::SoftmaxCrossEntropyLoss(reduce) =>
                                                                                            todo!(),
                                crate::utils::onnx::operators::Operator::Softplus(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::Softsign(unary) => todo!(),
                                crate::utils::onnx::operators::Operator::Contiguous(unary) => todo!(),
crate::utils::onnx::operators::Operator::InvPermute(permute) => todo!(),
                            }
                        }
                        if let Some(graph) = model.graph.as_ref() {
                            for output in graph.output.iter() {
                                if let Some(tensor) = tensors.get(output.name()) {
                                    res.insert(output.name().to_string(), tensor.clone());
                                }
                            }
                        }
                    }
                }
                Ok(res)
            }
            execute(self, &inputs)
        };
        spindle::with_lock(num_threads, func)
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
                            "Conv" => {
                                let ops = conv_init(node, &mut node_degree);
                                for op in ops {
                                    operators.push(op);
                                }
                            }
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
