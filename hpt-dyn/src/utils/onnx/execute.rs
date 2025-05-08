#![allow(unused)]

use core::panic;
use std::collections::{HashMap, HashSet};

use hpt_common::error::{base::TensorError, common::CommonError, shape::ShapeError};
use hpt_traits::tensor::TensorInfo;

use super::{
    fwd::*,
    init::*,
    map_dtype::to_dtype,
    operators::{Conv2dFused, Operator, TensorFormat},
};
use crate::{
    Tensor,
    onnx::{TensorShapeProto, tensor_shape_proto, type_proto},
    ops::models::onnx::{Initialized, OnnxModel},
};
use petgraph::{
    dot::{Config, Dot},
    stable_graph::StableGraph,
};

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
                        node_degree,
                    }) => {
                        let mut node_degree = node_degree
                            .iter()
                            .map(|(k, v)| (k.as_str(), *v))
                            .collect::<HashMap<&str, u32>>();
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

                        let mut total_conv = std::time::Duration::from_secs(0);
                        let mut total_relu = std::time::Duration::from_secs(0);
                        let mut total_add = std::time::Duration::from_secs(0);
                        let mut total_maxpool = std::time::Duration::from_secs(0);
                        let mut total_transpose = std::time::Duration::from_secs(0);
                        let mut total_conv_fused = std::time::Duration::from_secs(0);
                        for operator in operators.iter() {
                            // println!("operator: {:?}", operator);
                            match operator {
                                Operator::Constant(_) => {}
                                Operator::Abs(unary) => todo!(),
                                Operator::Acos(unary) => todo!(),
                                Operator::Acosh(unary) => todo!(),
                                Operator::Add(base) => {
                                    let now = std::time::Instant::now();
                                    add_fwd(&base.base, &mut tensors, &mut node_degree)?;
                                    total_add += now.elapsed();
                                }
                                Operator::And(binary) => todo!(),
                                Operator::ArgMax(arg_reduce) => todo!(),
                                Operator::ArgMin(arg_reduce) => todo!(),
                                Operator::Asin(unary) => todo!(),
                                Operator::Asinh(unary) => todo!(),
                                Operator::Atan(unary) => todo!(),
                                Operator::Atanh(unary) => todo!(),
                                Operator::AveragePool(pooling) => {
                                    avgpool_fwd(&pooling.base, &mut tensors)?
                                }
                                Operator::BatchNormalization(batch_normalization) => todo!(),
                                Operator::BitShift(binary) => todo!(),
                                Operator::BitwiseAnd(binary) => todo!(),
                                Operator::BitwiseNot(unary) => todo!(),
                                Operator::BitwiseOr(binary) => todo!(),
                                Operator::BitwiseXor(binary) => todo!(),
                                Operator::Cast(cast) => todo!(),
                                Operator::Ceil(unary) => todo!(),
                                Operator::Concat(concat) => concat_fwd(&concat.base, &mut tensors)?,
                                Operator::Conv2d(conv2d) => {
                                    let now = std::time::Instant::now();
                                    conv_fwd(&conv2d.base, &mut tensors, &mut node_degree)?;
                                    let duration = now.elapsed();
                                    total_conv += duration;
                                }
                                Operator::Conv2dInteger(conv2d) => {
                                    conv_fwd(&conv2d.base, &mut tensors, &mut node_degree)?
                                }
                                Operator::Cos(unary) => todo!(),
                                Operator::Cosh(unary) => todo!(),
                                Operator::ConstantOfShape(constant_of_shape) => {
                                    constant_of_shape_fwd(&constant_of_shape.base, &mut tensors)?
                                }
                                Operator::Div(binary) => todo!(),
                                Operator::Dropout(dropout) => todo!(),
                                Operator::Equal(binary) => todo!(),
                                Operator::Erf(unary) => todo!(),
                                Operator::Exp(unary) => todo!(),
                                Operator::Expand(expand) => todo!(),
                                Operator::EyeLike(eye_like) => todo!(),
                                Operator::Flatten(flatten) => {
                                    flatten_fwd(&flatten.base, &mut tensors, &mut node_degree)?
                                }
                                Operator::Floor(unary) => todo!(),
                                Operator::Gather(gather) => gather_fwd(&gather.base, &mut tensors)?,
                                Operator::Gemm(gemm) => {
                                    gemm_fwd(&gemm.base, &mut tensors, &mut node_degree)?
                                }
                                Operator::GlobalAveragePool(pooling) => {
                                    global_avgpool_fwd(&pooling.base, &mut tensors)?
                                }
                                Operator::GlobalMaxPool(pooling) => {
                                    global_maxpool_fwd(&pooling.base, &mut tensors)?
                                }
                                Operator::Greater(binary) => todo!(),
                                Operator::Identity(identity) => {
                                    identity_fwd(&identity.base, &mut tensors)?
                                }
                                Operator::If(_) => todo!(),
                                Operator::IsInf(unary) => todo!(),
                                Operator::IsNaN(unary) => todo!(),
                                Operator::Less(binary) => todo!(),
                                Operator::Log(unary) => todo!(),
                                Operator::Loop(_) => todo!(),
                                Operator::Lstm(lstm) => lstm_fwd(&lstm.base, &mut tensors)?,
                                Operator::MatMul(matmul) => {
                                    matmul_fwd(&matmul.base, &mut tensors, &mut node_degree)?
                                }
                                Operator::MatMulInteger(matmul) => todo!(),
                                Operator::Max(binary) => todo!(),
                                Operator::MaxPool(pooling) => {
                                    let now = std::time::Instant::now();
                                    maxpool_fwd(&pooling.base, &mut tensors, &mut node_degree)?;
                                    total_maxpool += now.elapsed();
                                }
                                Operator::Mean(reduce) => todo!(),
                                Operator::Min(binary) => todo!(),
                                Operator::Mod(binary) => todo!(),
                                Operator::Mul(binary) => todo!(),
                                Operator::Neg(unary) => todo!(),
                                Operator::Not(unary) => todo!(),
                                Operator::OneHot(one_hot) => todo!(),
                                Operator::Or(binary) => todo!(),
                                Operator::Pad(pad) => todo!(),
                                Operator::Pow(binary) => todo!(),
                                Operator::RandomNormal(random_normal) => todo!(),
                                Operator::RandomNormalLike(random_normal) => todo!(),
                                Operator::RandomUniform(random_uniform) => todo!(),
                                Operator::RandomUniformLike(random_uniform) => todo!(),
                                Operator::Reciprocal(unary) => todo!(),
                                Operator::ReduceMax(reduce) => todo!(),
                                Operator::ReduceMean(reduce) => todo!(),
                                Operator::ReduceMin(reduce) => todo!(),
                                Operator::ReduceProd(reduce) => todo!(),
                                Operator::ReduceSum(reduce) => todo!(),
                                Operator::Reshape(reshape) => todo!(),
                                Operator::Round(unary) => todo!(),
                                Operator::Sigmoid(unary) => todo!(),
                                Operator::Sign(unary) => todo!(),
                                Operator::Sin(unary) => todo!(),
                                Operator::Sinh(unary) => todo!(),
                                Operator::Slice(slice) => slice_fwd(&slice.base, &mut tensors)?,
                                Operator::Split(split) => todo!(),
                                Operator::Sqrt(unary) => todo!(),
                                Operator::Squeeze(squeeze) => {
                                    squeeze_fwd(&squeeze.base, &mut tensors)?
                                }
                                Operator::Sub(binary) => todo!(),
                                Operator::Sum(reduce) => todo!(),
                                Operator::Shape(unary) => {
                                    shape_fwd(&unary.base, &mut tensors, &mut node_degree)?
                                }
                                Operator::Tan(unary) => todo!(),
                                Operator::Tanh(unary) => todo!(),
                                Operator::Transpose(permute) => {
                                    let now = std::time::Instant::now();
                                    transpose_fwd(&permute.base, &mut tensors, &mut node_degree)?;
                                    total_transpose += now.elapsed();
                                }
                                Operator::Trilu(unary) => todo!(),
                                Operator::Unsqueeze(unsqueeze) => {
                                    unsqueeze_fwd(&unsqueeze.base, &mut tensors)?
                                }
                                Operator::Where(where_op) => todo!(),
                                Operator::Xor(binary) => todo!(),
                                Operator::Bernoulli(bernoulli) => todo!(),
                                Operator::BlackmanWindow(unary) => todo!(),
                                Operator::CastLike(cast) => todo!(),
                                Operator::Celu(unary) => todo!(),
                                Operator::Clip(clip) => todo!(),
                                Operator::Elu(unary) => todo!(),
                                Operator::Gelu(unary) => todo!(),
                                Operator::GreaterOrEqual(binary) => todo!(),
                                Operator::HammingWindow(unary) => todo!(),
                                Operator::HannWindow(unary) => todo!(),
                                Operator::HardSigmoid(unary) => todo!(),
                                Operator::HardSwish(unary) => todo!(),
                                Operator::LayerNormalization(layer_normalization) => todo!(),
                                Operator::LeakyRelu(unary) => todo!(),
                                Operator::LessOrEqual(binary) => todo!(),
                                Operator::LogSoftmax(reduce) => todo!(),
                                Operator::Mish(unary) => todo!(),
                                Operator::ReduceL1(reduce) => todo!(),
                                Operator::ReduceL2(reduce) => todo!(),
                                Operator::ReduceLogSum(reduce) => todo!(),
                                Operator::ReduceLogSumExp(reduce) => todo!(),
                                Operator::ReduceSumSquare(reduce) => todo!(),
                                Operator::Relu(unary) => {
                                    let now = std::time::Instant::now();
                                    relu_fwd(&unary.base, &mut tensors, &mut node_degree)?;
                                    total_relu += now.elapsed();
                                }
                                Operator::Selu(unary) => todo!(),
                                Operator::Shrink(unary) => todo!(),
                                Operator::Softmax(reduce) => todo!(),
                                Operator::SoftmaxCrossEntropyLoss(reduce) => todo!(),
                                Operator::Softplus(unary) => todo!(),
                                Operator::Softsign(unary) => todo!(),
                                Operator::Contiguous(unary) => todo!(),
                                Operator::InvPermute(permute) => todo!(),
                                Operator::PermuteContiguous(permute) => {
                                    permute_contiguous_fwd(&permute.base, &mut tensors)?
                                }
                                Operator::Conv2dFused(base) => {
                                    let now = std::time::Instant::now();
                                    conv_fused_fwd(&base.base, &mut tensors, &mut node_degree)?;
                                    total_conv_fused += now.elapsed();
                                }
                            }
                        }
                        if let Some(graph) = model.graph.as_ref() {
                            for output in graph.output.iter() {
                                if let Some(tensor) = tensors.get(output.name()) {
                                    res.insert(output.name().to_string(), tensor.clone());
                                }
                            }
                        }
                        // println!("total_conv time: {:?}", total_conv);
                        // println!("total_relu time: {:?}", total_relu);
                        // println!("total_add time: {:?}", total_add);
                        // println!("total_maxpool time: {:?}", total_maxpool);
                        // println!("total_transpose time: {:?}", total_transpose);
                        // println!("total_conv_fused time: {:?}", total_conv_fused);
                    }
                }
                Ok(res)
            }
            execute(self, &inputs)
        };
        func()
        // spindle::with_lock(num_threads, func)
    }

    pub fn initialize(self) -> Result<Self, TensorError> {
        match self {
            OnnxModel::Model(mut model_proto) => {
                let mut initializer_map = HashMap::new();
                let mut node_degree = HashMap::new();
                let mut tensor_to_node = HashMap::new();

                let mut all_inputs = HashSet::new();
                let mut operators = Vec::new();
                let mut new_operators = Vec::new();
                let mut formats = HashMap::new();

                if let Some(graph) = model_proto.graph.as_mut() {
                    for input in graph.input.iter() {
                        let name = input.name();
                        all_inputs.insert(name.to_string());
                        formats.insert(name.to_string(), TensorFormat::NCHW);
                    }
                    for initializer in graph.initializer.iter() {
                        let name = initializer.name();
                        all_inputs.insert(name.to_string());
                        formats.insert(name.to_string(), TensorFormat::NCHW);
                    }
                    for node in graph.node.iter() {
                        match node.op_type() {
                            "Conv" => {
                                let ops = conv_init(node, &mut formats);
                                for op in ops {
                                    operators.push(op);
                                }
                            }
                            "MaxPool" | "GlobalAveragePool" | "GlobalMaxPool" | "AveragePool" => {
                                let ops = pooling_init(node, &mut formats);
                                for op in ops {
                                    operators.push(op);
                                }
                            }
                            "Abs" | "Acos" | "Acosh" | "Asin" | "Asinh" | "Atan" | "Atanh"
                            | "BitwiseNot" | "Ceil" | "Cos" | "Cosh" | "Erf" | "Exp" | "Floor"
                            | "IsInf" | "IsNaN" | "Log" | "Neg" | "Not" | "Reciprocal"
                            | "Round" | "Sigmoid" | "Sign" | "Sin" | "Sinh" | "Sqrt" | "Tan"
                            | "Tanh" | "Gelu" | "HardSigmoid" | "HardSwish" | "LeakyRelu"
                            | "Mish" | "Shrink" | "Relu" | "Softplus" | "Softsign" | "Shape" => {
                                operators.push(unary_init(node, &mut formats));
                            }
                            "Add" | "Sub" | "Mul" | "Div" | "Max" | "Min" => {
                                let ops = binary_init(node, &mut formats);
                                for op in ops {
                                    operators.push(op);
                                }
                            }
                            "Gather" => {
                                let ops = gather_init(node, &mut formats);
                                for op in ops {
                                    operators.push(op);
                                }
                            }
                            "Constant" => operators.push(constant_init(
                                node,
                                &mut initializer_map,
                                &mut formats,
                            )?),
                            "Gemm" => {
                                let ops = gemm_init(node, &mut formats);
                                for op in ops {
                                    operators.push(op);
                                }
                            }
                            "MatMul" => {
                                let ops = matmul_init(node, &mut formats);
                                for op in ops {
                                    operators.push(op);
                                }
                            }
                            "Unsqueeze" => {
                                let ops = unsqueeze_init(node, &mut initializer_map, &mut formats);
                                for op in ops {
                                    operators.push(op);
                                }
                            }
                            "Squeeze" => {
                                let ops = squeeze_init(node, &mut initializer_map, &mut formats);
                                for op in ops {
                                    operators.push(op);
                                }
                            }
                            "Concat" => {
                                let ops = concat_init(node, &mut formats);
                                for op in ops {
                                    operators.push(op);
                                }
                            }
                            "ConstantOfShape" => {
                                let ops = const_of_shape_init(node, &mut formats)?;
                                for op in ops {
                                    operators.push(op);
                                }
                            }
                            "Transpose" => {
                                let ops = transpose_init(node, &mut formats);
                                for op in ops {
                                    operators.push(op);
                                }
                            }
                            "Slice" => {
                                let ops = slice_init(node, &mut formats);
                            }
                            "LSTM" => {
                                let ops = lstm_init(node, &mut formats);
                                for op in ops {
                                    operators.push(op);
                                }
                            }
                            "Identity" => operators.push(identity_init(node, &mut formats)),
                            "Flatten" => {
                                let ops = flatten_init(node, &mut formats);
                                for op in ops {
                                    operators.push(op);
                                }
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

                    for operator in operators.iter() {
                        operator.tensor_to_node(&mut tensor_to_node);
                        operator.fill_node_degree(&mut node_degree);
                    }

                    println!("tensor_to_node: {:#?}", tensor_to_node);

                    let mut stablegraph = StableGraph::new();
                    let mut node_indices = HashMap::new();
                    for node in operators.iter() {
                        let idx = stablegraph.add_node(node.clone());
                        node_indices.insert(node.id(), idx);
                    }

                    for node in operators.iter() {
                        let current_idx = node_indices[node.id()];

                        let inputs = node.inputs();

                        for input in node.inputs() {
                            if let Some(b) = tensor_to_node.get(input) {
                                stablegraph.add_edge(
                                    *node_indices.get(b.as_str()).expect(&format!(
                                        "node {} not found",
                                        tensor_to_node[input]
                                    )),
                                    current_idx,
                                    (),
                                );
                            }
                        }
                        if inputs.is_empty() {
                            stablegraph.add_edge(current_idx, current_idx, ());
                        }
                    }

                    for operator in operators {
                        match &operator {
                            Operator::Relu(unary)
                            | Operator::LeakyRelu(unary)
                            | Operator::Sigmoid(unary)
                            | Operator::Gelu(unary)
                            | Operator::Tanh(unary) => {
                                if let Some(prev) = new_operators.pop() {
                                    if let Operator::Conv2d(conv2d) = prev {
                                        if &unary.base.input == &conv2d.base.output
                                            && node_degree[conv2d.base.output.as_str()] == 1
                                        {
                                            let conv2d_fused = Conv2dFused {
                                                input: conv2d.base.input.clone(),
                                                output: unary.base.output.clone(),
                                                kernel: conv2d.base.kernel.clone(),
                                                bias: conv2d.base.bias.clone(),
                                                pads: conv2d.base.pads.clone(),
                                                strides: conv2d.base.strides.clone(),
                                                dilations: conv2d.base.dilations.clone(),
                                                group: conv2d.base.group,
                                                activation: match operator {
                                                    Operator::Relu(_) => {
                                                        super::operators::ConvActivation::Relu
                                                    }
                                                    Operator::LeakyRelu(_) => {
                                                        super::operators::ConvActivation::LeakyRelu
                                                    }
                                                    Operator::Sigmoid(_) => {
                                                        super::operators::ConvActivation::Sigmoid
                                                    }
                                                    Operator::Gelu(_) => {
                                                        super::operators::ConvActivation::Gelu
                                                    }
                                                    Operator::Tanh(_) => {
                                                        super::operators::ConvActivation::Tanh
                                                    }
                                                    _ => unimplemented!(),
                                                },
                                            };
                                            new_operators.push(Operator::Conv2dFused(
                                                super::operators::Base {
                                                    base: conv2d_fused,
                                                    id: conv2d.id,
                                                },
                                            ));
                                        } else {
                                            new_operators.push(Operator::Conv2d(conv2d));
                                            new_operators.push(operator.clone());
                                        }
                                    } else {
                                        new_operators.push(prev);
                                        new_operators.push(operator.clone());
                                    }
                                } else {
                                    new_operators.push(operator.clone());
                                }
                            }
                            _ => new_operators.push(operator.clone()),
                        }
                    }

                    fn generate_online_graphviz_link<N, E>(graph: &StableGraph<N, E>) -> String
                    where
                        N: std::fmt::Debug,
                        E: std::fmt::Debug,
                    {
                        let dot = Dot::with_config(&graph, &[Config::EdgeNoLabel]);
                        let dot_string = format!("{:?}", dot);

                        // URL编码DOT字符串
                        let encoded = urlencoding::encode(&dot_string);
                        format!("https://dreampuf.github.io/GraphvizOnline/#{}", encoded)
                    }
                    let url = generate_online_graphviz_link(&stablegraph);
                    println!("可以通过以下链接在线查看图表: [{}]", url);
                }

                Ok(OnnxModel::Initialized(Initialized {
                    model: model_proto,
                    initializer_map,
                    permutes: HashMap::new(),
                    operators: new_operators,
                    node_degree,
                }))
            }
            OnnxModel::Initialized(_) => panic!("model already initialized"),
        }
    }
}
