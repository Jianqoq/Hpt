#![allow(unused)]

use core::panic;
use std::collections::{ HashMap, HashSet };

use hpt_common::error::{ base::TensorError, common::CommonError, shape::ShapeError };
use hpt_traits::tensor::TensorInfo;

use super::{
    fwd::*,
    init::*,
    map_dtype::to_dtype,
    operators::{ Conv2dFused, Operator, TensorFormat },
    run_init::run_init,
};
use crate::{
    onnx::{ tensor_shape_proto, type_proto, TensorShapeProto },
    ops::models::onnx::{ Initialized, OnnxModel },
    utils::onnx::{
        build_graph::build_graph,
        optimize::{ constant_fold::pre_transpose, fuse::fuse_conv_unary },
        run_fwd::run_fwd,
    },
    Tensor,
};
use petgraph::{
    dot::{ Config, Dot },
    stable_graph::StableGraph,
    visit::{ EdgeRef, IntoNodeReferences },
    Direction::{ Incoming, Outgoing },
};

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
                    "dimension mismatch: tensor '{}' has shape [{}] but model expects shape [{}]",
                    tensor_name,
                    tensor
                        .shape()
                        .iter()
                        .map(|d| d.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                    onnx_shape.dim
                        .iter()
                        .map(|d| {
                            match &d.value {
                                Some(value) =>
                                    match value {
                                        tensor_shape_proto::dimension::Value::DimValue(dim) =>
                                            dim.to_string(),
                                        tensor_shape_proto::dimension::Value::DimParam(param) =>
                                            param.to_string(),
                                    }
                                None => "?".to_string(),
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
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
        inputs: &HashMap<String, Tensor>
    ) -> Result<HashMap<String, Tensor>, TensorError> {
        let func = || {
            fn execute(
                model: &OnnxModel,
                inputs: &HashMap<String, Tensor>
            ) -> Result<HashMap<String, Tensor>, TensorError> {
                let mut res = HashMap::new();
                match model {
                    OnnxModel::Model(_) => panic!("model not initialized"),
                    OnnxModel::Initialized(
                        Initialized { model, initializer_map, permutes, operators, node_degree },
                    ) => {
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
                                                tensor.permute(&permute)?.contiguous()?
                                            );
                                        }
                                    }
                                } else {
                                    panic!("input {:?} not found", inp.name);
                                }
                            }
                        }

                        run_fwd(&operators, &mut tensors, &mut node_degree)?;
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
        func()
    }

    pub fn initialize(self) -> Result<Self, TensorError> {
        match self {
            OnnxModel::Model(mut model_proto) => {
                let mut initializer_map = HashMap::new();
                let mut node_degree = HashMap::new();
                let mut tensor_to_node: HashMap<&str, &str> = HashMap::new();

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
                    run_init(&graph.node, &mut formats, &mut operators, &mut initializer_map)?;
                    for initializer in graph.initializer.iter_mut() {
                        let name = initializer.name().to_string();
                        let tensor = Tensor::from_onnx_tensor(initializer, &None)?;
                        initializer_map.insert(name, tensor);
                    }

                    for operator in operators.iter() {
                        operator.tensor_to_node(&mut tensor_to_node);
                        operator.fill_node_degree(&mut node_degree);
                    }

                    let mut stablegraph = build_graph(&operators, &tensor_to_node);
                    pre_transpose(&mut stablegraph, &mut initializer_map);
                    fuse_conv_unary(&mut stablegraph);
                    if let Ok(topo_order) = petgraph::algo::toposort(&stablegraph, None) {
                        for node in topo_order.iter() {
                            new_operators.push(stablegraph[*node].clone());
                        }
                    } else {
                        panic!("toposort failed");
                    }

                }

                Ok(
                    OnnxModel::Initialized(Initialized {
                        model: model_proto,
                        initializer_map,
                        permutes: HashMap::new(),
                        operators: new_operators,
                        node_degree,
                    })
                )
            }
            OnnxModel::Initialized(_) => panic!("model already initialized"),
        }
    }
}
