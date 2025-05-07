use hpt_common::error::{base::TensorError, onnx::OnnxError};
use hpt_types::{dtype::DType, into_scalar::Cast};

use super::{
    map_dtype::to_dtype,
    operators::{
        Binary, Concat, Gather, Gemm, Lstm, Matmul, Operator, Permute, Slice, Squeeze, Unary,
    },
};
use crate::{
    Tensor,
    onnx::{NodeProto, TensorProto, attribute_proto},
    utils::onnx::operators::{ConstantOfShape, Conv2d},
};
use hpt_traits::tensor::TensorInfo;
use std::collections::HashMap;

fn bytes_to_string(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).expect("invalid utf-8 sequence")
}

impl From<&TensorProto> for Tensor {
    fn from(tensor: &TensorProto) -> Self {
        let dtype = to_dtype(tensor.data_type());
        if tensor.float_data.len() > 0 {
            let raw = unsafe {
                std::slice::from_raw_parts(
                    tensor.float_data.as_ptr() as *const u8,
                    tensor.float_data.len() * std::mem::size_of::<f32>(),
                )
            };

            let tensor = Tensor::empty(&tensor.dims, dtype, crate::Device::Cpu)
                .expect("cannot create   empty tensor");
            let ptr = tensor.data.cast::<u8>().ptr;
            unsafe {
                ptr.copy_from(raw.as_ptr(), raw.len());
            }
            tensor
        } else if tensor.int32_data.len() > 0 {
            let raw = unsafe {
                std::slice::from_raw_parts(
                    tensor.int32_data.as_ptr() as *const u8,
                    tensor.int32_data.len() * std::mem::size_of::<i32>(),
                )
            };

            let tensor = Tensor::empty(&tensor.dims, dtype, crate::Device::Cpu)
                .expect("cannot create empty tensor");
            let ptr = tensor.data.cast::<u8>().ptr;
            unsafe {
                ptr.copy_from(raw.as_ptr(), raw.len());
            }
            tensor
        } else if tensor.int64_data.len() > 0 {
            let i64_data = tensor.int64_data.as_slice();

            let tensor = Tensor::empty(&tensor.dims, dtype, crate::Device::Cpu)
                .expect("cannot create empty tensor");
            let ptr = tensor.data.cast::<i64>().ptr;
            unsafe {
                ptr.copy_from(i64_data.as_ptr(), i64_data.len());
            }
            tensor
        } else if let Some(raw) = &tensor.raw_data {
            let tensor = Tensor::empty(&tensor.dims, dtype, crate::Device::Cpu)
                .expect("cannot create empty tensor");
            let ptr = tensor.data.cast::<u8>().ptr;
            unsafe {
                ptr.copy_from(raw.as_ptr(), raw.len() * std::mem::size_of::<u8>());
            }
            tensor
        } else {
            panic!("cannot find data in constant tensor attribute");
        }
    }
}

pub(crate) fn get_tensor_from_attribute(
    node: &NodeProto,
    attribute_index: usize,
) -> Result<Tensor, TensorError> {
    let ty =
        attribute_proto::AttributeType::try_from(node.attribute[attribute_index].r#type.unwrap())
            .unwrap();
    match ty {
        attribute_proto::AttributeType::Int => {
            if let Some(i) = node.attribute[0].i {
                let tensor = Tensor::empty(&[1], DType::I64, crate::Device::Cpu)?;
                let mut ptr = tensor.data.cast::<i64>();
                ptr[0] = i;
                Ok(tensor)
            } else {
                Err(OnnxError::new("constant int attribute not found".to_string()).into())
            }
        }
        attribute_proto::AttributeType::Float => {
            if let Some(f) = node.attribute[0].f {
                let tensor = Tensor::empty(&[1], DType::F32, crate::Device::Cpu)?;
                let mut ptr = tensor.data.cast::<f32>();
                ptr[0] = f;
                Ok(tensor)
            } else {
                Err(OnnxError::new("constant float attribute not found".to_string()).into())
            }
        }
        attribute_proto::AttributeType::Tensor => {
            if let Some(tensor) = &node.attribute[0].t {
                Ok(tensor.into())
            } else {
                Err(OnnxError::new(
                    "AttributeType is Tensor but can't find TensorProto".to_string(),
                )
                .into())
            }
        }
        _ => unimplemented!("constant ty {:?} not implemented", ty),
    }
}

pub(crate) fn conv_init(node: &NodeProto, node_degree: &mut HashMap<String, u32>) -> [Operator; 5] {
    let input_name = node.input[0].as_str();
    let kernel_name = node.input[1].as_str();
    let bias_name = node.input[2].as_str();
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

    let inp_permute = Operator::Transpose(Permute {
        input: input_name.to_string(),
        output: node.output[0].to_string(),
        perm: vec![0, 2, 3, 1],
    });

    let kernel_permute = Operator::Transpose(Permute {
        input: kernel_name.to_string(),
        output: node.output[0].to_string(),
        perm: vec![2, 3, 1, 0],
    });

    let tmp_output_name = format!("tmp_{}", node.output[0]);

    let out_permute = Operator::Transpose(Permute {
        input: tmp_output_name.clone(),
        output: tmp_output_name.clone(),
        perm: vec![0, 2, 3, 1],
    });

    let contiguous = Operator::Contiguous(Unary {
        input: tmp_output_name.clone(),
        output: node.output[0].to_string(),
    });

    let conv2d = Operator::Conv2d(Conv2d {
        input: input_name.to_string(),
        output: node.output[0].to_string(),
        kernel: kernel_name.to_string(),
        bias: bias_name.to_string(),
        pads: [(pads[0], pads[1]), (pads[2], pads[3])],
        strides: [strides[0], strides[1]],
        dilations: [dilations[0], dilations[1]],
        group: node.attribute[1].i.unwrap_or(1),
    });

    [inp_permute, kernel_permute, conv2d, out_permute, contiguous]
}

pub(crate) fn unary_init(node: &NodeProto, node_degree: &mut HashMap<String, u32>) -> Operator {
    let input_name = node.input[0].as_str();
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
        "Shape" => Operator::Shape(unary),
        _ => unimplemented!("unary operator {} not implemented", node.op_type()),
    }
}

pub(crate) fn binary_init(node: &NodeProto, node_degree: &mut HashMap<String, u32>) -> Operator {
    let input_name = node.input[0].as_str();
    let input2_name = node.input[1].as_str();
    *node_degree.entry(input_name.to_string()).or_insert(0) += 1;
    *node_degree.entry(input2_name.to_string()).or_insert(0) += 1;
    let binary = Binary {
        input1: input_name.to_string(),
        input2: input2_name.to_string(),
        output: node.output[0].to_string(),
    };
    match node.op_type() {
        "Add" => Operator::Add(binary),
        "Sub" => Operator::Sub(binary),
        "Mul" => Operator::Mul(binary),
        "Div" => Operator::Div(binary),
        "Max" => Operator::Max(binary),
        "Min" => Operator::Min(binary),
        _ => unimplemented!("unary operator {} not implemented", node.op_type()),
    }
}

pub(crate) fn gemm_init(node: &NodeProto, node_degree: &mut HashMap<String, u32>) -> Operator {
    let a = node.input[0].as_str().to_string();
    let b = node.input[1].as_str().to_string();
    let bias = if node.input[2].as_str() == "" {
        None
    } else {
        Some(node.input[2].as_str().to_string())
    };
    *node_degree.entry(a.to_string()).or_insert(0) += 1;
    *node_degree.entry(b.to_string()).or_insert(0) += 1;
    let output = node.output[0].as_str().to_string();
    let alpha = node.attribute[0].f.unwrap_or(1.0) as f64;
    let beta = node.attribute[1].f.unwrap_or(1.0) as f64;
    let trans_a = node.attribute[2].i.unwrap_or(0) != 0;
    let trans_b = node.attribute[3].i.unwrap_or(0) != 0;
    Operator::Gemm(Gemm {
        a,
        b,
        output,
        alpha,
        beta,
        trans_a,
        trans_b,
        bias,
    })
}

pub(crate) fn matmul_init(node: &NodeProto, node_degree: &mut HashMap<String, u32>) -> Operator {
    let a = node.input[0].as_str().to_string();
    let b = node.input[1].as_str().to_string();
    *node_degree.entry(a.to_string()).or_insert(0) += 1;
    *node_degree.entry(b.to_string()).or_insert(0) += 1;
    let output = node.output[0].as_str().to_string();
    Operator::MatMul(Matmul { a, b, output })
}

pub(crate) fn pooling_init(node: &NodeProto, node_degree: &mut HashMap<String, u32>) -> Operator {
    let input_name = node.input[0].as_str();
    *node_degree.entry(input_name.to_string()).or_insert(0) += 1;
    // Operator::Pooling(Pooling {
    //     input: input_name.to_string(),
    //     output: node.output[0].to_string(),
    //     kernel_shape: node.attribute[0].ints.as_slice().to_vec(),
    //     pads: todo!(),
    //     strides: todo!(),
    //     ceil_mode: todo!(),
    // })
    unimplemented!()
}

pub(crate) fn gather_init(node: &NodeProto, node_degree: &mut HashMap<String, u32>) -> Operator {
    let input_name = node.input[0].as_str();
    let indices_name = node.input[1].as_str();
    *node_degree.entry(input_name.to_string()).or_insert(0) += 1;
    *node_degree.entry(indices_name.to_string()).or_insert(0) += 1;
    let gather = Gather {
        input: input_name.to_string(),
        indices: indices_name.to_string(),
        output: node.output[0].to_string(),
        axis: node.attribute[0].i.unwrap_or(0),
    };
    Operator::Gather(gather)
}

pub(crate) fn constant_init(
    node: &NodeProto,
    initializer_map: &mut HashMap<String, Tensor>,
) -> Result<Operator, TensorError> {
    let output_name = node.output[0].as_str();
    let tensor = get_tensor_from_attribute(&node, 0)?;
    initializer_map.insert(output_name.to_string(), tensor);

    Ok(Operator::Constant)
}

pub(crate) fn squeeze_init(
    node: &NodeProto,
    node_degree: &mut HashMap<String, u32>,
    initializer_map: &mut HashMap<String, Tensor>,
) -> Operator {
    let input_name = node.input[0].as_str();
    let axes = node.input[1].as_str();
    *node_degree.entry(input_name.to_string()).or_insert(0) += 1;
    *node_degree.entry(axes.to_string()).or_insert(0) += 1;
    let axes_tensor = initializer_map.get(axes).expect("axes tensor not found");
    Operator::Squeeze(Squeeze {
        input: input_name.to_string(),
        output: node.output[0].to_string(),
        axes: axes_tensor.as_slice().to_vec(),
    })
}

pub(crate) fn unsqueeze_init(
    node: &NodeProto,
    node_degree: &mut HashMap<String, u32>,
    initializer_map: &mut HashMap<String, Tensor>,
) -> Operator {
    let input_name = node.input[0].as_str();
    let axes = node.input[1].as_str();
    *node_degree.entry(input_name.to_string()).or_insert(0) += 1;
    *node_degree.entry(axes.to_string()).or_insert(0) += 1;
    let axes_tensor = initializer_map.get(axes).expect("axes tensor not found");
    Operator::Unsqueeze(Squeeze {
        input: input_name.to_string(),
        output: node.output[0].to_string(),
        axes: axes_tensor.as_slice().to_vec(),
    })
}

pub(crate) fn concat_init(node: &NodeProto, node_degree: &mut HashMap<String, u32>) -> Operator {
    let input_names = node
        .input
        .iter()
        .map(|s| {
            *node_degree.entry(s.as_str().to_string()).or_insert(0) += 1;
            s.as_str().to_string()
        })
        .collect::<Vec<_>>();
    let output_name = node.output[0].as_str();
    Operator::Concat(Concat {
        inputs: input_names,
        output: output_name.to_string(),
        axis: node.attribute[0].i.expect("concat axis not found"),
    })
}

pub(crate) fn const_of_shape_init(
    node: &NodeProto,
    node_degree: &mut HashMap<String, u32>,
) -> Result<Operator, TensorError> {
    let input_name = node.input[0].as_str();
    *node_degree.entry(input_name.to_string()).or_insert(0) += 1;
    if let Ok(tensor) = get_tensor_from_attribute(&node, 0) {
        assert_eq!(tensor.size(), 1);
        let value = match tensor.dtype {
            #[cfg(feature = "bool")]
            DType::Bool => tensor.as_slice::<bool>()[0].cast(),
            #[cfg(feature = "i8")]
            DType::I8 => tensor.as_slice::<i8>()[0].cast(),
            #[cfg(feature = "u8")]
            DType::U8 => tensor.as_slice::<u8>()[0].cast(),
            #[cfg(feature = "i16")]
            DType::I16 => tensor.as_slice::<i16>()[0].cast(),
            #[cfg(feature = "u16")]
            DType::U16 => tensor.as_slice::<u16>()[0].cast(),
            #[cfg(feature = "i32")]
            DType::I32 => tensor.as_slice::<i32>()[0].cast(),
            #[cfg(feature = "u32")]
            DType::U32 => tensor.as_slice::<u32>()[0].cast(),
            #[cfg(feature = "i64")]
            DType::I64 => tensor.as_slice::<i64>()[0].cast(),
            #[cfg(feature = "f32")]
            DType::F32 => tensor.as_slice::<f32>()[0].cast(),
            #[cfg(feature = "f16")]
            DType::F16 => tensor.as_slice::<half::f16>()[0].cast(),
            #[cfg(feature = "bf16")]
            DType::BF16 => tensor.as_slice::<half::bf16>()[0].cast(),
            #[cfg(feature = "u64")]
            DType::U64 => tensor.as_slice::<u64>()[0].cast(),
            #[cfg(feature = "f64")]
            DType::F64 => tensor.as_slice::<f64>()[0].cast(),
            _ => panic!("unsupported dtype {:?}", tensor.dtype),
        };
        Ok(Operator::ConstantOfShape(ConstantOfShape {
            output: node.output[0].to_string(),
            input: input_name.to_string(),
            value,
            dtype: tensor.dtype,
        }))
    } else {
        Ok(Operator::ConstantOfShape(ConstantOfShape {
            output: node.output[0].to_string(),
            input: input_name.to_string(),
            value: 0.0,
            dtype: DType::F32,
        }))
    }
}

pub(crate) fn transpose_init(node: &NodeProto, node_degree: &mut HashMap<String, u32>) -> Operator {
    let input_name = node.input[0].as_str();
    *node_degree.entry(input_name.to_string()).or_insert(0) += 1;
    Operator::Transpose(Permute {
        input: input_name.to_string(),
        output: node.output[0].to_string(),
        perm: node.attribute[0].ints.as_slice().to_vec(),
    })
}

pub(crate) fn slice_init(node: &NodeProto, node_degree: &mut HashMap<String, u32>) -> Operator {
    let input_name = node.input[0].as_str();
    let starts_name = node.input[1].as_str();
    let ends_name = node.input[2].as_str();
    *node_degree.entry(input_name.to_string()).or_insert(0) += 1;
    *node_degree.entry(starts_name.to_string()).or_insert(0) += 1;
    *node_degree.entry(ends_name.to_string()).or_insert(0) += 1;

    let axes_name = if let Some(axes_name) = node.input.get(3) {
        *node_degree
            .entry(axes_name.as_str().to_string())
            .or_insert(0) += 1;
        Some(axes_name.as_str().to_string())
    } else {
        None
    };

    let steps_name = if let Some(steps_name) = node.input.get(4) {
        *node_degree
            .entry(steps_name.as_str().to_string())
            .or_insert(0) += 1;
        Some(steps_name.as_str().to_string())
    } else {
        None
    };

    Operator::Slice(Slice {
        input: input_name.to_string(),
        output: node.output[0].to_string(),
        starts: starts_name.to_string(),
        ends: ends_name.to_string(),
        steps: steps_name,
        axes: axes_name,
    })
}

pub(crate) fn lstm_init(node: &NodeProto, node_degree: &mut HashMap<String, u32>) -> Operator {
    let x = node.input[0].as_str();
    let w = node.input[1].as_str();
    let r = node.input[2].as_str();
    let b = if node.input[3].as_str() == "" {
        None
    } else {
        *node_degree
            .entry(node.input[3].as_str().to_string())
            .or_insert(0) += 1;
        Some(node.input[3].as_str().to_string())
    };
    let sequence_lens = if node.input[4].as_str() == "" {
        None
    } else {
        *node_degree
            .entry(node.input[4].as_str().to_string())
            .or_insert(0) += 1;
        Some(node.input[4].as_str().to_string())
    };
    let initial_h = if node.input[5].as_str() == "" {
        None
    } else {
        *node_degree
            .entry(node.input[5].as_str().to_string())
            .or_insert(0) += 1;
        Some(node.input[5].as_str().to_string())
    };
    let initial_c = if node.input[6].as_str() == "" {
        None
    } else {
        *node_degree
            .entry(node.input[6].as_str().to_string())
            .or_insert(0) += 1;
        Some(node.input[6].as_str().to_string())
    };
    let p = node.input.get(7).map(|s| {
        *node_degree.entry(s.as_str().to_string()).or_insert(0) += 1;
        s.as_str().to_string()
    });
    let y = if node.output[0].as_str() == "" {
        None
    } else {
        *node_degree
            .entry(node.output[0].as_str().to_string())
            .or_insert(0) += 1;
        Some(node.output[0].as_str().to_string())
    };
    let y_h = if node.output[1].as_str() == "" {
        None
    } else {
        *node_degree
            .entry(node.output[1].as_str().to_string())
            .or_insert(0) += 1;
        Some(node.output[1].as_str().to_string())
    };
    let y_c = if node.output[2].as_str() == "" {
        None
    } else {
        *node_degree
            .entry(node.output[2].as_str().to_string())
            .or_insert(0) += 1;
        Some(node.output[2].as_str().to_string())
    };

    let activation_alpha = if let Some(alpha_attri) = node
        .attribute
        .iter()
        .find(|x| x.name() == "activation_alpha")
    {
        Some(alpha_attri.floats.as_slice().to_vec())
    } else {
        None
    };

    let activation_beta = if let Some(beta_attri) = node
        .attribute
        .iter()
        .find(|x| x.name() == "activation_beta")
    {
        Some(beta_attri.floats.as_slice().to_vec())
    } else {
        None
    };

    let activations = if let Some(activations_attri) =
        node.attribute.iter().find(|x| x.name() == "activations")
    {
        Some(
            activations_attri
                .strings
                .iter()
                .map(|x| bytes_to_string(x))
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };

    let clip = if let Some(clip_attri) = node.attribute.iter().find(|x| x.name() == "clip") {
        clip_attri.f
    } else {
        None
    };

    let direction =
        if let Some(direction_attri) = node.attribute.iter().find(|x| x.name() == "direction") {
            bytes_to_string(direction_attri.s())
        } else {
            "forward".to_string()
        };

    let hidden_size = node
        .attribute
        .iter()
        .find(|x| x.name() == "hidden_size")
        .expect("hidden_size not found")
        .i();

    let input_forget = node
        .attribute
        .iter()
        .find(|x| x.name() == "input_forget")
        .map(|x| x.i.unwrap_or(0))
        .unwrap_or(0)
        != 0;

    let layout = node
        .attribute
        .iter()
        .find(|x| x.name() == "layout")
        .map(|x| x.i.unwrap_or(0))
        .unwrap_or(0);

    Operator::Lstm(Lstm {
        x: x.to_string(),
        w: w.to_string(),
        r: r.to_string(),
        b,
        sequence_lens,
        initial_h,
        initial_c,
        p,
        activation_alpha,
        activation_beta,
        activations,
        clip,
        direction,
        hidden_size,
        input_forget,
        layout,
        y,
        y_h,
        y_c,
    })
}

pub(crate) fn permute_init(node: &NodeProto, node_degree: &mut HashMap<String, u32>) -> Operator {
    let input_name = node.input[0].as_str();
    *node_degree.entry(input_name.to_string()).or_insert(0) += 1;
    Operator::Transpose(Permute {
        input: input_name.to_string(),
        output: node.output[0].to_string(),
        perm: node.attribute[0].ints.as_slice().to_vec(),
    })
}

pub(crate) fn contiguous_init(
    node: &NodeProto,
    node_degree: &mut HashMap<String, u32>,
) -> Operator {
    let input_name = node.input[0].as_str();
    *node_degree.entry(input_name.to_string()).or_insert(0) += 1;
    Operator::Contiguous(Unary {
        input: input_name.to_string(),
        output: node.output[0].to_string(),
    })
}
