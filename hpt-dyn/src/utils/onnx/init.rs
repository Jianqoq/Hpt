use hpt_common::error::{ base::TensorError, onnx::OnnxError };
use hpt_types::{ dtype::DType, into_scalar::Cast };

use super::{
    map_dtype::to_dtype,
    operators::{
        AutoPad, Base, BatchNormalization, Binary, Concat, Elu, Expand, Flatten, Gather, Gemm, LayerNormalization, Lstm, Matmul, Operator, Permute, Pooling, Reduce, Reshape, Slice, Softmax, Squeeze, TensorFormat, Unary
    },
};
use crate::{
    Tensor,
    onnx::{ NodeProto, TensorProto, attribute_proto },
    utils::onnx::operators::{ ConstantOfShape, Conv2d },
};
use hpt_traits::tensor::TensorInfo;
use std::collections::HashMap;

macro_rules! get {
    ($map:expr, $key:expr) => {
        $map.get($key)
            .unwrap_or_else(|| panic!("key {} not found in map", $key))
    };
}

fn bytes_to_string(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).expect("invalid utf-8 sequence")
}

fn parse_string_attribute(
    node: &NodeProto,
    arg_idx: &mut usize,
    target: &str,
    default: &str
) -> String {
    if let Some(attr) = node.attribute.get(*arg_idx) {
        if attr.name() == target {
            let res = if let Some(s) = &attr.s {
                if s.is_empty() { default.to_string() } else { bytes_to_string(s.as_slice()) }
            } else {
                default.to_string()
            };
            *arg_idx += 1;
            res
        } else {
            default.to_string()
        }
    } else {
        default.to_string()
    }
}

fn parse_strings_attribute(
    node: &NodeProto,
    arg_idx: &mut usize,
    target: &str,
    default: Vec<String>
) -> Vec<String> {
    let mut res = vec![];
    if node.attribute[*arg_idx].name() == target {
        for (idx, s) in node.attribute[*arg_idx].strings.iter().enumerate() {
            if s.is_empty() {
                res.push(default[idx].clone());
            } else {
                res.push(bytes_to_string(s.as_slice()));
            }
        }
        if default.len() > node.attribute[*arg_idx].strings.len() {
            for s in default.iter().skip(node.attribute[*arg_idx].strings.len()) {
                res.push(s.to_string());
            }
        }
        *arg_idx += 1;
        res
    } else {
        default
    }
}

fn parse_int_attribute(node: &NodeProto, arg_idx: &mut usize, target: &str, default: i64) -> i64 {
    if let Some(attr) = node.attribute.get(*arg_idx) {
        if attr.name() == target {
            let res = attr.i.unwrap_or(default);
            *arg_idx += 1;
            res
        } else {
            default
        }
    } else {
        default
    }
}

fn parse_int_attribute_required(node: &NodeProto, arg_idx: &mut usize, target: &str) -> i64 {
    if let Some(attr) = node.attribute.get(*arg_idx) {
        if attr.name() == target {
            let res = attr.i.expect(format!("expect {} but not found", target).as_str());
            *arg_idx += 1;
            res
        } else {
            panic!("expect {} but not found", target);
        }
    } else {
        panic!("expect {} but not found", target);
    }
}

fn parse_float_attribute(node: &NodeProto, arg_idx: &mut usize, target: &str, default: f32) -> f32 {
    if node.attribute[*arg_idx].name() == target {
        let res = node.attribute[*arg_idx].f.unwrap_or(default);
        *arg_idx += 1;
        res
    } else {
        default
    }
}

fn parse_floats_attribute(
    node: &NodeProto,
    arg_idx: &mut usize,
    target: &str,
    default: Vec<f32>
) -> Vec<f32> {
    let mut res = vec![];
    if node.attribute[*arg_idx].name() == target {
        for f in node.attribute[*arg_idx].floats.iter() {
            res.push(*f);
        }
        if default.len() > node.attribute[*arg_idx].floats.len() {
            for f in default.iter().skip(node.attribute[*arg_idx].floats.len()) {
                res.push(*f);
            }
        }
        *arg_idx += 1;
        res
    } else {
        default
    }
}

fn parse_ints_attribute(
    node: &NodeProto,
    arg_idx: &mut usize,
    target: &str,
    default: Vec<i64>
) -> Vec<i64> {
    let mut res = vec![];
    if let Some(attr) = node.attribute.get(*arg_idx) {
        if attr.name() == target {
            for f in attr.ints.iter() {
                res.push(*f);
            }
            if default.len() > attr.ints.len() {
                for f in default.iter().skip(attr.ints.len()) {
                    res.push(*f);
                }
            }
            *arg_idx += 1;
            res
        } else {
            default
        }
    } else {
        default
    }
}

fn try_pc(
    ret: &mut Vec<Operator>,
    input: &str,
    node_name: &str,
    formats: &mut HashMap<String, TensorFormat>
) -> Option<String> {
    if let Some(format) = formats.get(input) {
        if *format == TensorFormat::NHWC {
            ret.push(
                Operator::PermuteContiguous(Base {
                    base: Permute {
                        input: input.to_string(),
                        output: format!("{}_{}_permute", input, node_name),
                        perm: vec![0, 2, 3, 1],
                    },
                    id: format!("{}_{}_permute", input, node_name),
                })
            );
            Some(format!("{}_{}_permute", input, node_name))
        } else if *format == TensorFormat::HWCO {
            ret.push(
                Operator::PermuteContiguous(Base {
                    base: Permute {
                        input: input.to_string(),
                        output: format!("{}_{}_permute", input, node_name),
                        perm: vec![3, 2, 0, 1],
                    },
                    id: format!("{}_{}_permute", input, node_name),
                })
            );
            Some(format!("{}_{}_permute", input, node_name))
        } else {
            None
        }
    } else {
        panic!("input {} not found in formats", input);
    }
}

fn try_conv_pc(
    ret: &mut Vec<Operator>,
    input: &str,
    node_name: &str,
    formats: &mut HashMap<String, TensorFormat>
) {
    if let Some(format) = formats.get(input) {
        if *format == TensorFormat::NCHW {
            ret.push(
                Operator::PermuteContiguous(Base {
                    base: Permute {
                        input: input.to_string(),
                        output: format!("{}_{}_permute", input, node_name),
                        perm: vec![0, 2, 3, 1],
                    },
                    id: format!("{}_{}_permute", input, node_name),
                })
            );
        } else if *format == TensorFormat::HWCO || *format == TensorFormat::OCHW {
            panic!("invalid format for conv permute contiguous: {:?}", format);
        }
    } else {
        panic!("input {} not found in formats", input);
    }
}

fn insert_default_format(formats: &mut HashMap<String, TensorFormat>, output: &str) {
    formats.insert(output.to_string(), TensorFormat::NCHW);
}

fn insert_conv_format(formats: &mut HashMap<String, TensorFormat>, output: &str) {
    formats.insert(output.to_string(), TensorFormat::NHWC);
}

impl From<&TensorProto> for Tensor {
    fn from(tensor: &TensorProto) -> Self {
        let dtype = to_dtype(tensor.data_type());
        if tensor.float_data.len() > 0 {
            let raw = unsafe {
                std::slice::from_raw_parts(
                    tensor.float_data.as_ptr() as *const u8,
                    tensor.float_data.len() * std::mem::size_of::<f32>()
                )
            };

            let tensor = Tensor::empty(&tensor.dims, dtype, crate::Device::Cpu).expect(
                "cannot create   empty tensor"
            );
            let ptr = tensor.data.cast::<u8>().ptr;
            unsafe {
                ptr.copy_from(raw.as_ptr(), raw.len());
            }
            tensor
        } else if tensor.int32_data.len() > 0 {
            let raw = unsafe {
                std::slice::from_raw_parts(
                    tensor.int32_data.as_ptr() as *const u8,
                    tensor.int32_data.len() * std::mem::size_of::<i32>()
                )
            };

            let tensor = Tensor::empty(&tensor.dims, dtype, crate::Device::Cpu).expect(
                "cannot create empty tensor"
            );
            let ptr = tensor.data.cast::<u8>().ptr;
            unsafe {
                ptr.copy_from(raw.as_ptr(), raw.len());
            }
            tensor
        } else if tensor.int64_data.len() > 0 {
            let i64_data = tensor.int64_data.as_slice();

            let tensor = Tensor::empty(&tensor.dims, dtype, crate::Device::Cpu).expect(
                "cannot create empty tensor"
            );
            let ptr = tensor.data.cast::<i64>().ptr;
            unsafe {
                ptr.copy_from(i64_data.as_ptr(), i64_data.len());
            }
            tensor
        } else if let Some(raw) = &tensor.raw_data {
            let tensor = Tensor::empty(&tensor.dims, dtype, crate::Device::Cpu).expect(
                "cannot create empty tensor"
            );
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
    attribute_index: usize
) -> Result<Tensor, TensorError> {
    let ty = attribute_proto::AttributeType
        ::try_from(node.attribute[attribute_index].r#type.unwrap())
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
                Err(
                    OnnxError::new(
                        "AttributeType is Tensor but can't find TensorProto".to_string()
                    ).into()
                )
            }
        }
        _ => unimplemented!("constant ty {:?} not implemented", ty),
    }
}

pub(crate) fn conv_init(
    node: &NodeProto,

    formats: &mut HashMap<String, TensorFormat>
) -> Vec<Operator> {
    let input_name = node.input[0].as_str();
    let kernel_name = node.input[1].as_str();
    let bias_name = if let Some(bias_name) = node.input.get(2) {
        Some(bias_name.clone())
    } else {
        None
    };

    assert_eq!(node.attribute[0].name(), "dilations");
    let dilations = &node.attribute[0].ints.as_slice();
    assert_eq!(dilations.len(), 2);
    assert_eq!(node.attribute[3].name(), "pads");
    let pads = node.attribute[3].ints.as_slice();
    assert_eq!(node.attribute[4].name(), "strides");
    let strides = node.attribute[4].ints.as_slice();
    assert_eq!(strides.len(), 2);

    let inp_permute = Operator::Transpose(Base {
        base: Permute {
            input: input_name.to_string(),
            output: format!("{}_inp_permute", node.name()),
            perm: vec![0, 2, 3, 1],
        },
        id: format!("{}_inp_permute", node.name()),
    });

    let kernel_permute = Operator::Transpose(Base {
        base: Permute {
            input: kernel_name.to_string(),
            output: format!("{}_kernel_permute", node.name()),
            perm: vec![2, 3, 1, 0],
        },
        id: format!("{}_kernel_permute", node.name()),
    });

    let mut conv2d = Conv2d {
        input: input_name.to_string(),
        output: node.output[0].to_string(),
        kernel: kernel_name.to_string(),
        bias: bias_name,
        pads: [
            (pads[0], pads[1]),
            (pads[2], pads[3]),
        ],
        strides: [strides[0], strides[1]],
        dilations: [dilations[0], dilations[1]],
        group: node.attribute[1].i.unwrap_or(1),
    };

    let mut ret = vec![];

    if *get!(formats, input_name) != TensorFormat::NHWC {
        ret.push(inp_permute);
        conv2d.input = format!("{}_inp_permute", node.name());
    }
    if *get!(formats, kernel_name) != TensorFormat::HWCO {
        ret.push(kernel_permute);
        conv2d.kernel = format!("{}_kernel_permute", node.name());
    }
    let conv2d = Operator::Conv2d(Base {
        base: conv2d,
        id: node.name().to_string(),
    });
    ret.push(conv2d);
    formats.insert(node.output[0].to_string(), TensorFormat::NHWC);

    ret
}

pub(crate) fn unary_init(
    node: &NodeProto,
    formats: &mut HashMap<String, TensorFormat>
) -> Operator {
    let input_name = node.input[0].as_str();
    let unary = Unary {
        input: input_name.to_string(),
        output: node.output[0].to_string(),
    };
    let name = node.name().to_string();
    let base = Base {
        base: unary,
        id: name,
    };
    formats.insert(node.output[0].to_string(), formats[input_name]);
    match node.op_type() {
        "Abs" => Operator::Abs(base),
        "Acos" => Operator::Acos(base),
        "Acosh" => Operator::Acosh(base),
        "Asin" => Operator::Asin(base),
        "Asinh" => Operator::Asinh(base),
        "Atan" => Operator::Atan(base),
        "Atanh" => Operator::Atanh(base),
        "BitwiseNot" => Operator::BitwiseNot(base),
        "Ceil" => Operator::Ceil(base),
        "Cos" => Operator::Cos(base),
        "Cosh" => Operator::Cosh(base),
        "Erf" => Operator::Erf(base),
        "Exp" => Operator::Exp(base),
        "Floor" => Operator::Floor(base),
        "IsInf" => Operator::IsInf(base),
        "IsNaN" => Operator::IsNaN(base),
        "Log" => Operator::Log(base),
        "Neg" => Operator::Neg(base),
        "Not" => Operator::Not(base),
        "Reciprocal" => Operator::Reciprocal(base),
        "Round" => Operator::Round(base),
        "Sigmoid" => Operator::Sigmoid(base),
        "Sign" => Operator::Sign(base),
        "Sin" => Operator::Sin(base),
        "Sinh" => Operator::Sinh(base),
        "Sqrt" => Operator::Sqrt(base),
        "Tan" => Operator::Tan(base),
        "Tanh" => Operator::Tanh(base),
        "Gelu" => Operator::Gelu(base),
        "HardSigmoid" => Operator::HardSigmoid(base),
        "HardSwish" => Operator::HardSwish(base),
        "LeakyRelu" => Operator::LeakyRelu(base),
        "Mish" => Operator::Mish(base),
        "Shrink" => Operator::Shrink(base),
        "Relu" => Operator::Relu(base),
        "Softplus" => Operator::Softplus(base),
        "Softsign" => Operator::Softsign(base),
        "Shape" => Operator::Shape(base),
        _ => unimplemented!("unary operator {} not implemented", node.op_type()),
    }
}

pub(crate) fn selu_init(node: &NodeProto, formats: &mut HashMap<String, TensorFormat>) -> Operator {
    let input_name = node.input[0].as_str();
    let mut arg_idx = 0;
    let alpha = parse_float_attribute(
        node,
        &mut arg_idx,
        "alpha",
        1.6732632423543772848170429916717
    ) as f64;
    let gamma = parse_float_attribute(
        node,
        &mut arg_idx,
        "gamma",
        1.0507009873554804934193349852946
    ) as f64;
    let name = node.name().to_string();
    formats.insert(node.output[0].to_string(), formats[input_name]);
    Operator::Selu(Base {
        base: Elu {
            input: input_name.to_string(),
            output: node.output[0].to_string(),
            alpha,
            gamma,
        },
        id: name,
    })
}

pub(crate) fn elu_init(node: &NodeProto, formats: &mut HashMap<String, TensorFormat>) -> Operator {
    let input_name = node.input[0].as_str();
    let mut arg_idx = 0;
    let alpha = parse_float_attribute(node, &mut arg_idx, "alpha", 1.0) as f64;
    let name = node.name().to_string();
    formats.insert(node.output[0].to_string(), formats[input_name]);
    Operator::Elu(Base {
        base: Elu {
            input: input_name.to_string(),
            output: node.output[0].to_string(),
            alpha,
            gamma: 0.0,
        },
        id: name,
    })
}

pub(crate) fn binary_init(
    node: &NodeProto,

    formats: &mut HashMap<String, TensorFormat>
) -> Vec<Operator> {
    let input_name = node.input[0].as_str();
    let input2_name = node.input[1].as_str();
    let name = node.name().to_string();
    let mut ret = vec![];
    let binary = if get!(formats, input_name) == get!(formats, input2_name) {
        formats.insert(node.output[0].to_string(), *get!(formats, input_name));
        Binary {
            input1: input_name.to_string(),
            input2: input2_name.to_string(),
            output: node.output[0].to_string(),
        }
    } else {
        let new_input_name = try_pc(&mut ret, input_name, &name, formats);
        let new_input2_name = try_pc(&mut ret, input2_name, &name, formats);
        insert_default_format(formats, &node.output[0]);
        Binary {
            input1: new_input_name.unwrap_or(input_name.to_string()),
            input2: new_input2_name.unwrap_or(input2_name.to_string()),
            output: node.output[0].to_string(),
        }
    };
    let base = Base {
        base: binary,
        id: name,
    };

    match node.op_type() {
        "Add" => ret.push(Operator::Add(base)),
        "Sub" => ret.push(Operator::Sub(base)),
        "Mul" => ret.push(Operator::Mul(base)),
        "Div" => ret.push(Operator::Div(base)),
        "GreaterOrEqual" => ret.push(Operator::GreaterOrEqual(base)),
        "LessOrEqual" => ret.push(Operator::LessOrEqual(base)),
        "Equal" => ret.push(Operator::Equal(base)),
        "Greater" => ret.push(Operator::Greater(base)),
        "Less" => ret.push(Operator::Less(base)),
        "BitwiseOr" => ret.push(Operator::BitwiseOr(base)),
        "BitwiseAnd" => ret.push(Operator::BitwiseAnd(base)),
        "BitwiseXor" => ret.push(Operator::BitwiseXor(base)),
        "Mod" => ret.push(Operator::Mod(base)),
        "Pow" => ret.push(Operator::Pow(base)),
        _ => unimplemented!("unary operator {} not implemented", node.op_type()),
    }
    ret
}

pub(crate) fn gemm_init(
    node: &NodeProto,

    formats: &mut HashMap<String, TensorFormat>
) -> Vec<Operator> {
    let a = node.input[0].as_str().to_string();
    let b = node.input[1].as_str().to_string();
    let bias = if node.input[2].as_str() == "" {
        None
    } else {
        Some(node.input[2].as_str().to_string())
    };
    let mut arg_idx = 0;
    let alpha = parse_float_attribute(node, &mut arg_idx, "alpha", 1.0) as f64;
    let beta = parse_float_attribute(node, &mut arg_idx, "beta", 1.0) as f64;
    let trans_a = parse_int_attribute(node, &mut arg_idx, "transA", 0) == 1;
    let trans_b = parse_int_attribute(node, &mut arg_idx, "transB", 0) == 1;

    let output = node.output[0].as_str().to_string();
    let name = node.name().to_string();
    let mut ret = vec![];
    let new_a_name = try_pc(&mut ret, a.as_str(), &name, formats);
    let new_b_name = try_pc(&mut ret, b.as_str(), &name, formats);
    insert_default_format(formats, &output);
    let gemm = Gemm {
        a: new_a_name.unwrap_or(a.to_string()),
        b: new_b_name.unwrap_or(b.to_string()),
        output,
        alpha,
        beta,
        trans_a,
        trans_b,
        bias,
    };
    ret.push(
        Operator::Gemm(Base {
            base: gemm,
            id: name,
        })
    );
    ret
}

pub(crate) fn matmul_init(
    node: &NodeProto,

    formats: &mut HashMap<String, TensorFormat>
) -> Vec<Operator> {
    let a = node.input[0].as_str().to_string();
    let b = node.input[1].as_str().to_string();
    let output = node.output[0].as_str().to_string();
    let name = node.name().to_string();
    let mut ret = vec![];
    let new_a_name = try_pc(&mut ret, a.as_str(), &name, formats);
    let new_b_name = try_pc(&mut ret, b.as_str(), &name, formats);
    insert_default_format(formats, &output);
    let matmul = Matmul {
        a: new_a_name.unwrap_or(a.to_string()),
        b: new_b_name.unwrap_or(b.to_string()),
        output,
    };
    ret.push(
        Operator::MatMul(Base {
            base: matmul,
            id: name,
        })
    );
    ret
}

pub(crate) fn pooling_init(
    node: &NodeProto,

    formats: &mut HashMap<String, TensorFormat>
) -> Vec<Operator> {
    let input_name = node.input[0].as_str();
    let mut arg_idx = 0;
    let auto_pad_str = parse_string_attribute(node, &mut arg_idx, "auto_pad", "NOTSET");
    let auto_pad = match auto_pad_str.as_str() {
        "NOTSET" => AutoPad::NOTSET,
        "SAME_UPPER" => AutoPad::SAME_UPPER,
        "SAME_LOWER" => AutoPad::SAME_LOWER,
        "VALID" => AutoPad::VALID,
        _ => unimplemented!("unsupported auto_pad: {}", auto_pad_str),
    };
    let ceil_mode = parse_int_attribute(node, &mut arg_idx, "ceil_mode", 0) != 0;
    let dilations = parse_ints_attribute(node, &mut arg_idx, "dilations", vec![]);
    let kernel_shape = parse_ints_attribute(node, &mut arg_idx, "kernel_shape", vec![]);
    let pads = parse_ints_attribute(node, &mut arg_idx, "pads", vec![]);
    let storage_order = parse_int_attribute(node, &mut arg_idx, "storage_order", 0) != 0;
    let strides = parse_ints_attribute(node, &mut arg_idx, "strides", vec![]);
    let pooling = Pooling {
        input: input_name.to_string(),
        output: node.output[0].to_string(),
        auto_pad,
        kernel_shape,
        pads,
        strides,
        dilations,
        ceil_mode,
        storage_order,
    };
    let mut ret = vec![];
    try_conv_pc(&mut ret, input_name, node.name(), formats);

    insert_conv_format(formats, &node.output[0]);
    match node.op_type() {
        "MaxPool" =>
            ret.push(
                Operator::MaxPool(Base {
                    base: pooling,
                    id: node.name().to_string(),
                })
            ),
        "AveragePool" =>
            ret.push(
                Operator::AveragePool(Base {
                    base: pooling,
                    id: node.name().to_string(),
                })
            ),
        "GlobalMaxPool" =>
            ret.push(
                Operator::GlobalMaxPool(Base {
                    base: pooling,
                    id: node.name().to_string(),
                })
            ),
        "GlobalAveragePool" =>
            ret.push(
                Operator::GlobalAveragePool(Base {
                    base: pooling,
                    id: node.name().to_string(),
                })
            ),
        _ => unimplemented!("unsupported pooling operator: {}", node.op_type()),
    }
    ret
}

pub(crate) fn gather_init(
    node: &NodeProto,

    formats: &mut HashMap<String, TensorFormat>
) -> Vec<Operator> {
    let input_name = node.input[0].as_str();
    let indices_name = node.input[1].as_str();
    let name = node.name().to_string();
    let mut ret = vec![];
    let new_input_name = try_pc(&mut ret, input_name, &name, formats);
    let new_indices_name = try_pc(&mut ret, indices_name, &name, formats);
    let gather = Gather {
        input: new_input_name.unwrap_or(input_name.to_string()),
        indices: new_indices_name.unwrap_or(indices_name.to_string()),
        output: node.output[0].to_string(),
        axis: node.attribute[0].i.unwrap_or(0),
    };
    insert_default_format(formats, &gather.output);
    ret.push(
        Operator::Gather(Base {
            base: gather,
            id: name,
        })
    );
    ret
}

pub(crate) fn constant_init(
    node: &NodeProto,
    initializer_map: &mut HashMap<String, Tensor>,
    formats: &mut HashMap<String, TensorFormat>
) -> Result<Operator, TensorError> {
    let output_name = node.output[0].as_str();
    let tensor = get_tensor_from_attribute(&node, 0)?;
    initializer_map.insert(output_name.to_string(), tensor);
    let name = node.name().to_string();
    insert_default_format(formats, &output_name);
    Ok(
        Operator::Constant(Base {
            base: output_name.to_string(),
            id: name,
        })
    )
}

pub(crate) fn squeeze_init(
    node: &NodeProto,

    initializer_map: &mut HashMap<String, Tensor>,
    formats: &mut HashMap<String, TensorFormat>
) -> Vec<Operator> {
    let input_name = node.input[0].as_str();
    let axes = node.input[1].as_str();
    let axes_tensor = initializer_map.get(axes).expect("axes tensor not found");
    let name = node.name().to_string();
    let mut ret = vec![];
    let new_input_name = try_pc(&mut ret, input_name, &name, formats);
    let squeeze = Squeeze {
        input: new_input_name.unwrap_or(input_name.to_string()),
        output: node.output[0].to_string(),
        axes: axes_tensor.as_slice().to_vec(),
    };
    insert_default_format(formats, &squeeze.output);
    ret.push(
        Operator::Squeeze(Base {
            base: squeeze,
            id: name,
        })
    );
    ret
}

pub(crate) fn unsqueeze_init(
    node: &NodeProto,

    initializer_map: &mut HashMap<String, Tensor>,
    formats: &mut HashMap<String, TensorFormat>
) -> Vec<Operator> {
    let input_name = node.input[0].as_str();
    let axes = node.input[1].as_str();
    let axes_tensor = initializer_map.get(axes).expect("axes tensor not found");
    let name = node.name().to_string();
    let mut ret = vec![];
    let new_input_name = try_pc(&mut ret, input_name, &name, formats);
    let squeeze = Squeeze {
        input: new_input_name.unwrap_or(input_name.to_string()),
        output: node.output[0].to_string(),
        axes: axes_tensor.as_slice().to_vec(),
    };
    insert_default_format(formats, &squeeze.output);
    ret.push(
        Operator::Unsqueeze(Base {
            base: squeeze,
            id: name,
        })
    );
    ret
}

pub(crate) fn concat_init(
    node: &NodeProto,

    formats: &mut HashMap<String, TensorFormat>
) -> Vec<Operator> {
    let mut input_names = node.input
        .iter()
        .map(|s| s.as_str().to_string())
        .collect::<Vec<_>>();
    let output_name = node.output[0].as_str();
    let name = node.name().to_string();
    let mut ret = vec![];
    for input_name in input_names.iter_mut() {
        let new_input_name = try_pc(&mut ret, &input_name, &name, formats);
        *input_name = new_input_name.unwrap_or(input_name.to_string());
    }
    let concat = Concat {
        inputs: input_names.clone(),
        output: output_name.to_string(),
        axis: node.attribute[0].i.expect("concat axis not found"),
    };
    insert_default_format(formats, &output_name);
    ret.push(
        Operator::Concat(Base {
            base: concat,
            id: name,
        })
    );
    ret
}

pub(crate) fn const_of_shape_init(
    node: &NodeProto,

    formats: &mut HashMap<String, TensorFormat>
) -> Result<Vec<Operator>, TensorError> {
    let input_name = node.input[0].as_str();
    let name = node.name().to_string();
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
        let mut ret = vec![];
        let new_input_name = try_pc(&mut ret, input_name, &name, formats);
        let constant_of_shape = ConstantOfShape {
            output: node.output[0].to_string(),
            input: new_input_name.unwrap_or(input_name.to_string()),
            value,
            dtype: tensor.dtype,
        };
        insert_default_format(formats, &constant_of_shape.output);
        ret.push(
            Operator::ConstantOfShape(Base {
                base: constant_of_shape,
                id: name,
            })
        );
        Ok(ret)
    } else {
        let mut ret = vec![];
        let new_input_name = try_pc(&mut ret, input_name, &name, formats);
        let constant_of_shape = ConstantOfShape {
            output: node.output[0].to_string(),
            input: new_input_name.unwrap_or(input_name.to_string()),
            value: 0.0,
            dtype: DType::F32,
        };
        insert_default_format(formats, &constant_of_shape.output);
        ret.push(
            Operator::ConstantOfShape(Base {
                base: constant_of_shape,
                id: name,
            })
        );
        Ok(ret)
    }
}

pub(crate) fn transpose_init(
    node: &NodeProto,

    formats: &mut HashMap<String, TensorFormat>
) -> Vec<Operator> {
    let input_name = node.input[0].as_str();
    let name = node.name().to_string();
    let mut ret = vec![];
    let new_input_name = try_pc(&mut ret, input_name, &name, formats);
    let permute = Permute {
        input: new_input_name.unwrap_or(input_name.to_string()),
        output: node.output[0].to_string(),
        perm: node.attribute[0].ints.as_slice().to_vec(),
    };
    insert_default_format(formats, &permute.output);
    ret.push(
        Operator::Transpose(Base {
            base: permute,
            id: name,
        })
    );
    ret
}

pub(crate) fn slice_init(
    node: &NodeProto,

    formats: &mut HashMap<String, TensorFormat>
) -> Vec<Operator> {
    let input_name = node.input[0].as_str();
    let starts_name = node.input[1].as_str();
    let ends_name = node.input[2].as_str();

    let axes_name = if let Some(axes_name) = node.input.get(3) {
        Some(axes_name.as_str().to_string())
    } else {
        None
    };

    let steps_name = if let Some(steps_name) = node.input.get(4) {
        Some(steps_name.as_str().to_string())
    } else {
        None
    };
    let name = node.name().to_string();

    let mut ret = vec![];
    let new_input_name = try_pc(&mut ret, input_name, &name, formats);
    let slice = Slice {
        input: new_input_name.unwrap_or(input_name.to_string()),
        output: node.output[0].to_string(),
        starts: starts_name.to_string(),
        ends: ends_name.to_string(),
        steps: steps_name,
        axes: axes_name,
    };
    insert_default_format(formats, &slice.output);
    ret.push(
        Operator::Slice(Base {
            base: slice,
            id: name,
        })
    );
    ret
}

pub(crate) fn lstm_init(
    node: &NodeProto,
    formats: &mut HashMap<String, TensorFormat>
) -> Vec<Operator> {
    let x = node.input[0].as_str();
    let w = node.input[1].as_str();
    let r = node.input[2].as_str();
    let mut b = if node.input[3].as_str() == "" {
        None
    } else {
        Some(node.input[3].as_str().to_string())
    };
    let mut sequence_lens = if node.input[4].as_str() == "" {
        None
    } else {
        Some(node.input[4].as_str().to_string())
    };
    let mut initial_h = if node.input[5].as_str() == "" {
        None
    } else {
        Some(node.input[5].as_str().to_string())
    };
    let mut initial_c = if node.input[6].as_str() == "" {
        None
    } else {
        Some(node.input[6].as_str().to_string())
    };
    let mut p = node.input.get(7).map(|s| s.as_str().to_string());
    let y = if node.output[0].as_str() == "" {
        None
    } else {
        Some(node.output[0].as_str().to_string())
    };
    let y_h = if node.output[1].as_str() == "" {
        None
    } else {
        Some(node.output[1].as_str().to_string())
    };
    let y_c = if node.output[2].as_str() == "" {
        None
    } else {
        Some(node.output[2].as_str().to_string())
    };

    let mut arg_idx = 0;

    let activation_alpha = parse_floats_attribute(node, &mut arg_idx, "activation_alpha", vec![]);
    let activation_beta = parse_floats_attribute(node, &mut arg_idx, "activation_beta", vec![]);
    let activations = parse_strings_attribute(
        node,
        &mut arg_idx,
        "activations",
        vec!["Sigmoid".to_string(), "Tanh".to_string(), "Tanh".to_string()]
    );

    let clip = {
        let arg_idx: &mut usize = &mut arg_idx;
        if node.attribute[*arg_idx].name() == "clip" {
            let res = node.attribute[*arg_idx].f;
            *arg_idx += 1;
            res
        } else {
            None
        }
    };
    let direction = parse_string_attribute(node, &mut arg_idx, "direction", "forward");
    let hidden_size = parse_int_attribute_required(node, &mut arg_idx, "hidden_size");
    let input_forget = parse_int_attribute(node, &mut arg_idx, "input_forget", 0) != 0;
    let layout = parse_int_attribute(node, &mut arg_idx, "layout", 0);

    let name = node.name().to_string();
    let mut ret = vec![];
    let new_x = try_pc(&mut ret, x, &name, formats);
    let new_w = try_pc(&mut ret, w, &name, formats);
    let new_r = try_pc(&mut ret, r, &name, formats);
    if let Some(b) = &mut b {
        let new_b = try_pc(&mut ret, b.as_str(), &name, formats);
        *b = new_b.unwrap_or(b.to_string());
    }
    if let Some(sequence_lens) = &mut sequence_lens {
        let new_sequence_lens = try_pc(&mut ret, sequence_lens.as_str(), &name, formats);
        *sequence_lens = new_sequence_lens.unwrap_or(sequence_lens.to_string());
    }
    if let Some(initial_h) = &mut initial_h {
        let new_initial_h = try_pc(&mut ret, initial_h.as_str(), &name, formats);
        *initial_h = new_initial_h.unwrap_or(initial_h.to_string());
    }
    if let Some(initial_c) = &mut initial_c {
        let new_initial_c = try_pc(&mut ret, initial_c.as_str(), &name, formats);
        *initial_c = new_initial_c.unwrap_or(initial_c.to_string());
    }
    if let Some(p) = &mut p {
        let new_p = try_pc(&mut ret, p.as_str(), &name, formats);
        *p = new_p.unwrap_or(p.to_string());
    }
    if let Some(y) = &y {
        insert_default_format(formats, y);
    }
    if let Some(y_h) = &y_h {
        insert_default_format(formats, y_h);
    }
    if let Some(y_c) = &y_c {
        insert_default_format(formats, y_c);
    }
    let lstm = Lstm {
        x: new_x.unwrap_or(x.to_string()),
        w: new_w.unwrap_or(w.to_string()),
        r: new_r.unwrap_or(r.to_string()),
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
    };
    ret.push(
        Operator::Lstm(Base {
            base: lstm,
            id: name,
        })
    );
    ret
}

pub(crate) fn identity_init(
    node: &NodeProto,

    formats: &mut HashMap<String, TensorFormat>
) -> Operator {
    let input_name = node.input[0].as_str();
    let name = node.name().to_string();
    formats.insert(node.output[0].to_string(), TensorFormat::NCHW);
    let unary = Unary {
        input: input_name.to_string(),
        output: node.output[0].to_string(),
    };
    Operator::Identity(Base {
        base: unary,
        id: name,
    })
}

pub(crate) fn flatten_init(
    node: &NodeProto,
    formats: &mut HashMap<String, TensorFormat>
) -> Vec<Operator> {
    let input_name = node.input[0].as_str();
    let name = node.name().to_string();
    let mut ret = vec![];
    let new_input_name = try_pc(&mut ret, input_name, &name, formats);
    let flatten = Flatten {
        input: new_input_name.unwrap_or(input_name.to_string()),
        output: node.output[0].to_string(),
        start_dim: parse_int_attribute(node, &mut 0, "start_dim", 1),
    };
    insert_default_format(formats, &flatten.output);
    ret.push(
        Operator::Flatten(Base {
            base: flatten,
            id: name,
        })
    );
    ret
}

#[duplicate::duplicate_item(
    reduce_func    operator_enum;
    [reduce_sum_init]     [ReduceSum];
    [reduce_prod_init]    [ReduceProd];
    [reduce_mean_init]    [ReduceMean];
    [reduce_max_init]     [ReduceMax];
    [reduce_min_init]     [ReduceMin];
    [reduce_l1_init]      [ReduceL1];
    [reduce_l2_init]      [ReduceL2];
    [reduce_log_sum_init] [ReduceLogSum];
    [reduce_log_sum_exp_init] [ReduceLogSumExp];
    [reduce_sum_square_init] [ReduceSumSquare];
)]
pub(crate) fn reduce_func(
    node: &NodeProto,
    formats: &mut HashMap<String, TensorFormat>
) -> Vec<Operator> {
    let input_name = node.input[0].as_str();
    let axes_name = if let Some(axes_name) = node.input.get(1) {
        Some(axes_name.clone())
    } else {
        None
    };
    let name = node.name().to_string();
    let mut ret = vec![];
    let new_input_name = try_pc(&mut ret, input_name, &name, formats);
    let mut arg_idx = 0;
    let keepdims = parse_int_attribute(node, &mut arg_idx, "keepdims", 1) == 1;
    let reduce_all_if_empty_axes =
        parse_int_attribute(node, &mut arg_idx, "noop_with_empty_axes", 0) == 0;
    let reduce = Reduce {
        input: new_input_name.unwrap_or(input_name.to_string()),
        output: node.output[0].to_string(),
        axes: axes_name,
        keepdims,
        reduce_all_if_empty_axes,
    };
    insert_default_format(formats, &reduce.output);
    ret.push(
        Operator::operator_enum(Base {
            base: reduce,
            id: name,
        })
    );
    ret
}

pub(crate) fn reshape_init(
    node: &NodeProto,
    formats: &mut HashMap<String, TensorFormat>
) -> Vec<Operator> {
    let input_name = node.input[0].as_str();
    let shape_name = node.input[1].as_str();
    let name = node.name().to_string();
    let mut ret = vec![];
    let new_input_name = try_pc(&mut ret, input_name, &name, formats);
    let new_shape_name = try_pc(&mut ret, shape_name, &name, formats);

    let allow_zero = parse_int_attribute(node, &mut 0, "allow_zero", 0) == 0;

    let reshape = Reshape {
        input: new_input_name.unwrap_or(input_name.to_string()),
        output: node.output[0].to_string(),
        shape: new_shape_name.unwrap_or(shape_name.to_string()),
        allow_zero,
    };
    insert_default_format(formats, &reshape.output);
    ret.push(
        Operator::Reshape(Base {
            base: reshape,
            id: name,
        })
    );
    ret
}

#[duplicate::duplicate_item(
    func_name               operator_enum;
    [softmax_init]          [Softmax];
    [log_softmax_init]      [LogSoftmax];
)]
pub(crate) fn func_name(
    node: &NodeProto,
    formats: &mut HashMap<String, TensorFormat>
) -> Vec<Operator> {
    let input_name = node.input[0].as_str();
    let name = node.name().to_string();
    let mut ret = vec![];
    let new_input_name = try_pc(&mut ret, input_name, &name, formats);
    let softmax = Softmax {
        input: new_input_name.unwrap_or(input_name.to_string()),
        output: node.output[0].to_string(),
        axis: parse_int_attribute(node, &mut 0, "axis", -1),
    };
    insert_default_format(formats, &softmax.output);
    ret.push(
        Operator::operator_enum(Base {
            base: softmax,
            id: name,
        })
    );
    ret
}

pub(crate) fn layernorm_init(
    node: &NodeProto,
    formats: &mut HashMap<String, TensorFormat>
) -> Vec<Operator> {
    let x = node.input[0].as_str();
    let scale = node.input[1].as_str();
    let bias = if let Some(bias) = node.input.get(2) {
        Some(bias.as_str().to_string())
    } else {
        None
    };
    let name = node.name().to_string();
    let mut ret = vec![];
    let new_x = try_pc(&mut ret, x, &name, formats);
    let new_scale = try_pc(&mut ret, scale, &name, formats);
    let new_bias = if let Some(bias) = &bias {
        Some(try_pc(&mut ret, bias.as_str(), &name, formats).unwrap_or(bias.to_string()))
    } else {
        None
    };
    let mut arg_idx = 0;
    let axis = parse_int_attribute(node, &mut arg_idx, "axis", -1);
    let epsilon = parse_float_attribute(node, &mut arg_idx, "epsilon", 1e-5) as f64;
    let stash_type = parse_int_attribute(node, &mut arg_idx, "stash_type", 1);
    let layernorm = LayerNormalization {
        input: new_x.unwrap_or(x.to_string()),
        output: node.output[0].to_string(),
        scale: new_scale.unwrap_or(scale.to_string()),
        bias: if let Some(bias) = &new_bias {
            Some(bias.clone())
        } else {
            bias
        },
        epsilon,
        axis,
        stash_type,
    };
    insert_default_format(formats, &layernorm.output);
    ret.push(
        Operator::LayerNormalization(Base {
            base: layernorm,
            id: name,
        })
    );
    ret
}

pub(crate) fn bn_init(
    node: &NodeProto,
    formats: &mut HashMap<String, TensorFormat>
) -> Vec<Operator> {
    let x = node.input[0].as_str();
    let scale = node.input[1].as_str();
    let bias = node.input[2].as_str();
    let input_mean = node.input[3].as_str();
    let input_var = node.input[4].as_str();
    let name = node.name().to_string();
    let mut ret = vec![];
    let new_x = try_pc(&mut ret, x, &name, formats);
    let new_scale = try_pc(&mut ret, scale, &name, formats);
    let new_bias = try_pc(&mut ret, bias, &name, formats);
    let new_input_mean = try_pc(&mut ret, input_mean, &name, formats);
    let new_input_var = try_pc(&mut ret, input_var, &name, formats);
    let mut arg_idx = 0;
    let epsilon = parse_float_attribute(node, &mut arg_idx, "epsilon", 1e-5) as f64;
    let momentum = parse_float_attribute(node, &mut arg_idx, "momentum", 0.9) as f64;
    let spatial = parse_int_attribute(node, &mut arg_idx, "spatial", 0) == 0;

    let bn = BatchNormalization {
        input: new_x.unwrap_or(x.to_string()),
        y: node.output[0].to_string(),
        running_mean: node.output.get(1).map(|s| s.as_str().to_string()),
        running_var: node.output.get(2).map(|s| s.as_str().to_string()),
        scale: new_scale.unwrap_or(scale.to_string()),
        bias: new_bias.unwrap_or(bias.to_string()),
        input_mean: new_input_mean.unwrap_or(input_mean.to_string()),
        input_variance: new_input_var.unwrap_or(input_var.to_string()),
        epsilon,
        momentum,
        spatial,
    };
    insert_default_format(formats, &bn.y);
    if let Some(running_mean) = &bn.running_mean {
        insert_default_format(formats, running_mean);
    }
    if let Some(running_var) = &bn.running_var {
        insert_default_format(formats, running_var);
    }
    ret.push(
        Operator::BatchNormalization(Base {
            base: bn,
            id: name,
        })
    );
    ret
}

pub(crate) fn expand_init(
    node: &NodeProto,
    formats: &mut HashMap<String, TensorFormat>
) -> Vec<Operator> {
    let input_name = node.input[0].as_str();
    let shape_name = node.input[1].as_str();
    let name = node.name().to_string();
    let mut ret = vec![];
    let new_input_name = try_pc(&mut ret, input_name, &name, formats);
    let new_shape_name = try_pc(&mut ret, shape_name, &name, formats);
    let expand = Expand {
        input: new_input_name.unwrap_or(input_name.to_string()),
        output: node.output[0].to_string(),
        dims: new_shape_name.unwrap_or(shape_name.to_string()),
    };
    insert_default_format(formats, &expand.output);
    ret.push(
        Operator::Expand(Base {
            base: expand,
            id: name,
        })
    );
    ret
}
