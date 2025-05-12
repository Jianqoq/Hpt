use std::collections::HashMap;

use hpt_common::error::base::TensorError;

use crate::{ onnx::NodeProto, Tensor };

use super::operators::{ Operator, TensorFormat };

use crate::utils::onnx::init::*;

pub(super) fn run_init(
    nodes: &[NodeProto],
    formats: &mut HashMap<String, TensorFormat>,
    operators: &mut Vec<Operator>,
    initializer_map: &mut HashMap<String, Tensor>
) -> Result<(), TensorError> {
    for node in nodes.iter() {
        let ops = match node.op_type() {
            "Conv" => conv_init(node, formats),
            "MaxPool" | "GlobalAveragePool" | "GlobalMaxPool" | "AveragePool" =>
                pooling_init(node, formats),
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
            | "Softsign"
            | "Shape" => vec![unary_init(node, formats)],
            | "Add"
            | "Sub"
            | "Mul"
            | "Div"
            | "Mod"
            | "Pow"
            | "GreaterOrEqual"
            | "LessOrEqual"
            | "Equal"
            | "Greater"
            | "Less"
            | "BitwiseOr"
            | "BitwiseAnd"
            | "BitwiseXor" => binary_init(node, formats),
            "Gather" => gather_init(node, formats),
            "Constant" => vec![constant_init(node, initializer_map, formats)?],
            "Gemm" => gemm_init(node, formats),
            "MatMul" => matmul_init(node, formats),
            "Unsqueeze" => unsqueeze_init(node, initializer_map, formats),
            "Squeeze" => squeeze_init(node, initializer_map, formats),
            "Concat" => concat_init(node, formats),
            "ConstantOfShape" => const_of_shape_init(node, formats)?,
            "Transpose" => transpose_init(node, formats),
            "Slice" => slice_init(node, formats),
            "LSTM" => lstm_init(node, formats),
            "Identity" => vec![identity_init(node, formats)],
            "Flatten" => flatten_init(node, formats),
            "Selu" => vec![selu_init(node, formats)],
            "ReduceSum" => reduce_sum_init(node, formats),
            "ReduceProd" => reduce_prod_init(node, formats),
            "ReduceMean" => reduce_mean_init(node, formats),
            "ReduceMax" => reduce_max_init(node, formats),
            "ReduceMin" => reduce_min_init(node, formats),
            "ReduceL1" => reduce_l1_init(node, formats),
            "ReduceL2" => reduce_l2_init(node, formats),
            "ReduceLogSum" => reduce_log_sum_init(node, formats),
            "ReduceLogSumExp" => reduce_log_sum_exp_init(node, formats),
            "ReduceSumSquare" => reduce_sum_square_init(node, formats),
            "Reshape" => reshape_init(node, formats),
            "Elu" => vec![elu_init(node, formats)],
            "Softmax" => softmax_init(node, formats),
            "LogSoftmax" => log_softmax_init(node, formats),
            "LayerNormalization" => layernorm_init(node, formats),
            "BatchNormalization" => bn_init(node, formats),
            "Expand" => expand_init(node, formats),
            "Cast" => cast_init(node, formats),
            _ => unimplemented!("unsupported op when initializing: {:?}", node.op_type()),
        };
        for op in ops {
            operators.push(op);
        }
    }
    Ok(())
}
