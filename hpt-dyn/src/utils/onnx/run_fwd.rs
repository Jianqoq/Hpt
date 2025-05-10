use std::collections::HashMap;

use hpt_common::error::base::TensorError;

use crate::Tensor;

use super::operators::Operator;

use crate::utils::onnx::fwd::*;

pub(super) fn run_fwd<'a>(
    operators: &'a [Operator],
    tensors: &mut HashMap<&'a str, Tensor>,
    node_degree: &mut HashMap<&'a str, u32>
) -> Result<(), TensorError> {
    let mut total_conv = std::time::Duration::from_secs(0);
    let mut total_conv_fused = std::time::Duration::from_secs(0);
    for operator in operators.iter() {
        match operator {
            Operator::Constant(_) => {}
            Operator::Abs(unary) => abs_fwd(&unary.base, tensors, node_degree)?,
            Operator::Acos(unary) => acos_fwd(&unary.base, tensors, node_degree)?,
            Operator::Acosh(unary) => acosh_fwd(&unary.base, tensors, node_degree)?,
            Operator::Add(base) => add_fwd(&base.base, tensors, node_degree)?,
            Operator::And(binary) => todo!(),
            Operator::ArgMax(arg_reduce) => todo!(),
            Operator::ArgMin(arg_reduce) => todo!(),
            Operator::Asin(unary) => asin_fwd(&unary.base, tensors, node_degree)?,
            Operator::Asinh(unary) => asinh_fwd(&unary.base, tensors, node_degree)?,
            Operator::Atan(unary) => atan_fwd(&unary.base, tensors, node_degree)?,
            Operator::Atanh(unary) => atanh_fwd(&unary.base, tensors, node_degree)?,
            Operator::AveragePool(pooling) => avgpool_fwd(&pooling.base, tensors)?,
            Operator::BatchNormalization(batch_normalization) => todo!(),
            Operator::BitShift(binary) => todo!(),
            Operator::BitwiseAnd(binary) => todo!(),
            Operator::BitwiseNot(unary) => todo!(),
            Operator::BitwiseOr(binary) => todo!(),
            Operator::BitwiseXor(binary) => todo!(),
            Operator::Cast(cast) => todo!(),
            Operator::Ceil(unary) => todo!(),
            Operator::Concat(concat) => concat_fwd(&concat.base, tensors)?,
            Operator::Conv2d(conv2d) => {
                let start = std::time::Instant::now();
                conv_fwd(&conv2d.base, tensors, node_degree)?;
                total_conv += start.elapsed();
            },
            Operator::Conv2dInteger(conv2d) => conv_fwd(&conv2d.base, tensors, node_degree)?,
            Operator::Cos(unary) => cos_fwd(&unary.base, tensors, node_degree)?,
            Operator::Cosh(unary) => cosh_fwd(&unary.base, tensors, node_degree)?,
            Operator::ConstantOfShape(constant_of_shape) => constant_of_shape_fwd(&constant_of_shape.base, tensors)?,
            Operator::Div(binary) => todo!(),
            Operator::Dropout(dropout) => todo!(),
            Operator::Equal(binary) => todo!(),
            Operator::Erf(unary) => todo!(),
            Operator::Exp(unary) => exp_fwd(&unary.base, tensors, node_degree)?,
            Operator::Expand(expand) => todo!(),
            Operator::EyeLike(eye_like) => todo!(),
            Operator::Flatten(flatten) => flatten_fwd(&flatten.base, tensors, node_degree)?,
            Operator::Floor(unary) => floor_fwd(&unary.base, tensors, node_degree)?,
            Operator::Gather(gather) => gather_fwd(&gather.base, tensors)?,
            Operator::Gemm(gemm) => gemm_fwd(&gemm.base, tensors, node_degree)?,
            Operator::GlobalAveragePool(pooling) => global_avgpool_fwd(&pooling.base, tensors)?,
            Operator::GlobalMaxPool(pooling) => global_maxpool_fwd(&pooling.base, tensors)?,
            Operator::Greater(binary) => todo!(),
            Operator::Identity(identity) => identity_fwd(&identity.base, tensors)?,
            Operator::If(_) => todo!(),
            Operator::IsInf(unary) => todo!(),
            Operator::IsNaN(unary) => todo!(),
            Operator::Less(binary) => todo!(),
            Operator::Log(unary) => ln_fwd(&unary.base, tensors, node_degree)?,
            Operator::Loop(_) => todo!(),
            Operator::Lstm(lstm) => lstm_fwd(&lstm.base, tensors)?,
            Operator::MatMul(matmul) => matmul_fwd(&matmul.base, tensors, node_degree)?,
            Operator::MatMulInteger(matmul) => matmul_fwd(&matmul.base, tensors, node_degree)?,
            Operator::Max(binary) => todo!(),
            Operator::MaxPool(pooling) => maxpool_fwd(&pooling.base, tensors, node_degree)?,
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
            Operator::Round(unary) => round_fwd(&unary.base, tensors, node_degree)?,
            Operator::Sigmoid(unary) => sigmoid_fwd(&unary.base, tensors, node_degree)?,
            Operator::Sign(unary) => sign_fwd(&unary.base, tensors, node_degree)?,
            Operator::Sin(unary) => sin_fwd(&unary.base, tensors, node_degree)?,
            Operator::Sinh(unary) => sinh_fwd(&unary.base, tensors, node_degree)?,
            Operator::Slice(slice) => slice_fwd(&slice.base, tensors)?,
            Operator::Split(split) => todo!(),
            Operator::Sqrt(unary) => sqrt_fwd(&unary.base, tensors, node_degree)?,
            Operator::Squeeze(squeeze) => squeeze_fwd(&squeeze.base, tensors)?,
            Operator::Sub(binary) => todo!(),
            Operator::Sum(reduce) => todo!(),
            Operator::Shape(unary) => shape_fwd(&unary.base, tensors, node_degree)?,
            Operator::Tan(unary) => tan_fwd(&unary.base, tensors, node_degree)?,
            Operator::Tanh(unary) => tanh_fwd(&unary.base, tensors, node_degree)?,
            Operator::Transpose(permute) => transpose_fwd(&permute.base, tensors, node_degree)?,
            Operator::Trilu(unary) => todo!(),
            Operator::Unsqueeze(unsqueeze) => unsqueeze_fwd(&unsqueeze.base, tensors)?,
            Operator::Where(where_op) => todo!(),
            Operator::Xor(binary) => todo!(),
            Operator::Bernoulli(bernoulli) => todo!(),
            Operator::BlackmanWindow(unary) => todo!(),
            Operator::CastLike(cast) => todo!(),
            Operator::Celu(unary) => todo!(),
            Operator::Clip(clip) => todo!(),
            Operator::Elu(unary) => todo!(),
            Operator::Gelu(unary) => gelu_fwd(&unary.base, tensors, node_degree)?,
            Operator::GreaterOrEqual(binary) => todo!(),
            Operator::HammingWindow(unary) => todo!(),
            Operator::HannWindow(unary) => todo!(),
            Operator::HardSigmoid(unary) => todo!(),
            Operator::HardSwish(unary) => todo!(),
            Operator::LayerNormalization(layer_normalization) => todo!(),
            Operator::LeakyRelu(unary) => todo!(),
            Operator::LessOrEqual(binary) => todo!(),
            Operator::LogSoftmax(reduce) => todo!(),
            Operator::Mish(unary) => mish_fwd(&unary.base, tensors, node_degree)?,
            Operator::ReduceL1(reduce) => todo!(),
            Operator::ReduceL2(reduce) => todo!(),
            Operator::ReduceLogSum(reduce) => todo!(),
            Operator::ReduceLogSumExp(reduce) => todo!(),
            Operator::ReduceSumSquare(reduce) => todo!(),
            Operator::Relu(unary) => relu_fwd(&unary.base, tensors, node_degree)?,
            Operator::Selu(unary) => selu_fwd(&unary.base, tensors, node_degree)?,
            Operator::Shrink(unary) => todo!(),
            Operator::Softmax(reduce) => todo!(),
            Operator::SoftmaxCrossEntropyLoss(reduce) => todo!(),
            Operator::Softplus(unary) => softplus_fwd(&unary.base, tensors, node_degree)?,
            Operator::Softsign(unary) => softsign_fwd(&unary.base, tensors, node_degree)?,
            Operator::Contiguous(unary) => todo!(),
            Operator::InvPermute(permute) => todo!(),
            Operator::PermuteContiguous(permute) => permute_contiguous_fwd(&permute.base, tensors)?,
            Operator::Conv2dFused(base) => {
                let start = std::time::Instant::now();
                conv_fused_fwd(&base.base, tensors, node_degree)?;
                total_conv_fused += start.elapsed();
            },
        }
    }
    println!("total_conv time: {:?}", total_conv + total_conv_fused);
    Ok(())
}
