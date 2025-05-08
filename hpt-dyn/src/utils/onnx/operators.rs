// #![allow(unused)]

use std::collections::HashMap;

use hpt_types::dtype::DType;

#[derive(Debug, Clone)]
pub(crate) enum ConvActivation {
    Relu,
    LeakyRelu,
    Gelu,
    Sigmoid,
    Tanh,
}

#[derive(Debug, Clone)]
pub(crate) struct Unary {
    pub(crate) input: String,
    pub(crate) output: String,
}

#[derive(Debug, Clone)]
pub(crate) struct Elu {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) alpha: f64,
    pub(crate) gamma: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct Binary {
    pub(crate) input1: String,
    pub(crate) input2: String,
    pub(crate) output: String,
}

#[derive(Debug, Clone)]
pub(crate) struct ArgReduce {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) axis: i64,
    pub(crate) keepdims: bool,
    pub(crate) select_last_index: bool,
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AutoPad {
    NOTSET,
    SAME_UPPER,
    SAME_LOWER,
    VALID,
}

#[derive(Debug, Clone)]
pub(crate) struct Pooling {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) auto_pad: AutoPad,
    pub(crate) kernel_shape: Vec<i64>,
    pub(crate) pads: Vec<i64>,
    pub(crate) strides: Vec<i64>,
    pub(crate) dilations: Vec<i64>,
    pub(crate) ceil_mode: bool,
    pub(crate) storage_order: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct BatchNormalization {
    pub(crate) input: String,
    pub(crate) y: String,
    pub(crate) running_mean: Option<String>,
    pub(crate) running_var: Option<String>,
    pub(crate) scale: String,
    pub(crate) bias: String,
    pub(crate) input_mean: String,
    pub(crate) input_variance: String,
    pub(crate) epsilon: f64,
    pub(crate) momentum: f64,
    pub(crate) spatial: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct Concat {
    pub(crate) inputs: Vec<String>,
    pub(crate) output: String,
    pub(crate) axis: i64,
}

#[derive(Debug, Clone)]
pub(crate) struct Conv2d {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) kernel: String,
    pub(crate) bias: Option<String>,
    pub(crate) pads: [(i64, i64); 2],
    pub(crate) strides: [i64; 2],
    pub(crate) dilations: [i64; 2],
    pub(crate) group: i64,
}

#[derive(Debug, Clone)]
pub(crate) struct Conv2dFused {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) kernel: String,
    pub(crate) bias: Option<String>,
    pub(crate) pads: [(i64, i64); 2],
    pub(crate) strides: [i64; 2],
    pub(crate) dilations: [i64; 2],
    pub(crate) group: i64,
    pub(crate) activation: ConvActivation,
}

#[derive(Debug, Clone)]
pub(crate) struct Dropout {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) ratio: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct Expand {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) dims: String,
}

#[derive(Debug, Clone)]
pub(crate) struct EyeLike {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) shape: Vec<i64>,
}

#[derive(Debug, Clone)]
pub(crate) struct Flatten {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) start_dim: i64,
}

#[derive(Debug, Clone)]
pub(crate) struct Gemm {
    pub(crate) a: String,
    pub(crate) b: String,
    pub(crate) output: String,
    pub(crate) alpha: f64,
    pub(crate) beta: f64,
    pub(crate) trans_a: bool,
    pub(crate) trans_b: bool,
    pub(crate) bias: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct Matmul {
    pub(crate) a: String,
    pub(crate) b: String,
    pub(crate) output: String,
}

#[derive(Debug, Clone)]
pub(crate) struct Reduce {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) axes: Option<String>,
    pub(crate) keepdims: bool,
    pub(crate) reduce_all_if_empty_axes: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct Softmax {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) axis: i64,
}

#[derive(Debug, Clone)]
pub(crate) struct OneHot {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) axis: i64,
}

#[derive(Debug, Clone)]
pub(crate) struct Pad {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) pads: Vec<i64>,
    pub(crate) value: f64,
    pub(crate) axes: Vec<i64>,
}

#[derive(Debug, Clone)]
pub(crate) struct RandomNormal {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) mean: f64,
    pub(crate) scale: f64,
    pub(crate) seed: i64,
    pub(crate) shape: Vec<i64>,
    pub(crate) dtype: DType,
}

#[derive(Debug, Clone)]
pub(crate) struct RandomUniform {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) low: f64,
    pub(crate) high: f64,
    pub(crate) seed: i64,
    pub(crate) shape: Vec<i64>,
    pub(crate) dtype: DType,
}

#[derive(Debug, Clone)]
pub(crate) struct Reshape {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) shape: String,
    pub(crate) allow_zero: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct Slice {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) starts: String,
    pub(crate) ends: String,
    pub(crate) steps: Option<String>,
    pub(crate) axes: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct Split {
    pub(crate) input: String,
    pub(crate) outputs: Vec<String>,
    pub(crate) axis: i64,
}

#[derive(Debug, Clone)]
pub(crate) struct Squeeze {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) axes: Vec<i64>,
}

#[derive(Debug, Clone)]
pub(crate) struct Permute {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) perm: Vec<i64>,
}

#[derive(Debug, Clone)]
pub(crate) struct Where {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) condition: String,
}

#[derive(Debug, Clone)]
pub(crate) struct Bernoulli {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) dtype: DType,
    pub(crate) seed: i64,
}

#[derive(Debug, Clone)]
pub(crate) struct Cast {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) to: DType,
}

#[derive(Debug, Clone)]
pub(crate) struct Clip {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) min: f64,
    pub(crate) max: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct LayerNormalization {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) scale: String,
    pub(crate) bias: Option<String>,
    pub(crate) epsilon: f64,
    pub(crate) stash_type: i64,
    pub(crate) axis: i64,
}

#[derive(Debug, Clone)]
pub(crate) struct Gather {
    pub(crate) input: String,
    pub(crate) indices: String,
    pub(crate) output: String,
    pub(crate) axis: i64,
}

#[derive(Debug, Clone)]
pub(crate) struct Constant {
    pub(crate) output: String,
    pub(crate) data: Vec<u8>,
    pub(crate) dtype: DType,
}

#[derive(Debug, Clone)]
pub(crate) struct ConstantOfShape {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) value: f64,
    pub(crate) dtype: DType,
}

#[derive(Debug, Clone)]
pub(crate) struct Lstm {
    pub(crate) x: String,
    pub(crate) w: String,
    pub(crate) r: String,
    pub(crate) b: Option<String>,
    pub(crate) sequence_lens: Option<String>,
    pub(crate) initial_h: Option<String>,
    pub(crate) initial_c: Option<String>,
    pub(crate) p: Option<String>,
    pub(crate) activation_alpha: Vec<f32>,
    pub(crate) activation_beta: Vec<f32>,
    pub(crate) activations: Vec<String>,
    pub(crate) clip: Option<f32>,
    pub(crate) direction: String,
    pub(crate) hidden_size: i64,
    pub(crate) input_forget: bool,
    pub(crate) layout: i64,
    pub(crate) y: Option<String>,
    pub(crate) y_h: Option<String>,
    pub(crate) y_c: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TensorFormat {
    NCHW,
    NHWC,
    HWCO,
    OCHW,
}

#[derive(Debug, Clone)]
pub(crate) struct Base<T> {
    pub(crate) base: T,
    pub(crate) id: String,
}

#[derive(Clone)]
pub(crate) enum Operator {
    Constant(Base<String>),
    Contiguous(Base<Unary>),
    Abs(Base<Unary>),
    Acos(Base<Unary>),
    Acosh(Base<Unary>),
    Add(Base<Binary>),
    And(Base<Binary>),
    ArgMax(Base<ArgReduce>),
    ArgMin(Base<ArgReduce>),
    Asin(Base<Unary>),
    Asinh(Base<Unary>),
    Atan(Base<Unary>),
    Atanh(Base<Unary>),
    AveragePool(Base<Pooling>),
    BatchNormalization(Base<BatchNormalization>),
    BitShift(Base<Binary>),
    BitwiseAnd(Base<Binary>),
    BitwiseNot(Base<Unary>),
    BitwiseOr(Base<Binary>),
    BitwiseXor(Base<Binary>),
    Cast(Base<Cast>),
    Ceil(Base<Unary>),
    Concat(Base<Concat>),
    Conv2d(Base<Conv2d>),
    Conv2dInteger(Base<Conv2d>),
    Cos(Base<Unary>),
    Cosh(Base<Unary>),
    ConstantOfShape(Base<ConstantOfShape>),
    Div(Base<Binary>),
    Dropout(Base<Dropout>),
    Equal(Base<Binary>),
    Erf(Base<Unary>),
    Exp(Base<Unary>),
    Expand(Base<Expand>),
    EyeLike(Base<EyeLike>),
    Flatten(Base<Flatten>),
    Floor(Base<Unary>),
    Gather(Base<Gather>),
    Gemm(Base<Gemm>),
    GlobalAveragePool(Base<Pooling>),
    GlobalMaxPool(Base<Pooling>),
    Greater(Base<Binary>),
    Identity(Base<Unary>),
    If(Base<String>),
    IsInf(Base<Unary>),
    IsNaN(Base<Unary>),
    Less(Base<Binary>),
    Log(Base<Unary>),
    Loop(Base<String>),
    Lstm(Base<Lstm>),
    MatMul(Base<Matmul>),
    MatMulInteger(Base<Matmul>),
    Max(Base<Binary>),
    MaxPool(Base<Pooling>),
    Mean(Base<Reduce>),
    Min(Base<Binary>),
    Mod(Base<Binary>),
    Mul(Base<Binary>),
    Neg(Base<Unary>),
    Not(Base<Unary>),
    OneHot(Base<OneHot>),
    Or(Base<Binary>),
    Pad(Base<Pad>),
    Pow(Base<Binary>),
    RandomNormal(Base<RandomNormal>),
    RandomNormalLike(Base<RandomNormal>),
    RandomUniform(Base<RandomUniform>),
    RandomUniformLike(Base<RandomUniform>),
    Reciprocal(Base<Unary>),
    ReduceMax(Base<Reduce>),
    ReduceMean(Base<Reduce>),
    ReduceMin(Base<Reduce>),
    ReduceProd(Base<Reduce>),
    ReduceSum(Base<Reduce>),
    Reshape(Base<Reshape>),
    Round(Base<Unary>),
    Sigmoid(Base<Unary>),
    Sign(Base<Unary>),
    Sin(Base<Unary>),
    Sinh(Base<Unary>),
    Slice(Base<Slice>),
    Split(Base<Split>),
    Sqrt(Base<Unary>),
    Squeeze(Base<Squeeze>),
    Sub(Base<Binary>),
    Sum(Base<Reduce>),
    Shape(Base<Unary>),
    Tan(Base<Unary>),
    Tanh(Base<Unary>),
    Transpose(Base<Permute>),
    InvPermute(Base<Permute>),
    PermuteContiguous(Base<Permute>),
    Trilu(Base<Unary>),
    Unsqueeze(Base<Squeeze>),
    Where(Base<Where>),
    Xor(Base<Binary>),
    Bernoulli(Base<Bernoulli>),
    BlackmanWindow(Base<Unary>),
    CastLike(Base<Cast>),
    Celu(Base<Unary>),
    Clip(Base<Clip>),
    Elu(Base<Elu>),
    Gelu(Base<Unary>),
    GreaterOrEqual(Base<Binary>),
    HammingWindow(Base<Unary>),
    HannWindow(Base<Unary>),
    HardSigmoid(Base<Unary>),
    HardSwish(Base<Unary>),
    LayerNormalization(Base<LayerNormalization>),
    LeakyRelu(Base<Unary>),
    LessOrEqual(Base<Binary>),
    LogSoftmax(Base<Softmax>),
    Mish(Base<Unary>),
    ReduceL1(Base<Reduce>),
    ReduceL2(Base<Reduce>),
    ReduceLogSum(Base<Reduce>),
    ReduceLogSumExp(Base<Reduce>),
    ReduceSumSquare(Base<Reduce>),
    Relu(Base<Unary>),
    Selu(Base<Elu>),
    Shrink(Base<Unary>),
    Softmax(Base<Softmax>),
    SoftmaxCrossEntropyLoss(Base<Reduce>),
    Softplus(Base<Unary>),
    Softsign(Base<Unary>),
    Conv2dFused(Base<Conv2dFused>),
}

impl Operator {
    pub(super) fn id(&self) -> &str {
        match self {
            Operator::Constant(base) => &base.id,
            Operator::Contiguous(base) => &base.id,
            Operator::Abs(base) => &base.id,
            Operator::Acos(base) => &base.id,
            Operator::Acosh(base) => &base.id,
            Operator::Add(base) => &base.id,
            Operator::And(base) => &base.id,
            Operator::ArgMax(base) => &base.id,
            Operator::ArgMin(base) => &base.id,
            Operator::Asin(base) => &base.id,
            Operator::Asinh(base) => &base.id,
            Operator::Atan(base) => &base.id,
            Operator::Atanh(base) => &base.id,
            Operator::AveragePool(base) => &base.id,
            Operator::BatchNormalization(base) => &base.id,
            Operator::BitShift(base) => &base.id,
            Operator::BitwiseAnd(base) => &base.id,
            Operator::BitwiseNot(base) => &base.id,
            Operator::BitwiseOr(base) => &base.id,
            Operator::BitwiseXor(base) => &base.id,
            Operator::Cast(base) => &base.id,
            Operator::Ceil(base) => &base.id,
            Operator::Concat(base) => &base.id,
            Operator::Conv2d(base) => &base.id,
            Operator::Conv2dInteger(base) => &base.id,
            Operator::Cos(base) => &base.id,
            Operator::Cosh(base) => &base.id,
            Operator::ConstantOfShape(base) => &base.id,
            Operator::Div(base) => &base.id,
            Operator::Dropout(base) => &base.id,
            Operator::Equal(base) => &base.id,
            Operator::Erf(base) => &base.id,
            Operator::Exp(base) => &base.id,
            Operator::Expand(base) => &base.id,
            Operator::EyeLike(base) => &base.id,
            Operator::Flatten(base) => &base.id,
            Operator::Floor(base) => &base.id,
            Operator::Gather(base) => &base.id,
            Operator::Gemm(base) => &base.id,
            Operator::GlobalAveragePool(base) => &base.id,
            Operator::GlobalMaxPool(base) => &base.id,
            Operator::Greater(base) => &base.id,
            Operator::Identity(base) => &base.id,
            Operator::If(base) => &base.id,
            Operator::IsInf(base) => &base.id,
            Operator::IsNaN(base) => &base.id,
            Operator::Less(base) => &base.id,
            Operator::Log(base) => &base.id,
            Operator::Loop(base) => &base.id,
            Operator::Lstm(base) => &base.id,
            Operator::MatMul(base) => &base.id,
            Operator::MatMulInteger(base) => &base.id,
            Operator::Max(base) => &base.id,
            Operator::MaxPool(base) => &base.id,
            Operator::Mean(base) => &base.id,
            Operator::Min(base) => &base.id,
            Operator::Mod(base) => &base.id,
            Operator::Mul(base) => &base.id,
            Operator::Neg(base) => &base.id,
            Operator::Not(base) => &base.id,
            Operator::OneHot(base) => &base.id,
            Operator::Or(base) => &base.id,
            Operator::Pad(base) => &base.id,
            Operator::Pow(base) => &base.id,
            Operator::RandomNormal(base) => &base.id,
            Operator::RandomNormalLike(base) => &base.id,
            Operator::RandomUniform(base) => &base.id,
            Operator::RandomUniformLike(base) => &base.id,
            Operator::Reciprocal(base) => &base.id,
            Operator::ReduceMax(base) => &base.id,
            Operator::ReduceMean(base) => &base.id,
            Operator::ReduceMin(base) => &base.id,
            Operator::ReduceProd(base) => &base.id,
            Operator::ReduceSum(base) => &base.id,
            Operator::Reshape(base) => &base.id,
            Operator::Round(base) => &base.id,
            Operator::Sigmoid(base) => &base.id,
            Operator::Sign(base) => &base.id,
            Operator::Sin(base) => &base.id,
            Operator::Sinh(base) => &base.id,
            Operator::Slice(base) => &base.id,
            Operator::Split(base) => &base.id,
            Operator::Sqrt(base) => &base.id,
            Operator::Squeeze(base) => &base.id,
            Operator::Sub(base) => &base.id,
            Operator::Sum(base) => &base.id,
            Operator::Shape(base) => &base.id,
            Operator::Tan(base) => &base.id,
            Operator::Tanh(base) => &base.id,
            Operator::Transpose(base) => &base.id,
            Operator::InvPermute(base) => &base.id,
            Operator::PermuteContiguous(base) => &base.id,
            Operator::Trilu(base) => &base.id,
            Operator::Unsqueeze(base) => &base.id,
            Operator::Where(base) => &base.id,
            Operator::Xor(base) => &base.id,
            Operator::Bernoulli(base) => &base.id,
            Operator::BlackmanWindow(base) => &base.id,
            Operator::CastLike(base) => &base.id,
            Operator::Celu(base) => &base.id,
            Operator::Clip(base) => &base.id,
            Operator::Elu(base) => &base.id,
            Operator::Gelu(base) => &base.id,
            Operator::GreaterOrEqual(base) => &base.id,
            Operator::HammingWindow(base) => &base.id,
            Operator::HannWindow(base) => &base.id,
            Operator::HardSigmoid(base) => &base.id,
            Operator::HardSwish(base) => &base.id,
            Operator::LayerNormalization(base) => &base.id,
            Operator::LeakyRelu(base) => &base.id,
            Operator::LessOrEqual(base) => &base.id,
            Operator::LogSoftmax(base) => &base.id,
            Operator::Mish(base) => &base.id,
            Operator::ReduceL1(base) => &base.id,
            Operator::ReduceL2(base) => &base.id,
            Operator::ReduceLogSum(base) => &base.id,
            Operator::ReduceLogSumExp(base) => &base.id,
            Operator::ReduceSumSquare(base) => &base.id,
            Operator::Relu(base) => &base.id,
            Operator::Selu(base) => &base.id,
            Operator::Shrink(base) => &base.id,
            Operator::Softmax(base) => &base.id,
            Operator::SoftmaxCrossEntropyLoss(base) => &base.id,
            Operator::Softplus(base) => &base.id,
            Operator::Softsign(base) => &base.id,
            Operator::Conv2dFused(base) => &base.id,
        }
    }

    pub(super) fn inputs(&self) -> Vec<&str> {
        match self {
            Operator::Constant(_) => vec![],
            Operator::Contiguous(base) => vec![base.base.input.as_str()],
            Operator::Abs(base) => vec![base.base.input.as_str()],
            Operator::Acos(base) => vec![base.base.input.as_str()],
            Operator::Acosh(base) => vec![base.base.input.as_str()],
            Operator::Add(base) => vec![base.base.input1.as_str(), base.base.input2.as_str()],
            Operator::And(base) => vec![base.base.input1.as_str(), base.base.input2.as_str()],
            Operator::ArgMax(base) => vec![base.base.input.as_str()],
            Operator::ArgMin(base) => vec![base.base.input.as_str()],
            Operator::Asin(base) => vec![base.base.input.as_str()],
            Operator::Asinh(base) => vec![base.base.input.as_str()],
            Operator::Atan(base) => vec![base.base.input.as_str()],
            Operator::Atanh(base) => vec![base.base.input.as_str()],
            Operator::AveragePool(base) => vec![base.base.input.as_str()],
            Operator::BatchNormalization(base) =>
                vec![
                    base.base.input.as_str(),
                    base.base.scale.as_str(),
                    base.base.bias.as_str(),
                    base.base.input_mean.as_str(),
                    base.base.input_variance.as_str()
                ],
            Operator::BitShift(base) => vec![base.base.input1.as_str(), base.base.input2.as_str()],
            Operator::BitwiseAnd(base) => {
                vec![base.base.input1.as_str(), base.base.input2.as_str()]
            }
            Operator::BitwiseNot(base) => vec![base.base.input.as_str()],
            Operator::BitwiseOr(base) => vec![base.base.input1.as_str(), base.base.input2.as_str()],
            Operator::BitwiseXor(base) => {
                vec![base.base.input1.as_str(), base.base.input2.as_str()]
            }
            Operator::Cast(base) => vec![base.base.input.as_str()],
            Operator::Ceil(base) => vec![base.base.input.as_str()],
            Operator::Concat(base) =>
                base.base.inputs
                    .iter()
                    .map(|s| s.as_str())
                    .collect(),
            Operator::Conv2d(base) => {
                if let Some(bias) = base.base.bias.as_ref() {
                    vec![base.base.input.as_str(), base.base.kernel.as_str(), bias.as_str()]
                } else {
                    vec![base.base.input.as_str(), base.base.kernel.as_str()]
                }
            }
            Operator::Conv2dInteger(base) => {
                if let Some(bias) = base.base.bias.as_ref() {
                    vec![base.base.input.as_str(), base.base.kernel.as_str(), bias.as_str()]
                } else {
                    vec![base.base.input.as_str(), base.base.kernel.as_str()]
                }
            }
            Operator::Cos(base) => vec![base.base.input.as_str()],
            Operator::Cosh(base) => vec![base.base.input.as_str()],
            Operator::ConstantOfShape(base) => vec![base.base.input.as_str()],
            Operator::Div(base) => vec![base.base.input1.as_str(), base.base.input2.as_str()],
            Operator::Dropout(base) => vec![base.base.input.as_str()],
            Operator::Equal(base) => vec![base.base.input1.as_str(), base.base.input2.as_str()],
            Operator::Erf(base) => vec![base.base.input.as_str()],
            Operator::Exp(base) => vec![base.base.input.as_str()],
            Operator::Expand(base) => vec![base.base.input.as_str()],
            Operator::EyeLike(base) => vec![base.base.input.as_str()],
            Operator::Flatten(base) => vec![base.base.input.as_str()],
            Operator::Floor(base) => vec![base.base.input.as_str()],
            Operator::Gather(base) => vec![base.base.input.as_str(), base.base.indices.as_str()],
            Operator::Gemm(base) => vec![base.base.a.as_str(), base.base.b.as_str()],
            Operator::GlobalAveragePool(base) => vec![base.base.input.as_str()],
            Operator::GlobalMaxPool(base) => vec![base.base.input.as_str()],
            Operator::Greater(base) => vec![base.base.input1.as_str(), base.base.input2.as_str()],
            Operator::Identity(base) => vec![base.base.input.as_str()],
            Operator::If(base) => vec![base.base.as_str()],
            Operator::IsInf(base) => vec![base.base.input.as_str()],
            Operator::IsNaN(base) => vec![base.base.input.as_str()],
            Operator::Less(base) => vec![base.base.input1.as_str(), base.base.input2.as_str()],
            Operator::Log(base) => vec![base.base.input.as_str()],
            Operator::Loop(base) => vec![base.base.as_str()],
            Operator::Lstm(base) =>
                vec![base.base.x.as_str(), base.base.w.as_str(), base.base.r.as_str()],
            Operator::MatMul(base) => vec![base.base.a.as_str(), base.base.b.as_str()],
            Operator::MatMulInteger(base) => vec![base.base.a.as_str(), base.base.b.as_str()],
            Operator::Max(base) => vec![base.base.input1.as_str(), base.base.input2.as_str()],
            Operator::MaxPool(base) => vec![base.base.input.as_str()],
            Operator::Mean(base) => vec![base.base.input.as_str()],
            Operator::Min(base) => vec![base.base.input1.as_str(), base.base.input2.as_str()],
            Operator::Mod(base) => vec![base.base.input1.as_str(), base.base.input2.as_str()],
            Operator::Mul(base) => vec![base.base.input1.as_str(), base.base.input2.as_str()],
            Operator::Neg(base) => vec![base.base.input.as_str()],
            Operator::Not(base) => vec![base.base.input.as_str()],
            Operator::OneHot(base) => vec![base.base.input.as_str()],
            Operator::Or(base) => vec![base.base.input1.as_str(), base.base.input2.as_str()],
            Operator::Pad(base) => vec![base.base.input.as_str()],
            Operator::Pow(base) => vec![base.base.input1.as_str(), base.base.input2.as_str()],
            Operator::RandomNormal(base) => vec![base.base.input.as_str()],
            Operator::RandomNormalLike(base) => vec![base.base.input.as_str()],
            Operator::RandomUniform(base) => vec![base.base.input.as_str()],
            Operator::RandomUniformLike(base) => vec![base.base.input.as_str()],
            Operator::Reciprocal(base) => vec![base.base.input.as_str()],
            | Operator::ReduceL1(base)
            | Operator::ReduceL2(base)
            | Operator::ReduceLogSum(base)
            | Operator::ReduceLogSumExp(base)
            | Operator::ReduceSumSquare(base)
            | Operator::ReduceMax(base)
            | Operator::ReduceMean(base)
            | Operator::ReduceMin(base)
            | Operator::ReduceProd(base)
            | Operator::ReduceSum(base) => if let Some(axes) = base.base.axes.as_ref() {
                vec![base.base.input.as_str(), axes.as_str()]
            } else {
                vec![base.base.input.as_str()]
            }
            Operator::Reshape(base) => vec![base.base.input.as_str(), base.base.shape.as_str()],
            Operator::Round(base) => vec![base.base.input.as_str()],
            Operator::Sigmoid(base) => vec![base.base.input.as_str()],
            Operator::Sign(base) => vec![base.base.input.as_str()],
            Operator::Sin(base) => vec![base.base.input.as_str()],
            Operator::Sinh(base) => vec![base.base.input.as_str()],
            Operator::Slice(base) => vec![base.base.input.as_str()],
            Operator::Split(base) => vec![base.base.input.as_str()],
            Operator::Sqrt(base) => vec![base.base.input.as_str()],
            Operator::Squeeze(base) => vec![base.base.input.as_str()],
            Operator::Sub(base) => vec![base.base.input1.as_str(), base.base.input2.as_str()],
            Operator::Sum(base) => vec![base.base.input.as_str()],
            Operator::Shape(base) => vec![base.base.input.as_str()],
            Operator::Tan(base) => vec![base.base.input.as_str()],
            Operator::Tanh(base) => vec![base.base.input.as_str()],
            Operator::Transpose(base) => vec![base.base.input.as_str()],
            Operator::InvPermute(base) => vec![base.base.input.as_str()],
            Operator::PermuteContiguous(base) => vec![base.base.input.as_str()],
            Operator::Trilu(base) => vec![base.base.input.as_str()],
            Operator::Unsqueeze(base) => vec![base.base.input.as_str()],
            Operator::Where(base) => vec![base.base.input.as_str(), base.base.condition.as_str()],
            Operator::Xor(base) => vec![base.base.input1.as_str(), base.base.input2.as_str()],
            Operator::Bernoulli(base) => vec![base.base.input.as_str()],
            Operator::BlackmanWindow(base) => vec![base.base.input.as_str()],
            Operator::CastLike(base) => vec![base.base.input.as_str()],
            Operator::Celu(base) => vec![base.base.input.as_str()],
            Operator::Clip(base) => vec![base.base.input.as_str()],
            Operator::Elu(base) => vec![base.base.input.as_str()],
            Operator::Gelu(base) => vec![base.base.input.as_str()],
            Operator::GreaterOrEqual(base) => {
                vec![base.base.input1.as_str(), base.base.input2.as_str()]
            }
            Operator::HammingWindow(base) => vec![base.base.input.as_str()],
            Operator::HannWindow(base) => vec![base.base.input.as_str()],
            Operator::HardSigmoid(base) => vec![base.base.input.as_str()],
            Operator::HardSwish(base) => vec![base.base.input.as_str()],
            Operator::LayerNormalization(base) => vec![base.base.input.as_str()],
            Operator::LeakyRelu(base) => vec![base.base.input.as_str()],
            Operator::LessOrEqual(base) => {
                vec![base.base.input1.as_str(), base.base.input2.as_str()]
            }
            Operator::Mish(base) => vec![base.base.input.as_str()],
            Operator::Relu(base) => vec![base.base.input.as_str()],
            Operator::Selu(base) => vec![base.base.input.as_str()],
            Operator::Shrink(base) => vec![base.base.input.as_str()],
            Operator::Softmax(base) | Operator::LogSoftmax(base) => vec![base.base.input.as_str()],
            Operator::SoftmaxCrossEntropyLoss(base) => vec![base.base.input.as_str()],
            Operator::Softplus(base) => vec![base.base.input.as_str()],
            Operator::Softsign(base) => vec![base.base.input.as_str()],
            Operator::Conv2dFused(base) => {
                if let Some(bias) = base.base.bias.as_ref() {
                    vec![base.base.input.as_str(), base.base.kernel.as_str(), bias.as_str()]
                } else {
                    vec![base.base.input.as_str(), base.base.kernel.as_str()]
                }
            }
        }
    }

    pub(super) fn outputs(&self) -> Vec<&str> {
        match self {
            Operator::Constant(base) => vec![base.base.as_str()],
            Operator::Contiguous(base) => vec![base.base.output.as_str()],
            Operator::Abs(base) => vec![base.base.output.as_str()],
            Operator::Acos(base) => vec![base.base.output.as_str()],
            Operator::Acosh(base) => vec![base.base.output.as_str()],
            Operator::Add(base) => vec![base.base.output.as_str()],
            Operator::And(base) => vec![base.base.output.as_str()],
            Operator::ArgMax(base) => vec![base.base.output.as_str()],
            Operator::ArgMin(base) => vec![base.base.output.as_str()],
            Operator::Asin(base) => vec![base.base.output.as_str()],
            Operator::Asinh(base) => vec![base.base.output.as_str()],
            Operator::Atan(base) => vec![base.base.output.as_str()],
            Operator::Atanh(base) => vec![base.base.output.as_str()],
            Operator::AveragePool(base) => vec![base.base.output.as_str()],
            Operator::BatchNormalization(base) => {
                let mut ret = vec![base.base.y.as_str()];
                if let Some(running_mean) = base.base.running_mean.as_ref() {
                    ret.push(running_mean.as_str());
                }
                if let Some(running_var) = base.base.running_var.as_ref() {
                    ret.push(running_var.as_str());
                }
                ret
            }
            Operator::BitShift(base) => vec![base.base.output.as_str()],
            Operator::BitwiseAnd(base) => { vec![base.base.output.as_str()] }
            Operator::BitwiseNot(base) => vec![base.base.output.as_str()],
            Operator::BitwiseOr(base) => vec![base.base.output.as_str()],
            Operator::BitwiseXor(base) => { vec![base.base.output.as_str()] }
            Operator::Cast(base) => vec![base.base.output.as_str()],
            Operator::Ceil(base) => vec![base.base.output.as_str()],
            Operator::Concat(base) =>
                base.base.inputs
                    .iter()
                    .map(|s| s.as_str())
                    .collect(),
            Operator::Conv2d(base) => vec![base.base.output.as_str()],
            Operator::Conv2dInteger(base) => { vec![base.base.output.as_str()] }
            Operator::Cos(base) => vec![base.base.output.as_str()],
            Operator::Cosh(base) => vec![base.base.output.as_str()],
            Operator::ConstantOfShape(base) => vec![base.base.output.as_str()],
            Operator::Div(base) => vec![base.base.output.as_str()],
            Operator::Dropout(base) => vec![base.base.output.as_str()],
            Operator::Equal(base) => vec![base.base.output.as_str()],
            Operator::Erf(base) => vec![base.base.output.as_str()],
            Operator::Exp(base) => vec![base.base.output.as_str()],
            Operator::Expand(base) => vec![base.base.output.as_str()],
            Operator::EyeLike(base) => vec![base.base.output.as_str()],
            Operator::Flatten(base) => vec![base.base.output.as_str()],
            Operator::Floor(base) => vec![base.base.output.as_str()],
            Operator::Gather(base) => vec![base.base.output.as_str()],
            Operator::Gemm(base) => vec![base.base.output.as_str()],
            Operator::GlobalAveragePool(base) => vec![base.base.output.as_str()],
            Operator::GlobalMaxPool(base) => vec![base.base.output.as_str()],
            Operator::Greater(base) => vec![base.base.output.as_str()],
            Operator::Identity(base) => vec![base.base.output.as_str()],
            Operator::If(_) => vec![],
            Operator::IsInf(base) => vec![base.base.output.as_str()],
            Operator::IsNaN(base) => vec![base.base.output.as_str()],
            Operator::Less(base) => vec![base.base.output.as_str()],
            Operator::Log(base) => vec![base.base.output.as_str()],
            Operator::Loop(base) => vec![base.base.as_str()],
            Operator::Lstm(base) =>
                vec![
                    base.base.y
                        .as_ref()
                        .map(|x| x.as_str())
                        .unwrap_or(""),
                    base.base.y_c
                        .as_ref()
                        .map(|x| x.as_str())
                        .unwrap_or(""),
                    base.base.y_h
                        .as_ref()
                        .map(|x| x.as_str())
                        .unwrap_or("")
                ],
            Operator::MatMul(base) => vec![base.base.output.as_str()],
            Operator::MatMulInteger(base) => vec![base.base.output.as_str()],
            Operator::Max(base) => vec![base.base.output.as_str()],
            Operator::MaxPool(base) => vec![base.base.output.as_str()],
            Operator::Mean(base) => vec![base.base.output.as_str()],
            Operator::Min(base) => vec![base.base.output.as_str()],
            Operator::Mod(base) => vec![base.base.output.as_str()],
            Operator::Mul(base) => vec![base.base.output.as_str()],
            Operator::Neg(base) => vec![base.base.output.as_str()],
            Operator::Not(base) => vec![base.base.output.as_str()],
            Operator::OneHot(base) => vec![base.base.output.as_str()],
            Operator::Or(base) => vec![base.base.output.as_str()],
            Operator::Pad(base) => vec![base.base.output.as_str()],
            Operator::Pow(base) => vec![base.base.output.as_str()],
            Operator::RandomNormal(base) => vec![base.base.output.as_str()],
            Operator::RandomNormalLike(base) => vec![base.base.output.as_str()],
            Operator::RandomUniform(base) => vec![base.base.output.as_str()],
            Operator::RandomUniformLike(base) => vec![base.base.output.as_str()],
            Operator::Reciprocal(base) => vec![base.base.output.as_str()],
            Operator::ReduceMax(base) => vec![base.base.output.as_str()],
            Operator::ReduceMean(base) => vec![base.base.output.as_str()],
            Operator::ReduceMin(base) => vec![base.base.output.as_str()],
            Operator::ReduceProd(base) => vec![base.base.output.as_str()],
            Operator::ReduceSum(base) => vec![base.base.output.as_str()],
            Operator::Reshape(base) => vec![base.base.output.as_str()],
            Operator::Round(base) => vec![base.base.output.as_str()],
            Operator::Sigmoid(base) => vec![base.base.output.as_str()],
            Operator::Sign(base) => vec![base.base.output.as_str()],
            Operator::Sin(base) => vec![base.base.output.as_str()],
            Operator::Sinh(base) => vec![base.base.output.as_str()],
            Operator::Slice(base) => vec![base.base.output.as_str()],
            Operator::Split(base) =>
                base.base.outputs
                    .iter()
                    .map(|x| x.as_str())
                    .collect(),
            Operator::Sqrt(base) => vec![base.base.output.as_str()],
            Operator::Squeeze(base) => vec![base.base.output.as_str()],
            Operator::Sub(base) => vec![base.base.output.as_str()],
            Operator::Sum(base) => vec![base.base.output.as_str()],
            Operator::Shape(base) => vec![base.base.output.as_str()],
            Operator::Tan(base) => vec![base.base.output.as_str()],
            Operator::Tanh(base) => vec![base.base.output.as_str()],
            Operator::Transpose(base) => vec![base.base.output.as_str()],
            Operator::InvPermute(base) => vec![base.base.output.as_str()],
            Operator::PermuteContiguous(base) => vec![base.base.output.as_str()],
            Operator::Trilu(base) => vec![base.base.output.as_str()],
            Operator::Unsqueeze(base) => vec![base.base.output.as_str()],
            Operator::Where(base) => vec![base.base.output.as_str()],
            Operator::Xor(base) => vec![base.base.output.as_str()],
            Operator::Bernoulli(base) => vec![base.base.output.as_str()],
            Operator::BlackmanWindow(base) => vec![base.base.output.as_str()],
            Operator::CastLike(base) => vec![base.base.output.as_str()],
            Operator::Celu(base) => vec![base.base.output.as_str()],
            Operator::Clip(base) => vec![base.base.output.as_str()],
            Operator::Elu(base) => vec![base.base.output.as_str()],
            Operator::Gelu(base) => vec![base.base.output.as_str()],
            Operator::GreaterOrEqual(base) => { vec![base.base.output.as_str()] }
            Operator::HammingWindow(base) => vec![base.base.output.as_str()],
            Operator::HannWindow(base) => vec![base.base.output.as_str()],
            Operator::HardSigmoid(base) => vec![base.base.output.as_str()],
            Operator::HardSwish(base) => vec![base.base.output.as_str()],
            Operator::LayerNormalization(base) => vec![base.base.output.as_str()],
            Operator::LeakyRelu(base) => vec![base.base.output.as_str()],
            Operator::LessOrEqual(base) => { vec![base.base.output.as_str()] }
            Operator::LogSoftmax(base) => vec![base.base.output.as_str()],
            Operator::Mish(base) => vec![base.base.output.as_str()],
            Operator::ReduceL1(base) => vec![base.base.output.as_str()],
            Operator::ReduceL2(base) => vec![base.base.output.as_str()],
            Operator::ReduceLogSum(base) => vec![base.base.output.as_str()],
            Operator::ReduceLogSumExp(base) => vec![base.base.output.as_str()],
            Operator::ReduceSumSquare(base) => vec![base.base.output.as_str()],
            Operator::Relu(base) => vec![base.base.output.as_str()],
            Operator::Selu(base) => vec![base.base.output.as_str()],
            Operator::Shrink(base) => vec![base.base.output.as_str()],
            Operator::Softmax(base) => vec![base.base.output.as_str()],
            Operator::SoftmaxCrossEntropyLoss(base) => vec![base.base.output.as_str()],
            Operator::Softplus(base) => vec![base.base.output.as_str()],
            Operator::Softsign(base) => vec![base.base.output.as_str()],
            Operator::Conv2dFused(base) => vec![base.base.output.as_str()],
        }
    }

    pub(super) fn tensor_to_node<'a>(&'a self, tensor_to_node: &mut HashMap<&'a str, &'a str>) {
        let outputs = self.outputs();
        let id = self.id();
        for output in outputs {
            tensor_to_node.insert(output, id);
        }
    }

    pub(super) fn fill_node_degree(&self, node_degree: &mut HashMap<String, u32>) {
        let inputs = self.inputs();
        for input in inputs {
            if let Some(degree) = node_degree.get_mut(input) {
                *degree += 1;
            } else {
                node_degree.insert(input.to_string(), 1);
            }
        }
    }
}

impl std::fmt::Debug for Operator {
    #[allow(unused_variables)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operator::Constant(base) => write!(f, "Constant"),
            Operator::Contiguous(base) => write!(f, "Contiguous"),
            Operator::Abs(base) => write!(f, "Abs"),
            Operator::Acos(base) => write!(f, "Acos"),
            Operator::Acosh(base) => write!(f, "Acosh"),
            Operator::Add(base) => write!(f, "Add"),
            Operator::And(base) => write!(f, "And"),
            Operator::ArgMax(base) => write!(f, "ArgMax"),
            Operator::ArgMin(base) => write!(f, "ArgMin"),
            Operator::Asin(base) => write!(f, "Asin"),
            Operator::Asinh(base) => write!(f, "Asinh"),
            Operator::Atan(base) => write!(f, "Atan"),
            Operator::Atanh(base) => write!(f, "Atanh"),
            Operator::AveragePool(base) => write!(f, "AveragePool"),
            Operator::BatchNormalization(base) => { write!(f, "BatchNormalization") }
            Operator::BitShift(base) => write!(f, "BitShift"),
            Operator::BitwiseAnd(base) => write!(f, "BitwiseAnd"),
            Operator::BitwiseNot(base) => write!(f, "BitwiseNot"),
            Operator::BitwiseOr(base) => write!(f, "BitwiseOr"),
            Operator::BitwiseXor(base) => write!(f, "BitwiseXor"),
            Operator::Cast(base) => write!(f, "Cast"),
            Operator::Ceil(base) => write!(f, "Ceil"),
            Operator::Concat(base) => write!(f, "Concat"),
            Operator::Conv2d(base) => write!(f, "Conv2d"),
            Operator::Conv2dInteger(base) => write!(f, "Conv2dInteger"),
            Operator::Cos(base) => write!(f, "Cos"),
            Operator::Cosh(base) => write!(f, "Cosh"),
            Operator::ConstantOfShape(base) => write!(f, "ConstantOfShape"),
            Operator::Div(base) => write!(f, "Div"),
            Operator::Dropout(base) => write!(f, "Dropout"),
            Operator::Equal(base) => write!(f, "Equal"),
            Operator::Erf(base) => write!(f, "Erf"),
            Operator::Exp(base) => write!(f, "Exp"),
            Operator::Expand(base) => write!(f, "Expand"),
            Operator::EyeLike(base) => write!(f, "EyeLike"),
            Operator::Flatten(base) => write!(f, "Flatten"),
            Operator::Floor(base) => write!(f, "Floor"),
            Operator::Gather(base) => write!(f, "Gather"),
            Operator::Gemm(base) => write!(f, "Gemm"),
            Operator::GlobalAveragePool(base) => { write!(f, "GlobalAveragePool") }
            Operator::GlobalMaxPool(base) => write!(f, "GlobalMaxPool"),
            Operator::Greater(base) => write!(f, "Greater"),
            Operator::Identity(base) => write!(f, "Identity"),
            Operator::If(base) => write!(f, "If"),
            Operator::IsInf(base) => write!(f, "IsInf"),
            Operator::IsNaN(base) => write!(f, "IsNaN"),
            Operator::Less(base) => write!(f, "Less"),
            Operator::Log(base) => write!(f, "Log"),
            Operator::Loop(base) => write!(f, "Loop"),
            Operator::Lstm(base) => write!(f, "Lstm"),
            Operator::MatMul(base) => write!(f, "MatMul"),
            Operator::MatMulInteger(base) => write!(f, "MatMulInteger"),
            Operator::Max(base) => write!(f, "Max"),
            Operator::MaxPool(base) => write!(f, "MaxPool"),
            Operator::Mean(base) => write!(f, "Mean"),
            Operator::Min(base) => write!(f, "Min"),
            Operator::Mod(base) => write!(f, "Mod"),
            Operator::Mul(base) => write!(f, "Mul"),
            Operator::Neg(base) => write!(f, "Neg"),
            Operator::Not(base) => write!(f, "Not"),
            Operator::OneHot(base) => write!(f, "OneHot"),
            Operator::Or(base) => write!(f, "Or"),
            Operator::Pad(base) => write!(f, "Pad"),
            Operator::Pow(base) => write!(f, "Pow"),
            Operator::RandomNormal(base) => write!(f, "RandomNormal"),
            Operator::RandomNormalLike(base) => write!(f, "RandomNormalLike"),
            Operator::RandomUniform(base) => write!(f, "RandomUniform"),
            Operator::RandomUniformLike(base) => write!(f, "RandomUniformLike"),
            Operator::Reciprocal(base) => write!(f, "Reciprocal"),
            Operator::ReduceMax(base) => write!(f, "ReduceMax"),
            Operator::ReduceMean(base) => write!(f, "ReduceMean"),
            Operator::ReduceMin(base) => write!(f, "ReduceMin"),
            Operator::ReduceProd(base) => write!(f, "ReduceProd"),
            Operator::ReduceSum(base) => write!(f, "ReduceSum"),
            Operator::Reshape(base) => write!(f, "Reshape"),
            Operator::Round(base) => write!(f, "Round"),
            Operator::Sigmoid(base) => write!(f, "Sigmoid"),
            Operator::Sign(base) => write!(f, "Sign"),
            Operator::Sin(base) => write!(f, "Sin"),
            Operator::Sinh(base) => write!(f, "Sinh"),
            Operator::Slice(base) => write!(f, "Slice"),
            Operator::Split(base) => write!(f, "Split"),
            Operator::Sqrt(base) => write!(f, "Sqrt"),
            Operator::Squeeze(base) => write!(f, "Squeeze"),
            Operator::Sub(base) => write!(f, "Sub"),
            Operator::Sum(base) => write!(f, "Sum"),
            Operator::Shape(base) => write!(f, "Shape"),
            Operator::Tan(base) => write!(f, "Tan"),
            Operator::Tanh(base) => write!(f, "Tanh"),
            Operator::Transpose(base) => write!(f, "Transpose"),
            Operator::InvPermute(base) => write!(f, "InvPermute"),
            Operator::PermuteContiguous(base) => write!(f, "PermuteContiguous"),
            Operator::Trilu(base) => write!(f, "Trilu"),
            Operator::Unsqueeze(base) => write!(f, "Unsqueeze"),
            Operator::Where(base) => write!(f, "Where"),
            Operator::Xor(base) => write!(f, "Xor"),
            Operator::Bernoulli(base) => write!(f, "Bernoulli"),
            Operator::BlackmanWindow(base) => write!(f, "BlackmanWindow"),
            Operator::CastLike(base) => write!(f, "CastLike"),
            Operator::Celu(base) => write!(f, "Celu"),
            Operator::Clip(base) => write!(f, "Clip"),
            Operator::Elu(base) => write!(f, "Elu"),
            Operator::Gelu(base) => write!(f, "Gelu"),
            Operator::GreaterOrEqual(base) => write!(f, "GreaterOrEqual"),
            Operator::HammingWindow(base) => write!(f, "HammingWindow"),
            Operator::HannWindow(base) => write!(f, "HannWindow"),
            Operator::HardSigmoid(base) => write!(f, "HardSigmoid"),
            Operator::HardSwish(base) => write!(f, "HardSwish"),
            Operator::LayerNormalization(base) => { write!(f, "LayerNormalization") }
            Operator::LeakyRelu(base) => write!(f, "LeakyRelu"),
            Operator::LessOrEqual(base) => write!(f, "LessOrEqual"),
            Operator::LogSoftmax(base) => write!(f, "LogSoftmax"),
            Operator::Mish(base) => write!(f, "Mish"),
            Operator::ReduceL1(base) => write!(f, "ReduceL1"),
            Operator::ReduceL2(base) => write!(f, "ReduceL2"),
            Operator::ReduceLogSum(base) => write!(f, "ReduceLogSum"),
            Operator::ReduceLogSumExp(base) => write!(f, "ReduceLogSumExp"),
            Operator::ReduceSumSquare(base) => write!(f, "ReduceSumSquare"),
            Operator::Relu(base) => write!(f, "Relu"),
            Operator::Selu(base) => write!(f, "Selu"),
            Operator::Shrink(base) => write!(f, "Shrink"),
            Operator::Softmax(base) => write!(f, "Softmax"),
            Operator::SoftmaxCrossEntropyLoss(base) => { write!(f, "SoftmaxCrossEntropyLoss") }
            Operator::Softplus(base) => write!(f, "Softplus"),
            Operator::Softsign(base) => write!(f, "Softsign"),
            Operator::Conv2dFused(base) => write!(f, "Conv2dFused"),
        }
    }
}
