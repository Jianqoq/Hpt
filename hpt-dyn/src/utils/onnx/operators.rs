#![allow(unused)]

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
    pub(crate) saturate: bool,
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

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum OperatorEnum {
    Constant,
    Contiguous,
    Abs,
    Acos,
    Acosh,
    Add,
    And,
    ArgMax,
    ArgMin,
    Asin,
    Asinh,
    Atan,
    Atanh,
    AveragePool,
    BatchNormalization,
    BitShift,
    BitwiseAnd,
    BitwiseNot,
    BitwiseOr,
    BitwiseXor,
    Cast,
    Ceil,
    Concat,
    Conv2d,
    Conv2dInteger,
    Cos,
    Cosh,
    ConstantOfShape,
    Div,
    Dropout,
    Equal,
    Erf,
    Exp,
    Expand,
    EyeLike,
    Flatten,
    Floor,
    Gather,
    Gemm,
    GlobalAveragePool,
    GlobalMaxPool,
    Greater,
    Identity,
    If,
    IsInf,
    IsNaN,
    Less,
    Log,
    Loop,
    Lstm,
    MatMul,
    MatMulInteger,
    Max,
    MaxPool,
    Mean,
    Min,
    Mod,
    Mul,
    Neg,
    Not,
    OneHot,
    Or,
    Pad,
    Pow,
    RandomNormal,
    RandomNormalLike,
    RandomUniform,
    RandomUniformLike,
    Reciprocal,
    ReduceMax,
    ReduceMean,
    ReduceMin,
    ReduceProd,
    ReduceSum,
    Reshape,
    Round,
    Sigmoid,
    Sign,
    Sin,
    Sinh,
    Slice,
    Split,
    Sqrt,
    Squeeze,
    Sub,
    Sum,
    Shape,
    Tan,
    Tanh,
    Transpose,
    InvPermute,
    PermuteContiguous,
    Trilu,
    Unsqueeze,
    Where,
    Xor,
    Bernoulli,
    BlackmanWindow,
    CastLike,
    Celu,
    Clip,
    Elu,
    Gelu,
    GreaterOrEqual,
    HammingWindow,
    HannWindow,
    HardSigmoid,
    HardSwish,
    LayerNormalization,
    LeakyRelu,
    LessOrEqual,
    LogSoftmax,
    Mish,
    ReduceL1,
    ReduceL2,
    ReduceLogSum,
    ReduceLogSumExp,
    ReduceSumSquare,
    Relu,
    Selu,
    Shrink,
    Softmax,
    SoftmaxCrossEntropyLoss,
    Softplus,
    Softsign,
    Conv2dFused,
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
            Operator::BatchNormalization(base) => vec![
                base.base.input.as_str(),
                base.base.scale.as_str(),
                base.base.bias.as_str(),
                base.base.input_mean.as_str(),
                base.base.input_variance.as_str(),
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
            Operator::Concat(base) => base.base.inputs.iter().map(|s| s.as_str()).collect(),
            Operator::Conv2d(base) => {
                if let Some(bias) = base.base.bias.as_ref() {
                    vec![
                        base.base.input.as_str(),
                        base.base.kernel.as_str(),
                        bias.as_str(),
                    ]
                } else {
                    vec![base.base.input.as_str(), base.base.kernel.as_str()]
                }
            }
            Operator::Conv2dInteger(base) => {
                if let Some(bias) = base.base.bias.as_ref() {
                    vec![
                        base.base.input.as_str(),
                        base.base.kernel.as_str(),
                        bias.as_str(),
                    ]
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
            Operator::Lstm(base) => {
                let mut vec = vec![
                    base.base.x.as_str(),
                    base.base.w.as_str(),
                    base.base.r.as_str(),
                ];
                if let Some(init_h) = &base.base.initial_h {
                    vec.push(init_h.as_str());
                }
                if let Some(init_c) = &base.base.initial_c {
                    vec.push(init_c.as_str());
                }
                if let Some(peephole) = &base.base.p {
                    vec.push(peephole.as_str());
                }
                if let Some(b) = &base.base.b {
                    vec.push(b);
                }
                if let Some(sequence) = &base.base.sequence_lens {
                    vec.push(sequence.as_str());
                }
                vec
            }
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
            Operator::ReduceL1(base)
            | Operator::ReduceL2(base)
            | Operator::ReduceLogSum(base)
            | Operator::ReduceLogSumExp(base)
            | Operator::ReduceSumSquare(base)
            | Operator::ReduceMax(base)
            | Operator::ReduceMean(base)
            | Operator::ReduceMin(base)
            | Operator::ReduceProd(base)
            | Operator::ReduceSum(base) => {
                if let Some(axes) = base.base.axes.as_ref() {
                    vec![base.base.input.as_str(), axes.as_str()]
                } else {
                    vec![base.base.input.as_str()]
                }
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
                    vec![
                        base.base.input.as_str(),
                        base.base.kernel.as_str(),
                        bias.as_str(),
                    ]
                } else {
                    vec![base.base.input.as_str(), base.base.kernel.as_str()]
                }
            }
        }
    }

    pub(super) fn inputs_mut(&mut self) -> Vec<&mut String> {
        match self {
            Operator::Constant(_) => vec![],
            Operator::Contiguous(base) => vec![&mut base.base.input],
            Operator::Abs(base) => vec![&mut base.base.input],
            Operator::Acos(base) => vec![&mut base.base.input],
            Operator::Acosh(base) => vec![&mut base.base.input],
            Operator::Add(base) => vec![&mut base.base.input1, &mut base.base.input2],
            Operator::And(base) => vec![&mut base.base.input1, &mut base.base.input2],
            Operator::ArgMax(base) => vec![&mut base.base.input],
            Operator::ArgMin(base) => vec![&mut base.base.input],
            Operator::Asin(base) => vec![&mut base.base.input],
            Operator::Asinh(base) => vec![&mut base.base.input],
            Operator::Atan(base) => vec![&mut base.base.input],
            Operator::Atanh(base) => vec![&mut base.base.input],
            Operator::AveragePool(base) => vec![&mut base.base.input],
            Operator::BatchNormalization(base) => vec![
                &mut base.base.input,
                &mut base.base.scale,
                &mut base.base.bias,
                &mut base.base.input_mean,
                &mut base.base.input_variance,
            ],
            Operator::BitShift(base) => vec![&mut base.base.input1, &mut base.base.input2],
            Operator::BitwiseAnd(base) => {
                vec![&mut base.base.input1, &mut base.base.input2]
            }
            Operator::BitwiseNot(base) => vec![&mut base.base.input],
            Operator::BitwiseOr(base) => vec![&mut base.base.input1, &mut base.base.input2],
            Operator::BitwiseXor(base) => {
                vec![&mut base.base.input1, &mut base.base.input2]
            }
            Operator::Cast(base) => vec![&mut base.base.input],
            Operator::Ceil(base) => vec![&mut base.base.input],
            Operator::Concat(base) => base.base.inputs.iter_mut().map(|s| s).collect(),
            Operator::Conv2d(base) => {
                if let Some(bias) = base.base.bias.as_mut() {
                    vec![&mut base.base.input, &mut base.base.kernel, bias]
                } else {
                    vec![&mut base.base.input, &mut base.base.kernel]
                }
            }
            Operator::Conv2dInteger(base) => {
                if let Some(bias) = base.base.bias.as_mut() {
                    vec![&mut base.base.input, &mut base.base.kernel, bias]
                } else {
                    vec![&mut base.base.input, &mut base.base.kernel]
                }
            }
            Operator::Cos(base) => vec![&mut base.base.input],
            Operator::Cosh(base) => vec![&mut base.base.input],
            Operator::ConstantOfShape(base) => vec![&mut base.base.input],
            Operator::Div(base) => vec![&mut base.base.input1, &mut base.base.input2],
            Operator::Dropout(base) => vec![&mut base.base.input],
            Operator::Equal(base) => vec![&mut base.base.input1, &mut base.base.input2],
            Operator::Erf(base) => vec![&mut base.base.input],
            Operator::Exp(base) => vec![&mut base.base.input],
            Operator::Expand(base) => vec![&mut base.base.input],
            Operator::EyeLike(base) => vec![&mut base.base.input],
            Operator::Flatten(base) => vec![&mut base.base.input],
            Operator::Floor(base) => vec![&mut base.base.input],
            Operator::Gather(base) => vec![&mut base.base.input, &mut base.base.indices],
            Operator::Gemm(base) => vec![&mut base.base.a, &mut base.base.b],
            Operator::GlobalAveragePool(base) => vec![&mut base.base.input],
            Operator::GlobalMaxPool(base) => vec![&mut base.base.input],
            Operator::Greater(base) => vec![&mut base.base.input1, &mut base.base.input2],
            Operator::Identity(base) => vec![&mut base.base.input],
            Operator::If(base) => vec![&mut base.base],
            Operator::IsInf(base) => vec![&mut base.base.input],
            Operator::IsNaN(base) => vec![&mut base.base.input],
            Operator::Less(base) => vec![&mut base.base.input1, &mut base.base.input2],
            Operator::Log(base) => vec![&mut base.base.input],
            Operator::Loop(base) => vec![&mut base.base],
            Operator::Lstm(base) => {
                let mut vec = vec![&mut base.base.x, &mut base.base.w, &mut base.base.r];
                if let Some(init_h) = &mut base.base.initial_h {
                    vec.push(init_h);
                }
                if let Some(init_c) = &mut base.base.initial_c {
                    vec.push(init_c);
                }
                if let Some(p) = &mut base.base.p {
                    vec.push(p);
                }
                if let Some(b) = &mut base.base.b {
                    vec.push(b);
                }
                if let Some(sequence) = &mut base.base.sequence_lens {
                    vec.push(sequence);
                }
                vec
            },
            Operator::MatMul(base) => vec![&mut base.base.a, &mut base.base.b],
            Operator::MatMulInteger(base) => vec![&mut base.base.a, &mut base.base.b],
            Operator::Max(base) => vec![&mut base.base.input1, &mut base.base.input2],
            Operator::MaxPool(base) => vec![&mut base.base.input],
            Operator::Mean(base) => vec![&mut base.base.input],
            Operator::Min(base) => vec![&mut base.base.input1, &mut base.base.input2],
            Operator::Mod(base) => vec![&mut base.base.input1, &mut base.base.input2],
            Operator::Mul(base) => vec![&mut base.base.input1, &mut base.base.input2],
            Operator::Neg(base) => vec![&mut base.base.input],
            Operator::Not(base) => vec![&mut base.base.input],
            Operator::OneHot(base) => vec![&mut base.base.input],
            Operator::Or(base) => vec![&mut base.base.input1, &mut base.base.input2],
            Operator::Pad(base) => vec![&mut base.base.input],
            Operator::Pow(base) => vec![&mut base.base.input1, &mut base.base.input2],
            Operator::RandomNormal(base) => vec![&mut base.base.input],
            Operator::RandomNormalLike(base) => vec![&mut base.base.input],
            Operator::RandomUniform(base) => vec![&mut base.base.input],
            Operator::RandomUniformLike(base) => vec![&mut base.base.input],
            Operator::Reciprocal(base) => vec![&mut base.base.input],
            Operator::ReduceL1(base)
            | Operator::ReduceL2(base)
            | Operator::ReduceLogSum(base)
            | Operator::ReduceLogSumExp(base)
            | Operator::ReduceSumSquare(base)
            | Operator::ReduceMax(base)
            | Operator::ReduceMean(base)
            | Operator::ReduceMin(base)
            | Operator::ReduceProd(base)
            | Operator::ReduceSum(base) => {
                if let Some(axes) = base.base.axes.as_mut() {
                    vec![&mut base.base.input, axes]
                } else {
                    vec![&mut base.base.input]
                }
            }
            Operator::Reshape(base) => vec![&mut base.base.input, &mut base.base.shape],
            Operator::Round(base) => vec![&mut base.base.input],
            Operator::Sigmoid(base) => vec![&mut base.base.input],
            Operator::Sign(base) => vec![&mut base.base.input],
            Operator::Sin(base) => vec![&mut base.base.input],
            Operator::Sinh(base) => vec![&mut base.base.input],
            Operator::Slice(base) => vec![&mut base.base.input],
            Operator::Split(base) => vec![&mut base.base.input],
            Operator::Sqrt(base) => vec![&mut base.base.input],
            Operator::Squeeze(base) => vec![&mut base.base.input],
            Operator::Sub(base) => vec![&mut base.base.input1, &mut base.base.input2],
            Operator::Sum(base) => vec![&mut base.base.input],
            Operator::Shape(base) => vec![&mut base.base.input],
            Operator::Tan(base) => vec![&mut base.base.input],
            Operator::Tanh(base) => vec![&mut base.base.input],
            Operator::Transpose(base) => vec![&mut base.base.input],
            Operator::InvPermute(base) => vec![&mut base.base.input],
            Operator::PermuteContiguous(base) => vec![&mut base.base.input],
            Operator::Trilu(base) => vec![&mut base.base.input],
            Operator::Unsqueeze(base) => vec![&mut base.base.input],
            Operator::Where(base) => vec![&mut base.base.input, &mut base.base.condition],
            Operator::Xor(base) => vec![&mut base.base.input1, &mut base.base.input2],
            Operator::Bernoulli(base) => vec![&mut base.base.input],
            Operator::BlackmanWindow(base) => vec![&mut base.base.input],
            Operator::CastLike(base) => vec![&mut base.base.input],
            Operator::Celu(base) => vec![&mut base.base.input],
            Operator::Clip(base) => vec![&mut base.base.input],
            Operator::Elu(base) => vec![&mut base.base.input],
            Operator::Gelu(base) => vec![&mut base.base.input],
            Operator::GreaterOrEqual(base) => {
                vec![&mut base.base.input1, &mut base.base.input2]
            }
            Operator::HammingWindow(base) => vec![&mut base.base.input],
            Operator::HannWindow(base) => vec![&mut base.base.input],
            Operator::HardSigmoid(base) => vec![&mut base.base.input],
            Operator::HardSwish(base) => vec![&mut base.base.input],
            Operator::LayerNormalization(base) => vec![&mut base.base.input],
            Operator::LeakyRelu(base) => vec![&mut base.base.input],
            Operator::LessOrEqual(base) => {
                vec![&mut base.base.input1, &mut base.base.input2]
            }
            Operator::Mish(base) => vec![&mut base.base.input],
            Operator::Relu(base) => vec![&mut base.base.input],
            Operator::Selu(base) => vec![&mut base.base.input],
            Operator::Shrink(base) => vec![&mut base.base.input],
            Operator::Softmax(base) | Operator::LogSoftmax(base) => vec![&mut base.base.input],
            Operator::SoftmaxCrossEntropyLoss(base) => vec![&mut base.base.input],
            Operator::Softplus(base) => vec![&mut base.base.input],
            Operator::Softsign(base) => vec![&mut base.base.input],
            Operator::Conv2dFused(base) => {
                if let Some(bias) = base.base.bias.as_mut() {
                    vec![&mut base.base.input, &mut base.base.kernel, bias]
                } else {
                    vec![&mut base.base.input, &mut base.base.kernel]
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
            Operator::BitwiseAnd(base) => {
                vec![base.base.output.as_str()]
            }
            Operator::BitwiseNot(base) => vec![base.base.output.as_str()],
            Operator::BitwiseOr(base) => vec![base.base.output.as_str()],
            Operator::BitwiseXor(base) => {
                vec![base.base.output.as_str()]
            }
            Operator::Cast(base) => vec![base.base.output.as_str()],
            Operator::Ceil(base) => vec![base.base.output.as_str()],
            Operator::Concat(base) => vec![base.base.output.as_str()],
            Operator::Conv2d(base) => vec![base.base.output.as_str()],
            Operator::Conv2dInteger(base) => {
                vec![base.base.output.as_str()]
            }
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
            Operator::Lstm(base) => {
                let mut ret = vec![];
                if let Some(y) = base.base.y.as_ref() {
                    ret.push(y.as_str());
                }
                if let Some(y_c) = base.base.y_c.as_ref() {
                    ret.push(y_c.as_str());
                }
                if let Some(y_h) = base.base.y_h.as_ref() {
                    ret.push(y_h.as_str());
                }
                ret
            },
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
            Operator::Split(base) => base.base.outputs.iter().map(|x| x.as_str()).collect(),
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
            Operator::GreaterOrEqual(base) => {
                vec![base.base.output.as_str()]
            }
            Operator::HammingWindow(base) => vec![base.base.output.as_str()],
            Operator::HannWindow(base) => vec![base.base.output.as_str()],
            Operator::HardSigmoid(base) => vec![base.base.output.as_str()],
            Operator::HardSwish(base) => vec![base.base.output.as_str()],
            Operator::LayerNormalization(base) => vec![base.base.output.as_str()],
            Operator::LeakyRelu(base) => vec![base.base.output.as_str()],
            Operator::LessOrEqual(base) => {
                vec![base.base.output.as_str()]
            }
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

    pub(super) fn outputs_mut(&mut self) -> Vec<&mut String> {
        match self {
            Operator::Constant(base) => vec![&mut base.base],
            Operator::Contiguous(base) => vec![&mut base.base.output],
            Operator::Abs(base) => vec![&mut base.base.output],
            Operator::Acos(base) => vec![&mut base.base.output],
            Operator::Acosh(base) => vec![&mut base.base.output],
            Operator::Add(base) => vec![&mut base.base.output],
            Operator::And(base) => vec![&mut base.base.output],
            Operator::ArgMax(base) => vec![&mut base.base.output],
            Operator::ArgMin(base) => vec![&mut base.base.output],
            Operator::Asin(base) => vec![&mut base.base.output],
            Operator::Asinh(base) => vec![&mut base.base.output],
            Operator::Atan(base) => vec![&mut base.base.output],
            Operator::Atanh(base) => vec![&mut base.base.output],
            Operator::AveragePool(base) => vec![&mut base.base.output],
            Operator::BatchNormalization(base) => {
                let mut ret = vec![&mut base.base.y];
                if let Some(running_mean) = base.base.running_mean.as_mut() {
                    ret.push(running_mean);
                }
                if let Some(running_var) = base.base.running_var.as_mut() {
                    ret.push(running_var);
                }
                ret
            }
            Operator::BitShift(base) => vec![&mut base.base.output],
            Operator::BitwiseAnd(base) => {
                vec![&mut base.base.output]
            }
            Operator::BitwiseNot(base) => vec![&mut base.base.output],
            Operator::BitwiseOr(base) => vec![&mut base.base.output],
            Operator::BitwiseXor(base) => {
                vec![&mut base.base.output]
            }
            Operator::Cast(base) => vec![&mut base.base.output],
            Operator::Ceil(base) => vec![&mut base.base.output],
            Operator::Concat(base) => vec![&mut base.base.output],
            Operator::Conv2d(base) => vec![&mut base.base.output],
            Operator::Conv2dInteger(base) => {
                vec![&mut base.base.output]
            }
            Operator::Cos(base) => vec![&mut base.base.output],
            Operator::Cosh(base) => vec![&mut base.base.output],
            Operator::ConstantOfShape(base) => vec![&mut base.base.output],
            Operator::Div(base) => vec![&mut base.base.output],
            Operator::Dropout(base) => vec![&mut base.base.output],
            Operator::Equal(base) => vec![&mut base.base.output],
            Operator::Erf(base) => vec![&mut base.base.output],
            Operator::Exp(base) => vec![&mut base.base.output],
            Operator::Expand(base) => vec![&mut base.base.output],
            Operator::EyeLike(base) => vec![&mut base.base.output],
            Operator::Flatten(base) => vec![&mut base.base.output],
            Operator::Floor(base) => vec![&mut base.base.output],
            Operator::Gather(base) => vec![&mut base.base.output],
            Operator::Gemm(base) => vec![&mut base.base.output],
            Operator::GlobalAveragePool(base) => vec![&mut base.base.output],
            Operator::GlobalMaxPool(base) => vec![&mut base.base.output],
            Operator::Greater(base) => vec![&mut base.base.output],
            Operator::Identity(base) => vec![&mut base.base.output],
            Operator::If(_) => vec![],
            Operator::IsInf(base) => vec![&mut base.base.output],
            Operator::IsNaN(base) => vec![&mut base.base.output],
            Operator::Less(base) => vec![&mut base.base.output],
            Operator::Log(base) => vec![&mut base.base.output],
            Operator::Loop(base) => vec![],
            Operator::Lstm(base) => {
                let mut ret = vec![];
                if let Some(y) = base.base.y.as_mut() {
                    ret.push(y);
                }
                if let Some(y_c) = base.base.y_c.as_mut() {
                    ret.push(y_c);
                }
                if let Some(y_h) = base.base.y_h.as_mut() {
                    ret.push(y_h);
                }
                ret
            }
            Operator::MatMul(base) => vec![&mut base.base.output],
            Operator::MatMulInteger(base) => vec![&mut base.base.output],
            Operator::Max(base) => vec![&mut base.base.output],
            Operator::MaxPool(base) => vec![&mut base.base.output],
            Operator::Mean(base) => vec![&mut base.base.output],
            Operator::Min(base) => vec![&mut base.base.output],
            Operator::Mod(base) => vec![&mut base.base.output],
            Operator::Mul(base) => vec![&mut base.base.output],
            Operator::Neg(base) => vec![&mut base.base.output],
            Operator::Not(base) => vec![&mut base.base.output],
            Operator::OneHot(base) => vec![&mut base.base.output],
            Operator::Or(base) => vec![&mut base.base.output],
            Operator::Pad(base) => vec![&mut base.base.output],
            Operator::Pow(base) => vec![&mut base.base.output],
            Operator::RandomNormal(base) => vec![&mut base.base.output],
            Operator::RandomNormalLike(base) => vec![&mut base.base.output],
            Operator::RandomUniform(base) => vec![&mut base.base.output],
            Operator::RandomUniformLike(base) => vec![&mut base.base.output],
            Operator::Reciprocal(base) => vec![&mut base.base.output],
            Operator::ReduceMax(base) => vec![&mut base.base.output],
            Operator::ReduceMean(base) => vec![&mut base.base.output],
            Operator::ReduceMin(base) => vec![&mut base.base.output],
            Operator::ReduceProd(base) => vec![&mut base.base.output],
            Operator::ReduceSum(base) => vec![&mut base.base.output],
            Operator::Reshape(base) => vec![&mut base.base.output],
            Operator::Round(base) => vec![&mut base.base.output],
            Operator::Sigmoid(base) => vec![&mut base.base.output],
            Operator::Sign(base) => vec![&mut base.base.output],
            Operator::Sin(base) => vec![&mut base.base.output],
            Operator::Sinh(base) => vec![&mut base.base.output],
            Operator::Slice(base) => vec![&mut base.base.output],
            Operator::Split(base) => base.base.outputs.iter_mut().map(|x| x).collect(),
            Operator::Sqrt(base) => vec![&mut base.base.output],
            Operator::Squeeze(base) => vec![&mut base.base.output],
            Operator::Sub(base) => vec![&mut base.base.output],
            Operator::Sum(base) => vec![&mut base.base.output],
            Operator::Shape(base) => vec![&mut base.base.output],
            Operator::Tan(base) => vec![&mut base.base.output],
            Operator::Tanh(base) => vec![&mut base.base.output],
            Operator::Transpose(base) => vec![&mut base.base.output],
            Operator::InvPermute(base) => vec![&mut base.base.output],
            Operator::PermuteContiguous(base) => vec![&mut base.base.output],
            Operator::Trilu(base) => vec![&mut base.base.output],
            Operator::Unsqueeze(base) => vec![&mut base.base.output],
            Operator::Where(base) => vec![&mut base.base.output],
            Operator::Xor(base) => vec![&mut base.base.output],
            Operator::Bernoulli(base) => vec![&mut base.base.output],
            Operator::BlackmanWindow(base) => vec![&mut base.base.output],
            Operator::CastLike(base) => vec![&mut base.base.output],
            Operator::Celu(base) => vec![&mut base.base.output],
            Operator::Clip(base) => vec![&mut base.base.output],
            Operator::Elu(base) => vec![&mut base.base.output],
            Operator::Gelu(base) => vec![&mut base.base.output],
            Operator::GreaterOrEqual(base) => {
                vec![&mut base.base.output]
            }
            Operator::HammingWindow(base) => vec![&mut base.base.output],
            Operator::HannWindow(base) => vec![&mut base.base.output],
            Operator::HardSigmoid(base) => vec![&mut base.base.output],
            Operator::HardSwish(base) => vec![&mut base.base.output],
            Operator::LayerNormalization(base) => vec![&mut base.base.output],
            Operator::LeakyRelu(base) => vec![&mut base.base.output],
            Operator::LessOrEqual(base) => {
                vec![&mut base.base.output]
            }
            Operator::LogSoftmax(base) => vec![&mut base.base.output],
            Operator::Mish(base) => vec![&mut base.base.output],
            Operator::ReduceL1(base) => vec![&mut base.base.output],
            Operator::ReduceL2(base) => vec![&mut base.base.output],
            Operator::ReduceLogSum(base) => vec![&mut base.base.output],
            Operator::ReduceLogSumExp(base) => vec![&mut base.base.output],
            Operator::ReduceSumSquare(base) => vec![&mut base.base.output],
            Operator::Relu(base) => vec![&mut base.base.output],
            Operator::Selu(base) => vec![&mut base.base.output],
            Operator::Shrink(base) => vec![&mut base.base.output],
            Operator::Softmax(base) => vec![&mut base.base.output],
            Operator::SoftmaxCrossEntropyLoss(base) => vec![&mut base.base.output],
            Operator::Softplus(base) => vec![&mut base.base.output],
            Operator::Softsign(base) => vec![&mut base.base.output],
            Operator::Conv2dFused(base) => vec![&mut base.base.output],
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

    pub(super) fn to_enum(&self) -> OperatorEnum {
        match self {
            Operator::Constant(_) => OperatorEnum::Constant,
            Operator::Contiguous(_) => OperatorEnum::Contiguous,
            Operator::Abs(_) => OperatorEnum::Abs,
            Operator::Acos(_) => OperatorEnum::Acos,
            Operator::Acosh(_) => OperatorEnum::Acosh,
            Operator::Add(_) => OperatorEnum::Add,
            Operator::And(_) => OperatorEnum::And,
            Operator::ArgMax(_) => OperatorEnum::ArgMax,
            Operator::ArgMin(_) => OperatorEnum::ArgMin,
            Operator::Asin(_) => OperatorEnum::Asin,
            Operator::Asinh(_) => OperatorEnum::Asinh,
            Operator::Atan(_) => OperatorEnum::Atan,
            Operator::Atanh(_) => OperatorEnum::Atanh,
            Operator::AveragePool(_) => OperatorEnum::AveragePool,
            Operator::BatchNormalization(_) => OperatorEnum::BatchNormalization,
            Operator::BitShift(_) => OperatorEnum::BitShift,
            Operator::BitwiseAnd(_) => OperatorEnum::BitwiseAnd,
            Operator::BitwiseNot(_) => OperatorEnum::BitwiseNot,
            Operator::BitwiseOr(_) => OperatorEnum::BitwiseOr,
            Operator::BitwiseXor(_) => OperatorEnum::BitwiseXor,
            Operator::Cast(_) => OperatorEnum::Cast,
            Operator::Ceil(_) => OperatorEnum::Ceil,
            Operator::Concat(_) => OperatorEnum::Concat,
            Operator::Conv2d(_) => OperatorEnum::Conv2d,
            Operator::Conv2dInteger(_) => OperatorEnum::Conv2dInteger,
            Operator::Cos(_) => OperatorEnum::Cos,
            Operator::Cosh(_) => OperatorEnum::Cosh,
            Operator::ConstantOfShape(_) => OperatorEnum::ConstantOfShape,
            Operator::Div(_) => OperatorEnum::Div,
            Operator::Dropout(_) => OperatorEnum::Dropout,
            Operator::Equal(_) => OperatorEnum::Equal,
            Operator::Erf(_) => OperatorEnum::Erf,
            Operator::Exp(_) => OperatorEnum::Exp,
            Operator::Expand(_) => OperatorEnum::Expand,
            Operator::EyeLike(_) => OperatorEnum::EyeLike,
            Operator::Flatten(_) => OperatorEnum::Flatten,
            Operator::Floor(_) => OperatorEnum::Floor,
            Operator::Gather(_) => OperatorEnum::Gather,
            Operator::Gemm(_) => OperatorEnum::Gemm,
            Operator::GlobalAveragePool(_) => OperatorEnum::GlobalAveragePool,
            Operator::GlobalMaxPool(_) => OperatorEnum::GlobalMaxPool,
            Operator::Greater(_) => OperatorEnum::Greater,
            Operator::Identity(_) => OperatorEnum::Identity,
            Operator::If(_) => OperatorEnum::If,
            Operator::IsInf(_) => OperatorEnum::IsInf,
            Operator::IsNaN(_) => OperatorEnum::IsNaN,
            Operator::Less(_) => OperatorEnum::Less,
            Operator::Log(_) => OperatorEnum::Log,
            Operator::Loop(_) => OperatorEnum::Loop,
            Operator::Lstm(_) => OperatorEnum::Lstm,
            Operator::MatMul(_) => OperatorEnum::MatMul,
            Operator::MatMulInteger(_) => OperatorEnum::MatMulInteger,
            Operator::Max(_) => OperatorEnum::Max,
            Operator::MaxPool(_) => OperatorEnum::MaxPool,
            Operator::Mean(_) => OperatorEnum::Mean,
            Operator::Min(_) => OperatorEnum::Min,
            Operator::Mod(_) => OperatorEnum::Mod,
            Operator::Mul(_) => OperatorEnum::Mul,
            Operator::Neg(_) => OperatorEnum::Neg,
            Operator::Not(_) => OperatorEnum::Not,
            Operator::OneHot(_) => OperatorEnum::OneHot,
            Operator::Or(_) => OperatorEnum::Or,
            Operator::Pad(_) => OperatorEnum::Pad,
            Operator::Pow(_) => OperatorEnum::Pow,
            Operator::RandomNormal(_) => OperatorEnum::RandomNormal,
            Operator::RandomNormalLike(_) => OperatorEnum::RandomNormalLike,
            Operator::RandomUniform(_) => OperatorEnum::RandomUniform,
            Operator::RandomUniformLike(_) => OperatorEnum::RandomUniformLike,
            Operator::Reciprocal(_) => OperatorEnum::Reciprocal,
            Operator::ReduceMax(_) => OperatorEnum::ReduceMax,
            Operator::ReduceMean(_) => OperatorEnum::ReduceMean,
            Operator::ReduceMin(_) => OperatorEnum::ReduceMin,
            Operator::ReduceProd(_) => OperatorEnum::ReduceProd,
            Operator::ReduceSum(_) => OperatorEnum::ReduceSum,
            Operator::Reshape(_) => OperatorEnum::Reshape,
            Operator::Round(_) => OperatorEnum::Round,
            Operator::Sigmoid(_) => OperatorEnum::Sigmoid,
            Operator::Sign(_) => OperatorEnum::Sign,
            Operator::Sin(_) => OperatorEnum::Sin,
            Operator::Sinh(_) => OperatorEnum::Sinh,
            Operator::Slice(_) => OperatorEnum::Slice,
            Operator::Split(_) => OperatorEnum::Split,
            Operator::Sqrt(_) => OperatorEnum::Sqrt,
            Operator::Squeeze(_) => OperatorEnum::Squeeze,
            Operator::Sub(_) => OperatorEnum::Sub,
            Operator::Sum(_) => OperatorEnum::Sum,
            Operator::Shape(_) => OperatorEnum::Shape,
            Operator::Tan(_) => OperatorEnum::Tan,
            Operator::Tanh(_) => OperatorEnum::Tanh,
            Operator::Transpose(_) => OperatorEnum::Transpose,
            Operator::InvPermute(_) => OperatorEnum::InvPermute,
            Operator::PermuteContiguous(_) => OperatorEnum::PermuteContiguous,
            Operator::Trilu(_) => OperatorEnum::Trilu,
            Operator::Unsqueeze(_) => OperatorEnum::Unsqueeze,
            Operator::Where(_) => OperatorEnum::Where,
            Operator::Xor(_) => OperatorEnum::Xor,
            Operator::Bernoulli(_) => OperatorEnum::Bernoulli,
            Operator::BlackmanWindow(_) => OperatorEnum::BlackmanWindow,
            Operator::CastLike(_) => OperatorEnum::CastLike,
            Operator::Celu(_) => OperatorEnum::Celu,
            Operator::Clip(_) => OperatorEnum::Clip,
            Operator::Elu(_) => OperatorEnum::Elu,
            Operator::Gelu(_) => OperatorEnum::Gelu,
            Operator::GreaterOrEqual(_) => OperatorEnum::GreaterOrEqual,
            Operator::HammingWindow(_) => OperatorEnum::HammingWindow,
            Operator::HannWindow(_) => OperatorEnum::HannWindow,
            Operator::HardSigmoid(_) => OperatorEnum::HardSigmoid,
            Operator::HardSwish(_) => OperatorEnum::HardSwish,
            Operator::LayerNormalization(_) => OperatorEnum::LayerNormalization,
            Operator::LeakyRelu(_) => OperatorEnum::LeakyRelu,
            Operator::LessOrEqual(_) => OperatorEnum::LessOrEqual,
            Operator::LogSoftmax(_) => OperatorEnum::LogSoftmax,
            Operator::Mish(_) => OperatorEnum::Mish,
            Operator::ReduceL1(_) => OperatorEnum::ReduceL1,
            Operator::ReduceL2(_) => OperatorEnum::ReduceL2,
            Operator::ReduceLogSum(_) => OperatorEnum::ReduceLogSum,
            Operator::ReduceLogSumExp(_) => OperatorEnum::ReduceLogSumExp,
            Operator::ReduceSumSquare(_) => OperatorEnum::ReduceSumSquare,
            Operator::Relu(_) => OperatorEnum::Relu,
            Operator::Selu(_) => OperatorEnum::Selu,
            Operator::Shrink(_) => OperatorEnum::Shrink,
            Operator::Softmax(_) => OperatorEnum::Softmax,
            Operator::SoftmaxCrossEntropyLoss(_) => OperatorEnum::SoftmaxCrossEntropyLoss,
            Operator::Softplus(_) => OperatorEnum::Softplus,
            Operator::Softsign(_) => OperatorEnum::Softsign,
            Operator::Conv2dFused(_) => OperatorEnum::Conv2dFused,
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
            Operator::BatchNormalization(base) => {
                write!(f, "BatchNormalization")
            }
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
            Operator::GlobalAveragePool(base) => {
                write!(f, "GlobalAveragePool")
            }
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
            Operator::LayerNormalization(base) => {
                write!(f, "LayerNormalization")
            }
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
            Operator::SoftmaxCrossEntropyLoss(base) => {
                write!(f, "SoftmaxCrossEntropyLoss")
            }
            Operator::Softplus(base) => write!(f, "Softplus"),
            Operator::Softsign(base) => write!(f, "Softsign"),
            Operator::Conv2dFused(base) => write!(f, "Conv2dFused"),
        }
    }
}
