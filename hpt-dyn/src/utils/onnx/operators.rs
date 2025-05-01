use hpt_types::dtype::DType;

pub(crate) struct Unary {
    pub(crate) input: String,
    pub(crate) output: String,
}

pub(crate) struct Binary {
    pub(crate) input1: String,
    pub(crate) input2: String,
    pub(crate) output: String,
}

pub(crate) struct ArgReduce {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) axis: i64,
    pub(crate) keepdims: bool,
    pub(crate) select_last_index: bool,
}

pub(crate) struct Pooling {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) kernel_shape: Vec<i64>,
    pub(crate) pads: Vec<i64>,
    pub(crate) strides: Vec<i64>,
    pub(crate) ceil_mode: bool,
}

pub(crate) struct BatchNormalization {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) scale: String,
    pub(crate) bias: String,
    pub(crate) mean: String,
    pub(crate) variance: String,
    pub(crate) epsilon: f64,
    pub(crate) momentum: f64,
}

pub(crate) struct Concat {
    pub(crate) inputs: Vec<String>,
    pub(crate) output: String,
    pub(crate) axis: i64,
}

pub(crate) struct Conv2d {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) kernel: String,
    pub(crate) bias: String,
    pub(crate) pads: [(i64, i64); 2],
    pub(crate) strides: [i64; 2],
    pub(crate) dilations: [i64; 2],
    pub(crate) group: i64,
}

pub(crate) struct Dropout {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) ratio: f64,
}

pub(crate) struct Expand {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) dims: Vec<i64>,
}

pub(crate) struct EyeLike {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) shape: Vec<i64>,
}

pub(crate) struct Flatten {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) start_dim: i64,
    pub(crate) end_dim: i64,
}

pub(crate) struct Gemm {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) alpha: f64,
    pub(crate) beta: f64,
    pub(crate) trans_a: bool,
    pub(crate) trans_b: bool,
    pub(crate) bias: String,
}

pub(crate) struct Reduce {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) axes: Vec<i64>,
    pub(crate) keepdims: bool,
}

pub(crate) struct OneHot {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) axis: i64,
}

pub(crate) struct Pad {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) pads: Vec<i64>,
    pub(crate) value: f64,
    pub(crate) axes: Vec<i64>,
}

pub(crate) struct RandomNormal {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) mean: f64,
    pub(crate) scale: f64,
    pub(crate) seed: i64,
    pub(crate) shape: Vec<i64>,
    pub(crate) dtype: DType,
}

pub(crate) struct RandomUniform {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) low: f64,
    pub(crate) high: f64,
    pub(crate) seed: i64,
    pub(crate) shape: Vec<i64>,
    pub(crate) dtype: DType,
}

pub(crate) struct Reshape {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) shape: Vec<i64>,
}

pub(crate) struct Slice {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) selections: Vec<(i64, i64, i64)>,
}

pub(crate) struct Split {
    pub(crate) input: String,
    pub(crate) outputs: Vec<String>,
    pub(crate) axis: i64,
}

pub(crate) struct Squeeze {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) axes: Vec<i64>,
}

pub(crate) struct Permute {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) perm: Vec<i64>,
}

pub(crate) struct Where {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) condition: String,
}

pub(crate) struct Bernoulli {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) dtype: DType,
    pub(crate) seed: i64,
}

pub(crate) struct Cast {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) to: DType,
}

pub(crate) struct Clip {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) min: f64,
    pub(crate) max: f64,
}

pub(crate) struct LayerNormalization {
    pub(crate) input: String,
    pub(crate) output: String,
    pub(crate) scale: String,
    pub(crate) bias: String,
    pub(crate) epsilon: f64,
}

pub(crate) enum Operator {
    Abs(Unary),
    Acos(Unary),
    Acosh(Unary),
    Add(Binary),
    And(Binary),
    ArgMax(ArgReduce),
    ArgMin(ArgReduce),
    Asin(Unary),
    Asinh(Unary),
    Atan(Unary),
    Atanh(Unary),
    AveragePool(Pooling),
    BatchNormalization(BatchNormalization),
    BitShift(Binary),
    BitwiseAnd(Binary),
    BitwiseNot(Unary),
    BitwiseOr(Binary),
    BitwiseXor(Binary),
    Cast(Cast),
    Ceil(Unary),
    Concat(Concat),
    Conv2d(Conv2d),
    Conv2dInteger(Conv2d),
    Cos(Unary),
    Cosh(Unary),
    Div(Binary),
    Dropout(Dropout),
    Equal(Binary),
    Erf(Unary),
    Exp(Unary),
    Expand(Expand),
    EyeLike(EyeLike),
    Flatten(Flatten),
    Floor(Unary),
    Gemm(Gemm),
    GlobalAveragePool(Pooling),
    GlobalMaxPool(Pooling),
    Greater(Binary),
    Identity(EyeLike),
    If,
    IsInf(Unary),
    IsNaN(Unary),
    Less(Binary),
    Log(Unary),
    Loop,
    MatMul(Gemm),
    MatMulInteger(Gemm),
    Max(Binary),
    MaxPool(Pooling),
    Mean(Reduce),
    Min(Binary),
    Mod(Binary),
    Mul(Binary),
    Neg(Unary),
    Not(Unary),
    OneHot(OneHot),
    Or(Binary),
    Pad(Pad),
    Pow(Binary),
    RandomNormal(RandomNormal),
    RandomNormalLike(RandomNormal),
    RandomUniform(RandomUniform),
    RandomUniformLike(RandomUniform),
    Reciprocal(Unary),
    ReduceMax(Reduce),
    ReduceMean(Reduce),
    ReduceMin(Reduce),
    ReduceProd(Reduce),
    ReduceSum(Reduce),
    Reshape(Reshape),
    Round(Unary),
    Sigmoid(Unary),
    Sign(Unary),
    Sin(Unary),
    Sinh(Unary),
    Slice(Slice),
    Split(Split),
    Sqrt(Unary),
    Squeeze(Squeeze),
    Sub(Binary),
    Sum(Reduce),
    Tan(Unary),
    Tanh(Unary),
    Transpose(Permute),
    Trilu(Unary),
    Unsqueeze(Squeeze),
    Where(Where),
    Xor(Binary),
    Bernoulli(Bernoulli),
    BlackmanWindow(Unary),
    CastLike(Cast),
    Celu(Unary),
    Clip(Clip),
    Elu(Unary),
    Gelu(Unary),
    GreaterOrEqual(Binary),
    HammingWindow(Unary),
    HannWindow(Unary),
    HardSigmoid(Unary),
    HardSwish(Unary),
    LayerNormalization(LayerNormalization),
    LeakyRelu(Unary),
    LessOrEqual(Binary),
    LogSoftmax(Reduce),
    Mish(Unary),
    ReduceL1(Reduce),
    ReduceL2(Reduce),
    ReduceLogSum(Reduce),
    ReduceLogSumExp(Reduce),
    ReduceSumSquare(Reduce),
    Relu(Unary),
    Selu(Unary),
    Shrink(Unary),
    Softmax(Reduce),
    SoftmaxCrossEntropyLoss(Reduce),
    Softplus(Unary),
    Softsign(Unary),
}
