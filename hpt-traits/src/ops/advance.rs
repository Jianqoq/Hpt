use hpt_common::error::base::TensorError;

/// A trait contains advance operations
pub trait AdvancedOps {
    /// The type of the meta data
    type Meta;
    /// The type of the output tensor
    type Output;
    /// The type of the index tensor
    type IndexOutput;
    /// Pad the tensor
    fn pad(&self, pads: &[(i64, i64)], val: Self::Meta) -> Result<Self::Output, TensorError>;
    /// Topk the tensor
    fn topk(
        &self,
        k: i64,
        dim: i64,
        largest: bool,
        sorted: bool,
    ) -> Result<(Self::IndexOutput, Self::Output), TensorError>;
    /// Onehot the tensor
    fn onehot(
        &self,
        depth: usize,
        axis: i64,
        true_val: Self::Meta,
        false_val: Self::Meta,
    ) -> Result<Self::Output, TensorError>;
    /// Gather elements the tensor
    // fn gather_elements(
    //     &self,
    //     indices: &Self::IndexOutput,
    //     axis: i64,
    // ) -> Result<Self::Output, TensorError>;
    /// Scatter elements the tensor
    fn scatter(
        &self,
        indices: &Self::IndexOutput,
        axis: i64,
        src: &Self::Output,
    ) -> Result<Self::Output, TensorError>;
}

/// A trait for hardmax
pub trait HardMax<T> {
    /// The type of the output tensor
    type Output;
    /// Hardmax the tensor
    fn hardmax(&self, axis: i64) -> Result<Self::Output, TensorError>;
}

/// A trait for tensor where
pub trait TensorWhere {
    /// The type of the output tensor
    type Output;
    /// The type of the condition tensor
    type Condition;
    /// Where the tensor
    fn tensor_where(
        condition: &Self::Condition,
        x: &Self::Output,
        y: &Self::Output,
    ) -> Result<Self::Output, TensorError>;
}
