/// Enum to specify axes in tensor dot product operations.
///
/// Provides several ways to specify which axes the tensor dot product should be computed along.
///
/// # Variants
/// - `Int(i64)`: A single integer representing a single axis.
/// - `TupleScalar((i64, i64))`: A tuple of two integers representing axes for the first and second tensor, respectively.
/// - `TupleArray(([i64; N], [i64; N]))`: A tuple of two arrays, each representing a list of axes for each tensor.
/// - `ArrayArray([[i64; N]; 2])`: An array of two arrays, each representing a list of axes for each tensor.
///
/// # Examples
/// ```
/// use tensor_common::TensorDotArgs;
/// let axes = TensorDotArgs::<2>::TupleScalar((2, 2));
/// let axes = TensorDotArgs::<2>::TupleArray(([0i64, 1], [2, 3]));
/// ```
pub enum TensorDotArgs<const N: usize> {
    Int(i64),
    TupleScalar((i64, i64)),
    TupleArray(([i64; N], [i64; N])),
    ArrayArray([[i64; N]; 2]),
}

impl From<i64> for TensorDotArgs<1> {
    fn from(arg: i64) -> Self {
        Self::Int(arg)
    }
}

impl From<(i64, i64)> for TensorDotArgs<1> {
    fn from(arg: (i64, i64)) -> Self {
        Self::TupleScalar(arg)
    }
}

impl<const N: usize> From<([i64; N], [i64; N])> for TensorDotArgs<N> {
    fn from(arg: ([i64; N], [i64; N])) -> Self {
        Self::TupleArray(arg)
    }
}

impl<const N: usize> From<[[i64; N]; 2]> for TensorDotArgs<N> {
    fn from(arg: [[i64; N]; 2]) -> Self {
        Self::ArrayArray(arg)
    }
}

impl<const N: usize> From<TensorDotArgs<N>> for [Vec<i64>; 2] {
    fn from(val: TensorDotArgs<N>) -> Self {
        match val {
            TensorDotArgs::<N>::Int(i) => {
                let a: Vec<i64> = (-i..0).collect();
                let b: Vec<i64> = (0..i).collect();
                [a, b]
            }
            TensorDotArgs::<N>::TupleScalar((i, j)) => { [vec![i], vec![j]] }
            TensorDotArgs::<N>::TupleArray((i, j)) => [i.to_vec(), j.to_vec()],
            TensorDotArgs::<N>::ArrayArray(i) => [i[0].to_vec(), i[1].to_vec()],
        }
    }
}
