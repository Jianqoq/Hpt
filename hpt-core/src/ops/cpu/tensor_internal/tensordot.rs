use hpt_common::error::{base::TensorError, shape::ShapeError};
use hpt_traits::{CommonBounds, Matmul, ShapeManipulate, TensorDot, TensorInfo};
use hpt_types::type_promote::NormalOut;

use crate::tensor_base::_Tensor;

impl<A, B> TensorDot<_Tensor<B>> for _Tensor<A>
where
    A: CommonBounds + NormalOut<B>,
    B: CommonBounds,
    _Tensor<A>: Matmul<_Tensor<B>>,
    <_Tensor<A> as Matmul<_Tensor<B>>>::Output:
        ShapeManipulate<Output = <_Tensor<A> as Matmul<_Tensor<B>>>::Output>,
{
    type Output = <_Tensor<A> as Matmul<_Tensor<B>>>::Output;

    fn tensordot<const N: usize>(
        &self,
        rhs: &_Tensor<B>,
        axes: ([i64; N], [i64; N]),
    ) -> std::result::Result<Self::Output, TensorError> {
        let mut axes: [Vec<i64>; 2] = [axes.0.to_vec(), axes.1.to_vec()];
        let a_axes_dim = axes[0].len();
        let b_axes_dim = axes[1].len();
        let a_shape = &self.shape();
        let b_shape = &rhs.shape();
        let a_ndim = a_shape.len();
        let b_ndim = b_shape.len();
        ShapeError::check_dim(a_axes_dim, b_axes_dim)?;
        for i in 0..a_axes_dim {
            if axes[0][i] < 0 {
                axes[0][i] += a_ndim as i64;
                ShapeError::check_index_out_of_range(axes[0][i], a_ndim as i64)?;
            }
            if axes[1][i] < 0 {
                axes[1][i] += b_ndim as i64;
                ShapeError::check_index_out_of_range(axes[1][i], b_ndim as i64)?;
            }
            ShapeError::check_dim(
                a_shape[axes[0][i] as usize] as usize,
                b_shape[axes[1][i] as usize] as usize,
            )?;
        }
        let notin = (0..a_ndim as i64)
            .into_iter()
            .filter(|i| !axes[0].contains(i))
            .collect::<Vec<_>>();
        let mut new_axes_a = notin.clone();
        new_axes_a.extend(&axes[0]);
        let n2 = axes[0].iter().fold(1, |acc, x| acc * a_shape[*x as usize]);
        let n1 = notin.iter().fold(1, |acc, x| acc * a_shape[*x as usize]);
        let new_a_shape = vec![n1, n2];
        let mut olda = notin
            .into_iter()
            .map(|x| a_shape[x as usize])
            .collect::<Vec<_>>();
        let notin = (0..b_ndim as i64)
            .into_iter()
            .filter(|i| !axes[1].contains(i))
            .collect::<Vec<_>>();
        let mut new_axes_b = notin.clone();
        new_axes_b.extend(&axes[1]);
        let n2 = axes[1].iter().fold(1, |acc, x| acc * b_shape[*x as usize]);
        let n1 = notin.iter().fold(1, |acc, x| acc * b_shape[*x as usize]);
        let new_b_shape = vec![n2, n1];
        let oldb = notin
            .into_iter()
            .map(|x| b_shape[x as usize])
            .collect::<Vec<_>>();
        let new_a = self.permute(new_axes_a)?.reshape(new_a_shape)?;
        let new_b = rhs.permute(new_axes_b)?.reshape(new_b_shape)?;
        let res = new_a.matmul(new_b)?;
        olda.extend(&oldb);
        res.reshape(olda)
    }
}
