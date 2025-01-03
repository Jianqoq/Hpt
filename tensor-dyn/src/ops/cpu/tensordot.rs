use crate::tensor_base::_Tensor;
use tensor_common::err_handler::TensorError;
use tensor_common::tensordot_args::TensorDotArgs;
use tensor_traits::ops::binary::Matmul;
use tensor_traits::shape_manipulate::ShapeManipulate;
use tensor_traits::tensor::CommonBounds;
use tensor_traits::tensor::TensorInfo;
use tensor_types::type_promote::NormalOut;

use std::panic::Location;

pub(crate) fn _tensordot<A, B, G, const N: usize>(
    a: &_Tensor<A>,
    b: &_Tensor<B>,
    axes: G,
) -> std::result::Result<<_Tensor<A> as Matmul<_Tensor<B>>>::Output, TensorError>
where
    A: CommonBounds + NormalOut<B>,
    B: CommonBounds,
    <A as NormalOut<B>>::Output: CommonBounds,
    G: Into<TensorDotArgs<N>>,
    _Tensor<A>: Matmul<_Tensor<B>>,
    <_Tensor<A> as Matmul<_Tensor<B>>>::Output: ShapeManipulate<Output = <_Tensor<A> as Matmul<_Tensor<B>>>::Output>,
{
    let mut axes: [Vec<i64>; 2] = axes.into().into();
    let a_axes_dim = axes[0].len();
    let b_axes_dim = axes[1].len();
    let a_shape = &a.shape();
    let b_shape = &b.shape();
    let a_ndim = a_shape.len();
    let b_ndim = b_shape.len();
    if a_axes_dim != b_axes_dim {
        return Err(TensorError::NdimMismatched(
            a_axes_dim,
            b_axes_dim,
            Location::caller(),
        ));
    } else {
        for i in (0..a_axes_dim).into_iter() {
            if axes[0][i] < 0 {
                axes[0][i] += a_ndim as i64;
                if axes[0][i] < 0 {
                    return Err(TensorError::TensorDotAxesOutOfBounds(
                        0,
                        axes[0][i] as usize,
                        Location::caller(),
                    ));
                }
            }
            if axes[1][i] < 0 {
                axes[1][i] += b_ndim as i64;
                if axes[1][i] < 0 {
                    return Err(TensorError::TensorDotAxesOutOfBounds(
                        1,
                        axes[1][i] as usize,
                        Location::caller(),
                    ));
                }
            }
            if a_shape[axes[0][i] as usize] != b_shape[axes[1][i] as usize] {
                return Err(TensorError::TensorDotDimMismatched(
                    i,
                    a_shape[axes[0][i] as usize] as usize,
                    b_shape[axes[1][i] as usize] as usize,
                    Location::caller(),
                ));
            }
        }
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
    let new_a: _Tensor<A> = a.permute(new_axes_a)?.reshape(new_a_shape)?;
    let new_b: _Tensor<B> = b.permute(new_axes_b)?.reshape(new_b_shape)?;
    let res: <_Tensor<A> as Matmul<_Tensor<B>>>::Output = new_a.matmul(new_b)?;
    olda.extend(&oldb);
    res.reshape(olda)
}
