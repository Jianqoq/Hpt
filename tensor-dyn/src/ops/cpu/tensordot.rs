use crate::tensor_base::_Tensor;
use tensor_traits::ops::binary::Matmul;
use tensor_common::tensordot_args::TensorDotArgs;
use tensor_types::type_promote::NormalOut;
use tensor_traits::tensor::CommonBounds;
use tensor_traits::shape_manipulate::ShapeManipulate;
use tensor_traits::tensor::TensorInfo;

macro_rules! impl_tensordot {
    ($tensor_type:ident, $func_name:ident) => {
        pub(crate) fn $func_name<A, B, G, const N: usize>(
            a: &$tensor_type<A>,
            b: &$tensor_type<B>,
            axes: G
        )
            -> anyhow::Result<<$tensor_type<A> as Matmul<$tensor_type<B>>>::Output>
            where
                A: CommonBounds + NormalOut<B>,
                B: CommonBounds,
                <A as NormalOut<B>>::Output: CommonBounds,
                G: Into<TensorDotArgs<N>>,
                $tensor_type<A>: Matmul<$tensor_type<B>>,
                <$tensor_type<A> as Matmul<$tensor_type<B>>>::Output: ShapeManipulate
        {
            let mut axes: [Vec<i64>; 2] = axes.into().into();
            let a_axes_dim = axes[0].len();
            let b_axes_dim = axes[1].len();
            let a_shape = &a.shape();
            let b_shape = &b.shape();
            let a_ndim = a_shape.len();
            let b_ndim = b_shape.len();
            let mut equal = true;
            if a_axes_dim != b_axes_dim {
                equal = false;
            } else {
                for i in (0..a_axes_dim).into_iter() {
                    if axes[0][i] < 0 {
                        axes[0][i] += a_ndim as i64;
                        if axes[0][i] < 0 {
                            anyhow::bail!("axes[0][{}] out of bounds", i);
                        }
                    }
                    if axes[1][i] < 0 {
                        axes[1][i] += b_ndim as i64;
                        if axes[1][i] < 0 {
                            anyhow::bail!("axes[1][{}] out of bounds", i);
                        }
                    }
                    if a_shape[axes[0][i] as usize] != b_shape[axes[1][i] as usize] {
                        equal = false;
                        break;
                    }
                }
            }
            if !equal {
                anyhow::bail!("shape-mismatch for sum");
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
        
            let new_a: $tensor_type<A> = a.permute(new_axes_a)?.reshape(new_a_shape)?;
            let new_b: $tensor_type<B> = b.permute(new_axes_b)?.reshape(new_b_shape)?;
            let res: <$tensor_type<A> as Matmul<$tensor_type<B>>>::Output = new_a.matmul(new_b)?;
            olda.extend(&oldb);
            res.reshape(olda)
        }
    };
}

impl_tensordot!(_Tensor, _tensordot);

