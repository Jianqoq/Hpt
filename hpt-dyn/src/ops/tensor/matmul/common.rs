use hpt_common::{
    error::{base::TensorError, shape::ShapeError},
    layout::layout::Layout,
    shape::{
        shape::Shape,
        shape_utils::{compare_and_pad_shapes, predict_broadcast_shape},
    },
};
use hpt_traits::tensor::TensorInfo;
use hpt_types::dtype::DType;

use crate::tensor::Tensor;

#[inline(always)]
#[track_caller]
pub(crate) fn matmul_prepare(
    device: &crate::Device,
    lhs: &Layout,
    lhs_dtype: DType,
    rhs: &Layout,
    rhs_dtype: DType,
    out: Option<Tensor>,
) -> Result<Tensor, TensorError> {
    assert_eq!(lhs_dtype, rhs_dtype);
    if lhs.shape().len() == 2 && rhs.shape().len() == 2 {
        ShapeError::check_matmul(lhs.shape(), rhs.shape())?;
        let res = if let Some(out) = out {
            assert_eq!(out.dtype, lhs_dtype);
            ShapeError::check_inplace_out_layout_valid(
                &Shape::from([lhs.shape()[0], rhs.shape()[1]]),
                &out.layout(),
            )?;
            Ok(out)
        } else {
            Tensor::empty(&[lhs.shape()[0], rhs.shape()[1]], lhs_dtype, device.clone())
        };
        res
    } else {
        let (longer_shape, padded_short_shape) = compare_and_pad_shapes(&lhs.shape(), &rhs.shape());
        let a_shape;
        let b_shape;
        if lhs.shape().len() > rhs.shape().len() {
            a_shape = longer_shape;
            b_shape = padded_short_shape;
        } else {
            a_shape = padded_short_shape;
            b_shape = longer_shape;
        }
        ShapeError::check_matmul(lhs.shape(), rhs.shape())?;
        let mut res_shape =
            predict_broadcast_shape(&a_shape[..a_shape.len() - 2], &b_shape[..b_shape.len() - 2])?
                .to_vec();
        res_shape.push(a_shape[a_shape.len() - 2]);
        res_shape.push(b_shape[b_shape.len() - 1]);
        let res = if let Some(out) = out {
            ShapeError::check_inplace_out_layout_valid(&Shape::from(&res_shape), &out.layout())?;
            Ok(out)
        } else {
            Tensor::empty(&res_shape, lhs_dtype, device.clone())
        };
        res
    }
}

pub(crate) fn check_out_layout(
    lhs: &Layout,
    rhs: &Layout,
    out: &Layout,
) -> Result<(), TensorError> {
    if lhs.shape().len() == 2 && rhs.shape().len() == 2 {
        ShapeError::check_matmul(lhs.shape(), rhs.shape())?;
        ShapeError::check_inplace_out_layout_valid(
            &Shape::from([lhs.shape()[0], rhs.shape()[1]]),
            &out,
        )?;
        Ok(())
    } else {
        let (longer_shape, padded_short_shape) = compare_and_pad_shapes(&lhs.shape(), &rhs.shape());
        let a_shape;
        let b_shape;
        if lhs.shape().len() > rhs.shape().len() {
            a_shape = longer_shape;
            b_shape = padded_short_shape;
        } else {
            a_shape = padded_short_shape;
            b_shape = longer_shape;
        }
        ShapeError::check_matmul(lhs.shape(), rhs.shape())?;
        let mut res_shape =
            predict_broadcast_shape(&a_shape[..a_shape.len() - 2], &b_shape[..b_shape.len() - 2])?
                .to_vec();
        res_shape.push(a_shape[a_shape.len() - 2]);
        res_shape.push(b_shape[b_shape.len() - 1]);
        ShapeError::check_inplace_out_layout_valid(&Shape::from(&res_shape), &out)?;
        Ok(())
    }
}
