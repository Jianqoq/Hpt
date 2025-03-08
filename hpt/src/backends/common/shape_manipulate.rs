use std::{marker::PhantomData, panic::Location};

use hpt_allocator::{
    traits::{Allocator, AllocatorOutputRetrive},
    BackendTy, Buffer,
};
use hpt_common::{
    axis::axis::{process_axes, Axis},
    error::{base::TensorError, shape::ShapeError},
    layout::layout::Layout,
    shape::shape::Shape,
    shape::shape_utils::{try_pad_shape, yield_one_after, yield_one_before},
};
use hpt_traits::{ops::slice::Slice, tensor::CommonBounds};

use crate::tensor_base::_Tensor;

#[track_caller]
pub(crate) fn reshape<
    S: Into<Shape>,
    T: Clone,
    B: BackendTy + Buffer + Clone,
    const DEVICE_ID: usize,
    Al,
>(
    a: &_Tensor<T, B, DEVICE_ID, Al>,
    shape: S,
    contiguous: fn(
        &_Tensor<T, B, DEVICE_ID, Al>,
    ) -> std::result::Result<_Tensor<T, B, DEVICE_ID, Al>, TensorError>,
) -> std::result::Result<_Tensor<T, B, DEVICE_ID, Al>, TensorError>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    let shape: Shape = shape.into();
    ShapeError::check_size_match(a.layout.size() as i64, shape.size())?;
    if let Ok(new_layout) = a.layout.inplace_reshape(&shape) {
        Ok(_Tensor {
            data: a.data.clone(),
            parent: a.parent.clone(),
            mem_layout: a.mem_layout.clone(),
            layout: new_layout,
            _backend: a._backend.clone(),
            phantom: PhantomData,
        })
    } else {
        reshape(&contiguous(a)?, shape, contiguous)
    }
}
#[track_caller]
pub(crate) fn squeeze<
    A: Into<Axis>,
    T: Clone,
    B: BackendTy + Buffer + Clone,
    const DEVICE_ID: usize,
    Al,
>(
    a: &_Tensor<T, B, DEVICE_ID, Al>,
    axes: A,
    contiguous: fn(
        &_Tensor<T, B, DEVICE_ID, Al>,
    ) -> std::result::Result<_Tensor<T, B, DEVICE_ID, Al>, TensorError>,
) -> std::result::Result<_Tensor<T, B, DEVICE_ID, Al>, TensorError>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    let axes: Vec<usize> = process_axes(axes, a.layout.ndim())?;
    for i in 0..axes.len() {
        if a.layout.shape()[axes[i]] != 1 {
            return Err(ShapeError::SqueezeError {
                axis: axes[i],
                shape: a.layout.shape().clone(),
                location: Location::caller(),
            }
            .into());
        }
    }
    let new_shape: Vec<i64> = a
        .layout
        .shape()
        .iter()
        .enumerate()
        .filter(|&(i, _)| !axes.contains(&i))
        .map(|(_, &x)| x)
        .collect();
    reshape(&a, new_shape, contiguous)
}
#[track_caller]
pub(crate) fn unsqueeze<
    A: Into<Axis>,
    T: Clone,
    B: BackendTy + Buffer + Clone,
    const DEVICE_ID: usize,
    Al,
>(
    a: &_Tensor<T, B, DEVICE_ID, Al>,
    axes: A,
    contiguous: fn(
        &_Tensor<T, B, DEVICE_ID, Al>,
    ) -> std::result::Result<_Tensor<T, B, DEVICE_ID, Al>, TensorError>,
) -> std::result::Result<_Tensor<T, B, DEVICE_ID, Al>, TensorError>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    let mut res_shape: Vec<i64> = a.layout.shape().to_vec();
    let axes: Axis = axes.into();
    assert_eq!(axes.axes.len(), 1);
    let mut new_axes = Vec::with_capacity(axes.axes.len());
    for i in &axes.axes {
        if *i < 0 {
            let new_i = *i + a.layout.ndim() as i64;
            if new_i < 0 || new_i > a.layout.ndim() as i64 {
                return Err(ShapeError::DimOutOfRange {
                    expected: 0..a.layout.ndim() as i64 + 1,
                    actual: new_i,
                    location: core::panic::Location::caller(),
                }
                .into());
            } else {
                new_axes.push(new_i as usize);
            }
        } else {
            if *i > a.layout.ndim() as i64 {
                return Err(ShapeError::DimOutOfRange {
                    expected: 0..a.layout.ndim() as i64 + 1,
                    actual: *i,
                    location: core::panic::Location::caller(),
                }
                .into());
            } else {
                new_axes.push(*i as usize);
            }
        }
    }
    new_axes.iter().for_each(|&x| {
        res_shape = yield_one_before(&res_shape, x);
    });
    reshape(&a, res_shape, contiguous)
}
#[track_caller]
pub(crate) fn permute<
    A: Into<Axis>,
    T: Clone,
    B: BackendTy + Buffer + Clone,
    const DEVICE_ID: usize,
    Al,
>(
    a: &_Tensor<T, B, DEVICE_ID, Al>,
    axes: A,
    permute_method: fn(&Layout, A) -> std::result::Result<Layout, TensorError>,
) -> std::result::Result<_Tensor<T, B, DEVICE_ID, Al>, TensorError>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    let permuted_layout = permute_method(&a.layout, axes)?;
    Ok(_Tensor {
        data: a.data.clone(),
        layout: permuted_layout,
        parent: a.parent.clone(),
        mem_layout: a.mem_layout.clone(),
        _backend: a._backend.clone(),
        phantom: PhantomData,
    })
}
#[track_caller]
pub(crate) fn expand<
    S: Into<Shape>,
    T: Clone,
    B: BackendTy + Buffer + Clone,
    const DEVICE_ID: usize,
    Al,
>(
    a: &_Tensor<T, B, DEVICE_ID, Al>,
    shape: S,
) -> std::result::Result<_Tensor<T, B, DEVICE_ID, Al>, TensorError>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    let res_shape = Shape::from(shape.into());
    let res_strides = a.layout.expand_strides(&res_shape)?;
    Ok(_Tensor {
        data: a.data.clone(),
        parent: a.parent.clone(),
        mem_layout: a.mem_layout.clone(),
        layout: Layout::new(res_shape, res_strides),
        _backend: a._backend.clone(),
        phantom: PhantomData,
    })
}
#[track_caller]
pub(crate) fn transpose<T: Clone, B: BackendTy + Buffer + Clone, const DEVICE_ID: usize, Al>(
    a: &_Tensor<T, B, DEVICE_ID, Al>,
    axis1: i64,
    axis2: i64,
) -> std::result::Result<_Tensor<T, B, DEVICE_ID, Al>, TensorError>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    ShapeError::check_ndim_enough(
        "transpose expected 2 dimensions.".to_string(),
        2,
        a.layout.ndim(),
    )?;
    permute(a, vec![axis1, axis2], |layout, axes| layout.permute(axes))
}
#[track_caller]
pub(crate) fn t<T: Clone, B: BackendTy + Buffer + Clone, const DEVICE_ID: usize, Al>(
    a: &_Tensor<T, B, DEVICE_ID, Al>,
) -> std::result::Result<_Tensor<T, B, DEVICE_ID, Al>, TensorError>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    if a.layout.ndim() > 2 {
        let mut axes = (0..a.layout.ndim() as i64).collect::<Vec<i64>>();
        axes.swap(a.layout.ndim() - 1, a.layout.ndim() - 2);
        return permute(a, axes, |layout, axes| layout.permute(axes));
    }
    transpose(a, 1, 0)
}
#[track_caller]
pub(crate) fn mt<T: Clone, B: BackendTy + Buffer + Clone, const DEVICE_ID: usize, Al>(
    a: &_Tensor<T, B, DEVICE_ID, Al>,
) -> std::result::Result<_Tensor<T, B, DEVICE_ID, Al>, TensorError>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    permute(
        a,
        (0..a.layout.ndim() as i64).rev().collect::<Vec<i64>>(),
        |layout, axes| layout.permute(axes),
    )
}
#[track_caller]
pub(crate) fn flip<
    A: Into<Axis>,
    T: Clone,
    B: BackendTy + Buffer + Clone,
    const DEVICE_ID: usize,
    Al,
>(
    a: &_Tensor<T, B, DEVICE_ID, Al>,
    axes: A,
) -> std::result::Result<_Tensor<T, B, DEVICE_ID, Al>, TensorError>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    let axes = process_axes(axes, a.layout.ndim())?;
    let mut new_strides = a.layout.strides().to_vec();
    let mut ptr = a.data.clone();
    for &i in axes.iter() {
        ptr.offset(new_strides[i] * (a.layout.shape()[i] - 1));
        new_strides[i] = -new_strides[i];
    }
    if a.parent.is_none() {
        Ok(_Tensor {
            data: ptr,
            parent: Some(a.data.clone()),
            mem_layout: a.mem_layout.clone(),
            layout: Layout::new(a.layout.shape().clone(), new_strides),
            _backend: a._backend.clone(),
            phantom: PhantomData,
        })
    } else {
        Ok(_Tensor {
            data: ptr,
            parent: a.parent.clone(),
            mem_layout: a.mem_layout.clone(),
            layout: Layout::new(a.layout.shape().clone(), new_strides),
            _backend: a._backend.clone(),
            phantom: PhantomData,
        })
    }
}
#[track_caller]
pub(crate) fn fliplr<T: Clone, B: BackendTy + Buffer + Clone, const DEVICE_ID: usize, Al>(
    a: &_Tensor<T, B, DEVICE_ID, Al>,
) -> std::result::Result<_Tensor<T, B, DEVICE_ID, Al>, TensorError>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    ShapeError::check_ndim_enough(
        "fliplr expected 2 dimensions.".to_string(),
        2,
        a.layout.ndim(),
    )?;
    flip(a, 1)
}
#[track_caller]
pub(crate) fn flipud<T: Clone, B: BackendTy + Buffer + Clone, const DEVICE_ID: usize, Al>(
    a: &_Tensor<T, B, DEVICE_ID, Al>,
) -> std::result::Result<_Tensor<T, B, DEVICE_ID, Al>, TensorError>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    ShapeError::check_ndim_enough(
        "flipud expected 2 dimensions.".to_string(),
        2,
        a.layout.ndim(),
    )?;
    flip(a, 0)
}
#[track_caller]
pub(crate) fn tile<
    S: Into<Axis>,
    T: Clone,
    B: BackendTy + Buffer + Clone,
    const DEVICE_ID: usize,
    Al,
>(
    a: &_Tensor<T, B, DEVICE_ID, Al>,
    repeats: S,
    contiguous: fn(
        &_Tensor<T, B, DEVICE_ID, Al>,
    ) -> std::result::Result<_Tensor<T, B, DEVICE_ID, Al>, TensorError>,
) -> std::result::Result<_Tensor<T, B, DEVICE_ID, Al>, TensorError>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    let repeats: Axis = repeats.into();
    ShapeError::check_index_out_of_range((repeats.axes.len() - 1) as i64, a.layout.ndim() as i64)?;
    let repeats: Vec<i64> = repeats
        .axes
        .into_iter()
        .map(|x| x as i64)
        .collect::<Vec<i64>>();
    let final_repeats;
    let mut final_shape;
    if repeats.len() > a.layout.ndim() {
        final_shape = try_pad_shape(a.layout.shape().as_ref(), repeats.len());
        final_repeats = repeats.clone();
    } else {
        final_shape = a.layout.shape().to_vec();
        final_repeats = try_pad_shape(repeats.as_ref(), a.layout.ndim());
    }
    let mut res = reshape(a, &final_shape, contiguous)?;
    let mut cnt = 0;
    for (idx, &i) in final_repeats.iter().enumerate() {
        if i == 1 {
            continue;
        } else {
            let tmp_shape = yield_one_before(res.layout.shape().as_ref(), idx);
            res = reshape(&res, &tmp_shape, contiguous)?;
            res = repeat(&res, i as usize, (idx + cnt) as i16, contiguous)?;
            final_shape[idx] *= i;
            cnt += 1;
        }
    }
    reshape(&res, final_shape, contiguous)
}
#[track_caller]
pub(crate) fn repeat<T: Clone, B: BackendTy + Buffer + Clone, const DEVICE_ID: usize, Al>(
    a: &_Tensor<T, B, DEVICE_ID, Al>,
    repeats: usize,
    axes: i16,
    contiguous: fn(
        &_Tensor<T, B, DEVICE_ID, Al>,
    ) -> std::result::Result<_Tensor<T, B, DEVICE_ID, Al>, TensorError>,
) -> std::result::Result<_Tensor<T, B, DEVICE_ID, Al>, TensorError>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    let mut val: usize = axes as usize;
    if axes < 0 {
        val = a.layout.ndim() + (axes as usize);
    }
    let mut new_shape = yield_one_after(&a.layout.shape(), val);
    let mut new_tensor = reshape(a, &new_shape, contiguous)?;
    new_shape[val + 1] *= repeats as i64;
    new_tensor = expand(&new_tensor, new_shape)?;
    new_shape = a.layout.shape().to_vec();
    new_shape[val] *= repeats as i64;
    Ok(reshape(&contiguous(&new_tensor)?, new_shape, contiguous)?)
}
#[track_caller]
pub(crate) fn split<T: CommonBounds, B: BackendTy + Buffer + Clone, const DEVICE_ID: usize, Al>(
    a: &_Tensor<T, B, DEVICE_ID, Al>,
    indices_or_sections: &[i64],
    axis: i64,
) -> std::result::Result<Vec<_Tensor<T, B, DEVICE_ID, Al>>, TensorError>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    let mut new_axis = axis;
    if axis < 0 {
        new_axis = (a.layout.ndim() as i64) + axis;
    }
    assert!(new_axis >= 0);
    let mut reses = vec![];
    let mut tmp: Vec<(i64, i64, i64)> = Vec::with_capacity(a.layout.ndim());
    for _ in 0..a.layout.ndim() {
        tmp.push((0, 0x7FFFFFFFFFFFFFFF, 1));
    }
    let mut prev = 0;
    for &i in indices_or_sections.iter() {
        tmp[axis as usize] = (prev, i, 1);
        prev = i;
        reses.push(a.slice(&tmp)?);
    }
    let last = *indices_or_sections.last().unwrap();
    tmp[axis as usize] = (last, a.layout.shape()[axis as usize], 1);
    let remain = a.slice(&tmp)?;
    reses.push(remain);
    Ok(reses)
}
#[track_caller]
pub(crate) fn dsplit<T: CommonBounds, B: BackendTy + Buffer + Clone, const DEVICE_ID: usize, Al>(
    a: &_Tensor<T, B, DEVICE_ID, Al>,
    indices: &[i64],
) -> std::result::Result<Vec<_Tensor<T, B, DEVICE_ID, Al>>, TensorError>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    ShapeError::check_ndim_enough(
        "dsplit required input has 3 dimensions.".to_string(),
        3,
        a.layout.ndim(),
    )?;
    split(a, indices, 2)
}
#[track_caller]
pub(crate) fn hsplit<T: CommonBounds, B: BackendTy + Buffer + Clone, const DEVICE_ID: usize, Al>(
    a: &_Tensor<T, B, DEVICE_ID, Al>,
    indices: &[i64],
) -> std::result::Result<Vec<_Tensor<T, B, DEVICE_ID, Al>>, TensorError>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    ShapeError::check_ndim_enough(
        "hsplit required input has 2 dimensions.".to_string(),
        2,
        a.layout.ndim(),
    )?;
    split(a, indices, 1)
}
#[track_caller]
pub(crate) fn vsplit<T: CommonBounds, B: BackendTy + Buffer + Clone, const DEVICE_ID: usize, Al>(
    a: &_Tensor<T, B, DEVICE_ID, Al>,
    indices: &[i64],
) -> std::result::Result<Vec<_Tensor<T, B, DEVICE_ID, Al>>, TensorError>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    ShapeError::check_ndim_enough(
        "vsplit required input has 1 dimension.".to_string(),
        1,
        a.layout.ndim(),
    )?;
    split(a, indices, 0)
}
#[track_caller]
pub(crate) fn swap_axes<
    T: CommonBounds,
    B: BackendTy + Buffer + Clone,
    const DEVICE_ID: usize,
    Al,
>(
    a: &_Tensor<T, B, DEVICE_ID, Al>,
    mut axis1: i64,
    mut axis2: i64,
) -> std::result::Result<_Tensor<T, B, DEVICE_ID, Al>, TensorError>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    if axis1 < 0 {
        axis1 += a.layout.ndim() as i64;
    }
    if axis2 < 0 {
        axis2 += a.layout.ndim() as i64;
    }
    ShapeError::check_index_out_of_range(axis1, a.layout.ndim() as i64)?;
    ShapeError::check_index_out_of_range(axis2, a.layout.ndim() as i64)?;
    let mut new_shape = a.layout.shape().to_vec();
    let mut new_strides = a.layout.strides().to_vec();
    new_shape.swap(axis1 as usize, axis2 as usize);
    new_strides.swap(axis1 as usize, axis2 as usize);
    let layout = Layout::new(new_shape, new_strides);
    Ok(_Tensor {
        data: a.data.clone(),
        layout,
        parent: a.parent.clone(),
        mem_layout: a.mem_layout.clone(),
        _backend: a._backend.clone(),
        phantom: PhantomData,
    })
}
#[track_caller]
pub(crate) fn flatten<
    A,
    T: CommonBounds,
    B: BackendTy + Buffer + Clone,
    const DEVICE_ID: usize,
    Al,
>(
    a: &_Tensor<T, B, DEVICE_ID, Al>,
    start_dim: A,
    end_dim: A,
    contiguous: fn(
        &_Tensor<T, B, DEVICE_ID, Al>,
    ) -> std::result::Result<_Tensor<T, B, DEVICE_ID, Al>, TensorError>,
) -> std::result::Result<_Tensor<T, B, DEVICE_ID, Al>, TensorError>
where
    A: Into<Option<usize>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    let start = start_dim.into().unwrap_or(0);
    let end = end_dim.into().unwrap_or(a.layout.ndim() - 1);
    let shape = a.layout.shape();
    ShapeError::check_index_out_of_range(start as i64, a.layout.ndim() as i64)?;
    ShapeError::check_index_out_of_range(end as i64, a.layout.ndim() as i64)?;
    let flattened_dim = shape[start..=end].iter().product::<i64>();
    let mut new_shape = Vec::new();
    for (i, &dim) in shape.iter().enumerate() {
        if i < start {
            new_shape.push(dim);
        } else if i == start {
            new_shape.push(flattened_dim);
        } else if i > end {
            new_shape.push(dim);
        }
    }
    reshape(a, new_shape, contiguous)
}
