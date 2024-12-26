use std::panic::Location;

use tensor_common::{
    axis::{process_axes, Axis},
    err_handler::ErrHandler,
    layout::Layout,
    shape::Shape,
    shape_utils::{try_pad_shape, yield_one_after, yield_one_before},
    slice::Slice,
};
use tensor_traits::CommonBounds;

use crate::{tensor_base::_Tensor, BackendTy, Buffer};

pub(crate) fn reshape<
    S: Into<Shape>,
    T: Clone,
    B: BackendTy + Buffer + Clone,
    const DEVICE_ID: usize,
>(
    a: &_Tensor<T, B, DEVICE_ID>,
    shape: S,
    contiguous: fn(
        &_Tensor<T, B, DEVICE_ID>,
    ) -> std::result::Result<_Tensor<T, B, DEVICE_ID>, ErrHandler>,
) -> std::result::Result<_Tensor<T, B, DEVICE_ID>, ErrHandler> {
    let shape: Shape = shape.into();
    if shape.size() != (a.layout.size() as i64) {
        return Err(ErrHandler::ReshapeError(
            a.layout.shape().clone(),
            shape.clone(),
            a.layout.size() as usize,
            shape.size() as usize,
            Location::caller(),
        )
        .into());
    }
    if let Ok(new_layout) = a.layout.inplace_reshape(&shape) {
        Ok(_Tensor {
            data: a.data.clone(),
            parent: a.parent.clone(),
            mem_layout: a.mem_layout.clone(),
            layout: new_layout,
            _backend: a._backend.clone(),
        })
    } else {
        reshape(&contiguous(a)?, shape, contiguous)
    }
}

pub(crate) fn squeeze<
    A: Into<Axis>,
    T: Clone,
    B: BackendTy + Buffer + Clone,
    const DEVICE_ID: usize,
>(
    a: &_Tensor<T, B, DEVICE_ID>,
    axes: A,
    contiguous: fn(
        &_Tensor<T, B, DEVICE_ID>,
    ) -> std::result::Result<_Tensor<T, B, DEVICE_ID>, ErrHandler>,
) -> std::result::Result<_Tensor<T, B, DEVICE_ID>, ErrHandler> {
    let axes: Vec<usize> = process_axes(axes, a.layout.ndim())?;
    for i in 0..axes.len() {
        if a.layout.shape()[axes[i]] != 1 {
            return Err(ErrHandler::SqueezeError(
                axes[i],
                a.layout.shape().clone(),
                Location::caller(),
            )
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

pub(crate) fn unsqueeze<
    A: Into<Axis>,
    T: Clone,
    B: BackendTy + Buffer + Clone,
    const DEVICE_ID: usize,
>(
    a: &_Tensor<T, B, DEVICE_ID>,
    axes: A,
    contiguous: fn(
        &_Tensor<T, B, DEVICE_ID>,
    ) -> std::result::Result<_Tensor<T, B, DEVICE_ID>, ErrHandler>,
) -> std::result::Result<_Tensor<T, B, DEVICE_ID>, ErrHandler> {
    let mut res_shape: Vec<i64> = a.layout.shape().to_vec();
    let axes: Vec<usize> = process_axes(axes, a.layout.ndim())?;
    axes.iter().for_each(|&x| {
        res_shape = yield_one_before(&res_shape, x);
    });
    reshape(&a, res_shape, contiguous)
}

pub(crate) fn permute<
    A: Into<Axis>,
    T: Clone,
    B: BackendTy + Buffer + Clone,
    const DEVICE_ID: usize,
>(
    a: &_Tensor<T, B, DEVICE_ID>,
    axes: A,
    permute_method: fn(&Layout, A) -> std::result::Result<Layout, ErrHandler>,
) -> std::result::Result<_Tensor<T, B, DEVICE_ID>, ErrHandler> {
    let permuted_layout = permute_method(&a.layout, axes)?;
    Ok(_Tensor {
        data: a.data.clone(),
        layout: permuted_layout,
        parent: a.parent.clone(),
        mem_layout: a.mem_layout.clone(),
        _backend: a._backend.clone(),
    })
}

pub(crate) fn expand<
    S: Into<Shape>,
    T: Clone,
    B: BackendTy + Buffer + Clone,
    const DEVICE_ID: usize,
>(
    a: &_Tensor<T, B, DEVICE_ID>,
    shape: S,
) -> std::result::Result<_Tensor<T, B, DEVICE_ID>, ErrHandler> {
    let res_shape = Shape::from(shape.into());
    let res_strides = a.layout.expand_strides(&res_shape);
    Ok(_Tensor {
        data: a.data.clone(),
        parent: a.parent.clone(),
        mem_layout: a.mem_layout.clone(),
        layout: Layout::new(res_shape, res_strides),
        _backend: a._backend.clone(),
    })
}

pub(crate) fn transpose<T: Clone, B: BackendTy + Buffer + Clone, const DEVICE_ID: usize>(
    a: &_Tensor<T, B, DEVICE_ID>,
    axis1: i64,
    axis2: i64,
) -> std::result::Result<_Tensor<T, B, DEVICE_ID>, ErrHandler> {
    if a.layout.ndim() < 2 {
        Err(ErrHandler::TransposeError(
            a.layout.shape().clone(),
            a.layout.ndim(),
            Location::caller(),
        )
        .into())
    } else {
        permute(a, vec![axis1, axis2], |layout, axes| layout.permute(axes))
    }
}

pub(crate) fn t<T: Clone, B: BackendTy + Buffer + Clone, const DEVICE_ID: usize>(
    a: &_Tensor<T, B, DEVICE_ID>,
) -> std::result::Result<_Tensor<T, B, DEVICE_ID>, ErrHandler> {
    if a.layout.ndim() > 2 {
        let mut axes = (0..a.layout.ndim() as i64).collect::<Vec<i64>>();
        axes.swap(a.layout.ndim() - 1, a.layout.ndim() - 2);
        return permute(a, axes, |layout, axes| layout.permute(axes));
    }
    transpose(a, 1, 0)
}

pub(crate) fn mt<T: Clone, B: BackendTy + Buffer + Clone, const DEVICE_ID: usize>(
    a: &_Tensor<T, B, DEVICE_ID>,
) -> std::result::Result<_Tensor<T, B, DEVICE_ID>, ErrHandler> {
    permute(
        a,
        (0..a.layout.ndim() as i64).rev().collect::<Vec<i64>>(),
        |layout, axes| layout.permute(axes),
    )
}

pub(crate) fn flip<
    A: Into<Axis>,
    T: Clone,
    B: BackendTy + Buffer + Clone,
    const DEVICE_ID: usize,
>(
    a: &_Tensor<T, B, DEVICE_ID>,
    axes: A,
) -> std::result::Result<_Tensor<T, B, DEVICE_ID>, ErrHandler> {
    let axes = process_axes(axes, a.layout.ndim())?;
    let mut new_strides = a.layout.strides().to_vec();
    let mut ptr = a.data.clone();
    for &i in axes.iter() {
        new_strides[i] = -new_strides[i];
        ptr.offset(a.layout.strides()[i]);
    }
    if a.parent.is_none() {
        Ok(_Tensor {
            data: ptr,
            parent: Some(a.data.clone()),
            mem_layout: a.mem_layout.clone(),
            layout: Layout::new(a.layout.shape().clone(), new_strides),
            _backend: a._backend.clone(),
        })
    } else {
        Ok(_Tensor {
            data: ptr,
            parent: a.parent.clone(),
            mem_layout: a.mem_layout.clone(),
            layout: Layout::new(a.layout.shape().clone(), new_strides),
            _backend: a._backend.clone(),
        })
    }
}

pub(crate) fn fliplr<T: Clone, B: BackendTy + Buffer + Clone, const DEVICE_ID: usize>(
    a: &_Tensor<T, B, DEVICE_ID>,
) -> std::result::Result<_Tensor<T, B, DEVICE_ID>, ErrHandler> {
    if a.layout.ndim() < 2 {
        return Err(ErrHandler::NdimNotEnough(2, a.layout.ndim(), Location::caller()).into());
    }
    flip(a, 1)
}

pub(crate) fn flipud<T: Clone, B: BackendTy + Buffer + Clone, const DEVICE_ID: usize>(
    a: &_Tensor<T, B, DEVICE_ID>,
) -> std::result::Result<_Tensor<T, B, DEVICE_ID>, ErrHandler> {
    if a.layout.ndim() < 1 {
        return Err(ErrHandler::NdimNotEnough(1, a.layout.ndim(), Location::caller()).into());
    }
    flip(a, 0)
}

pub(crate) fn tile<
    S: Into<Axis>,
    T: Clone,
    B: BackendTy + Buffer + Clone,
    const DEVICE_ID: usize,
>(
    a: &_Tensor<T, B, DEVICE_ID>,
    repeats: S,
    contiguous: fn(
        &_Tensor<T, B, DEVICE_ID>,
    ) -> std::result::Result<_Tensor<T, B, DEVICE_ID>, ErrHandler>,
) -> std::result::Result<_Tensor<T, B, DEVICE_ID>, ErrHandler> {
    let repeats: Axis = repeats.into();
    ErrHandler::check_index_in_range(a.layout.ndim(), (repeats.axes.len() - 1) as i64)?;
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

pub(crate) fn repeat<T: Clone, B: BackendTy + Buffer + Clone, const DEVICE_ID: usize>(
    a: &_Tensor<T, B, DEVICE_ID>,
    repeats: usize,
    axes: i16,
    contiguous: fn(
        &_Tensor<T, B, DEVICE_ID>,
    ) -> std::result::Result<_Tensor<T, B, DEVICE_ID>, ErrHandler>,
) -> std::result::Result<_Tensor<T, B, DEVICE_ID>, ErrHandler> {
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

pub(crate) fn split<T: CommonBounds, B: BackendTy + Buffer + Clone, const DEVICE_ID: usize>(
    a: &_Tensor<T, B, DEVICE_ID>,
    indices_or_sections: &[i64],
    axis: i64,
) -> std::result::Result<Vec<_Tensor<T, B, DEVICE_ID>>, ErrHandler> {
    let mut new_axis = axis;
    if axis < 0 {
        new_axis = (a.layout.ndim() as i64) + axis;
    }
    assert!(new_axis >= 0);
    let mut reses = vec![];
    let mut tmp: Vec<Slice> = Vec::with_capacity(a.layout.ndim());
    for _ in 0..a.layout.ndim() {
        tmp.push(Slice::Full);
    }
    let mut prev = 0;
    for &i in indices_or_sections.iter() {
        tmp[axis as usize] = Slice::Range((prev, i));
        prev = i;
        reses.push(a.slice(&tmp)?);
    }
    let last = *indices_or_sections.last().unwrap();
    tmp[axis as usize] = Slice::Range((last, a.layout.shape()[axis as usize]));
    let remain = a.slice(&tmp)?;
    reses.push(remain);
    Ok(reses)
}

pub(crate) fn dsplit<T: CommonBounds, B: BackendTy + Buffer + Clone, const DEVICE_ID: usize>(
    a: &_Tensor<T, B, DEVICE_ID>,
    indices: &[i64],
) -> std::result::Result<Vec<_Tensor<T, B, DEVICE_ID>>, ErrHandler> {
    if a.layout.shape().len() < 3 {
        return Err(ErrHandler::NdimNotEnough(3, a.layout.ndim(), Location::caller()).into());
    }
    split(a, indices, 2)
}

pub(crate) fn hsplit<T: CommonBounds, B: BackendTy + Buffer + Clone, const DEVICE_ID: usize>(
    a: &_Tensor<T, B, DEVICE_ID>,
    indices: &[i64],
) -> std::result::Result<Vec<_Tensor<T, B, DEVICE_ID>>, ErrHandler> {
    if a.layout.shape().len() < 2 {
        return Err(ErrHandler::NdimNotEnough(2, a.layout.ndim(), Location::caller()).into());
    }
    split(a, indices, 1)
}

pub(crate) fn vsplit<T: CommonBounds, B: BackendTy + Buffer + Clone, const DEVICE_ID: usize>(
    a: &_Tensor<T, B, DEVICE_ID>,
    indices: &[i64],
) -> std::result::Result<Vec<_Tensor<T, B, DEVICE_ID>>, ErrHandler> {
    if a.layout.shape().len() < 1 {
        return Err(ErrHandler::NdimNotEnough(1, a.layout.ndim(), Location::caller()).into());
    }
    split(a, indices, 0)
}

pub(crate) fn swap_axes<T: CommonBounds, B: BackendTy + Buffer + Clone, const DEVICE_ID: usize>(
    a: &_Tensor<T, B, DEVICE_ID>,
    mut axis1: i64,
    mut axis2: i64,
) -> std::result::Result<_Tensor<T, B, DEVICE_ID>, ErrHandler> {
    ErrHandler::check_index_in_range_mut(a.layout.ndim(), &mut axis1)?;
    ErrHandler::check_index_in_range_mut(a.layout.ndim(), &mut axis2)?;
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
    })
}

pub(crate) fn flatten<A, T: CommonBounds, B: BackendTy + Buffer + Clone, const DEVICE_ID: usize>(
    a: &_Tensor<T, B, DEVICE_ID>,
    start_dim: A,
    end_dim: A,
    contiguous: fn(
        &_Tensor<T, B, DEVICE_ID>,
    ) -> std::result::Result<_Tensor<T, B, DEVICE_ID>, ErrHandler>,
) -> std::result::Result<_Tensor<T, B, DEVICE_ID>, ErrHandler>
where
    A: Into<Option<usize>>,
{
    let start = start_dim.into().unwrap_or(0);
    let end = end_dim.into().unwrap_or(a.layout.ndim() - 1);
    let shape = a.layout.shape();
    ErrHandler::check_index_in_range(a.layout.ndim(), start as i64)?;
    ErrHandler::check_index_in_range(a.layout.ndim(), end as i64)?;
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
