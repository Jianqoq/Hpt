#![allow(unused)]

use std::sync::Arc;

use hpt_common::{Pointer, layout::layout::Layout};

pub(crate) fn dispatch_loop_progress_update(
    layout: &Layout,
) -> Arc<dyn Fn(&mut [i64]) -> i64 + Send + Sync> {
    let shape = layout.shape();
    let strides = layout.strides();
    match layout.ndim() {
        2 => {
            let shape = [shape[0] - 1];
            let strides = [strides[0]];
            Arc::new(move |prg| update_2d(prg, shape, strides))
        }
        3 => {
            let shape = [shape[0] - 1, shape[1] - 1];
            let strides = [strides[0], strides[1]];
            Arc::new(move |prg| update_3d(prg, shape, strides))
        }
        4 => {
            let shape = [shape[0] - 1, shape[1] - 1, shape[2] - 1];
            let strides = [
                strides[0],
                strides[1],
                strides[2],
            ];
            Arc::new(move |prg| update_4d(prg, shape, strides))
        }
        _ => {
            let mut shape = shape.to_vec();
            let mut strides = strides.to_vec();
            shape.iter_mut().for_each(|s| *s -= 1);
            Arc::new(move |prg| updata_nd(prg, &shape, &strides))
        }
    }
}

pub(crate) fn dispatch_map_global_idx(layout: &Layout) -> Arc<dyn Fn(i64) -> i64 + Send + Sync> {
    let shape = layout.shape();
    let strides = layout.strides();
    match layout.ndim() {
        1 => {
            let shape = shape[0];
            let strides = strides[0];
            Arc::new(move |global_idx| map_global_1d(shape, strides, global_idx))
        }
        2 => {
            let shape = [shape[0], shape[1]];
            let strides = [strides[0], strides[1]];
            Arc::new(move |global_idx| map_global_2d(shape, strides, global_idx))
        }
        3 => {
            let shape = [shape[0], shape[1], shape[2]];
            let strides = [strides[0], strides[1], strides[2]];
            Arc::new(move |global_idx| map_global_3d(shape, strides, global_idx))
        }
        4 => {
            let shape = [shape[0], shape[1], shape[2], shape[3]];
            let strides = [strides[0], strides[1], strides[2], strides[3]];
            Arc::new(move |global_idx| map_global_4d(shape, strides, global_idx))
        }
        _ => {
            let mut shape = shape.clone();
            let mut strides = strides.clone();
            Arc::new(move |global_idx| map_global_idx(&shape, &strides, global_idx))
        }
    }
}

pub(crate) fn dispatch_map_gp(layout: &Layout) -> Arc<dyn Fn(i64) -> (i64, Vec<i64>) + Send + Sync> {
    let shape = layout.shape();
    let strides = layout.strides();
    match layout.ndim() {
        1 => {
            let shape = shape[0];
            let strides = strides[0];
            Arc::new(move |global_idx| map_gp_1d(shape, strides, global_idx))
        }
        2 => {
            let shape = [shape[0], shape[1]];
            let strides = [strides[0], strides[1]];
            Arc::new(move |global_idx| map_gp_2d(shape, strides, global_idx))
        }
        3 => {
            let shape = [shape[0], shape[1], shape[2]];
            let strides = [strides[0], strides[1], strides[2]];
            Arc::new(move |global_idx| map_gp_3d(shape, strides, global_idx))
        }
        4 => {
            let shape = [shape[0], shape[1], shape[2], shape[3]];
            let strides = [strides[0], strides[1], strides[2], strides[3]];
            Arc::new(move |global_idx| map_gp_4d(shape, strides, global_idx))
        }
        _ => {
            let mut shape = shape.clone();
            let mut strides = strides.clone();
            Arc::new(move |global_idx| map_gp_nd(&shape, &strides, global_idx))
        }
    }
}

pub(crate) fn update_2d(
    prg: &mut [i64],
    shape: [i64; 1],
    strides: [i64; 1],
) -> i64 {
    assert_eq!(prg.len(), 2);

    let mut offset = 0;
    if prg[0] < shape[0] {
        prg[0] += 1;
        offset += strides[0];
    } else {
        prg[0] = 0;
        offset += -strides[0] * shape[0];
    }
    offset
}

pub(crate) fn update_3d(
    prg: &mut [i64],
    shape: [i64; 2],
    strides: [i64; 2],
) -> i64 {
    assert_eq!(prg.len(), 3);

    let mut offset = 0;
    if prg[1] < shape[1] {
        prg[1] += 1;
        offset += strides[1];
        return offset;
    } else {
        prg[1] = 0;
        offset += -strides[1] * shape[1];
    }

    if prg[0] < shape[0] {
        prg[0] += 1;
        offset += strides[0];
    } else {
        prg[0] = 0;
        offset += -strides[0] * shape[0];
    }
    offset
}

pub(crate) fn update_4d(
    prg: &mut [i64],
    shape: [i64; 3],
    strides: [i64; 3],
) -> i64 {
    assert_eq!(prg.len(), 4);

    let mut offset = 0;
    if prg[2] < shape[2] {
        prg[2] += 1;
        offset += strides[2];
        return offset;
    } else {
        prg[2] = 0;
        offset += -strides[2] * shape[2];
    }

    if prg[1] < shape[1] {
        prg[1] += 1;
        offset += strides[1];
        return offset;
    } else {
        prg[1] = 0;
        offset += -strides[1] * shape[1];
    }

    if prg[0] < shape[0] {
        prg[0] += 1;
        offset += strides[0];
    } else {
        prg[0] = 0;
        offset += -strides[0] * shape[0];
    }
    offset
}

pub(crate) fn updata_nd(
    prg: &mut [i64],
    shape: &[i64],
    strides: &[i64],
) -> i64 {
    assert_eq!(prg.len(), shape.len());
    assert_eq!(strides.len(), shape.len());
    let mut offset = 0;
    for j in (0..(shape.len() as i64) - 1).rev() {
        let j = j as usize;
        if prg[j] < shape[j] {
            prg[j] += 1;
            offset += strides[j];
            break;
        } else {
            prg[j] = 0;
            offset += -strides[j] * shape[j];
        }
    }
    offset
}

pub(crate) fn cal_idx_2d(prg: &[i64], strides: [i64; 2]) -> i64 {
    #[cfg(feature = "bound_check")]
    {
        prg[0] * strides[0] + prg[1] * strides[1]
    }
    #[cfg(not(feature = "bound_check"))]
    unsafe {
        *prg.get_unchecked(0) * strides[0] + *prg.get_unchecked(1) * strides[1]
    }
}

pub(crate) fn cal_idx_3d(prg: &[i64], strides: [i64; 3]) -> i64 {
    #[cfg(feature = "bound_check")]
    {
        prg[0] * strides[0] + prg[1] * strides[1] + prg[2] * strides[2]
    }
    #[cfg(not(feature = "bound_check"))]
    unsafe {
        *prg.get_unchecked(0) * strides[0]
            + *prg.get_unchecked(1) * strides[1]
            + *prg.get_unchecked(2) * strides[2]
    }
}

pub(crate) fn cal_idx_4d(prg: &[i64], strides: [i64; 4]) -> i64 {
    #[cfg(feature = "bound_check")]
    {
        prg[0] * strides[0] + prg[1] * strides[1] + prg[2] * strides[2] + prg[3] * strides[3]
    }
    #[cfg(not(feature = "bound_check"))]
    unsafe {
        *prg.get_unchecked(0) * strides[0]
            + *prg.get_unchecked(1) * strides[1]
            + *prg.get_unchecked(2) * strides[2]
            + *prg.get_unchecked(3) * strides[3]
    }
}

pub(crate) fn cal_idx_nd(prg: &[i64], strides: &[i64]) -> i64 {
    assert_eq!(prg.len(), strides.len());
    #[cfg(feature = "bound_check")]
    {
        prg.iter().zip(strides.iter()).map(|(p, s)| p * s).sum()
    }
    #[cfg(not(feature = "bound_check"))]
    unsafe {
        prg.iter().zip(strides.iter()).map(|(p, s)| *p * *s).sum()
    }
}

pub(crate) fn map_global_1d(shape: i64, strides: i64, global_idx: i64) -> i64 {
    (global_idx % shape) * strides
}

pub(crate) fn map_global_2d(shape: [i64; 2], strides: [i64; 2], mut global_idx: i64) -> i64 {
    let mut ret = 0;
    ret += (global_idx % shape[1]) * strides[1];
    global_idx /= shape[1];
    ret += (global_idx % shape[0]) * strides[0];
    ret
}

pub(crate) fn map_global_3d(shape: [i64; 3], strides: [i64; 3], mut global_idx: i64) -> i64 {
    let mut ret = 0;
    ret += (global_idx % shape[2]) * strides[2];
    global_idx /= shape[2];
    ret += (global_idx % shape[1]) * strides[1];
    global_idx /= shape[1];
    ret += (global_idx % shape[0]) * strides[0];
    ret
}

pub(crate) fn map_global_4d(shape: [i64; 4], strides: [i64; 4], mut global_idx: i64) -> i64 {
    let mut global_idx = global_idx as i64;
    let mut ret = 0;
    ret += (global_idx % shape[3]) * strides[3];
    global_idx /= shape[3];
    ret += (global_idx % shape[2]) * strides[2];
    global_idx /= shape[2];
    ret += (global_idx % shape[1]) * strides[1];
    global_idx /= shape[1];
    ret += (global_idx % shape[0]) * strides[0];
    ret
}

pub(crate) fn map_global_idx(shapes: &[i64], strides: &[i64], mut global_idx: i64) -> i64 {
    #[cfg(feature = "bound_check")]
    {
        assert_eq!(strides.len(), shapes.len());
    }
    let mut ret = 0;
    shapes
        .iter()
        .rev()
        .zip(strides.iter().rev())
        .for_each(|(dim, stride)| {
            ret += (global_idx % dim) * stride;
            global_idx /= dim;
        });
    ret
}

pub(crate) fn map_gp_1d(shape: i64, strides: i64, global_idx: i64) -> (i64, Vec<i64>) {
    let rem = global_idx % shape;
    (rem * strides, vec![rem])
}

pub(crate) fn map_gp_2d(
    shape: [i64; 2],
    strides: [i64; 2],
    mut global_idx: i64,
) -> (i64, Vec<i64>) {
    let mut vec = [0, 0];
    let mut ret = 0;
    let mut rem = global_idx % shape[1];
    ret += rem * strides[1];
    global_idx /= shape[1];
    vec[1] = rem;
    rem = global_idx % shape[0];
    ret += rem * strides[0];
    vec[0] = rem;
    (ret, vec.to_vec())
}

pub(crate) fn map_gp_3d(
    shape: [i64; 3],
    strides: [i64; 3],
    mut global_idx: i64,
) -> (i64, Vec<i64>) {
    let mut vec = [0, 0, 0];
    let mut ret = 0;
    let mut rem = global_idx % shape[2];
    ret += rem * strides[2];
    global_idx /= shape[2];
    vec[2] = rem;
    rem = global_idx % shape[1];
    ret += rem * strides[1];
    global_idx /= shape[1];
    vec[1] = rem;
    rem = global_idx % shape[0];
    ret += rem * strides[0];
    vec[0] = rem;
    (ret, vec.to_vec())
}

pub(crate) fn map_gp_4d(
    shape: [i64; 4],
    strides: [i64; 4],
    mut global_idx: i64,
) -> (i64, Vec<i64>) {
    let mut vec = [0, 0, 0, 0];
    let mut ret = 0;
    let mut rem = global_idx % shape[3];
    ret += rem * strides[3];
    global_idx /= shape[3];
    vec[3] = rem;
    rem = global_idx % shape[2];
    ret += rem * strides[2];
    global_idx /= shape[2];
    vec[2] = rem;
    rem = global_idx % shape[1];
    ret += rem * strides[1];
    global_idx /= shape[1];
    vec[1] = rem;
    rem = global_idx % shape[0];
    ret += rem * strides[0];
    vec[0] = rem;
    (ret, vec.to_vec())
}

pub(crate) fn map_gp_nd(shapes: &[i64], strides: &[i64], mut global_idx: i64) -> (i64, Vec<i64>) {
    #[cfg(feature = "bound_check")]
    {
        assert_eq!(strides.len(), shapes.len());
    }
    let mut vec = Vec::with_capacity(shapes.len());
    let mut ret = 0;
    shapes
        .iter()
        .rev()
        .zip(strides.iter().rev())
        .for_each(|(dim, stride)| {
            let rem = global_idx % dim;
            ret += rem * stride;
            vec.push(rem);
            global_idx /= dim;
        });
    (ret, vec)
}
