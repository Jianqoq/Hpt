use std::collections::HashMap;

use hpt_common::error::base::TensorError;
use hpt_traits::tensor::TensorInfo;
use hpt_types::dtype::DType;

use super::operators::{
    Binary, Concat, ConstantOfShape, Gather, Lstm, Matmul, Permute, Slice, Squeeze, Unary,
};
use crate::Tensor;

pub(crate) fn shape_fwd<'a>(
    unary: &'a Unary,
    tensors: &mut HashMap<&'a str, Tensor>,
) -> Result<(), TensorError> {
    let inp = tensors[unary.input.as_str()].shape();
    let out = Tensor::from_vec(inp.to_vec(), &[inp.len() as i64])?;
    tensors.insert(unary.output.as_str(), out);
    Ok(())
}

pub(crate) fn gather_fwd<'a>(
    gather: &'a Gather,
    tensors: &mut HashMap<&'a str, Tensor>,
) -> Result<(), TensorError> {
    let inp = &tensors[gather.input.as_str()];
    let indices = &tensors[gather.indices.as_str()];
    let out = inp.gather(indices, gather.axis)?;
    tensors.insert(gather.output.as_str(), out);
    Ok(())
}

pub(crate) fn unsqueeze_fwd<'a>(
    unsqueeze: &'a Squeeze,
    tensors: &mut HashMap<&'a str, Tensor>,
) -> Result<(), TensorError> {
    let inp = &tensors[unsqueeze.input.as_str()];
    let out = inp.unsqueeze(&unsqueeze.axes)?;
    tensors.insert(unsqueeze.output.as_str(), out);
    Ok(())
}

pub(crate) fn squeeze_fwd<'a>(
    squeeze: &'a Squeeze,
    tensors: &mut HashMap<&'a str, Tensor>,
) -> Result<(), TensorError> {
    let inp = &tensors[squeeze.input.as_str()];
    let out = inp.squeeze(&squeeze.axes)?;
    tensors.insert(squeeze.output.as_str(), out);
    Ok(())
}

pub(crate) fn concat_fwd<'a>(
    concat: &'a Concat,
    tensors: &mut HashMap<&'a str, Tensor>,
) -> Result<(), TensorError> {
    let inp = concat
        .inputs
        .iter()
        .map(|s| &tensors[s.as_str()])
        .collect::<Vec<_>>();
    let out = Tensor::concat(inp, concat.axis, false)?;
    tensors.insert(concat.output.as_str(), out);
    Ok(())
}

pub(crate) fn constant_of_shape_fwd<'a>(
    constant_of_shape: &'a ConstantOfShape,
    tensors: &mut HashMap<&'a str, Tensor>,
) -> Result<(), TensorError> {
    let shape = &tensors[constant_of_shape.input.as_str()];
    let shape = unsafe {
        std::slice::from_raw_parts(shape.data.cast::<i64>().ptr, shape.layout.size() as usize)
    };
    let out = Tensor::full(
        shape,
        constant_of_shape.value,
        constant_of_shape.dtype,
        crate::Device::Cpu,
    )?;
    tensors.insert(constant_of_shape.output.as_str(), out);
    Ok(())
}

pub(crate) fn slice_fwd<'a>(
    slice: &'a Slice,
    tensors: &mut HashMap<&'a str, Tensor>,
) -> Result<(), TensorError> {
    let inp = &tensors[slice.input.as_str()];
    let starts = &tensors[slice.starts.as_str()];
    assert_eq!(starts.dtype, DType::I64);
    let starts = unsafe {
        std::slice::from_raw_parts(starts.data.cast::<i64>().ptr, starts.layout.size() as usize)
    };
    let ends = &tensors[slice.ends.as_str()];
    assert_eq!(ends.dtype, DType::I64);
    let ends = unsafe {
        std::slice::from_raw_parts(ends.data.cast::<i64>().ptr, ends.layout.size() as usize)
    };
    let mut selections = vec![(0i64, 0x7FFFFFFFFFFFFFFFi64, 1i64); inp.ndim()];
    if let Some(axes_name) = &slice.axes {
        let axes = &tensors[axes_name.as_str()];
        assert_eq!(axes.dtype, DType::I64);
        let axes = unsafe {
            std::slice::from_raw_parts(axes.data.cast::<i64>().ptr, axes.layout.size() as usize)
        };
        assert_eq!(axes.len(), starts.len());
        assert_eq!(axes.len(), ends.len());
        if let Some(steps_name) = &slice.steps {
            let steps = &tensors[steps_name.as_str()];
            assert_eq!(steps.dtype, DType::I64);
            let steps = unsafe {
                std::slice::from_raw_parts(
                    steps.data.cast::<i64>().ptr,
                    steps.layout.size() as usize,
                )
            };
            assert_eq!(steps.len(), starts.len());
            for (i, &axis) in axes.iter().enumerate() {
                selections[axis as usize] = (starts[i], ends[i], steps[i]);
            }
        } else {
            for (i, &axis) in axes.iter().enumerate() {
                selections[axis as usize] = (starts[i], ends[i], 1);
            }
        }
    } else {
        assert_eq!(starts.len(), ends.len());
        if let Some(steps_name) = &slice.steps {
            let steps = &tensors[steps_name.as_str()];
            assert_eq!(steps.dtype, DType::I64);
            let steps = unsafe {
                std::slice::from_raw_parts(
                    steps.data.cast::<i64>().ptr,
                    steps.layout.size() as usize,
                )
            };
            assert_eq!(steps.len(), starts.len());
            for (i, ((start, end), step)) in
                starts.iter().zip(ends.iter()).zip(steps.iter()).enumerate()
            {
                selections[i] = (*start, *end, *step);
            }
        } else {
            for (i, (start, end)) in starts.iter().zip(ends.iter()).enumerate() {
                selections[i] = (*start, *end, 1);
            }
        }
    }
    let out = inp.slice(&selections)?;
    tensors.insert(slice.output.as_str(), out);
    Ok(())
}

pub(crate) fn transpose_fwd<'a>(
    transpose: &'a Permute,
    tensors: &mut HashMap<&'a str, Tensor>,
) -> Result<(), TensorError> {
    let inp = &tensors[transpose.input.as_str()];
    let out = inp.permute(&transpose.perm)?;
    tensors.insert(transpose.output.as_str(), out);
    Ok(())
}

pub(crate) fn lstm_fwd<'a>(
    lstm: &'a Lstm,
    tensors: &mut HashMap<&'a str, Tensor>,
) -> Result<(), TensorError> {
    let x = &tensors[lstm.x.as_str()];
    let w = &tensors[lstm.w.as_str()];
    let r = &tensors[lstm.r.as_str()];
    let b = if let Some(b) = &lstm.b {
        Some(&tensors[b.as_str()])
    } else {
        None
    };
    let seq_lens = if let Some(sequence_lens) = &lstm.sequence_lens {
        Some(&tensors[sequence_lens.as_str()])
    } else {
        None
    };
    let init_h = if let Some(initial_h) = &lstm.initial_h {
        Some(&tensors[initial_h.as_str()])
    } else {
        None
    };
    let init_c = if let Some(initial_c) = &lstm.initial_c {
        Some(&tensors[initial_c.as_str()])
    } else {
        None
    };
    let p = if let Some(p) = &lstm.p {
        Some(&tensors[p.as_str()])
    } else {
        None
    };
    let (y, y_h, y_c) = x.lstm(
        w,
        r,
        b,
        seq_lens,
        init_h,
        init_c,
        p,
        lstm.direction.as_str(),
    )?;
    if let Some(y_name) = &lstm.y {
        tensors.insert(y_name.as_str(), y);
    }
    if let Some(y_h_name) = &lstm.y_h {
        tensors.insert(y_h_name.as_str(), y_h);
    }
    if let Some(y_c_name) = &lstm.y_c {
        tensors.insert(y_c_name.as_str(), y_c);
    }
    Ok(())
}

pub(crate) fn matmul_fwd<'a>(
    matmul: &'a Matmul,
    tensors: &mut HashMap<&'a str, Tensor>,
) -> Result<(), TensorError> {
    let a = &tensors[matmul.a.as_str()];
    let b = &tensors[matmul.b.as_str()];
    let out = a.matmul(b)?;
    tensors.insert(matmul.output.as_str(), out);
    Ok(())
}

pub(crate) fn add_fwd<'a>(
    add: &'a Binary,
    tensors: &mut HashMap<&'a str, Tensor>,
) -> Result<(), TensorError> {
    let a = &tensors[add.input1.as_str()];
    let b = &tensors[add.input2.as_str()];
    let out = a + b;
    tensors.insert(add.output.as_str(), out);
    Ok(())
}
