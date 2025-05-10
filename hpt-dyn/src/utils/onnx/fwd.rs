use std::collections::HashMap;

use hpt_common::error::base::TensorError;
use hpt_traits::tensor::TensorInfo;
use hpt_types::dtype::DType;

use super::operators::{
    Binary, Concat, ConstantOfShape, Conv2d, Conv2dFused, Elu, Flatten, Gather, Gemm, Lstm, Matmul, Permute, Pooling, Slice, Squeeze, Unary
};
use crate::Tensor;

macro_rules! get {
    ($map:expr, $key:expr) => {
        $map.get($key)
            .unwrap_or_else(|| panic!("key {} not found in map", $key))
    };
}

macro_rules! try_remove_node {
    ($name:expr, $node_degree:expr, $tensors:expr) => {
        if let Some(degree) = $node_degree.get_mut($name) {
            *degree -= 1;
            if $node_degree[$name] == 0 {
                Some($tensors.remove($name).expect("failed to remove node"))
            } else {
                None
            }
        } else {
            None
        }
    };
}

#[inline]
pub(crate) fn shape_fwd<'a>(
    unary: &'a Unary,
    tensors: &mut HashMap<&'a str, Tensor>,
    node_degree: &mut HashMap<&'a str, u32>,
) -> Result<(), TensorError> {
    let inp = tensors[unary.input.as_str()].shape();
    let out = Tensor::from_vec(inp.to_vec(), &[inp.len() as i64])?;
    tensors.insert(unary.output.as_str(), out);
    try_remove_node!(unary.input.as_str(), node_degree, tensors);
    Ok(())
}

#[inline]
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

#[inline]
pub(crate) fn unsqueeze_fwd<'a>(
    unsqueeze: &'a Squeeze,
    tensors: &mut HashMap<&'a str, Tensor>,
) -> Result<(), TensorError> {
    let inp = &tensors[unsqueeze.input.as_str()];
    let out = inp.unsqueeze(&unsqueeze.axes)?;
    tensors.insert(unsqueeze.output.as_str(), out);
    Ok(())
}

#[inline]
pub(crate) fn squeeze_fwd<'a>(
    squeeze: &'a Squeeze,
    tensors: &mut HashMap<&'a str, Tensor>,
) -> Result<(), TensorError> {
    let inp = &tensors[squeeze.input.as_str()];
    let out = inp.squeeze(&squeeze.axes)?;
    tensors.insert(squeeze.output.as_str(), out);
    Ok(())
}

#[inline]
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

#[inline]
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
    node_degree: &mut HashMap<&'a str, u32>,
) -> Result<(), TensorError> {
    let inp = &tensors[transpose.input.as_str()];
    let out = inp.permute(&transpose.perm)?;
    tensors.insert(transpose.output.as_str(), out);
    try_remove_node!(transpose.input.as_str(), node_degree, tensors);
    Ok(())
}

#[inline]
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

#[inline]
pub(crate) fn matmul_fwd<'a>(
    matmul: &'a Matmul,
    tensors: &mut HashMap<&'a str, Tensor>,
    node_degree: &mut HashMap<&'a str, u32>,
) -> Result<(), TensorError> {
    let a = &tensors[matmul.a.as_str()];
    let b = &tensors[matmul.b.as_str()];
    let out = a.matmul(b)?;
    tensors.insert(matmul.output.as_str(), out);
    try_remove_node!(matmul.a.as_str(), node_degree, tensors);
    try_remove_node!(matmul.b.as_str(), node_degree, tensors);
    Ok(())
}

#[inline]
pub(crate) fn gemm_fwd<'a>(
    gemm: &'a Gemm,
    tensors: &mut HashMap<&'a str, Tensor>,
    node_degree: &mut HashMap<&'a str, u32>,
) -> Result<(), TensorError> {
    let a = &tensors[gemm.a.as_str()];
    let b = &tensors[gemm.b.as_str()];
    let a = if gemm.trans_a { a.t()? } else { a.clone() };
    let b = if gemm.trans_b { b.t()? } else { b.clone() };
    let out = if let Some(bias) = &gemm.bias {
        a.gemm(&b, Some(&tensors[bias.as_str()]), gemm.alpha, gemm.beta)?
    } else {
        a.gemm(&b, None, gemm.alpha, gemm.beta)?
    };
    tensors.insert(gemm.output.as_str(), out);
    try_remove_node!(gemm.a.as_str(), node_degree, tensors);
    try_remove_node!(gemm.b.as_str(), node_degree, tensors);
    if let Some(bias) = &gemm.bias {
        try_remove_node!(bias.as_str(), node_degree, tensors);
    }
    Ok(())
}

#[inline]
pub(crate) fn add_fwd<'a>(
    add: &'a Binary,
    tensors: &mut HashMap<&'a str, Tensor>,
    node_degree: &mut HashMap<&'a str, u32>,
) -> Result<(), TensorError> {
    let a_remove = try_remove_node!(add.input1.as_str(), node_degree, tensors);
    let b_remove = try_remove_node!(add.input2.as_str(), node_degree, tensors);
    let out = match (a_remove, b_remove) {
        (None, None) => {
            let a = &tensors[add.input1.as_str()];
            let b = &tensors[add.input2.as_str()];
            a + b
        },
        (None, Some(b)) => {
            let a = &tensors[add.input1.as_str()];
            let broadcast_layout = a.layout.broadcast(&b.shape())?;
            if b.is_contiguous() && b.parent.is_none() && broadcast_layout == b.layout {
                let mut out = b.clone();
                a.add_(&b, &mut out)?;
                out
            } else {
                a + &b
            }
        }
        (Some(a), None) => {
            let b = &tensors[add.input2.as_str()];
            let broadcast_layout = a.layout.broadcast(&b.shape())?;
            if a.is_contiguous() && a.parent.is_none() && broadcast_layout == a.layout {
                let mut out = a.clone();
                a.add_(&b, &mut out)?
            } else {
                &a + b
            }
        }
        (Some(a), Some(b)) => {
            let broadcast_layout = a.layout.broadcast(&b.shape())?;
            if a.is_contiguous() && a.parent.is_none() && broadcast_layout == a.layout {
                let mut out = a.clone();
                a.add_(&b, &mut out)?
            } else if b.is_contiguous() && b.parent.is_none() && broadcast_layout == b.layout {
                let mut out = b.clone();
                a.add_(&b, &mut out)?
            } else {
                a + b
            }
        }
    };
    tensors.insert(add.output.as_str(), out);
    Ok(())
}

#[inline]
pub(crate) fn conv_fwd<'a>(
    conv: &'a Conv2d,
    tensors: &mut HashMap<&'a str, Tensor>,
    node_degree: &mut HashMap<&'a str, u32>,
) -> Result<(), TensorError> {
    let inp = &tensors[conv.input.as_str()];
    let kernel = &tensors[conv.kernel.as_str()];
    let bias = if let Some(bias) = &conv.bias {
        Some(&tensors[bias.as_str()])
    } else {
        None
    };
    let pads = conv.pads;
    let dilations = conv.dilations;
    let steps = conv.strides;
    let group = conv.group;
    let out = if group == 1 {
        inp.conv2d(kernel, bias, steps, pads, dilations)?
    } else {
        inp.conv2d_group(kernel, bias, steps, pads, dilations, group)?
    };
    tensors.insert(conv.output.as_str(), out);
    try_remove_node!(conv.input.as_str(), node_degree, tensors);
    try_remove_node!(conv.kernel.as_str(), node_degree, tensors);
    if let Some(bias) = &conv.bias {
        try_remove_node!(bias.as_str(), node_degree, tensors);
    }
    Ok(())
}

#[inline]
pub(crate) fn conv_fused_fwd<'a>(
    conv: &'a Conv2dFused,
    tensors: &mut HashMap<&'a str, Tensor>,
    node_degree: &mut HashMap<&'a str, u32>,
) -> Result<(), TensorError> {
    let inp = &tensors[conv.input.as_str()];
    let kernel = &tensors[conv.kernel.as_str()];
    let bias = if let Some(bias) = &conv.bias {
        Some(&tensors[bias.as_str()])
    } else {
        None
    };
    let pads = conv.pads;
    let dilations = conv.dilations;
    let steps = conv.strides;
    let group = conv.group;
    let out = if group == 1 {
        use hpt_types::type_promote::NormalOutUnary;
        use hpt_types::type_promote::FloatOutUnary;
        macro_rules! post_conv {
            ($dtype: ty, $activation: ident) => {{
                inp.conv2d_post::<$dtype>(
                    kernel,
                    bias,
                    steps,
                    pads,
                    dilations,
                    Some(|x| x.$activation()),
                    Some(|x| x.$activation()),
                )?
            }};
        }
        macro_rules! arm {
            ($dtype: ty) => {
                match conv.activation {
                    super::operators::ConvActivation::Relu => post_conv!($dtype, _relu),
                    super::operators::ConvActivation::Gelu => post_conv!($dtype, _gelu),
                    super::operators::ConvActivation::Sigmoid => post_conv!($dtype, _sigmoid),
                    super::operators::ConvActivation::Tanh => post_conv!($dtype, _tanh),
                    _ => unimplemented!("conv fused fwd not implemented for {:?}", conv.activation),
                }
            };
        }
        match inp.dtype {
            #[cfg(feature = "bool")]
            DType::Bool => arm!(bool),
            #[cfg(feature = "i8")]
            DType::I8 => arm!(i8),
            #[cfg(feature = "u8")]
            DType::U8 => arm!(u8),
            #[cfg(feature = "i16")]
            DType::I16 => arm!(i16),
            #[cfg(feature = "u16")]
            DType::U16 => arm!(u16),
            #[cfg(feature = "i32")]
            DType::I32 => arm!(i32),
            #[cfg(feature = "u32")]
            DType::U32 => arm!(u32),
            #[cfg(feature = "u64")]
            DType::U64 => arm!(u64),
            #[cfg(feature = "f32")]
            DType::F32 => arm!(f32),
            #[cfg(feature = "f16")]
            DType::F16 => arm!(half::f16),
            #[cfg(feature = "bf16")]
            DType::BF16 => arm!(half::bf16),
            #[cfg(feature = "f64")]
            DType::F64 => arm!(f64),
            _ => unimplemented!("conv fused fwd not implemented for {:?}", inp.dtype),
        }
    } else {
        inp.conv2d_group(kernel, bias, steps, pads, dilations, group)?
    };
    tensors.insert(conv.output.as_str(), out);
    try_remove_node!(conv.input.as_str(), node_degree, tensors);
    try_remove_node!(conv.kernel.as_str(), node_degree, tensors);
    if let Some(bias) = &conv.bias {
        try_remove_node!(bias.as_str(), node_degree, tensors);
    }
    Ok(())
}

#[inline]
pub(crate) fn maxpool_fwd<'a>(
    maxpool: &'a Pooling,
    tensors: &mut HashMap<&'a str, Tensor>,
    node_degree: &mut HashMap<&'a str, u32>,
) -> Result<(), TensorError> {
    let inp = &tensors[maxpool.input.as_str()];
    let out = inp.maxpool2d(
        &maxpool.kernel_shape,
        [maxpool.strides[0], maxpool.strides[1]],
        [
            (maxpool.pads[0], maxpool.pads[0]),
            (maxpool.pads[1], maxpool.pads[1]),
        ],
        [maxpool.dilations[0], maxpool.dilations[1]],
    )?;
    tensors.insert(maxpool.output.as_str(), out);
    try_remove_node!(maxpool.input.as_str(), node_degree, tensors);
    Ok(())
}

#[inline]
pub(crate) fn avgpool_fwd<'a>(
    avgpool: &'a Pooling,
    tensors: &mut HashMap<&'a str, Tensor>,
) -> Result<(), TensorError> {
    let inp = &tensors[avgpool.input.as_str()];
    let out = inp.avgpool2d(
        &avgpool.kernel_shape,
        [avgpool.strides[0], avgpool.strides[1]],
        [
            (avgpool.pads[0], avgpool.pads[0]),
            (avgpool.pads[1], avgpool.pads[1]),
        ],
        [avgpool.dilations[0], avgpool.dilations[1]],
    )?;
    tensors.insert(avgpool.output.as_str(), out);
    Ok(())
}

#[inline]
pub(crate) fn global_avgpool_fwd<'a>(
    global_avgpool: &'a Pooling,
    tensors: &mut HashMap<&'a str, Tensor>,
) -> Result<(), TensorError> {
    let inp = &tensors[global_avgpool.input.as_str()];
    let out = inp.adaptive_avgpool2d([1, 1])?;
    tensors.insert(global_avgpool.output.as_str(), out);
    Ok(())
}

#[inline]
pub(crate) fn global_maxpool_fwd<'a>(
    global_maxpool: &'a Pooling,
    tensors: &mut HashMap<&'a str, Tensor>,
) -> Result<(), TensorError> {
    let inp = &tensors[global_maxpool.input.as_str()];
    let out = inp.adaptive_maxpool2d([1, 1])?;
    tensors.insert(global_maxpool.output.as_str(), out);
    Ok(())
}

#[duplicate::duplicate_item(
    func_name               method                  out_method;
    [relu_fwd]              [relu]                  [relu_];
    [sigmoid_fwd]           [sigmoid]               [sigmoid_];
    [gelu_fwd]              [gelu]                  [gelu_];
    [softplus_fwd]          [softplus]              [softplus_];
    [softsign_fwd]          [softsign]              [softsign_];
    [sin_fwd]               [sin]                   [sin_];
    [cos_fwd]               [cos]                   [cos_];
    [tan_fwd]               [tan]                   [tan_];
    [asin_fwd]              [asin]                  [asin_];
    [acos_fwd]              [acos]                  [acos_];
    [atan_fwd]              [atan]                  [atan_];
    [sinh_fwd]              [sinh]                  [sinh_];
    [cosh_fwd]              [cosh]                  [cosh_];
    [tanh_fwd]              [tanh]                  [tanh_];
    [asinh_fwd]             [asinh]                 [asinh_];
    [acosh_fwd]             [acosh]                 [acosh_];
    [atanh_fwd]             [atanh]                 [atanh_];
    [exp_fwd]               [exp]                   [exp_];
    [abs_fwd]               [abs]                   [abs_];
    [floor_fwd]             [floor]                 [floor_];
    [ln_fwd]                [ln]                    [ln_];
    [sqrt_fwd]              [sqrt]                  [sqrt_];
    [round_fwd]             [round]                 [round_];
    [sign_fwd]              [signum]                [signum_];
    [mish_fwd]              [mish]                  [mish_];
)]
#[inline]
pub(crate) fn func_name<'a>(
    method: &'a Unary,
    tensors: &mut HashMap<&'a str, Tensor>,
    node_degree: &mut HashMap<&'a str, u32>,
) -> Result<(), TensorError> {
    let inp_remove = try_remove_node!(method.input.as_str(), node_degree, tensors);
    let out = if let Some(inp) = inp_remove {
        if inp.is_contiguous() && inp.parent.is_none() {
            inp.out_method(&mut inp.clone())?
        } else {
            inp.method()?
        }
    } else {
        let inp = &tensors[method.input.as_str()];
        inp.relu()?
    };
    tensors.insert(method.output.as_str(), out);
    Ok(())
}

#[inline]
pub(crate) fn selu_fwd<'a>(
    method: &'a Elu,
    tensors: &mut HashMap<&'a str, Tensor>,
    node_degree: &mut HashMap<&'a str, u32>,
) -> Result<(), TensorError> {
    let inp_remove = try_remove_node!(method.input.as_str(), node_degree, tensors);
    let out = if let Some(inp) = inp_remove {
        if inp.is_contiguous() && inp.parent.is_none() {
            inp.selu_(method.alpha, method.gamma, &mut inp.clone())?
        } else {
            inp.selu(method.alpha, method.gamma)?
        }
    } else {
        let inp = &tensors[method.input.as_str()];
        inp.selu(method.alpha, method.gamma)?
    };
    tensors.insert(method.output.as_str(), out);
    Ok(())
}

#[inline]
pub(crate) fn permute_contiguous_fwd<'a>(
    permute_contiguous: &'a Permute,
    tensors: &mut HashMap<&'a str, Tensor>,
) -> Result<(), TensorError> {
    let inp = &tensors[permute_contiguous.input.as_str()];
    let out = inp.permute(&permute_contiguous.perm)?;
    let out = out.contiguous()?;
    tensors.insert(permute_contiguous.output.as_str(), out);
    Ok(())
}

#[inline]
pub(crate) fn identity_fwd<'a>(
    identity: &'a Unary,
    tensors: &mut HashMap<&'a str, Tensor>,
) -> Result<(), TensorError> {
    let inp = get!(tensors, identity.input.as_str());
    tensors.insert(identity.output.as_str(), inp.clone());
    Ok(())
}

#[inline]
pub(crate) fn flatten_fwd<'a>(
    flatten: &'a Flatten,
    tensors: &mut HashMap<&'a str, Tensor>,
    node_degree: &mut HashMap<&'a str, u32>,
) -> Result<(), TensorError> {
    let inp = &tensors[flatten.input.as_str()];
    let out = inp.flatten(flatten.start_dim, (inp.ndim() - 1) as i64)?;
    tensors.insert(flatten.output.as_str(), out);
    try_remove_node!(flatten.input.as_str(), node_degree, tensors);
    Ok(())
}
