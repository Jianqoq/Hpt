use hpt_common::{
    error::{base::TensorError, shape::ShapeError},
    shape::shape::Shape,
};
use hpt_traits::{ops::creation::TensorCreator, tensor::CommonBounds};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::tensor_base::_Tensor;
use hpt_types::REGNUM;
use hpt_allocator::{
    traits::{Allocator, AllocatorOutputRetrive},
    Cpu,
};
use hpt_traits::tensor::TensorInfo;
use hpt_types::{into_scalar::Cast, traits::VecTrait};

#[track_caller]
pub(crate) fn pooling_template<T: CommonBounds, O: CommonBounds, const DEVICE: usize, A>(
    img: &_Tensor<T, Cpu, DEVICE, A>,
    kernels_shape: &Shape,
    steps: [i64; 2],
    padding: [(i64, i64); 2],
    dilation: [i64; 2],
    scalar_op: impl Fn(T, T) -> T + Send + Sync,
    vec_op: impl Fn(T::Vec, T::Vec) -> T::Vec + Send + Sync,
    post_scalar_op: impl Fn(T) -> O + Send + Sync,
    post_vec_op: impl Fn(T::Vec) -> O::Vec + Send + Sync,
) -> Result<_Tensor<O, Cpu, DEVICE, A>, TensorError>
where
    A: Allocator + Send + Sync,
    A::Output: AllocatorOutputRetrive,
{
    ShapeError::check_contiguous(
        "pooling input must be contiguous".to_string(),
        &img.layout(),
    )?;
    let img_shape = img.shape();
    ShapeError::check_dim(4, img_shape.len())?;
    let batch = img_shape[0];
    let img_height = img_shape[1];
    let img_width = img_shape[2];
    let in_channels = img_shape[3];
    let kernel_height = kernels_shape[0];
    let kernel_width = kernels_shape[1];
    let (step_width, step_height) = (steps[0], steps[1]);
    let ((ph_start, ph_end), (pw_start, pw_end)) = (padding[0], padding[1]);
    let (dh, dw) = (dilation[0], dilation[1]);

    let out_height =
        (img_height + ph_start + ph_end - dh * (kernel_height - 1) - 1) / step_height + 1;
    let out_width = (img_width + pw_start + pw_end - dw * (kernel_width - 1) - 1) / step_width + 1;
    let img = img.clone();
    if out_height <= 0 || out_width <= 0 {
        return Err((ShapeError::ConvError {
            message: if out_height <= 0 {
                "output height <= 0".to_string()
            } else {
                "output width <= 0".to_string()
            },
            location: core::panic::Location::caller(),
        })
        .into());
    }
    let output = _Tensor::<O, Cpu, DEVICE, A>::empty([batch, out_height, out_width, in_channels])?;
    let out = output.ptr::<O>();
    let inp = img.ptr::<T>();

    let osb = output.strides()[0]; // batch
    let osh = output.strides()[1]; // height
    let osw = output.strides()[2]; // width

    let isb = img.strides()[0]; // batch
    let ish = img.strides()[1]; // height
    let isw = img.strides()[2]; // width

    let out_size = batch * out_height * out_width;

    const IC_BLOCK_SIZE: usize = REGNUM / 2;
    let in_channel_remain = in_channels % ((IC_BLOCK_SIZE * T::Vec::SIZE) as i64);
    let same_vec_size = T::Vec::SIZE == O::Vec::SIZE;
    (0..out_size).into_par_iter().for_each(|idx| {
        let out = out.clone();
        let b = idx / (out_height * out_width);
        let h = (idx / out_width) % out_height;
        let w = idx % out_width;

        for ii in (0..in_channels - in_channel_remain).step_by(IC_BLOCK_SIZE * T::Vec::SIZE) {
            let mut res_vecs = [T::Vec::splat(T::ZERO); IC_BLOCK_SIZE];
            for kh in 0..kernel_height {
                if h * step_height + kh * dh < ph_start
                    || h * step_height + kh * dh - ph_start >= img_height
                {
                    continue;
                }
                for kw in 0..kernel_width {
                    if w * step_width + kw * dw < pw_start
                        || w * step_width + kw * dw - pw_start >= img_width
                    {
                        continue;
                    }
                    let mut inp_vecs = [T::Vec::splat(T::ZERO); IC_BLOCK_SIZE];
                    for (idx, vec) in inp_vecs.iter_mut().enumerate() {
                        let i = ii + ((idx * T::Vec::SIZE) as i64);
                        let inp_idx = b * isb
                            + (h * step_height + kh * dh - ph_start) * ish
                            + (w * step_width + kw * dw - pw_start) * isw
                            + i;
                        *vec = unsafe { T::Vec::from_ptr(&inp[inp_idx]) };
                    }
                    for idx in 0..IC_BLOCK_SIZE {
                        res_vecs[idx] = vec_op(res_vecs[idx], inp_vecs[idx]);
                    }
                }
            }
            if same_vec_size {
                for (idx, vec) in res_vecs.into_iter().enumerate() {
                    let i = ii + ((idx * T::Vec::SIZE) as i64);
                    let out_idx = b * osb + h * osh + w * osw + i;
                    let out_vec = (unsafe { out.ptr.add(out_idx as usize) }) as *mut O::Vec;
                    unsafe {
                        out_vec.write_unaligned(post_vec_op(vec.read_unaligned()));
                    }
                }
            } else {
                for vec in res_vecs {
                    for i in 0..T::Vec::SIZE {
                        let out_idx = b * osb + h * osh + w * osw + ii + i as i64;
                        let out = (unsafe { out.ptr.add(out_idx as usize) }) as *mut O;
                        unsafe {
                            out.write(post_scalar_op(vec[i]));
                        }
                    }
                }
            }
        }

        let remain = in_channel_remain % (T::Vec::SIZE as i64);
        for ii in (in_channels - in_channel_remain..in_channels - remain).step_by(T::Vec::SIZE) {
            let mut res_vecs = T::Vec::splat(T::ZERO);
            for kh in 0..kernel_height {
                if h * step_height + kh * dh < ph_start
                    || h * step_height + kh * dh - ph_start >= img_height
                {
                    continue;
                }
                for kw in 0..kernel_width {
                    if w * step_width + kw * dw < pw_start
                        || w * step_width + kw * dw - pw_start >= img_width
                    {
                        continue;
                    }
                    let i = ii;
                    let inp_idx = b * isb
                        + (h * step_height + kh * dh - ph_start) * ish
                        + (w * step_width + kw * dw - pw_start) * isw
                        + i;
                    let inp_vec = unsafe { T::Vec::from_ptr(&inp[inp_idx]) };

                    res_vecs = vec_op(res_vecs, inp_vec);
                }
            }
            let i = ii;
            if same_vec_size {
                let out_idx = b * osb + h * osh + w * osw + i;
                let out_vec = (unsafe { out.ptr.add(out_idx as usize) }) as *mut O::Vec;
                unsafe {
                    out_vec.write_unaligned(post_vec_op(res_vecs.read_unaligned()));
                }
            } else {
                for i in 0..T::Vec::SIZE {
                    let out_idx = b * osb + h * osh + w * osw + ii + i as i64;
                    let out = (unsafe { out.ptr.add(out_idx as usize) }) as *mut O;
                    unsafe {
                        out.write(post_scalar_op(res_vecs[i]));
                    }
                }
            }
        }

        for ii in in_channels - remain..in_channels {
            let mut res = T::ZERO;
            for kh in 0..kernel_height {
                if h * step_height + kh * dh < ph_start
                    || h * step_height + kh * dh - ph_start >= img_height
                {
                    continue;
                }
                for kw in 0..kernel_width {
                    if w * step_width + kw * dw < pw_start
                        || w * step_width + kw * dw - pw_start >= img_width
                    {
                        continue;
                    }
                    let i = ii;
                    let inp_idx = b * isb
                        + (h * step_height + kh * dh - ph_start) * ish
                        + (w * step_width + kw * dw - pw_start) * isw
                        + i;

                    res = scalar_op(res, inp[inp_idx]);
                }
            }
            let i = ii;
            let out_idx = b * osb + h * osh + w * osw + i;
            let out = (unsafe { out.ptr.add(out_idx as usize) }) as *mut O;
            unsafe {
                out.write_unaligned(post_scalar_op(res));
            }
        }
    });

    Ok(output)
}

#[track_caller]
pub(crate) fn adaptive_pooling_template<T: CommonBounds, O: CommonBounds, const DEVICE: usize, A>(
    img: &_Tensor<T, Cpu, DEVICE, A>,
    output_size: [i64; 2],
    scalar_op: impl Fn(T, T) -> T + Send + Sync,
    vec_op: impl Fn(T::Vec, T::Vec) -> T::Vec + Send + Sync,
    post_scalar_op: impl Fn(T, O) -> O + Send + Sync,
    post_vec_op: impl Fn(T::Vec, O::Vec) -> O::Vec + Send + Sync,
) -> std::result::Result<_Tensor<O, Cpu, DEVICE, A>, TensorError>
where
    i64: Cast<O>,
    A: Allocator + Send + Sync,
    A::Output: AllocatorOutputRetrive,
{
    ShapeError::check_contiguous(
        "pooling input must be contiguous".to_string(),
        &img.layout(),
    )?;
    let img_shape = img.shape();
    ShapeError::check_dim(4, img_shape.len())?;
    let batch = img_shape[0];
    let img_height = img_shape[1];
    let img_width = img_shape[2];
    let in_channels = img_shape[3];

    let out_height = output_size[0];
    let out_width = output_size[1];
    let img = img.clone();
    if out_height <= 0 || out_width <= 0 {
        return Err(ShapeError::ConvError {
            message: if out_height <= 0 {
                "output height <= 0".to_string()
            } else {
                "output width <= 0".to_string()
            },
            location: core::panic::Location::caller(),
        }
        .into());
    }
    let output = _Tensor::<O, Cpu, DEVICE, A>::empty([batch, out_height, out_width, in_channels])?;
    let out = output.ptr::<O>();
    let inp = img.ptr::<T>();

    let osb = output.strides()[0]; // batch
    let osh = output.strides()[1]; // height
    let osw = output.strides()[2]; // width

    let isb = img.strides()[0]; // batch
    let ish = img.strides()[1]; // height
    let isw = img.strides()[2]; // width

    let out_size = batch * out_height * out_width;

    const IC_BLOCK_SIZE: usize = REGNUM / 2;
    let in_channel_remain = in_channels % ((IC_BLOCK_SIZE * T::Vec::SIZE) as i64);
    (0..out_size).into_par_iter().for_each(|idx: i64| {
        let out = out.clone();
        let b = idx / (out_height * out_width);
        let h = (idx / out_width) % out_height;
        let w = idx % out_width;
        let start_h = (h * img_height / out_height) as i64;
        let end_h = ((h + 1) * img_height + out_height - 1) / out_height as i64;
        let start_w = (w * img_width / out_width) as i64;
        let end_w = ((w + 1) * img_width + out_width - 1) / out_width as i64;
        let kernel_size: O = ((end_h - start_h) * (end_w - start_w)).cast();
        let kernel_size_vec = O::Vec::splat(kernel_size);
        let same_vec_size = T::Vec::SIZE == O::Vec::SIZE;
        for ii in (0..in_channels - in_channel_remain).step_by(IC_BLOCK_SIZE * T::Vec::SIZE) {
            let mut res_vecs = [T::Vec::splat(T::ZERO); IC_BLOCK_SIZE];
            for kh in start_h..end_h {
                for kw in start_w..end_w {
                    let mut inp_vecs = [T::Vec::splat(T::ZERO); IC_BLOCK_SIZE];
                    for (idx, vec) in inp_vecs.iter_mut().enumerate() {
                        let i = ii + ((idx * T::Vec::SIZE) as i64);
                        let inp_idx = b * isb + kh * ish + kw * isw + i;
                        *vec = unsafe { T::Vec::from_ptr(&inp[inp_idx]) };
                    }
                    for idx in 0..IC_BLOCK_SIZE {
                        res_vecs[idx] = vec_op(res_vecs[idx], inp_vecs[idx]);
                    }
                }
            }
            if same_vec_size {
                for (idx, vec) in res_vecs.iter().enumerate() {
                    let i = ii + ((idx * T::Vec::SIZE) as i64);
                    let out_idx = b * osb + h * osh + w * osw + i;
                    let out_vec = (unsafe { out.ptr.add(out_idx as usize) }) as *mut O::Vec;
                    unsafe {
                        out_vec.write_unaligned(post_vec_op(vec.read_unaligned(), kernel_size_vec));
                    }
                }
            } else {
                for vec in res_vecs {
                    for i in 0..T::Vec::SIZE {
                        let out_idx = b * osb + h * osh + w * osw + ii + i as i64;
                        let out = (unsafe { out.ptr.add(out_idx as usize) }) as *mut O;
                        unsafe {
                            out.write(post_scalar_op(vec[i], kernel_size));
                        }
                    }
                }
            }
        }

        let remain = in_channel_remain % (T::Vec::SIZE as i64);
        for ii in (in_channels - in_channel_remain..in_channels - remain).step_by(T::Vec::SIZE) {
            let mut res_vecs = T::Vec::splat(T::ZERO);
            for kh in start_h..end_h {
                for kw in start_w..end_w {
                    let i = ii;
                    let inp_idx = b * isb + kh * ish + kw * isw + i;
                    let inp_vec = unsafe { T::Vec::from_ptr(&inp[inp_idx]) };

                    res_vecs = vec_op(res_vecs, inp_vec);
                }
            }
            if same_vec_size {
                let i = ii;
                let out_idx = b * osb + h * osh + w * osw + i;
                let out_vec = (unsafe { out.ptr.add(out_idx as usize) }) as *mut O::Vec;
                unsafe {
                    out_vec
                        .write_unaligned(post_vec_op(res_vecs.read_unaligned(), kernel_size_vec));
                }
            } else {
                for i in 0..T::Vec::SIZE {
                    let out_idx = b * osb + h * osh + w * osw + ii + i as i64;
                    let out = (unsafe { out.ptr.add(out_idx as usize) }) as *mut O;
                    unsafe {
                        out.write(post_scalar_op(res_vecs[i], kernel_size));
                    }
                }
            }
        }

        for ii in in_channels - remain..in_channels {
            let mut res = T::ZERO;
            for kh in start_h..end_h {
                for kw in start_w..end_w {
                    let i = ii;
                    let inp_idx = b * isb + kh * ish + kw * isw + i;

                    res = scalar_op(res, inp[inp_idx]);
                }
            }
            let i = ii;
            let out_idx = b * osb + h * osh + w * osw + i;
            let out = (unsafe { out.ptr.add(out_idx as usize) }) as *mut O;
            unsafe {
                out.write_unaligned(post_scalar_op(res, kernel_size));
            }
        }
    });

    Ok(output)
}
