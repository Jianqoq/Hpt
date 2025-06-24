use hpt_common::{
    error::{ base::TensorError, shape::ShapeError },
    shape::shape_utils::mt_intervals,
};
use hpt_traits::tensor::{ CommonBounds, TensorInfo };
use hpt_types::type_promote::FloatOutBinary;
use hpt_types::type_promote::FloatOutUnary;
use hpt_types::{ traits::VecTrait, type_promote::NormalOut };
use rayon::iter::{ IntoParallelIterator, ParallelIterator };

use crate::Tensor;

/// input shape: [batch, height, width, channel]
pub(crate) fn batch_norm<T>(
    input: &Tensor, // shape: [batch, height, width, channel]
    mean: &Tensor, // shape: [channel]
    var: &Tensor, // shape: [channel]
    gamma: &Tensor, // shape: [channel]
    beta: &Tensor, // shape: [channel]
    eps: T,
    post_scalar: Option<fn(T) -> T>,
    post_vec: Option<fn(<T>::Vec) -> <T>::Vec>,
    out: Option<Tensor>
)
    -> Result<Tensor, TensorError>
    where
        T: CommonBounds + FloatOutBinary<Output = T> + FloatOutUnary<Output = T>,
        T::Vec: FloatOutBinary<Output = T::Vec> + FloatOutUnary<Output = T::Vec>
{
    ShapeError::check_contiguous(
        "batch norm requires contiguous input".to_string(),
        input.layout()
    )?;
    ShapeError::check_contiguous("batch norm requires contiguous mean".to_string(), mean.layout())?;
    ShapeError::check_contiguous("batch norm requires contiguous var".to_string(), var.layout())?;
    ShapeError::check_contiguous(
        "batch norm requires contiguous gamma".to_string(),
        gamma.layout()
    )?;
    ShapeError::check_contiguous("batch norm requires contiguous beta".to_string(), beta.layout())?;

    let res = if let Some(out) = out {
        ShapeError::check_inplace_out_layout_valid(input.shape(), &out.layout)?;
        out
    } else {
        input.empty_like()?
    };

    let batch = input.shape()[0];
    let height = input.shape()[1];
    let width = input.shape()[2];
    let channel = input.shape()[3];

    let outer_loop_size = batch * height * width;
    let num_threads = (outer_loop_size as usize).min(rayon::current_num_threads());

    let intervals = mt_intervals(outer_loop_size as usize, num_threads);
    let eps_vec = T::Vec::splat(eps);
    let post_scalar = post_scalar.unwrap_or(|x| x);
    let post_vec = post_vec.unwrap_or(|x| x);
    let inp_ptr = input.ptr::<T>();
    (0..num_threads).into_par_iter().for_each(|idx| {
        let (start, end) = intervals[idx];
        let inp_ptr = inp_ptr;
        let out_ptr = res.ptr::<T>();
        let mean_ptr = mean.ptr::<T>();
        let var_ptr = var.ptr::<T>();
        let gamma_ptr = gamma.ptr::<T>();
        let beta_ptr = beta.ptr::<T>();
        let mean_vec_ptr = mean_ptr.ptr as *const T::Vec;
        let var_vec_ptr = var_ptr.ptr as *const T::Vec;
        let gamma_vec_ptr = gamma_ptr.ptr as *const T::Vec;
        let beta_vec_ptr = beta_ptr.ptr as *const T::Vec;
        let rem = channel % (T::Vec::SIZE as i64);
        let num_vec = channel / (T::Vec::SIZE as i64);
        for i in start..end {
            let inp = inp_ptr + i * (channel as usize);
            let mut out = out_ptr + i * (channel as usize);
            let inp_vec_ptr = inp.ptr as *const T::Vec;
            let out_vec_ptr = out.ptr as *mut T::Vec;
            unsafe {
                for j in 0..num_vec {
                    let mean = mean_vec_ptr.offset(j as isize).read_unaligned();
                    let var = var_vec_ptr.offset(j as isize).read_unaligned();
                    let gamma = gamma_vec_ptr.offset(j as isize).read_unaligned();
                    let beta = beta_vec_ptr.offset(j as isize).read_unaligned();
                    let inp_vec = inp_vec_ptr.offset(j as isize).read_unaligned();
                    let res = inp_vec
                        ._sub(mean)
                        ._div(var._add(eps_vec)._sqrt())
                        ._mul(gamma)
                        ._add(beta);
                    out_vec_ptr.offset(j as isize).write_unaligned(post_vec(res));
                }
                for j in channel - rem..channel {
                    let mean = mean_ptr[j];
                    let var = var_ptr[j];
                    let gamma = gamma_ptr[j];
                    let beta = beta_ptr[j];
                    let inp = inp[j];
                    let res = inp._sub(mean)._div(var._add(eps)._sqrt())._mul(gamma)._add(beta);
                    out[j] = post_scalar(res);
                }
            }
        }
    });

    Ok(res)
}
