use crate::{Tensor, current_num_threads};
use hpt_common::error::base::TensorError;
use hpt_common::shape::shape_utils::mt_intervals;
use hpt_traits::tensor::TensorInfo;
use hpt_types::promote_float_unary;
use hpt_types::scalar::*;
use hpt_types::vector::*;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

pub(crate) fn unary_1operand<F1, F2>(
    output: &mut Tensor,
    inp: &Tensor,
    kernel: F1,
    simd_kernel: F2,
    unroll: usize,
) where
    F1: Fn(usize, usize) + Send + Sync,
    F2: Fn(usize, usize) + Send + Sync,
{
    assert!(
        inp.layout.ndim() > 0,
        "input tensor must have at least one dimension"
    );
    if inp.parent.is_some() {
        let inner_size = *inp.layout.shape().last().expect("inner size is None");
        let outer_size = inp.layout.size() / inner_size;

        let chunks = mt_intervals(outer_size as usize, current_num_threads());

        let lhs_sizeof = inp.dtype.sizeof() as i64;
        let out_sizeof = output.dtype.sizeof() as i64;
        let inp_ptrs = chunks
            .iter()
            .map(|&(start, _)| {
                let (offset, prg) = (inp.map_gp)(start as i64 * inner_size as i64);
                let mut ptr = inp.data;
                ptr += offset * lhs_sizeof as i64;
                (ptr, prg)
            })
            .collect::<Vec<_>>();

        let out_ptrs = chunks
            .iter()
            .map(|&(start, _)| {
                let offset = (output.map_global_idx)(start as i64 * inner_size as i64);
                let mut ptr = output.data;
                ptr += offset * out_sizeof as i64;
                ptr
            })
            .collect::<Vec<_>>();

        let lhs_last_stride = *inp.layout.strides().last().expect("last stride is None");
        let lhs_prg_update = inp.prg_update.as_ref();
        inp_ptrs
            .into_par_iter()
            .zip(out_ptrs.into_par_iter())
            .for_each(move |((mut lhs, mut lhs_prg), mut out)| {
                for i in 0..inner_size {
                    kernel(
                        out.offset_addr(i * out_sizeof),
                        lhs.offset_addr(i * lhs_last_stride * lhs_sizeof),
                    );
                }
                out += inner_size * out_sizeof;
                lhs_prg_update(&mut lhs_prg, &mut lhs);
            });
    } else {
        let out_sizeof = output.dtype.sizeof() as i64;
        let lhs_sizeof = inp.dtype.sizeof() as i64;
        let out = output.data;
        let lhs = inp.data;

        if out_sizeof == lhs_sizeof {
            let vec_size = inp.dtype.vec_size();
            let out_sizeof = output.dtype.sizeof();
            let vector_bytes = unroll * vec_size as usize * out_sizeof as usize;

            let intervals = mt_intervals(output.layout.size() as usize, current_num_threads());
            intervals.into_par_iter().for_each(|(start, end)| {
                let size = end - start;
                let num_loop = size / (unroll * vec_size as usize);
                let rem = size % (unroll * vec_size as usize);
                let lhs = lhs + start * out_sizeof;
                let out = out + start * out_sizeof;

                for i in 0..num_loop {
                    simd_kernel(
                        lhs.add_addr(i * vector_bytes),
                        out.add_addr(i * vector_bytes),
                    );
                }

                for i in end - rem..end {
                    kernel(lhs.add_addr(i * out_sizeof), out.add_addr(i * out_sizeof));
                }
            });
        } else {
            (0..inp.layout.size()).into_par_iter().for_each(|i| {
                kernel(
                    lhs.offset_addr(i * lhs_sizeof),
                    out.offset_addr(i * out_sizeof),
                );
            });
        }
    }
}

impl Tensor {
    #[duplicate::duplicate_item(
        func_name       kernel                  simd_kernel;
        [sin]           [dispatch_sin]          [dispatch_simd_sin];
        [cos]           [dispatch_cos]          [dispatch_simd_cos];
        [tan]           [dispatch_tan]          [dispatch_simd_tan];
        [asin]          [dispatch_asin]         [dispatch_simd_asin];
        [acos]          [dispatch_acos]         [dispatch_simd_acos];
        [atan]          [dispatch_atan]         [dispatch_simd_atan];
        [sinh]          [dispatch_sinh]         [dispatch_simd_sinh];
        [cosh]          [dispatch_cosh]         [dispatch_simd_cosh];
        [tanh]          [dispatch_tanh]         [dispatch_simd_tanh];
        [asinh]         [dispatch_asinh]        [dispatch_simd_asinh];
        [acosh]         [dispatch_acosh]        [dispatch_simd_acosh];
        [atanh]         [dispatch_atanh]        [dispatch_simd_atanh];
        [exp]           [dispatch_exp]          [dispatch_simd_exp];
        [exp2]          [dispatch_exp2]         [dispatch_simd_exp2];
        [expm1]         [dispatch_expm1]        [dispatch_simd_expm1];
        [ln]            [dispatch_ln]           [dispatch_simd_ln];
        [log1p]         [dispatch_log1p]        [dispatch_simd_log1p];
        [log2]          [dispatch_log2]         [dispatch_simd_log2];
        [log10]         [dispatch_log10]        [dispatch_simd_log10];
        [sqrt]          [dispatch_sqrt]         [dispatch_simd_sqrt];
        [cbrt]          [dispatch_cbrt]         [dispatch_simd_cbrt];
        [recip]         [dispatch_recip]        [dispatch_simd_recip];
        [erf]           [dispatch_erf]          [dispatch_simd_erf];
        [sigmoid]       [dispatch_sigmoid]      [dispatch_simd_sigmoid];
        [gelu]          [dispatch_gelu]         [dispatch_simd_gelu];
        [hard_sigmoid]  [dispatch_hard_sigmoid] [dispatch_simd_hard_sigmoid];
        [hard_swish]    [dispatch_hard_swish]   [dispatch_simd_hard_swish];
        [softplus]      [dispatch_softplus]     [dispatch_simd_softplus];
        [softsign]      [dispatch_softsign]     [dispatch_simd_softsign];
        [mish]          [dispatch_mish]         [dispatch_simd_mish];
    )]
    pub fn func_name(&self) -> Result<Self, TensorError> {
        let mut res = Tensor::empty(
            &self.layout.shape(),
            promote_float_unary(self.dtype),
            self.device.clone(),
        )?;
        let scalar_fn = kernel(self.dtype);
        let (simd_fn, unroll) = simd_kernel(self.dtype);

        unary_1operand(&mut res, &self, scalar_fn, simd_fn, unroll);
        Ok(res)
    }

    #[duplicate::duplicate_item(
        func_name       kernel                  simd_kernel;
        [celu]          [dispatch_celu]         [dispatch_simd_celu];
        [elu]           [dispatch_elu]          [dispatch_simd_elu];
    )]
    pub fn func_name(&self, alpha: f64) -> Result<Self, TensorError> {
        let mut res = Tensor::empty(
            &self.layout.shape(),
            promote_float_unary(self.dtype),
            self.device.clone(),
        )?;
        let scalar_fn = kernel(self.dtype, alpha);
        let (simd_fn, unroll) = simd_kernel(self.dtype, alpha);

        unary_1operand(
            &mut res,
            &self,
            scalar_fn.as_ref(),
            simd_fn.as_ref(),
            unroll,
        );
        Ok(res)
    }

    pub fn selu(&self) -> Result<Self, TensorError> {
        let mut res = Tensor::empty(
            &self.layout.shape(),
            promote_float_unary(self.dtype),
            self.device.clone(),
        )?;
        let scalar_fn = dispatch_selu(
            self.dtype,
            1.6732632423543772848170429916717,
            1.0507009873554804934193349852946,
        );
        let (simd_fn, unroll) = dispatch_simd_selu(
            self.dtype,
            1.6732632423543772848170429916717,
            1.0507009873554804934193349852946,
        );

        unary_1operand(
            &mut res,
            &self,
            scalar_fn.as_ref(),
            simd_fn.as_ref(),
            unroll,
        );
        Ok(res)
    }

    pub fn contiguous(&self) -> Result<Self, TensorError> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }
        let mut res = Tensor::empty(
            &self.layout.shape(),
            self.dtype.clone(),
            self.device.clone(),
        )?;
        let copy_fn = dispatch_copy(self.dtype);
        let (simd_copy_fn, unroll) = dispatch_simd_copy(self.dtype);

        unary_1operand(&mut res, &self, copy_fn, simd_copy_fn, unroll);
        Ok(res)
    }

    pub fn copy(&self) -> Result<Self, TensorError> {
        let mut res = Tensor::empty(
            &self.layout.shape(),
            self.dtype.clone(),
            self.device.clone(),
        )?;
        let copy_fn = dispatch_copy(self.dtype);
        let (simd_copy_fn, unroll) = dispatch_simd_copy(self.dtype);

        unary_1operand(&mut res, &self, copy_fn, simd_copy_fn, unroll);
        Ok(res)
    }
}
