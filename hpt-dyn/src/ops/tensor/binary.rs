use hpt_common::shape::shape_utils::mt_intervals;
use hpt_traits::tensor::TensorInfo;
use hpt_types::promote_normal_binary;
use hpt_types::scalar::*;
use hpt_types::vector::*;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

use crate::{Tensor, current_num_threads};

pub(crate) fn binary<F1, F2>(
    output: &mut Tensor,
    inp: &Tensor,
    other: &Tensor,
    kernel: F1,
    simd_kernel: F2,
    unroll: usize,
) where
    F1: Fn(usize, usize, usize) + Send + Sync,
    F2: Fn(usize, usize, usize) + Send + Sync,
{
    assert!(
        inp.layout.ndim() > 0,
        "input tensor must have at least one dimension"
    );
    let inp = inp.broadcast_to(output.shape()).expect("broadcast failed");
    let other = other
        .broadcast_to(output.shape())
        .expect("broadcast failed");

    if inp.parent.is_some()
        || other.parent.is_some()
        || !inp.is_contiguous()
        || !other.is_contiguous()
    {
        let inner_size = *output.layout.shape().last().expect("inner size is None");
        let outer_size = output.layout.size() / inner_size;

        let chunks = mt_intervals(outer_size as usize, current_num_threads());

        let lhs_sizeof = inp.dtype.sizeof() as i64;
        let rhs_sizeof = other.dtype.sizeof() as i64;
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

        let rhs_ptrs = chunks
            .iter()
            .map(|&(start, _)| {
                let (offset, prg) = (other.map_gp)(start as i64 * inner_size as i64);
                let mut ptr = other.data;
                ptr += offset * rhs_sizeof as i64;
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
        let rhs_last_stride = *other.layout.strides().last().expect("last stride is None");
        let lhs_prg_update = inp.prg_update.as_ref();
        let rhs_prg_update = other.prg_update.as_ref();
        if lhs_last_stride == 1
            && rhs_last_stride == 1
            && lhs_sizeof == rhs_sizeof
            && lhs_sizeof == out_sizeof
        {
            let unroll = unroll as i64;
            let vec_size = inp.dtype.vec_size() as i64;
            let num_vec = inner_size / (unroll * vec_size);
            let rem = inner_size % (unroll * vec_size);
            let vector_bytes = unroll * vec_size * out_sizeof;
            inp_ptrs
                .into_par_iter()
                .zip(rhs_ptrs.into_par_iter())
                .zip(out_ptrs.into_par_iter())
                .for_each(
                    move |(((mut lhs, mut lhs_prg), (mut rhs, mut rhs_prg)), mut out)| {
                        for i in 0..num_vec {
                            simd_kernel(
                                lhs.offset_addr(i * lhs_last_stride * vector_bytes),
                                rhs.offset_addr(i * rhs_last_stride * vector_bytes),
                                out.offset_addr(i * vector_bytes),
                            );
                        }
                        for i in inner_size - rem..inner_size {
                            kernel(
                                lhs.offset_addr(i * lhs_last_stride * lhs_sizeof),
                                rhs.offset_addr(i * rhs_last_stride * rhs_sizeof),
                                out.offset_addr(i * out_sizeof),
                            );
                        }
                        out += inner_size * out_sizeof;
                        lhs_prg_update(&mut lhs_prg, &mut lhs);
                        rhs_prg_update(&mut rhs_prg, &mut rhs);
                    },
                );
        } else {
            inp_ptrs
                .into_par_iter()
                .zip(rhs_ptrs.into_par_iter())
                .zip(out_ptrs.into_par_iter())
                .for_each(
                    move |(((mut lhs, mut lhs_prg), (mut rhs, mut rhs_prg)), mut out)| {
                        for i in 0..inner_size {
                            kernel(
                                out.offset_addr(i * out_sizeof),
                                lhs.offset_addr(i * lhs_last_stride * lhs_sizeof),
                                rhs.offset_addr(i * rhs_last_stride * rhs_sizeof),
                            );
                        }
                        out += inner_size * out_sizeof;
                        lhs_prg_update(&mut lhs_prg, &mut lhs);
                        rhs_prg_update(&mut rhs_prg, &mut rhs);
                    },
                );
        }
    } else {
        let out_sizeof = output.dtype.sizeof() as i64;
        let lhs_sizeof = inp.dtype.sizeof() as i64;
        let rhs_sizeof = other.dtype.sizeof() as i64;
        let out = output.data;
        let lhs = inp.data;
        let rhs = other.data;

        if out_sizeof == lhs_sizeof && out_sizeof == rhs_sizeof {
            let slice = inp.as_slice::<u8>();
            let slice_other = other.as_slice::<u8>();
            let slice_out = output.as_slice_mut::<u8>();

            let vec_size = inp.dtype.vec_size() as i64;

            let mut out_chunk =
                slice_out.par_chunks_exact_mut(unroll * vec_size as usize * out_sizeof as usize);
            let lhs_chunk =
                slice.par_chunks_exact(unroll * vec_size as usize * lhs_sizeof as usize);
            let rhs_chunk =
                slice_other.par_chunks_exact(unroll * vec_size as usize * rhs_sizeof as usize);
            out_chunk
                .remainder()
                .par_iter_mut()
                .zip(lhs_chunk.remainder().par_iter())
                .zip(rhs_chunk.remainder().par_iter())
                .for_each(|((out, lhs), rhs)| {
                    kernel(
                        lhs as *const u8 as usize,
                        rhs as *const u8 as usize,
                        out as *mut u8 as usize,
                    );
                });
            out_chunk
                .into_par_iter()
                .zip(lhs_chunk.into_par_iter())
                .zip(rhs_chunk.into_par_iter())
                .for_each(|((out, lhs), rhs)| {
                    let lhs_ptr = lhs.as_ptr();
                    let rhs_ptr = rhs.as_ptr();
                    let out_ptr = out.as_mut_ptr();
                    simd_kernel(lhs_ptr as usize, rhs_ptr as usize, out_ptr as usize);
                });
        } else {
            (0..inp.layout.size()).into_par_iter().for_each(|i| {
                kernel(
                    lhs.offset_addr(i * lhs_sizeof),
                    rhs.offset_addr(i * rhs_sizeof),
                    out.offset_addr(i * out_sizeof),
                );
            });
        }
    }
}

#[
    duplicate::duplicate_item(
        trait_name  method_name   scalar_dispatch   simd_dispatch           promote_method;
        [Add]       [add]         [dispatch_add]     [dispatch_simd_add]     [promote_normal_binary];
        [Sub]       [sub]         [dispatch_sub]     [dispatch_simd_sub]     [promote_normal_binary];
        [Mul]       [mul]         [dispatch_mul]     [dispatch_simd_mul]     [promote_normal_binary];
        [Rem]       [rem]         [dispatch_rem]     [dispatch_simd_rem]     [promote_normal_binary];
    )
]
impl std::ops::trait_name for Tensor {
    type Output = Tensor;

    fn method_name(self, rhs: Self) -> Self::Output {
        let res_layout = self
            .layout
            .broadcast(rhs.shape())
            .expect("broadcast failed");

        let mut res = Tensor::empty(
            &res_layout.shape(),
            promote_method(self.dtype, rhs.dtype),
            self.device.clone(),
        )
        .expect("failed to create empty tensor");

        let (simd_kernel, unroll) = simd_dispatch(self.dtype, rhs.dtype);
        let scalar_kernel = scalar_dispatch(self.dtype, rhs.dtype);
        binary(&mut res, &self, &rhs, scalar_kernel, simd_kernel, unroll);
        res
    }
}

#[
    duplicate::duplicate_item(
        trait_name  method_name   scalar_dispatch   simd_dispatch           promote_method;
        [Add]       [add]         [dispatch_add]     [dispatch_simd_add]     [promote_normal_binary];
        [Sub]       [sub]         [dispatch_sub]     [dispatch_simd_sub]     [promote_normal_binary];
        [Mul]       [mul]         [dispatch_mul]     [dispatch_simd_mul]     [promote_normal_binary];
        [Rem]       [rem]         [dispatch_rem]     [dispatch_simd_rem]     [promote_normal_binary];
    )
]
impl std::ops::trait_name<&Tensor> for &Tensor {
    type Output = Tensor;

    fn method_name(self, rhs: &Tensor) -> Self::Output {
        let res_layout = self
            .layout
            .broadcast(rhs.shape())
            .expect("broadcast failed");

        let mut res = Tensor::empty(
            &res_layout.shape(),
            promote_method(self.dtype, rhs.dtype),
            self.device.clone(),
        )
        .expect("failed to create empty tensor");

        let (simd_kernel, unroll) = simd_dispatch(self.dtype, rhs.dtype);
        let scalar_kernel = scalar_dispatch(self.dtype, rhs.dtype);
        binary(&mut res, &self, &rhs, scalar_kernel, simd_kernel, unroll);
        res
    }
}
