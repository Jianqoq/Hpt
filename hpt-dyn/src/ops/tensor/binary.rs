use std::borrow::Borrow;

use hpt_common::error::base::TensorError;
use hpt_common::error::shape::ShapeError;
use hpt_common::shape::shape_utils::mt_intervals;
use hpt_iterator::iterator_traits::ParStridedIteratorSimd;
use hpt_iterator::iterator_traits::ParStridedIteratorSimdZip;
use hpt_iterator::iterator_traits::ParStridedIteratorZip;
use hpt_iterator::TensorIterator;
use hpt_traits::tensor::CommonBounds;
use hpt_traits::tensor::TensorInfo;
use hpt_types::dtype::TypeCommon;
use hpt_types::promote_normal_binary;
use hpt_types::scalar::*;
use hpt_types::vector::*;
use rayon::iter::{
    IndexedParallelIterator,
    IntoParallelIterator,
    IntoParallelRefIterator,
    IntoParallelRefMutIterator,
    ParallelIterator,
};
use rayon::slice::{ ParallelSlice, ParallelSliceMut };

use crate::{ Tensor, current_num_threads };

pub(crate) fn binary_fn_with_out<T, O, F, F2>(
    lhs: &Tensor,
    rhs: &Tensor,
    f: F,
    f2: F2,
    out: Option<O>
)
    -> std::result::Result<Tensor, TensorError>
    where
        T: CommonBounds,
        O: Borrow<Tensor>,
        F: Fn(T, T) -> T + Sync + Send + Copy,
        F2: Fn(<T as TypeCommon>::Vec, <T as TypeCommon>::Vec) -> <T as TypeCommon>::Vec +
            Sync +
            Send +
            Copy
{
    use hpt_types::traits::*;
    use rayon::slice::{ ParallelSlice, ParallelSliceMut };
    if lhs.size() == 1 {
        let val = lhs.as_slice::<T>()[0];
        let val_vec = <T as TypeCommon>::Vec::splat(val);
        let mut res = if let Some(out) = out {
            ShapeError::check_inplace_out_layout_valid(rhs.shape(), &out.borrow().layout())?;
            let out: &Tensor = out.borrow();
            out.clone()
        } else {
            Tensor::empty(rhs.shape(), rhs.dtype, rhs.device.clone())?
        };
        if rhs.is_contiguous() {
            let remain = res.size() % <T as TypeCommon>::Vec::SIZE;
            res.as_slice_mut::<T>()
                .par_chunks_exact_mut(<T as TypeCommon>::Vec::SIZE)
                .zip(rhs.as_slice::<T>().par_chunks_exact(<T as TypeCommon>::Vec::SIZE))
                .for_each(|(a, b)| {
                    let inp = unsafe { <T as TypeCommon>::Vec::from_ptr(b.as_ptr()) };
                    let res: *const T = f2(val_vec, inp).as_ptr();
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            res,
                            a.as_mut_ptr(),
                            <T as TypeCommon>::Vec::SIZE
                        );
                    }
                });
            if remain > 0 {
                let ret_size = res.size();
                res.as_slice_mut::<T>()
                    [ret_size - remain..].iter_mut()
                    .zip(rhs.as_slice::<T>()[ret_size - remain..].iter())
                    .for_each(|(a, b)| {
                        *a = f(val, *b);
                    });
            }
        } else {
            res.par_iter_mut()
                .zip(rhs.par_iter())
                .for_each(|(a, b)| {
                    *a = f(val, b);
                });
        }
        Ok(res)
    } else if rhs.size() == 1 {
        let val = rhs.as_slice::<T>()[0];
        let val_vec = <T as TypeCommon>::Vec::splat(val);
        let mut res = if let Some(out) = out {
            ShapeError::check_inplace_out_layout_valid(lhs.shape(), &out.borrow().layout())?;
            let out: &Tensor = out.borrow();
            out.clone()
        } else {
            Tensor::empty(lhs.shape(), lhs.dtype, lhs.device.clone())?
        };
        if lhs.is_contiguous() {
            let remain = res.size() % <T as TypeCommon>::Vec::SIZE;
            res.as_slice_mut::<T>()
                .par_chunks_exact_mut(<T as TypeCommon>::Vec::SIZE)
                .zip(lhs.as_slice::<T>().par_chunks_exact(<T as TypeCommon>::Vec::SIZE))
                .for_each(|(a, lhs)| {
                    let inp = unsafe { <T as TypeCommon>::Vec::from_ptr(lhs.as_ptr()) };
                    let res: *const T = f2(inp, val_vec).as_ptr();
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            res,
                            a.as_mut_ptr(),
                            <T as TypeCommon>::Vec::SIZE
                        );
                    }
                });
            if remain > 0 {
                let ret_size = res.size();
                res.as_slice_mut::<T>()
                    [ret_size - remain..].iter_mut()
                    .zip(lhs.as_slice::<T>()[ret_size - remain..].iter())
                    .for_each(|(a, lhs)| {
                        *a = f(*lhs, val);
                    });
            }
        } else {
            res.par_iter_mut()
                .zip(lhs.par_iter())
                .for_each(|(a, lhs)| {
                    *a = f(lhs, val);
                });
        }
        Ok(res)
    } else {
        if rhs.is_contiguous() && lhs.is_contiguous() && rhs.shape() == lhs.shape() {
            let mut ret = if let Some(out) = out {
                ShapeError::check_inplace_out_layout_valid(rhs.shape(), &out.borrow().layout())?;
                let out: &Tensor = out.borrow();
                out.clone()
            } else {
                Tensor::empty(rhs.shape(), rhs.dtype, rhs.device.clone())?
            };
            let remain = ret.size() % <T as TypeCommon>::Vec::SIZE;
            ret.as_slice_mut::<T>()
                .par_chunks_exact_mut(<T as TypeCommon>::Vec::SIZE)
                .zip(lhs.as_slice::<T>().par_chunks_exact(<T as TypeCommon>::Vec::SIZE))
                .zip(rhs.as_slice::<T>().par_chunks_exact(<T as TypeCommon>::Vec::SIZE))
                .for_each(|((ret, lhs), rhs)| {
                    let a = unsafe { <T as TypeCommon>::Vec::from_ptr(lhs.as_ptr()) };
                    let b = unsafe { <T as TypeCommon>::Vec::from_ptr(rhs.as_ptr()) };
                    let res = f2(a, b);
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            res.as_ptr(),
                            ret.as_mut_ptr(),
                            <T as TypeCommon>::Vec::SIZE
                        );
                    }
                });
            if remain > 0 {
                let ret_size = ret.size();
                ret.as_slice_mut::<T>()
                    [ret_size - remain..].iter_mut()
                    .zip(lhs.as_slice::<T>()[ret_size - remain..].iter())
                    .zip(rhs.as_slice::<T>()[ret_size - remain..].iter())
                    .for_each(|((a, &lhs), &rhs)| {
                        *a = f(lhs, rhs);
                    });
            }
            Ok(ret)
        } else {
            let output_shape = lhs.layout().broadcast(rhs.shape())?;
            let mut res = if let Some(out) = out {
                ShapeError::check_inplace_out_layout_valid(
                    output_shape.shape(),
                    &out.borrow().layout()
                )?;
                let out: &Tensor = out.borrow();
                out.clone()
            } else {
                Tensor::empty(output_shape.shape(), lhs.dtype, lhs.device.clone())?
            };
            let iter = res.par_iter_mut_simd().zip(lhs.par_iter_simd()).zip(rhs.par_iter_simd());
            ParStridedIteratorSimd::for_each(
                iter,
                |((x, y), z)| {
                    *x = f(y, z);
                },
                |((x, y), z)| {
                    x.write_unaligned(f2(y, z));
                }
            );
            Ok(res)
        }
    }
}

pub(crate) fn binary<F1, F2>(
    output: &mut Tensor,
    inp: &Tensor,
    other: &Tensor,
    kernel: F1,
    simd_kernel: F2,
    unroll: usize
)
    where F1: Fn(usize, usize, usize) + Send + Sync, F2: Fn(usize, usize, usize) + Send + Sync
{
    assert!(inp.layout.ndim() > 0, "input tensor must have at least one dimension");
    let inp = inp.broadcast_to(output.shape()).expect("broadcast failed");
    let other = other.broadcast_to(output.shape()).expect("broadcast failed");

    if
        inp.parent.is_some() ||
        other.parent.is_some() ||
        !inp.is_contiguous() ||
        !other.is_contiguous()
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
                let (offset, prg) = (inp.map_gp)((start as i64) * (inner_size as i64));
                let mut ptr = inp.data;
                ptr += offset * (lhs_sizeof as i64);
                (ptr, prg)
            })
            .collect::<Vec<_>>();

        let rhs_ptrs = chunks
            .iter()
            .map(|&(start, _)| {
                let (offset, prg) = (other.map_gp)((start as i64) * (inner_size as i64));
                let mut ptr = other.data;
                ptr += offset * (rhs_sizeof as i64);
                (ptr, prg)
            })
            .collect::<Vec<_>>();

        let out_ptrs = chunks
            .iter()
            .map(|&(start, _)| {
                let offset = (output.map_global_idx)((start as i64) * (inner_size as i64));
                let mut ptr = output.data;
                ptr += offset * (out_sizeof as i64);
                ptr
            })
            .collect::<Vec<_>>();

        let lhs_last_stride = *inp.layout.strides().last().expect("last stride is None");
        let rhs_last_stride = *other.layout.strides().last().expect("last stride is None");
        let lhs_prg_update = inp.prg_update.as_ref();
        let rhs_prg_update = other.prg_update.as_ref();
        if
            lhs_last_stride == 1 &&
            rhs_last_stride == 1 &&
            lhs_sizeof == rhs_sizeof &&
            lhs_sizeof == out_sizeof
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
                .for_each(move |(((mut lhs, mut lhs_prg), (mut rhs, mut rhs_prg)), mut out)| {
                    for i in 0..num_vec {
                        simd_kernel(
                            lhs.offset_addr(i * lhs_last_stride * vector_bytes),
                            rhs.offset_addr(i * rhs_last_stride * vector_bytes),
                            out.offset_addr(i * vector_bytes)
                        );
                    }
                    for i in inner_size - rem..inner_size {
                        kernel(
                            lhs.offset_addr(i * lhs_last_stride * lhs_sizeof),
                            rhs.offset_addr(i * rhs_last_stride * rhs_sizeof),
                            out.offset_addr(i * out_sizeof)
                        );
                    }
                    out += inner_size * out_sizeof;
                    lhs_prg_update(&mut lhs_prg, &mut lhs);
                    rhs_prg_update(&mut rhs_prg, &mut rhs);
                });
        } else {
            inp_ptrs
                .into_par_iter()
                .zip(rhs_ptrs.into_par_iter())
                .zip(out_ptrs.into_par_iter())
                .for_each(move |(((mut lhs, mut lhs_prg), (mut rhs, mut rhs_prg)), mut out)| {
                    for i in 0..inner_size {
                        kernel(
                            out.offset_addr(i * out_sizeof),
                            lhs.offset_addr(i * lhs_last_stride * lhs_sizeof),
                            rhs.offset_addr(i * rhs_last_stride * rhs_sizeof)
                        );
                    }
                    out += inner_size * out_sizeof;
                    lhs_prg_update(&mut lhs_prg, &mut lhs);
                    rhs_prg_update(&mut rhs_prg, &mut rhs);
                });
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

            let mut out_chunk = slice_out.par_chunks_exact_mut(
                unroll * (vec_size as usize) * (out_sizeof as usize)
            );
            let lhs_chunk = slice.par_chunks_exact(
                unroll * (vec_size as usize) * (lhs_sizeof as usize)
            );
            let rhs_chunk = slice_other.par_chunks_exact(
                unroll * (vec_size as usize) * (rhs_sizeof as usize)
            );
            out_chunk
                .remainder()
                .par_iter_mut()
                .zip(lhs_chunk.remainder().par_iter())
                .zip(rhs_chunk.remainder().par_iter())
                .for_each(|((out, lhs), rhs)| {
                    kernel(
                        lhs as *const u8 as usize,
                        rhs as *const u8 as usize,
                        out as *mut u8 as usize
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
                    out.offset_addr(i * out_sizeof)
                );
            });
        }
    }
}

impl Tensor {
    pub fn add_(&self, other: &Tensor, out: &mut Tensor) -> Result<Tensor, ShapeError> {
        let res_layout = self.layout.broadcast(other.shape()).expect("broadcast failed");
        ShapeError::check_inplace_out_layout_valid(out.shape(), &res_layout)?;
        let (simd_kernel, unroll) = dispatch_simd_add(self.dtype, other.dtype);
        let scalar_kernel = dispatch_add(self.dtype, other.dtype);
        binary(out, self, other, scalar_kernel, simd_kernel, unroll);
        Ok(out.clone())
    }
}

#[duplicate::duplicate_item(
        trait_name  method_name   scalar_dispatch   simd_dispatch           promote_method;
        [Add]       [add]         [dispatch_add]     [dispatch_simd_add]     [promote_normal_binary];
        [Sub]       [sub]         [dispatch_sub]     [dispatch_simd_sub]     [promote_normal_binary];
        [Mul]       [mul]         [dispatch_mul]     [dispatch_simd_mul]     [promote_normal_binary];
        [Rem]       [rem]         [dispatch_rem]     [dispatch_simd_rem]     [promote_normal_binary];
    )]
impl std::ops::trait_name for Tensor {
    type Output = Tensor;

    fn method_name(self, rhs: Self) -> Self::Output {
        let res_layout = self.layout.broadcast(rhs.shape()).expect("broadcast failed");

        let mut res = Tensor::empty(
            &res_layout.shape(),
            promote_method(self.dtype, rhs.dtype),
            self.device.clone()
        ).expect("failed to create empty tensor");

        let (simd_kernel, unroll) = simd_dispatch(self.dtype, rhs.dtype);
        let scalar_kernel = scalar_dispatch(self.dtype, rhs.dtype);
        binary(&mut res, &self, &rhs, scalar_kernel, simd_kernel, unroll);
        res
    }
}

#[duplicate::duplicate_item(
        trait_name  method_name   scalar_dispatch   simd_dispatch           promote_method;
        [Add]       [add]         [dispatch_add]     [dispatch_simd_add]     [promote_normal_binary];
        [Sub]       [sub]         [dispatch_sub]     [dispatch_simd_sub]     [promote_normal_binary];
        [Mul]       [mul]         [dispatch_mul]     [dispatch_simd_mul]     [promote_normal_binary];
        [Rem]       [rem]         [dispatch_rem]     [dispatch_simd_rem]     [promote_normal_binary];
    )]
impl std::ops::trait_name<&Tensor> for &Tensor {
    type Output = Tensor;

    fn method_name(self, rhs: &Tensor) -> Self::Output {
        let res_layout = self.layout.broadcast(rhs.shape()).expect("broadcast failed");

        let mut res = Tensor::empty(
            &res_layout.shape(),
            promote_method(self.dtype, rhs.dtype),
            self.device.clone()
        ).expect("failed to create empty tensor");

        let (simd_kernel, unroll) = simd_dispatch(self.dtype, rhs.dtype);
        let scalar_kernel = scalar_dispatch(self.dtype, rhs.dtype);
        binary(&mut res, &self, &rhs, scalar_kernel, simd_kernel, unroll);
        res
    }
}
