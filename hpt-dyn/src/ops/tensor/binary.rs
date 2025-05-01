use std::borrow::Borrow;

use hpt_common::error::base::TensorError;
use hpt_common::error::shape::ShapeError;
use hpt_iterator::TensorIterator;
use hpt_iterator::iterator_traits::ParStridedIteratorSimd;
use hpt_iterator::iterator_traits::ParStridedIteratorSimdZip;
use hpt_iterator::iterator_traits::ParStridedIteratorZip;
use hpt_traits::tensor::CommonBounds;
use hpt_traits::tensor::TensorInfo;
use hpt_types::dtype::ToDType;
use hpt_types::dtype::TypeCommon;
use hpt_types::promote_float_binary;
use hpt_types::promote_normal_binary;
use hpt_types::type_promote::FloatOutBinary;
use hpt_types::type_promote::NormalOut;
use rayon::iter::{ IndexedParallelIterator, IntoParallelIterator, ParallelIterator };

use crate::Tensor;

use half::{ bf16, f16 };

pub(crate) fn binary_fn_with_out<A, B, K, O, F, F2>(
    lhs: &Tensor,
    rhs: &Tensor,
    f: F,
    f2: F2,
    out: Option<O>
)
    -> std::result::Result<Tensor, TensorError>
    where
        A: CommonBounds,
        B: CommonBounds,
        K: CommonBounds + ToDType,
        O: Borrow<Tensor>,
        F: Fn(A, B) -> K + Sync + Send + Copy,
        F2: Fn(<A as TypeCommon>::Vec, <B as TypeCommon>::Vec) -> <K as TypeCommon>::Vec +
            Sync +
            Send +
            Copy
{
    use hpt_types::traits::*;
    use rayon::slice::{ ParallelSlice, ParallelSliceMut };
    if lhs.size() == 1 {
        let val = lhs.as_slice::<A>()[0];
        let val_vec = <A as TypeCommon>::Vec::splat(val);
        let mut res = if let Some(out) = out {
            ShapeError::check_inplace_out_layout_valid(rhs.shape(), &out.borrow().layout())?;
            let out: &Tensor = out.borrow();
            assert_eq!(out.dtype, K::to_dtype());
            out.clone()
        } else {
            Tensor::empty(rhs.shape(), K::to_dtype(), rhs.device.clone())?
        };
        if rhs.is_contiguous() && B::BYTE_SIZE == K::BYTE_SIZE && A::BYTE_SIZE == K::BYTE_SIZE {
            let chunks = rhs.as_slice::<B>().par_chunks_exact(4 * <B as TypeCommon>::Vec::SIZE);
            let mut res_chunks = res
                .as_slice_mut::<K>()
                .par_chunks_exact_mut(4 * <K as TypeCommon>::Vec::SIZE);

            chunks
                .remainder()
                .into_iter()
                .zip(res_chunks.remainder().into_iter())
                .for_each(|(rhs, res)| {
                    *res = f(val, *rhs);
                });

            chunks
                .into_par_iter()
                .zip(res_chunks.into_par_iter())
                .for_each(|(rhs, res)| {
                    let out_ptr = res.as_mut_ptr() as *mut K::Vec;
                    let buffer_ptr = rhs.as_ptr() as *const B::Vec;
                    unsafe {
                        seq_macro::seq!(N in 0..4 {
                            out_ptr.add(N).write_unaligned(f2(val_vec, buffer_ptr.add(N).read_unaligned()));
                        });
                    }
                });
        } else {
            res.par_iter_mut()
                .zip(rhs.par_iter())
                .for_each(|(a, b)| {
                    *a = f(val, b);
                });
        }
        Ok(res)
    } else if rhs.size() == 1 {
        let val = rhs.as_slice::<B>()[0];
        let val_vec = <B as TypeCommon>::Vec::splat(val);
        let mut res = if let Some(out) = out {
            ShapeError::check_inplace_out_layout_valid(lhs.shape(), &out.borrow().layout())?;
            let out: &Tensor = out.borrow();
            assert_eq!(out.dtype, K::to_dtype());
            out.clone()
        } else {
            Tensor::empty(lhs.shape(), K::to_dtype(), lhs.device.clone())?
        };
        if lhs.is_contiguous() {
            let chunks = lhs.as_slice::<A>().par_chunks_exact(4 * <A as TypeCommon>::Vec::SIZE);
            let mut res_chunks = res
                .as_slice_mut::<K>()
                .par_chunks_exact_mut(4 * <K as TypeCommon>::Vec::SIZE);

            chunks
                .remainder()
                .into_iter()
                .zip(res_chunks.remainder().into_iter())
                .for_each(|(lhs, res)| {
                    *res = f(*lhs, val);
                });

            chunks
                .into_par_iter()
                .zip(res_chunks.into_par_iter())
                .for_each(|(lhs, res)| {
                    let out_ptr = res.as_mut_ptr() as *mut K::Vec;
                    let buffer_ptr = lhs.as_ptr() as *const A::Vec;
                    unsafe {
                        seq_macro::seq!(N in 0..4 {
                            out_ptr.add(N).write_unaligned(f2(buffer_ptr.add(N).read_unaligned(), val_vec));
                        });
                    }
                });
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
            let mut res = if let Some(out) = out {
                ShapeError::check_inplace_out_layout_valid(rhs.shape(), &out.borrow().layout())?;
                let out: &Tensor = out.borrow();
                out.clone()
            } else {
                Tensor::empty(rhs.shape(), K::to_dtype(), rhs.device.clone())?
            };
            if A::BYTE_SIZE == K::BYTE_SIZE && B::BYTE_SIZE == K::BYTE_SIZE {
                let lhs_chunks = lhs
                    .as_slice::<A>()
                    .par_chunks_exact(4 * <A as TypeCommon>::Vec::SIZE);
                let rhs_chunks = rhs
                    .as_slice::<B>()
                    .par_chunks_exact(4 * <B as TypeCommon>::Vec::SIZE);
                let mut res_chunks = res
                    .as_slice_mut::<K>()
                    .par_chunks_exact_mut(4 * <K as TypeCommon>::Vec::SIZE);

                lhs_chunks
                    .remainder()
                    .into_iter()
                    .zip(res_chunks.remainder().into_iter())
                    .zip(rhs_chunks.remainder().into_iter())
                    .for_each(|((lhs, res), rhs)| {
                        *res = f(*lhs, *rhs);
                    });

                lhs_chunks
                    .into_par_iter()
                    .zip(rhs_chunks.into_par_iter())
                    .zip(res_chunks.into_par_iter())
                    .for_each(|((lhs, rhs), res)| {
                        let out_ptr = res.as_mut_ptr() as *mut K::Vec;
                        let lhs_ptr = lhs.as_ptr() as *const A::Vec;
                        let rhs_ptr = rhs.as_ptr() as *const B::Vec;
                        unsafe {
                            seq_macro::seq!(N in 0..4 {
                                out_ptr.add(N).write_unaligned(
                                    f2(lhs_ptr.add(N).read_unaligned(), rhs_ptr.add(N).read_unaligned())
                                );
                            });
                        }
                    });
            } else {
                lhs.as_slice::<A>()
                    .into_par_iter()
                    .zip(rhs.as_slice::<B>().into_par_iter())
                    .zip(res.as_slice_mut::<K>().into_par_iter())
                    .for_each(|((lhs, rhs), res)| {
                        *res = f(*lhs, *rhs);
                    });
            }
            Ok(res)
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

impl Tensor {
    #[duplicate::duplicate_item(
        func_name       trait_name        method_name       description                                     test_code;
        [add_]          [NormalOut]       [_add]            ["Compute addition of two tensors"]             ["let res = a.add_(b, &mut c);"];
        [sub_]          [NormalOut]       [_sub]            ["Compute subtraction of two tensors"]          ["let res = a.sub_(b, &mut c);"];
        [mul_]          [NormalOut]       [_mul]            ["Compute multiplication of two tensors"]       ["let res = a.mul_(b, &mut c);"];
        [rem_]          [NormalOut]       [_rem]            ["Compute remainder of two tensors"]            ["let res = a.rem_(b, &mut c);"];
        [div_]          [FloatOutBinary]  [_div]            ["Compute division of two tensors"]             ["let res = a.div_(b, &mut c);"];
    )]
    pub fn func_name(&self, other: &Tensor, out: &mut Tensor) -> Result<Tensor, TensorError> {
        macro_rules! binary {
            ($lhs:ty, $rhs:ty) => {
        {
                type LHS = $lhs;
                type RHS = $rhs;
                type LHSVec = <LHS as TypeCommon>::Vec;
                type RHSVec = <RHS as TypeCommon>::Vec;
                let res_layout = self
                    .layout
                    .broadcast(other.shape())
                    .expect("broadcast failed");
                ShapeError::check_inplace_out_layout_valid(out.shape(), &res_layout)?;
                binary_fn_with_out(
                    self,
                    other,
                    |x: LHS, y: RHS| x.method_name(y),
                    |x: LHSVec, y: RHSVec| x.method_name(y),
                    Some(out),
                )
        }
            };
        }
        match (self.dtype, other.dtype) {
            (hpt_types::dtype::DType::I8, hpt_types::dtype::DType::I8) => binary!(i8, i8),
            (hpt_types::dtype::DType::I8, hpt_types::dtype::DType::U8) => binary!(i8, u8),
            (hpt_types::dtype::DType::I8, hpt_types::dtype::DType::F32) => binary!(i8, f32),
            (hpt_types::dtype::DType::I8, hpt_types::dtype::DType::F16) => binary!(i8, f16),
            (hpt_types::dtype::DType::I8, hpt_types::dtype::DType::BF16) => binary!(i8, bf16),
            (hpt_types::dtype::DType::U8, hpt_types::dtype::DType::I8) => binary!(u8, i8),
            (hpt_types::dtype::DType::U8, hpt_types::dtype::DType::U8) => binary!(u8, u8),
            (hpt_types::dtype::DType::U8, hpt_types::dtype::DType::F32) => binary!(u8, f32),
            (hpt_types::dtype::DType::U8, hpt_types::dtype::DType::F16) => binary!(u8, f16),
            (hpt_types::dtype::DType::U8, hpt_types::dtype::DType::BF16) => binary!(u8, bf16),
            (hpt_types::dtype::DType::F32, hpt_types::dtype::DType::I8) => binary!(f32, i8),
            (hpt_types::dtype::DType::F32, hpt_types::dtype::DType::U8) => binary!(f32, u8),
            (hpt_types::dtype::DType::F32, hpt_types::dtype::DType::F32) => binary!(f32, f32),
            (hpt_types::dtype::DType::F32, hpt_types::dtype::DType::F16) => binary!(f32, f16),
            (hpt_types::dtype::DType::F32, hpt_types::dtype::DType::BF16) => binary!(f32, bf16),
            (hpt_types::dtype::DType::F16, hpt_types::dtype::DType::I8) => binary!(f16, i8),
            (hpt_types::dtype::DType::F16, hpt_types::dtype::DType::U8) => binary!(f16, u8),
            (hpt_types::dtype::DType::F16, hpt_types::dtype::DType::F32) => binary!(f16, f32),
            (hpt_types::dtype::DType::F16, hpt_types::dtype::DType::F16) => binary!(f16, f16),
            (hpt_types::dtype::DType::F16, hpt_types::dtype::DType::BF16) => binary!(f16, bf16),
            (hpt_types::dtype::DType::BF16, hpt_types::dtype::DType::I8) => binary!(bf16, i8),
            (hpt_types::dtype::DType::BF16, hpt_types::dtype::DType::U8) => binary!(bf16, u8),
            (hpt_types::dtype::DType::BF16, hpt_types::dtype::DType::F32) => binary!(bf16, f32),
            (hpt_types::dtype::DType::BF16, hpt_types::dtype::DType::F16) => binary!(bf16, f16),
            (hpt_types::dtype::DType::BF16, hpt_types::dtype::DType::BF16) => binary!(bf16, bf16),
            _ => unimplemented!(),
        }
    }

    pub fn mul_add(&self, other: &Tensor, other2: &Tensor) -> Result<Tensor, TensorError> {
        assert_eq!(self.dtype, other.dtype);
        assert_eq!(self.dtype, other2.dtype);
        let res_layout = self.layout.broadcast(other.shape()).expect("broadcast failed");
        let mut res = Tensor::empty(res_layout.shape(), self.dtype, self.device.clone())?;
        macro_rules! mul_add {
            ($type: ty) => {
                {
                    use rayon::prelude::{ ParallelSlice, ParallelSliceMut };
                    use hpt_types::traits::VecTrait;
                    type T = $type;
                    if
                        self.is_contiguous() &&
                        other.is_contiguous() &&
                        other2.is_contiguous() &&
                        self.shape() == other.shape() &&
                        self.shape() == other2.shape()
                    {
                        let a_chunks = self
                            .as_slice::<T>()
                            .par_chunks_exact(4 * <T as TypeCommon>::Vec::SIZE);
                        let b_chunks = other
                            .as_slice::<T>()
                            .par_chunks_exact(4 * <T as TypeCommon>::Vec::SIZE);
                        let c_chunks = other2
                            .as_slice::<T>()
                            .par_chunks_exact(4 * <T as TypeCommon>::Vec::SIZE);
                        let mut res_chunks = res
                            .as_slice_mut::<T>()
                            .par_chunks_exact_mut(4 * <T as TypeCommon>::Vec::SIZE);
                        res_chunks
                            .remainder()
                            .into_iter()
                            .zip(a_chunks.remainder().into_iter())
                            .zip(b_chunks.remainder().into_iter())
                            .zip(c_chunks.remainder().into_iter())
                            .for_each(|(((res, a), b), c)| {
                                *res = a._mul_add(*b, *c);
                            });
                        res_chunks
                            .into_par_iter()
                            .zip(a_chunks.into_par_iter())
                            .zip(b_chunks.into_par_iter())
                            .zip(c_chunks.into_par_iter())
                            .for_each(|(((res, a), b), c)| {
                                let out_ptr = res.as_mut_ptr() as *mut <T as TypeCommon>::Vec;
                                let a_ptr = a.as_ptr() as *const <T as TypeCommon>::Vec;
                                let b_ptr = b.as_ptr() as *const <T as TypeCommon>::Vec;
                                let c_ptr = c.as_ptr() as *const <T as TypeCommon>::Vec;
                                unsafe {
                                    seq_macro::seq!(N in 0..4 {
                                    out_ptr.add(N).write_unaligned(
                                        a_ptr.add(N).read_unaligned().mul_add(
                                            b_ptr.add(N).read_unaligned(),
                                            c_ptr.add(N).read_unaligned()
                                        )
                                    );
                                });
                                }
                            });
                    } else {
                        let iter = res
                            .par_iter_mut_simd()
                            .zip(self.par_iter_simd())
                            .zip(other.par_iter_simd())
                            .zip(other2.par_iter_simd());
                        ParStridedIteratorSimd::for_each(
                            iter,
                            |(((res, a), b), c): (((&mut T, T), T), T)| {
                                *res = a._mul_add(b, c);
                            },
                            |(((res, a), b), c)| {
                                res.write_unaligned(a.mul_add(b, c));
                            }
                        );
                    }
                }
            };
        }
        match self.dtype {
            hpt_types::dtype::DType::Bool => mul_add!(bool),
            hpt_types::dtype::DType::I8 => mul_add!(i8),
            hpt_types::dtype::DType::U8 => mul_add!(u8),
            hpt_types::dtype::DType::I16 => mul_add!(i16),
            hpt_types::dtype::DType::U16 => mul_add!(u16),
            hpt_types::dtype::DType::I32 => mul_add!(i32),
            hpt_types::dtype::DType::U32 => mul_add!(u32),
            hpt_types::dtype::DType::I64 => mul_add!(i64),
            hpt_types::dtype::DType::F32 => mul_add!(f32),
            hpt_types::dtype::DType::F16 => mul_add!(f16),
            hpt_types::dtype::DType::BF16 => mul_add!(bf16),
        }
        Ok(res)
    }
}

#[duplicate::duplicate_item(
        trait_name  method_name  promote_method             func_name;
        [Add]       [add]        [promote_normal_binary]    [add_];
        [Sub]       [sub]        [promote_normal_binary]    [sub_];
        [Mul]       [mul]        [promote_normal_binary]    [mul_];
        [Rem]       [rem]        [promote_normal_binary]    [rem_];
        [Div]       [div]        [promote_float_binary]     [div_];
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

        self.func_name(&rhs, &mut res).expect(
            format!("{} failed", stringify!(method_name)).as_str()
        )
    }
}

#[duplicate::duplicate_item(
        trait_name  method_name   promote_method                func_name;
        [Add]       [add]         [promote_normal_binary]       [add_];
        [Sub]       [sub]         [promote_normal_binary]       [sub_];
        [Mul]       [mul]         [promote_normal_binary]       [mul_];
        [Rem]       [rem]         [promote_normal_binary]       [rem_];
        [Div]       [div]         [promote_float_binary]        [div_];
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

        self.func_name(&rhs, &mut res).expect(
            format!("{} failed", stringify!(method_name)).as_str()
        )
    }
}
