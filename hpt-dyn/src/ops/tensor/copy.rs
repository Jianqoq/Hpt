// use crate::Tensor;
// use hpt_traits::tensor::TensorInfo;
// use hpt_types::dtype::{ DType, TypeCommon };
// use hpt_iterator::TensorIterator;

// impl Tensor {
//     pub fn copy_from(&mut self, other: &Self) {
//         if self.layout.shape() != other.layout.shape() {
//             panic!("shape mismatch");
//         }
//         if self.dtype != other.dtype {
//             panic!("dtype mismatch");
//         }

//         if
//             self.is_contiguous() &&
//             other.is_contiguous() &&
//             self.parent.is_none() &&
//             other.parent.is_none()
//         {
//             macro_rules! copy_from_contiguous {
//                 ($t: ty) => {
//                     {
//                         use rayon::prelude::{ ParallelSlice, ParallelSliceMut };
//                         use rayon::iter::{
//                             IntoParallelIterator,
//                             IndexedParallelIterator,
//                             ParallelIterator,
//                         };
    
//                         use hpt_types::traits::VecTrait;
//                         type T = $t;
//                         type InVec = <T as TypeCommon>::Vec;
//                         let other_chunks = other
//                             .as_slice::<T>()
//                             .par_chunks_exact(4 * <T as TypeCommon>::Vec::SIZE);
//                         let mut res_chunks = self
//                             .as_slice_mut::<T>()
//                             .par_chunks_exact_mut(4 * <T as TypeCommon>::Vec::SIZE);
    
//                         other_chunks
//                             .remainder()
//                             .into_iter()
//                             .zip(res_chunks.remainder().into_iter())
//                             .for_each(|(lhs, res)| {
//                                 *res = *lhs;
//                             });
    
//                         other_chunks
//                             .into_par_iter()
//                             .zip(res_chunks.into_par_iter())
//                             .for_each(|(lhs, res)| {
//                                 let out_ptr = res.as_mut_ptr() as *mut InVec;
//                                 let lhs_ptr = lhs.as_ptr() as *const InVec;
//                                 unsafe {
//                                     out_ptr.write_unaligned(lhs_ptr.read_unaligned());
//                                     out_ptr.add(1).write_unaligned(lhs_ptr.add(1).read_unaligned());
//                                     out_ptr.add(2).write_unaligned(lhs_ptr.add(2).read_unaligned());
//                                     out_ptr.add(3).write_unaligned(lhs_ptr.add(3).read_unaligned());
//                                 }
//                             });
//                     }
//                 };
//             }
//             match self.dtype {
//                 DType::Bool => copy_from_contiguous!(bool),
//                 DType::I8 => copy_from_contiguous!(i8),
//                 DType::U8 => copy_from_contiguous!(u8),
//                 DType::I16 => copy_from_contiguous!(i16),
//                 DType::U16 => copy_from_contiguous!(u16),
//                 DType::I32 => copy_from_contiguous!(i32),
//                 DType::U32 => copy_from_contiguous!(u32),
//                 DType::I64 => copy_from_contiguous!(i64),
//                 DType::F32 => copy_from_contiguous!(f32),
//                 DType::F16 => copy_from_contiguous!(half::f16),
//                 DType::BF16 => copy_from_contiguous!(half::bf16),
//             }
//         } else {
//             macro_rules! copy_from {
//                 ($t:ty) => {
//                     {
//                         use hpt_iterator::iterator_traits::{ ParStridedIteratorSimd, ParStridedIteratorSimdZip };
//                         use hpt_common::utils::simd_ref::MutVec;
//                         type T = $t;
//                         type InVec = <T as TypeCommon>::Vec;
//                         let iter = self.par_iter_mut_simd().zip(other.par_iter_simd());
//                         ParStridedIteratorSimd::for_each(
//                             iter,
//                             |(x, y): (&mut T, T)| {
//                                 *x = y;
//                             },
//                             |(x, y): (MutVec<'_, InVec>, InVec)| {
//                                 x.write_unaligned(y);
//                             }
//                         );
//                     }
//                 };
//             }
//             match self.dtype {
//                 DType::Bool => copy_from!(bool),
//                 DType::I8 => copy_from!(i8),
//                 DType::U8 => copy_from!(u8),
//                 DType::I16 => copy_from!(i16),
//                 DType::U16 => copy_from!(u16),
//                 DType::I32 => copy_from!(i32),
//                 DType::U32 => copy_from!(u32),
//                 DType::I64 => copy_from!(i64),
//                 DType::F32 => copy_from!(f32),
//                 DType::F16 => copy_from!(half::f16),
//                 DType::BF16 => copy_from!(half::bf16),
//             }
//         }
//     }
// }
