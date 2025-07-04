use crate::{ Tensor, current_num_threads };
use hpt_common::{layout::layout::Layout, shape::shape_utils::mt_intervals, Pointer};
use hpt_traits::tensor::TensorInfo;
use hpt_types::dtype::{ DType, TypeCommon };

impl Tensor {
    pub fn copy_from(&mut self, other: &Self) {
        _copy_from(
            self.ptr::<u8>(),
            &self.layout,
            self.dtype,
            self.parent.is_none(),
            other.ptr::<u8>(),
            &other.layout,
            other.dtype,
            other.parent.is_none(),
            current_num_threads()
        );
    }
    pub(crate) fn copy_from_thr_specific(&mut self, other: &Self, num_threads: usize) {
        _copy_from(
            self.ptr::<u8>(),
            &self.layout,
            self.dtype,
            self.parent.is_none(),
            other.ptr::<u8>(),
            &other.layout,
            other.dtype,
            other.parent.is_none(),
            num_threads
        );
    }
}

pub(crate) fn _copy_from(
    dst: Pointer<u8>,
    dst_layout: &Layout,
    dst_dtype: DType,
    dst_has_parent: bool,
    src: Pointer<u8>,
    src_layout: &Layout,
    src_dtype: DType,
    src_has_parent: bool,
    num_threads: usize
) {
    if dst_layout.size() != src_layout.size() {
        panic!("size mismatch, self: {:?}, other: {:?}", dst_layout, src_layout);
    }
    if dst_dtype != src_dtype {
        panic!("dtype mismatch, self: {:?}, other: {:?}", dst_dtype, src_dtype);
    }

    if
        dst_layout.is_contiguous() &&
        src_layout.is_contiguous() &&
        !dst_has_parent &&
        !src_has_parent
    {
        macro_rules! copy_from_contiguous {
            ($t:ty) => {
        {
                use hpt_common::shape::shape_utils::mt_intervals;
                use hpt_types::traits::VecTrait;
                use rayon::iter::{IntoParallelIterator, ParallelIterator};
                type T = $t;
                type InVec = <T as TypeCommon>::Vec;
                let res_ptr = dst.cast::<T>();

                let func = |(start, end): (usize, usize)| {
                    let other_slice = &unsafe {
                        std::slice::from_raw_parts(src.cast::<T>().ptr as *const T, src_layout.size() as usize)
                    }[start..end];
                    let res_slice = unsafe {
                        std::slice::from_raw_parts_mut((res_ptr + start).ptr, end - start)
                    };

                    let other_chunks =
                        other_slice.chunks_exact(4 * <T as TypeCommon>::Vec::SIZE);
                    let res_chunks =
                        res_slice.chunks_exact_mut(4 * <T as TypeCommon>::Vec::SIZE);
                    other_chunks
                        .remainder()
                        .into_iter()
                        .zip(res_chunks.into_remainder().into_iter())
                        .for_each(|(lhs, res)| {
                            *res = *lhs;
                        });
                    let res_chunks =
                        res_slice.chunks_exact_mut(4 * <T as TypeCommon>::Vec::SIZE);
                    other_chunks
                        .into_iter()
                        .zip(res_chunks.into_iter())
                        .for_each(|(lhs, res)| {
                            let out_ptr = res.as_mut_ptr() as *mut InVec;
                            let lhs_ptr = lhs.as_ptr() as *const InVec;
                            unsafe {
                                out_ptr.write_unaligned(lhs_ptr.read_unaligned());
                                out_ptr
                                    .add(1)
                                    .write_unaligned(lhs_ptr.add(1).read_unaligned());
                                out_ptr
                                    .add(2)
                                    .write_unaligned(lhs_ptr.add(2).read_unaligned());
                                out_ptr
                                    .add(3)
                                    .write_unaligned(lhs_ptr.add(3).read_unaligned());
                            }
                        });
                };
                if num_threads == 1 {
                    func((0, src_layout.size() as usize));
                } else {
                    let intervals = mt_intervals(src_layout.size() as usize, num_threads);
                    intervals.into_par_iter().for_each(func);
                }
        }
            };
        }
        match dst_dtype {
            #[cfg(feature = "bool")]
            DType::Bool => copy_from_contiguous!(bool),
            #[cfg(feature = "i8")]
            DType::I8 => copy_from_contiguous!(i8),
            #[cfg(feature = "u8")]
            DType::U8 => copy_from_contiguous!(u8),
            #[cfg(feature = "i16")]
            DType::I16 => copy_from_contiguous!(i16),
            #[cfg(feature = "u16")]
            DType::U16 => copy_from_contiguous!(u16),
            #[cfg(feature = "i32")]
            DType::I32 => copy_from_contiguous!(i32),
            #[cfg(feature = "u32")]
            DType::U32 => copy_from_contiguous!(u32),
            #[cfg(feature = "i64")]
            DType::I64 => copy_from_contiguous!(i64),
            #[cfg(feature = "u64")]
            DType::U64 => copy_from_contiguous!(u64),
            #[cfg(feature = "f32")]
            DType::F32 => copy_from_contiguous!(f32),
            #[cfg(feature = "f16")]
            DType::F16 => copy_from_contiguous!(half::f16),
            #[cfg(feature = "bf16")]
            DType::BF16 => copy_from_contiguous!(half::bf16),
            #[cfg(feature = "f64")]
            DType::F64 => copy_from_contiguous!(f64),
            _ => panic!("unsupported dtype {:?}", dst_dtype),
        }
    } else {
        macro_rules! copy_from {
            ($t:ty) => {
        {
                use hpt_types::traits::VecTrait;
                use rayon::iter::IntoParallelIterator;
                use rayon::iter::ParallelIterator;
                use crate::utils::index_cal::{dispatch_loop_progress_update, dispatch_map_gp};
                type T = $t;
                type InVec = <T as TypeCommon>::Vec;

                let outer_loop_size = dst_layout.outer_loop_size();
                let inner_loop_size = dst_layout.inner_loop_size();

                let res_prg_update = dispatch_loop_progress_update(dst_layout);
                let input_prg_update = dispatch_loop_progress_update(src_layout);
                let res_map_gp = dispatch_map_gp(dst_layout);
                let input_map_gp = dispatch_map_gp(src_layout);

                let res_ptr = dst.cast::<T>();
                let input_ptr = src.cast::<T>();

                let can_vectorize =
                    dst_layout.last_stride() == 1 && src_layout.last_stride() == 1;

                let func = move |(start, end): (usize, usize)| {
                    let global_idx = start as i64 * inner_loop_size;
                    let (res_offset, mut res_prg) = res_map_gp(global_idx);
                    let (input_offset, mut input_prg) = input_map_gp(global_idx);
                    let mut res_ptr = res_ptr + res_offset;
                    let mut input_ptr = input_ptr + input_offset;
                    if can_vectorize {
                        let vec_size = InVec::SIZE;
                        let vec_count = (inner_loop_size as usize) / vec_size;
                        for _ in start..end {
                            let tmp_input_ptr = input_ptr.cast::<InVec>();
                            let tmp_res_ptr = res_ptr.cast::<InVec>();
                            for i in 0..vec_count {
                                let in_vec = (tmp_input_ptr + i).read_unaligned();
                                (tmp_res_ptr + i).write_unaligned(in_vec);
                            }
                            let tmp_input_ptr = input_ptr.cast::<T>();
                            let mut tmp_res_ptr = res_ptr.cast::<T>();
                            for i in vec_count * vec_size..inner_loop_size as usize {
                                tmp_res_ptr[i] = tmp_input_ptr[i];
                            }
                            res_ptr += res_prg_update(&mut res_prg);
                            input_ptr += input_prg_update(&mut input_prg);
                        }
                    } else {
                        for _ in start..end {
                            let tmp_input_ptr = input_ptr.cast::<T>();
                            let mut tmp_res_ptr = res_ptr.cast::<T>();
                            for i in 0..inner_loop_size {
                                tmp_res_ptr[i] = tmp_input_ptr[i];
                            }
                            res_ptr += res_prg_update(&mut res_prg);
                            input_ptr += input_prg_update(&mut input_prg);
                        }
                    }
                };
                if num_threads == 1 {
                    func((0, outer_loop_size as usize));
                } else {
                    let intervals = mt_intervals(outer_loop_size as usize, num_threads);
                    intervals.into_par_iter().for_each(func);
                }
        }
            };
        }
        match dst_dtype {
            DType::Bool => copy_from!(bool),
            #[cfg(feature = "i8")]
            DType::I8 => copy_from!(i8),
            #[cfg(feature = "u8")]
            DType::U8 => copy_from!(u8),
            #[cfg(feature = "i16")]
            DType::I16 => copy_from!(i16),
            #[cfg(feature = "u16")]
            DType::U16 => copy_from!(u16),
            #[cfg(feature = "i32")]
            DType::I32 => copy_from!(i32),
            #[cfg(feature = "u32")]
            DType::U32 => copy_from!(u32),
            #[cfg(feature = "i64")]
            DType::I64 => copy_from!(i64),
            #[cfg(feature = "u64")]
            DType::U64 => copy_from!(u64),
            #[cfg(feature = "f32")]
            DType::F32 => copy_from!(f32),
            #[cfg(feature = "f16")]
            DType::F16 => copy_from!(half::f16),
            #[cfg(feature = "bf16")]
            DType::BF16 => copy_from!(half::bf16),
            _ => panic!("unsupported dtype {:?}", dst_dtype),
        }
    }
}
