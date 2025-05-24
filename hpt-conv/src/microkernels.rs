/// conv2d micro kernel
#[macro_export]
macro_rules! conv2d_micro_kernel {
    ($name:ident, $dtype: ty, $vec_type: ty, $nr:expr, $mr:expr) => {
        fn $name(
            inp: crate::Pointer<$dtype>,
            kernel: crate::Pointer<$dtype>,
            mut out: crate::Pointer<$dtype>,
            icb: i64,
            osw: i64,
            kernel_idx: &mut i64,
            [k, j, l]: [i64; 3],
            [kh, kw]: [i64; 2],
            [step_height, step_width]: [i64; 2],
            _: [i64; 2],
            _: [i64; 2],
            [ish, isw]: [i64; 2],
            [owr, ocr]: [i64; 2],
            [dh, dw]: [i64; 2],
            first_ic_iter: bool,
        ) {
            use crate::VecTrait;
            let mut c_local = if first_ic_iter {
                [[<$vec_type>::splat(<$dtype>::ZERO); $nr]; $mr]
            } else {
                let mut c_local = [[<$vec_type>::splat(<$dtype>::ZERO); $nr]; $mr];
                if ocr == $nr * <$vec_type>::SIZE as i64 {
                    for mr in 0..owr {
                        unsafe {
                            let out_ptr = out.ptr.offset(((mr + k) * osw + j) as isize);
                            seq_macro::seq!(NR in 0..$nr {
                                let ptr = out_ptr.add(NR * <$vec_type>::SIZE) as *const $vec_type;
                                c_local[mr as usize][NR] = ptr.read_unaligned();
                            });
                        }
                    }
                } else {
                    for mr in 0..owr {
                        let reg = c_local[mr as usize].as_mut_ptr() as *mut $dtype;
                        for nr in 0..ocr as i64 {
                            let val = out[(mr + k) * osw + j + nr];
                            unsafe { reg.add(nr as usize).write(val) };
                        }
                    }
                }
                c_local
            };
            for n in 0..kh {
                let in_y = l * step_height + n * dh;
                for m in 0..kw {
                    for ii in 0..icb {
                        let kernel = unsafe { kernel.ptr.add(*kernel_idx as usize) };
                        seq_macro::seq!(NR in 0..$nr {
                            let b_vec~NR = unsafe {
                                    *(kernel.add(NR * <$vec_type>::SIZE)
                                        as *const $vec_type)
                                };
                            }
                        );
                        #[allow(unused_mut)]
                        let mut a_vec;
                        seq_macro::seq!(MR in 0..$mr {
                            let in_x = (k + MR) * step_width + m * dw;
                            a_vec = <$vec_type>::splat(inp[in_y * ish + in_x * isw + ii]);
                            seq_macro::seq!(NR in 0..$nr {
                                c_local[MR][NR] = a_vec.mul_add(b_vec~NR, c_local[MR][NR]);
                            });
                            }
                        );
                        *kernel_idx += $nr * <$vec_type>::SIZE as i64;
                    }
                }
            }
            if ocr == $nr * <$vec_type>::SIZE as i64 {
                for mr in 0..owr {
                    unsafe {
                        let out_ptr = out.ptr.offset(((mr + k) * osw + j) as isize);
                        seq_macro::seq!(NR in 0..$nr {
                            let vec = c_local[mr as usize][NR];
                            let ptr = out_ptr.add(NR * <$vec_type>::SIZE) as *mut $vec_type;
                            ptr.write_unaligned(vec);
                        });
                    }
                }
            } else {
                for mr in 0..owr {
                    let reg = c_local[mr as usize].as_ptr() as *const $dtype;
                    for nr in 0..ocr as i64 {
                        out[(mr + k) * osw + j + nr] =
                            unsafe { reg.add(nr as usize).read() };
                    }
                }
            }
        }
    };
}

/// conv2d mixed precision micro kernel
#[macro_export]
macro_rules! conv2d_mixed_precision_micro_kernel {
    ($name:ident, $dtype:ty, $vec_type:ty, $mix_type:ty, $mix_vec:ty, $nr:expr, $mr:expr, $nr2:expr) => {
        fn $name(
            inp: crate::Pointer<$mix_type>,
            kernel: crate::Pointer<$mix_type>,
            mut out: crate::Pointer<$dtype>,
            icb: i64,
            osw: i64,
            kernel_idx: &mut i64,
            [k, j, l]: [i64; 3],
            [kh, kw]: [i64; 2],
            [step_height, step_width]: [i64; 2],
            _: [i64; 2],
            _: [i64; 2],
            [ish, isw]: [i64; 2],
            [owr, ocr]: [i64; 2],
            [dh, dw]: [i64; 2],
            first_ic_iter: bool,
            vec_cast: fn(*mut $mix_vec, *const $dtype),
            vec_cast_back: fn(*mut $vec_type, *const $mix_vec),
            cast: fn($dtype) -> $mix_type,
            cast_back: fn(&mut $dtype, &$mix_type),
        ) {
            use crate::VecTrait;
            use matconv_simd::Zero;
            let mut c_local = if first_ic_iter {
                [[<$mix_vec>::splat(<$mix_type>::ZERO); $nr2]; $mr]
            } else {
                let mut c_local = [[<$mix_vec>::splat(<$mix_type>::ZERO); $nr2]; $mr];
                if ocr == $nr2 * <$mix_vec>::SIZE as i64 {
                    for mr in 0..owr {
                        let reg = c_local[mr as usize].as_mut_ptr() as *mut $mix_vec;
                        let out_ptr = unsafe { out.ptr.offset(((mr + k) * osw + j) as isize) } as *const $dtype;
                        seq_macro::seq!(NR in 0..$nr2 {
                            let mut res = <$mix_vec>::splat(<$mix_type>::ZERO);
                            vec_cast(&mut res, unsafe { out_ptr.add(NR * <$mix_vec>::SIZE) });
                            unsafe { reg.add(NR as usize).write(res) };
                        });
                    }
                } else {
                    for mr in 0..owr {
                        let reg = c_local[mr as usize].as_mut_ptr() as *mut $mix_type;
                        for nr in 0..ocr as i64 {
                            let val = cast(out[(mr + k) * osw + j + nr]);
                            unsafe { reg.add(nr as usize).write(val) };
                        }
                    }
                }
                c_local
            };
            for n in 0..kh {
                let in_y = l * step_height + n * dh;
                for m in 0..kw {
                    for ii in 0..icb {
                        let kernel = unsafe { kernel.ptr.add(*kernel_idx as usize) };
                        seq_macro::seq!(NR in 0..$nr2 {
                            let b_vec~NR = unsafe {
                                    *(kernel.add(NR * <$mix_vec>::SIZE)
                                        as *const $mix_vec)
                                };
                            }
                        );
                        #[allow(unused_mut)]
                        let mut a_vec;
                        seq_macro::seq!(MR in 0..$mr {
                            let in_x = (k + MR) * step_width + m * dw;
                            a_vec = <$mix_vec>::splat(inp[in_y * ish + in_x * isw + ii]);
                            seq_macro::seq!(NR in 0..$nr2 {
                                c_local[MR][NR] = a_vec.mul_add(b_vec~NR, c_local[MR][NR]);
                            });
                            }
                        );
                        *kernel_idx += $nr2 * <$mix_vec>::SIZE as i64;
                    }
                }
            }
            if ocr == $nr2 * <$mix_vec>::SIZE as i64 {
                for mr in 0..owr {
                    let out_ptr = unsafe { out.ptr.offset(((mr + k) * osw + j) as isize) } as *mut $vec_type;
                    for i in 0..$nr2 / (size_of::<$mix_type>() as i64 / size_of::<$dtype>() as i64) {
                        unsafe {
                            let c_local_ptr = c_local[mr as usize].as_ptr() as *const $mix_vec;
                            let mut res = <$vec_type>::splat(<$dtype>::ZERO);
                            vec_cast_back(&mut res, c_local_ptr.offset(i as isize));
                            out_ptr.offset(i as isize).write_unaligned(res);
                        }
                    }
                }
            } else {
                for mr in 0..owr {
                    let reg = c_local[mr as usize].as_ptr() as *const $mix_type;
                    for nr in 0..ocr as i64 {
                        unsafe {
                            let mut res = <$dtype>::ZERO;
                            cast_back(&mut res, &*reg.add(nr as usize) );
                            out[(mr + k) * osw + j + nr] = res;
                        }
                    }
                }
            }
        }
    };
}

/// conv2d micro kernel with padding
#[macro_export]
macro_rules! conv2d_micro_kernel_with_padding {
    ($name:ident, $dtype: ty, $vec_type: ty, $nr:expr, $mr:expr) => {
        fn $name(
            inp: crate::Pointer<$dtype>,
            kernel: crate::Pointer<$dtype>,
            mut out: crate::Pointer<$dtype>,
            icb: i64,
            osw: i64,
            kernel_idx: &mut i64,
            [k, j, l]: [i64; 3],
            [kh, kw]: [i64; 2],
            [step_height, step_width]: [i64; 2],
            [ph_start, pw_start]: [i64; 2],
            [img_height, img_width]: [i64; 2],
            [ish, isw]: [i64; 2],
            [owr, ocr]: [i64; 2],
            [dh, dw]: [i64; 2],
            first_ic_iter: bool,
        ) {
            use crate::VecTrait;
            let mut c_local = [[<$vec_type>::splat(<$dtype>::ZERO); $nr]; $mr];
            for n in 0..kh {
                let in_y = l * step_height + n * dh - ph_start;
                if in_y < 0 || in_y >= img_height {
                    *kernel_idx += icb * kw * $nr * <$vec_type>::SIZE as i64;
                    continue;
                }
                for m in 0..kw {
                    let max_in_x = (k + $mr - 1) * step_width + m * dw - pw_start;
                    let min_in_x = k * step_width + m * dw - pw_start;
                    if min_in_x >= 0 && max_in_x < img_width {
                        for ii in 0..icb {
                            let kernel = unsafe { kernel.ptr.add(*kernel_idx as usize) };
                            seq_macro::seq!(NR in 0..$nr {
                                let b_vec~NR = unsafe {
                                        *(kernel.add(NR * <$vec_type>::SIZE)
                                            as *const $vec_type)
                                    };
                                }
                            );
                            #[allow(unused_mut)]
                            let mut a_vec;
                            seq_macro::seq!(MR in 0..$mr {
                                let in_x = (k + MR) * step_width + m * dw - pw_start;
                                a_vec = <$vec_type>::splat(inp[in_y * ish + in_x * isw + ii]);
                                seq_macro::seq!(NR in 0..$nr {
                                    c_local[MR][NR] = a_vec.mul_add(b_vec~NR, c_local[MR][NR]);
                                });
                                }
                            );
                            *kernel_idx += $nr * <$vec_type>::SIZE as i64;
                        }
                    } else {
                        for ii in 0..icb {
                            let kernel = unsafe { kernel.ptr.add(*kernel_idx as usize) };
                            seq_macro::seq!(NR in 0..$nr {
                                let b_vec~NR = unsafe {
                                        *(kernel.add(NR * <$vec_type>::SIZE)
                                            as *const $vec_type)
                                    };
                                }
                            );
                            #[allow(unused_mut)]
                            let mut a_vec;
                            seq_macro::seq!(MR in 0..$mr {
                                let in_x = (k + MR) * step_width + m * dw - pw_start;
                                if in_x >= 0 && in_x < img_width {
                                    a_vec = <$vec_type>::splat(inp[in_y * ish + in_x * isw + ii]);
                                    seq_macro::seq!(NR in 0..$nr {
                                        c_local[MR][NR] = a_vec.mul_add(b_vec~NR, c_local[MR][NR]);
                                    });
                                }
                                }
                            );
                            *kernel_idx += $nr * <$vec_type>::SIZE as i64;
                        }
                    }
                }
            }
            if first_ic_iter {
                for mr in 0..owr {
                    let reg = c_local[mr as usize].as_ptr() as *const $dtype;
                    for nr in 0..ocr as i64 {
                        out[(mr + k) * osw + j + nr] =
                            unsafe { reg.add(nr as usize).read() };
                    }
                }
            } else {
                for mr in 0..owr {
                    let reg = c_local[mr as usize].as_ptr() as *const $dtype;
                    for nr in 0..ocr as i64 {
                        let val = out[(mr + k) * osw + j + nr];
                        out[(mr + k) * osw + j + nr] =
                            val.add(unsafe { reg.add(nr as usize).read() });
                    }
                }
            }
        }
    };
}

/// conv2d mixed precision micro kernel
#[macro_export]
macro_rules! conv2d_mixed_precision_micro_kernel_with_padding {
    ($name:ident, $dtype:ty, $vec_type:ty, $mix_type:ty, $mix_vec:ty, $nr:expr, $mr:expr, $nr2:expr) => {
        fn $name(
            inp: crate::Pointer<$mix_type>,
            kernel: crate::Pointer<$mix_type>,
            mut out: crate::Pointer<$dtype>,
            icb: i64,
            osw: i64,
            kernel_idx: &mut i64,
            [k, j, l]: [i64; 3],
            [kh, kw]: [i64; 2],
            [step_height, step_width]: [i64; 2],
            [ph_start, pw_start]: [i64; 2],
            [img_height, img_width]: [i64; 2],
            [ish, isw]: [i64; 2],
            [owr, ocr]: [i64; 2],
            [dh, dw]: [i64; 2],
            first_ic_iter: bool,
            vec_cast: fn(*mut $mix_vec, *const $dtype),
            vec_cast_back: fn(*mut $vec_type, *const $mix_vec),
            cast: fn($dtype) -> $mix_type,
            cast_back: fn(&mut $dtype, &$mix_type),
        ) {
            use crate::VecTrait;
            use matconv_simd::Zero;
            let mut c_local = if first_ic_iter {
                [[<$mix_vec>::splat(<$mix_type>::ZERO); $nr2]; $mr]
            } else {
                let mut c_local = [[<$mix_vec>::splat(<$mix_type>::ZERO); $nr2]; $mr];
                if ocr == $nr2 * <$mix_vec>::SIZE as i64 {
                    for mr in 0..owr {
                        let reg = c_local[mr as usize].as_mut_ptr() as *mut $mix_vec;
                        let out_ptr = unsafe { out.ptr.offset(((mr + k) * osw + j) as isize) } as *const $dtype;
                        seq_macro::seq!(NR in 0..$nr2 {
                            let mut res = $mix_vec::splat($mix_type::ZERO);
                            vec_cast(&mut res, unsafe { out_ptr.add(NR * $mix_vec::SIZE) });
                            unsafe { reg.add(NR as usize).write(res) };
                        });
                    }
                } else {
                    for mr in 0..owr {
                        let reg = c_local[mr as usize].as_mut_ptr() as *mut $mix_type;
                        for nr in 0..ocr as i64 {
                            let val = cast(out[(mr + k) * osw + j + nr]);
                            unsafe { reg.add(nr as usize).write(val) };
                        }
                    }
                }
                c_local
            };
            for n in 0..kh {
                let in_y = l * step_height + n * dh - ph_start;
                if in_y < 0 || in_y >= img_height {
                    *kernel_idx += icb * kw * $nr * <$mix_vec>::SIZE as i64;
                    continue;
                }
                for m in 0..kw {
                    let max_in_x = (k + $mr - 1) * step_width + m * dw - pw_start;
                    let min_in_x = k * step_width + m * dw - pw_start;
                    if min_in_x >= 0 && max_in_x < img_width {
                        for ii in 0..icb {
                            let kernel = unsafe { kernel.ptr.add(*kernel_idx as usize) };
                            seq_macro::seq!(NR in 0..$nr2 {
                                let b_vec~NR = unsafe {
                                        *(kernel.add(NR * <$mix_vec>::SIZE)
                                            as *const $mix_vec)
                                    };
                                }
                            );
                            #[allow(unused_mut)]
                            let mut a_vec;
                            seq_macro::seq!(MR in 0..$mr {
                                let in_x = (k + MR) * step_width + m * dw - pw_start;
                                a_vec = <$mix_vec>::splat(inp[in_y * ish + in_x * isw + ii]);
                                seq_macro::seq!(NR in 0..$nr2 {
                                    c_local[MR][NR] = a_vec.mul_add(b_vec~NR, c_local[MR][NR]);
                                });
                                }
                            );
                            *kernel_idx += $nr2 * <$mix_vec>::SIZE as i64;
                        }
                    } else {
                        for ii in 0..icb {
                            let kernel = unsafe { kernel.ptr.add(*kernel_idx as usize) };
                            seq_macro::seq!(NR in 0..$nr2 {
                                let b_vec~NR = unsafe {
                                        *(kernel.add(NR * <$mix_vec>::SIZE)
                                            as *const $mix_vec)
                                    };
                                }
                            );
                            #[allow(unused_mut)]
                            let mut a_vec;
                            seq_macro::seq!(MR in 0..$mr {
                                let in_x = (k + MR) * step_width + m * dw - pw_start;
                                if in_x >= 0 && in_x < img_width {
                                    a_vec = <$mix_vec>::splat(inp[in_y * ish + in_x * isw + ii]);
                                    seq_macro::seq!(NR in 0..$nr2 {
                                        c_local[MR][NR] = a_vec.mul_add(b_vec~NR, c_local[MR][NR]);
                                    });
                                }
                            });
                            *kernel_idx += $nr2 * <$mix_vec>::SIZE as i64;
                        }
                    }
                }
            }
            if ocr == $nr2 * <$mix_vec>::SIZE as i64 {
                for mr in 0..owr {
                    let out_ptr = unsafe { out.ptr.offset(((mr + k) * osw + j) as isize) } as *mut $vec_type;
                    for i in 0..$nr2 / (std::mem::size_of::<$mix_type>() as i64 / std::mem::size_of::<$dtype>() as i64) {
                        unsafe {
                            let c_local_ptr = c_local[mr as usize].as_ptr() as *const $mix_vec;
                            let mut res = <$vec_type>::splat(<$dtype>::ZERO);
                            vec_cast_back(&mut res, c_local_ptr.offset(i as isize));
                            out_ptr.offset(i as isize).write_unaligned(res);
                        }
                    }
                }
            } else {
                for mr in 0..owr {
                    let reg = c_local[mr as usize].as_ptr() as *const $mix_type;
                    for nr in 0..ocr as i64 {
                        unsafe {
                            let mut res = <$dtype>::ZERO;
                            cast_back(&mut res, &*reg.add(nr as usize));
                            out[(mr + k) * osw + j + nr] = res;
                        }
                    }
                }
            }
        }
    };
}

// /// conv2d neon micro kernel
// #[macro_export]
// macro_rules! conv2d_neon_micro_kernel {
//     ($name:ident, $nr:expr, $mr:expr) => {
//         fn $name<T: CommonBounds>(
//             inp: crate::Pointer<T>,
//             kernel: crate::Pointer<T>,
//             mut out: crate::Pointer<T>,
//             icb: i64,
//             osw: i64,
//             kernel_idx: &mut i64,
//             [k, j, _]: [i64; 3],
//             [kh, kw]: [i64; 2],
//             [_, step_width]: [i64; 2],
//             _: [i64; 2],
//             _: [i64; 2],
//             [ish, isw]: [i64; 2],
//             [owr, ocr]: [i64; 2],
//             first_ic_iter: bool,
//         ) {
//             use $crate::types::vectors::traits::VecTrait;
//             use $crate::types::math::NormalOut;
//             use CommonBounds;
//             use hpt_types::dtype::TypeCommon;
//             fn load_vec_neon<const NR: usize>(data: *const f32, mut to_write: *mut f32) {
//                 use std::{arch::aarch64::*, mem::transmute};
//                 unsafe {
//                     match NR {
//                         1 => {
//                             let res = vld1q_f32(data);
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 1]) = transmute(res);
//                         }
//                         2 => {
//                             let res = vld1q_f32_x2(data);
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 2]) = transmute(res);
//                         }
//                         3 => {
//                             let res = vld1q_f32_x3(data);
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 3]) = transmute(res);
//                         }
//                         4 => {
//                             let res = vld1q_f32_x4(data);
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
//                         }
//                         5 => {
//                             let res = vld1q_f32(data);
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 1]) = transmute(res);
//                             to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE);
//                             let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE));
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
//                         }
//                         6 => {
//                             let res = vld1q_f32_x2(data);
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 2]) = transmute(res);
//                             to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 2);
//                             let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE * 2));
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
//                         }
//                         7 => {
//                             let res = vld1q_f32_x3(data);
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 3]) = transmute(res);
//                             to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 3);
//                             let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE * 3));
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
//                         }
//                         8 => {
//                             let res = vld1q_f32_x4(data);
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
//                             to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 4);
//                             let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE * 4));
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
//                         }
//                         9 => {
//                             let res = vld1q_f32_x4(data);
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
//                             to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 4);
//                             let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE * 4));
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
//                             to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 4);
//                             let res = vld1q_f32(data.add(<f32 as TypeCommon>::Vec::SIZE * 8));
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 1]) = transmute(res);
//                         }
//                         10 => {
//                             let res = vld1q_f32_x4(data);
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
//                             to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 4);
//                             let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE * 4));
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
//                             to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 4);
//                             let res = vld1q_f32_x2(data.add(<f32 as TypeCommon>::Vec::SIZE * 8));
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 2]) = transmute(res);
//                         }
//                         _ => { unreachable!() }
//                     }
//                 }
//             }
//             fn make_a_vec<T: CommonBounds>(inp: crate::Pointer<T>, n: i64, k: i64, m: i64, ii: i64, ish: i64, step_width: i64, isw: i64) -> T::Vec {
//                 let mut a_vec = T::Vec::splat(T::ZERO);
//                 seq_macro::seq!(MR in 0..$mr {
//                     a_vec[MR] = inp[n * ish + ((k + MR) * step_width + m) * isw + ii];
//                 });
//                 a_vec
//             }
//             let mut c_local = if first_ic_iter {
//                 [[T::Vec::splat(T::ZERO); $nr]; $mr]
//             } else {
//                 let mut c_local = [[T::Vec::splat(T::ZERO); $nr]; $mr];
//                 for mr in 0..owr {
//                     let reg = c_local[mr as usize].as_mut_ptr() as *mut T;
//                     for nr in 0..ocr as i64 {
//                         let val = out[(mr + k) * osw + j + nr];
//                         unsafe { reg.add(nr as usize).write(val) };
//                     }
//                 }
//                 c_local
//             };
//             for n in 0..kh {
//                 for m in 0..kw {
//                     for ii in 0..icb {
//                         let kernel = unsafe { kernel.ptr.add(*kernel_idx as usize) };
//                         let mut b_vec = [<T as hpt_types::dtype::TypeCommon>::Vec::splat(<T>::ZERO); $nr];
//                         load_vec_neon::<$nr>(kernel as *const f32, b_vec.as_mut_ptr() as *mut f32);
//                         let a_vec = make_a_vec(inp, n, k, m, ii, ish, step_width, isw);
//                         seq_macro::seq!(MR in 0..$mr {
//                             seq_macro::seq!(NR in 0..$nr {
//                                 c_local[MR][NR] = b_vec[NR].mul_add_lane::<MR>(a_vec, c_local[MR][NR]);
//                             });
//                             }
//                         );
//                         *kernel_idx += $nr * T::Vec::SIZE as i64;
//                     }
//                 }
//             }
//             for mr in 0..owr {
//                 let reg = c_local[mr as usize].as_ptr() as *const T;
//                 for nr in 0..ocr as i64 {
//                     out[(mr + k) * osw + j + nr] =
//                         unsafe { reg.add(nr as usize).read() };
//                 }
//             }
//         }
//     };
// }

// /// conv2d mixed precision neon micro kernel
// #[macro_export]
// macro_rules! conv2d_mixed_precision_neon_micro_kernel {
//     ($name:ident, $nr:expr, $mr:expr, $nr2:expr) => {
//         fn $name<T: CommonBounds, IM: CommonBounds>(
//             inp: crate::Pointer<T>,
//             kernel: crate::Pointer<IM>,
//             mut out: crate::Pointer<T>,
//             icb: i64,
//             osw: i64,
//             kernel_idx: &mut i64,
//             [k, j, _]: [i64; 3],
//             [kh, kw]: [i64; 2],
//             [_, step_width]: [i64; 2],
//             _: [i64; 2],
//             _: [i64; 2],
//             [ish, isw]: [i64; 2],
//             [owr, ocr]: [i64; 2],
//             first_ic_iter: bool,
//             cast: fn(T) -> IM,
//             cast_back: fn(IM) -> T,
//         ) {
//             use $crate::types::vectors::traits::VecTrait;
//             use $crate::types::math::NormalOut;
//             use CommonBounds;
//             use hpt_types::dtype::TypeCommon;
//             fn load_vec_neon<const NR: usize>(data: *const f32, mut to_write: *mut f32) {
//                 use std::{arch::aarch64::*, mem::transmute};
//                 unsafe {
//                     match NR {
//                         1 => {
//                             let res = vld1q_f32(data);
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 1]) = transmute(res);
//                         }
//                         2 => {
//                             let res = vld1q_f32_x2(data);
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 2]) = transmute(res);
//                         }
//                         3 => {
//                             let res = vld1q_f32_x3(data);
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 3]) = transmute(res);
//                         }
//                         4 => {
//                             let res = vld1q_f32_x4(data);
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
//                         }
//                         5 => {
//                             let res = vld1q_f32(data);
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 1]) = transmute(res);
//                             to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE);
//                             let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE));
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
//                         }
//                         6 => {
//                             let res = vld1q_f32_x2(data);
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 2]) = transmute(res);
//                             to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 2);
//                             let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE * 2));
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
//                         }
//                         7 => {
//                             let res = vld1q_f32_x3(data);
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 3]) = transmute(res);
//                             to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 3);
//                             let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE * 3));
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
//                         }
//                         8 => {
//                             let res = vld1q_f32_x4(data);
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
//                             to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 4);
//                             let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE * 4));
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
//                         }
//                         9 => {
//                             let res = vld1q_f32_x4(data);
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
//                             to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 4);
//                             let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE * 4));
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
//                             to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 4);
//                             let res = vld1q_f32(data.add(<f32 as TypeCommon>::Vec::SIZE * 8));
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 1]) = transmute(res);
//                         }
//                         10 => {
//                             let res = vld1q_f32_x4(data);
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
//                             to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 4);
//                             let res = vld1q_f32_x4(data.add(<f32 as TypeCommon>::Vec::SIZE * 4));
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 4]) = transmute(res);
//                             to_write = to_write.add(<f32 as TypeCommon>::Vec::SIZE * 4);
//                             let res = vld1q_f32_x2(data.add(<f32 as TypeCommon>::Vec::SIZE * 8));
//                             *(to_write as *mut [<f32 as TypeCommon>::Vec; 2]) = transmute(res);
//                         }
//                         _ => { unreachable!() }
//                     }
//                 }
//             }
//             fn make_a_vec<T: CommonBounds, IM: CommonBounds>(inp: crate::Pointer<T>, n: i64, k: i64, m: i64, ii: i64, ish: i64, step_width: i64, isw: i64, cast: fn(T) -> IM) -> IM::Vec {
//                 let mut a_vec = IM::Vec::splat(IM::ZERO);
//                 seq_macro::seq!(MR in 0..$mr {
//                     a_vec[MR] = cast(inp[n * ish + ((k + MR) * step_width + m) * isw + ii]);
//                 });
//                 a_vec
//             }
//             let mut c_local = if first_ic_iter {
//                 [[IM::Vec::splat(IM::ZERO); $nr2]; $mr]
//             } else {
//                 let mut c_local = [[IM::Vec::splat(IM::ZERO); $nr2]; $mr];
//                 for mr in 0..owr {
//                     let reg = c_local[mr as usize].as_mut_ptr() as *mut IM;
//                     for nr in 0..ocr as i64 {
//                         let val = cast(out[(mr + k) * osw + j + nr]);
//                         unsafe { reg.add(nr as usize).write(val) };
//                     }
//                 }
//                 c_local
//             };
//             for n in 0..kh {
//                 for m in 0..kw {
//                     for ii in 0..icb {
//                         let kernel = unsafe { kernel.ptr.add(*kernel_idx as usize) };
//                         let mut b_vec = [<IM as hpt_types::dtype::TypeCommon>::Vec::splat(<IM>::ZERO); $nr2];
//                         load_vec_neon::<$nr2>(kernel as *const f32, b_vec.as_mut_ptr() as *mut f32);
//                         let a_vec = make_a_vec::<T, IM>(inp, n, k, m, ii, ish, step_width, isw, cast);
//                         seq_macro::seq!(MR in 0..$mr {
//                             seq_macro::seq!(NR in 0..$nr2 {
//                                 c_local[MR][NR] = b_vec[NR].mul_add_lane::<MR>(a_vec, c_local[MR][NR]);
//                             });
//                             }
//                         );
//                         *kernel_idx += $nr2 * IM::Vec::SIZE as i64;
//                     }
//                 }
//             }
//             for mr in 0..owr {
//                 let reg = c_local[mr as usize].as_ptr() as *const IM;
//                 for nr in 0..ocr as i64 {
//                     out[(mr + k) * osw + j + nr] =
//                         cast_back(unsafe { reg.add(nr as usize).read() });
//                 }
//             }
//         }
//     };
// }
