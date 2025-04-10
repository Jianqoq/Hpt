/// conv2d micro kernel
#[macro_export]
macro_rules! conv2d_micro_kernel {
    ($name: ident, $nr:expr, $mr:expr) => {
        fn $name<T: $crate::common::CommonBounds>(
            inp: $crate::common::Pointer<T>,
            kernel: $crate::common::Pointer<T>,
            mut out: $crate::common::Pointer<T>,
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
            first_ic_iter: bool,
        ) {
            use $crate::types::vectors::traits::VecTrait;
            use $crate::types::math::NormalOut;
            let mut c_local = if first_ic_iter {
                [[T::Vec::splat(T::ZERO); $nr]; $mr]
            } else {
                let mut c_local = [[T::Vec::splat(T::ZERO); $nr]; $mr];
                for mr in 0..owr {
                    let reg = c_local[mr as usize].as_mut_ptr() as *mut T;
                    for nr in 0..ocr as i64 {
                        let val = out[(mr + k) * osw + j + nr];
                        unsafe { reg.add(nr as usize).write(val) };
                    }
                }
                c_local
            };
            for n in 0..kh {
                for m in 0..kw {
                    for ii in 0..icb {
                        let kernel = unsafe { kernel.ptr.add(*kernel_idx as usize) };
                        $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                            let b_vec~NR = unsafe {
                                    *(kernel.add(NR * <T as $crate::types::TypeCommon>::Vec::SIZE)
                                        as *const <T as $crate::types::TypeCommon>::Vec)
                                };
                            }
                        );
                        #[allow(unused_mut)]
                        let mut a_vec;
                        $crate::re_exports::seq_macro::seq!(MR in 0..$mr {
                            a_vec = <T as $crate::types::TypeCommon>::Vec::splat(inp[n * ish + ((k + MR) * step_width + m) * isw + ii]);
                            $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                                c_local[MR][NR] = a_vec._mul_add(b_vec~NR, c_local[MR][NR]);
                            });
                            }
                        );
                        *kernel_idx += $nr * T::Vec::SIZE as i64;
                    }
                }
            }
            for mr in 0..owr {
                let reg = c_local[mr as usize].as_ptr() as *const T;
                for nr in 0..ocr as i64 {
                    out[(mr + k) * osw + j + nr] =
                        unsafe { reg.add(nr as usize).read() };
                }
            }
        }
    };
}

/// conv2d micro kernel with padding
#[macro_export]
macro_rules! conv2d_micro_kernel_with_padding {
    ($name: ident, $nr:expr, $mr:expr) => {
        fn $name<T: $crate::common::CommonBounds>(
            inp: $crate::common::Pointer<T>,
            kernel: $crate::common::Pointer<T>,
            mut out: $crate::common::Pointer<T>,
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
            first_ic_iter: bool,
        ) {
            use $crate::types::vectors::traits::VecTrait;
            use $crate::types::math::NormalOut;
            let mut c_local = [[T::Vec::splat(T::ZERO); $nr]; $mr];
            for n in 0..kh {
                let in_y = l * step_height + n - ph_start;
                if in_y < 0 || in_y >= img_height {
                    *kernel_idx += icb * kw * $nr * T::Vec::SIZE as i64;
                    continue;
                }
                for m in 0..kw {
                    let max_in_x = (k + $mr - 1) * step_width + m - pw_start;
                    let kernel = unsafe { kernel.ptr.add(*kernel_idx as usize) };
                    $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                        let b_vec~NR = unsafe {
                                *(kernel.add(NR * <T as $crate::types::TypeCommon>::Vec::SIZE)
                                    as *const <T as $crate::types::TypeCommon>::Vec)
                            };
                        }
                    );
                    if max_in_x >= 0 && max_in_x < img_width {
                        for ii in 0..icb {
                            #[allow(unused_mut)]
                            let mut a_vec;
                            $crate::re_exports::seq_macro::seq!(MR in 0..$mr {
                                let in_x = (k + MR) * step_width + m - pw_start;
                                a_vec = <T as $crate::types::TypeCommon>::Vec::splat(inp[n * ish + in_x * isw + ii]);
                                $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                                    c_local[MR][NR] = a_vec._mul_add(b_vec~NR, c_local[MR][NR]);
                                });
                                }
                            );
                            *kernel_idx += $nr * T::Vec::SIZE as i64;
                        }
                    } else {
                        for ii in 0..icb {
                        #[allow(unused_mut)]
                        let mut a_vec;
                        $crate::re_exports::seq_macro::seq!(MR in 0..$mr {
                            let in_x = (k + MR) * step_width + m - pw_start;
                            if in_x < 0 || in_x >= img_width {
                                continue;
                            }
                            a_vec = <T as $crate::types::TypeCommon>::Vec::splat(inp[n * ish + in_x * isw + ii]);
                            $crate::re_exports::seq_macro::seq!(NR in 0..$nr {
                                c_local[MR][NR] = a_vec._mul_add(b_vec~NR, c_local[MR][NR]);
                            });
                            }
                        );
                        *kernel_idx += $nr * T::Vec::SIZE as i64;
                    }
                    }
                }
            }
            if first_ic_iter {
                for mr in 0..owr {
                    let reg = c_local[mr as usize].as_ptr() as *const T;
                    for nr in 0..ocr as i64 {
                        out[(mr + k) * osw + j + nr] =
                            unsafe { reg.add(nr as usize).read() };
                    }
                }
            } else {
                for mr in 0..owr {
                    let reg = c_local[mr as usize].as_ptr() as *const T;
                    for nr in 0..ocr as i64 {
                        let val = out[(mr + k) * osw + j + nr];
                        out[(mr + k) * osw + j + nr] =
                            val._add(unsafe { reg.add(nr as usize).read() });
                    }
                }
            }
        }
    };
}
