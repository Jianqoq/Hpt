#[macro_export]
macro_rules! sum_kernel {
    (
        $_:expr,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_last_stride:ident,
        $shape_len:ident
    ) => {
        for _ in 0..$loop_size2 {
            let mut tmp = $result_ptr[0isize];
            for i in 0..$loop_size as i64 {
                let a_val = $a_ptr[i * $a_last_stride];
                tmp = a_val._add(tmp);
            }
            $result_ptr.modify(0, tmp);
            for j in (0..$shape_len - 1).rev() {
                if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                    $iterator.prg[j as usize] += 1;
                    $a_ptr.offset($iterator.strides[j as usize]);
                    break;
                } else {
                    $iterator.prg[j as usize] = 0;
                    $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
                }
            }
        }
        $result_ptr.add(1);
    };
    (
        $_:expr,
        $outer_idx:ident,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_stride:ident,
        $shape_len:ident
    ) => {
        for _ in 0..$loop_size2 {
            for i in 0..$loop_size  as i64{
                let a_val = $a_ptr[i * $a_stride];
                let result_val = $result_ptr[i];
                $result_ptr.modify(i, a_val._add(result_val));
            }
            for j in ($shape_len..=$iterator.a_shape.len() as i64 - 1).rev() {
                if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                    $iterator.prg[j as usize] += 1;
                    $a_ptr.offset($iterator.strides[j as usize]);
                    break;
                } else {
                    $iterator.prg[j as usize] = 0;
                    $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
                }
            }
        }
        for j in (0..$shape_len - 1).rev() {
            if $iterator.a_prg[j as usize] < $iterator.a_shape[j as usize] {
                $iterator.a_prg[j as usize] += 1;
                $a_ptr.offset($iterator.strides[j as usize]);
                break;
            } else {
                $iterator.a_prg[j as usize] = 0;
                $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
            }
        }
        $result_ptr.add($loop_size);
        $iterator.reset_prg();
    };
    (reduce_all, $result_data:ident, $inp_name:ident, $init_val:expr) => {
        let val = $inp_name.as_raw_mut().par_iter().fold(|| $init_val, |acc, &x| acc._add(x)).reduce(|| $init_val,|a, b| a._add(b));
        $result_data.write(val._add($result_data.read()));
    };
}

#[macro_export]
macro_rules! sum_with_cast_kernel {
    (
        $_:expr,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_last_stride:ident,
        $shape_len:ident
    ) => {
        for _ in 0..$loop_size2 {
            let mut tmp = $result_ptr[0isize];
            for i in 0..$loop_size as i64 {
                let a_val = $a_ptr[i * $a_last_stride];
                tmp = B::__from(a_val._add(tmp));
            }
            $result_ptr.modify(0, tmp);
            for j in (0..$shape_len - 1).rev() {
                if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                    $iterator.prg[j as usize] += 1;
                    $a_ptr.offset($iterator.strides[j as usize]);
                    break;
                } else {
                    $iterator.prg[j as usize] = 0;
                    $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
                }
            }
        }
        $result_ptr.add(1);
    };
    (
        $_:expr,
        $outer_idx:ident,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_stride:ident,
        $shape_len:ident
    ) => {
        for _ in 0..$loop_size2 {
            for i in 0..$loop_size  as i64{
                let a_val = $a_ptr[i * $a_stride];
                let result_val = $result_ptr[i];
                $result_ptr.modify(i, B::__from(a_val._add(result_val)));
            }
            for j in ($shape_len..=$iterator.a_shape.len() as i64 - 1).rev() {
                if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                    $iterator.prg[j as usize] += 1;
                    $a_ptr.offset($iterator.strides[j as usize]);
                    break;
                } else {
                    $iterator.prg[j as usize] = 0;
                    $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
                }
            }
        }
        for j in (0..$shape_len - 1).rev() {
            if $iterator.a_prg[j as usize] < $iterator.a_shape[j as usize] {
                $iterator.a_prg[j as usize] += 1;
                $a_ptr.offset($iterator.strides[j as usize]);
                break;
            } else {
                $iterator.a_prg[j as usize] = 0;
                $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
            }
        }
        $result_ptr.add($loop_size);
        $iterator.reset_prg();
    };
    (reduce_all, $result_data:ident, $inp_name:ident, $init_val:expr) => {
        let val = $inp_name.as_raw_mut().par_iter().fold(|| $init_val, |acc, &x| acc._add(x)).reduce(|| $init_val,|a, b| a._add(b));
        $result_data.write(B::__from(val._add($result_data.read())));
    };
}

#[macro_export]
macro_rules! reducel1_kernel {
    (
        $_:expr,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_last_stride:ident,
        $shape_len:ident
    ) => {
        for _ in 0..$loop_size2 {
            let mut tmp = $result_ptr[0isize];
            for i in 0..$loop_size as i64 {
                let a_val = $a_ptr[i * $a_last_stride]._abs();
                tmp = a_val._add(tmp);
            }
            $result_ptr.modify(0, tmp);
            for j in (0..$shape_len - 1).rev() {
                if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                    $iterator.prg[j as usize] += 1;
                    $a_ptr.offset($iterator.strides[j as usize]);
                    break;
                } else {
                    $iterator.prg[j as usize] = 0;
                    $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
                }
            }
        }
        $result_ptr.add(1);
    };
    (
        $_:expr,
        $outer_idx:ident,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_stride:ident,
        $shape_len:ident
    ) => {
        for _ in 0..$loop_size2 {
            for i in 0..$loop_size  as i64{
                let a_val = $a_ptr[i * $a_stride]._abs();
                let result_val = $result_ptr[i];
                $result_ptr.modify(i, a_val._add(result_val));
            }
            for j in ($shape_len..=$iterator.a_shape.len() as i64 - 1).rev() {
                if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                    $iterator.prg[j as usize] += 1;
                    $a_ptr.offset($iterator.strides[j as usize]);
                    break;
                } else {
                    $iterator.prg[j as usize] = 0;
                    $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
                }
            }
        }
        for j in (0..$shape_len - 1).rev() {
            if $iterator.a_prg[j as usize] < $iterator.a_shape[j as usize] {
                $iterator.a_prg[j as usize] += 1;
                $a_ptr.offset($iterator.strides[j as usize]);
                break;
            } else {
                $iterator.a_prg[j as usize] = 0;
                $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
            }
        }
        $result_ptr.add($loop_size);
        $iterator.reset_prg();
    };
    (reduce_all, $result_data:ident, $inp_name:ident, $init_val:expr) => {
        let val = $inp_name.as_raw_mut().par_iter().fold(|| $init_val, |acc, &x| acc._add(x._abs())).reduce(|| $init_val,|a, b| a._add(b._abs()));
        $result_data.write(val._add($result_data.read()));
    };
}

#[macro_export]
macro_rules! sum_square_kernel {
    (
        $_:expr,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_last_stride:ident,
        $shape_len:ident
    ) => {
        for _ in 0..$loop_size2 {
            let mut tmp = $result_ptr[0isize];
            for i in 0..$loop_size as i64 {
                let a_val = <T as NormalOut>::_square($a_ptr[i * $a_last_stride]);
                tmp = a_val._add(tmp);
            }
            $result_ptr.modify(0, tmp);
            for j in (0..$shape_len - 1).rev() {
                if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                    $iterator.prg[j as usize] += 1;
                    $a_ptr.offset($iterator.strides[j as usize]);
                    break;
                } else {
                    $iterator.prg[j as usize] = 0;
                    $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
                }
            }
        }
        $result_ptr.modify(0, $result_ptr.read());
        $result_ptr.add(1);
    };
    (
        $_:expr,
        $outer_idx:ident,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_stride:ident,
        $shape_len:ident
    ) => {
        for _ in 0..$loop_size2 {
            for i in 0..$loop_size  as i64{
                let a_val = <T as NormalOut>::_square($a_ptr[i * $a_stride]);
                let result_val = $result_ptr[i];
                $result_ptr.modify(i, a_val._add(result_val));
            }
            for j in ($shape_len..=$iterator.a_shape.len() as i64 - 1).rev() {
                if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                    $iterator.prg[j as usize] += 1;
                    $a_ptr.offset($iterator.strides[j as usize]);
                    break;
                } else {
                    $iterator.prg[j as usize] = 0;
                    $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
                }
            }
        }
        for j in (0..$shape_len - 1).rev() {
            if $iterator.a_prg[j as usize] < $iterator.a_shape[j as usize] {
                $iterator.a_prg[j as usize] += 1;
                $a_ptr.offset($iterator.strides[j as usize]);
                break;
            } else {
                $iterator.a_prg[j as usize] = 0;
                $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
            }
        }
        for i in 0..$loop_size as i64 {
            $result_ptr.modify(i, $result_ptr[i]);
        }
        $result_ptr.add($loop_size);
        $iterator.reset_prg();
    };
    (reduce_all, $result_data:ident, $inp_name:ident, $init_val:expr) => {
        let val = $inp_name.as_raw_mut().par_iter().fold(|| $init_val, |acc, &x| {
            let square = <T as NormalOut>::_square(x);
            acc._add(square)
        }).reduce(|| $init_val,|a, b| {
            let square = <T as NormalOut>::_square(b);
            a._add(square)
        });
        
        $result_data.write(val._add($result_data.read()));
    };
}

#[macro_export]
macro_rules! reducel2_kernel {
    (
        $_:expr,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_last_stride:ident,
        $shape_len:ident
    ) => {
        for _ in 0..$loop_size2 {
            let mut tmp = $result_ptr[0isize];
            for i in 0..$loop_size as i64 {
                let a_val = <T as NormalOut>::_square($a_ptr[i * $a_last_stride]);
                tmp = a_val._add(tmp);
            }
            $result_ptr.modify(0, tmp);
            for j in (0..$shape_len - 1).rev() {
                if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                    $iterator.prg[j as usize] += 1;
                    $a_ptr.offset($iterator.strides[j as usize]);
                    break;
                } else {
                    $iterator.prg[j as usize] = 0;
                    $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
                }
            }
        }
        $result_ptr.modify(0, $result_ptr.read()._sqrt());
        $result_ptr.add(1);
    };
    (
        $_:expr,
        $outer_idx:ident,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_stride:ident,
        $shape_len:ident
    ) => {
        for _ in 0..$loop_size2 {
            for i in 0..$loop_size  as i64{
                let a_val = <T as NormalOut>::_square($a_ptr[i * $a_stride]);
                let result_val = $result_ptr[i];
                $result_ptr.modify(i, a_val._add(result_val));
            }
            for j in ($shape_len..=$iterator.a_shape.len() as i64 - 1).rev() {
                if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                    $iterator.prg[j as usize] += 1;
                    $a_ptr.offset($iterator.strides[j as usize]);
                    break;
                } else {
                    $iterator.prg[j as usize] = 0;
                    $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
                }
            }
        }
        for j in (0..$shape_len - 1).rev() {
            if $iterator.a_prg[j as usize] < $iterator.a_shape[j as usize] {
                $iterator.a_prg[j as usize] += 1;
                $a_ptr.offset($iterator.strides[j as usize]);
                break;
            } else {
                $iterator.a_prg[j as usize] = 0;
                $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
            }
        }
        for i in 0..$loop_size as i64 {
            $result_ptr.modify(i, $result_ptr[i]._sqrt());
        }
        $result_ptr.add($loop_size);
        $iterator.reset_prg();
    };
    (reduce_all, $result_data:ident, $inp_name:ident, $init_val:expr) => {
        let val = $inp_name.as_raw_mut().par_iter().fold(|| $init_val, |acc, &x| {
            let square = <T as NormalOut>::_square(x);
            acc._add(square)
        }).reduce(|| $init_val,|a, b| {
            let square = <T as NormalOut>::_square(b);
            a._add(square)
        });
        
        $result_data.write(val._add($result_data.read())._sqrt());
    };
}

#[macro_export]
macro_rules! reducel3_kernel {
    (
        $_:expr,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_last_stride:ident,
        $shape_len:ident
    ) => {
        let three: <T as NormalOut>::Output = 3f64.into_scalar();
        for _ in 0..$loop_size2 {
            let mut tmp = $result_ptr[0isize];
            for i in 0..$loop_size as i64 {
                let a_val = <T as NormalOut>::_abs($a_ptr[i * $a_last_stride]);
                let a_val = a_val._pow(three);
                tmp = a_val._add(tmp);
            }
            $result_ptr.modify(0, tmp);
            for j in (0..$shape_len - 1).rev() {
                if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                    $iterator.prg[j as usize] += 1;
                    $a_ptr.offset($iterator.strides[j as usize]);
                    break;
                } else {
                    $iterator.prg[j as usize] = 0;
                    $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
                }
            }
        }
        $result_ptr.modify(0, $result_ptr.read()._cbrt());
        $result_ptr.add(1);
    };
    (
        $_:expr,
        $outer_idx:ident,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_stride:ident,
        $shape_len:ident
    ) => {
        let three: <T as NormalOut>::Output = 3f64.into_scalar();
        for _ in 0..$loop_size2 {
            for i in 0..$loop_size as i64 {
                let a_val = <T as NormalOut>::_abs($a_ptr[i * $a_stride]);
                let a_val = a_val._pow(three);
                let result_val = $result_ptr[i];
                $result_ptr.modify(i, a_val._add(result_val));
            }
            for j in ($shape_len..=$iterator.a_shape.len() as i64 - 1).rev() {
                if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                    $iterator.prg[j as usize] += 1;
                    $a_ptr.offset($iterator.strides[j as usize]);
                    break;
                } else {
                    $iterator.prg[j as usize] = 0;
                    $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
                }
            }
        }
        for j in (0..$shape_len - 1).rev() {
            if $iterator.a_prg[j as usize] < $iterator.a_shape[j as usize] {
                $iterator.a_prg[j as usize] += 1;
                $a_ptr.offset($iterator.strides[j as usize]);
                break;
            } else {
                $iterator.a_prg[j as usize] = 0;
                $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
            }
        }
        for i in 0..$loop_size as i64 {
            $result_ptr.modify(i, $result_ptr[i]._cbrt());
        }
        $result_ptr.add($loop_size);
        $iterator.reset_prg();
    };
    (reduce_all, $result_data:ident, $inp_name:ident, $init_val:expr) => {
        let three: <T as NormalOut>::Output = 3f64.into_scalar();
        let val = $inp_name.as_raw_mut().par_iter().fold(|| $init_val, |acc, &x| {
            let abs = <T as NormalOut>::_abs(x);
            let abs = abs._pow(three);
            acc._add(abs)
        }).reduce(|| $init_val,|a, b| {
            let abs = <T as NormalOut>::_abs(b);
            let abs = abs._pow(three);
            a._add(abs)
        });
        $result_data.write(val._add($result_data.read())._cbrt());
    };
}

#[macro_export]
macro_rules! cumsum_kernel {
    (
        $_:expr,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_last_stride:ident,
        $shape_len:ident
    ) => {
        for _ in 0..$loop_size2 {
            let mut tmp = $result_ptr[0isize];
            for i in 0..$loop_size {
                let a_val = $a_ptr[i * $a_last_stride];
                tmp = a_val._add(tmp);
            }
            $result_ptr.modify(0, tmp);
            for j in (0..$shape_len - 1).rev() {
                if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                    $iterator.prg[j as usize] += 1;
                    $a_ptr.offset($iterator.strides[j as usize]);
                    break;
                } else {
                    $iterator.prg[j as usize] = 0;
                    $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
                }
            }
        }
        $result_ptr.add(1);
    };
    (
        $_:expr,
        $outer_idx:ident,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_stride:ident,
        $shape_len:ident
    ) => {
        for _ in 0..$loop_size2 {
            for i in 0..$loop_size {
                let a_val = $a_ptr[i * $a_stride];
                let result_val = $result_ptr[i];
                $result_ptr.modify(i, a_val._add(result_val));
            }
            for j in ($shape_len..=$iterator.a_shape.len() as i64 - 1).rev() {
                if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                    $iterator.prg[j as usize] += 1;
                    $a_ptr.offset($iterator.strides[j as usize]);
                    break;
                } else {
                    $iterator.prg[j as usize] = 0;
                    $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
                }
            }
        }
        for j in (0..$shape_len - 1).rev() {
            if $iterator.a_prg[j as usize] < $iterator.a_shape[j as usize] {
                $iterator.a_prg[j as usize] += 1;
                $a_ptr.offset($iterator.strides[j as usize]);
                break;
            } else {
                $iterator.a_prg[j as usize] = 0;
                $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
            }
        }
        $result_ptr.add($loop_size);
        $iterator.reset_prg();
    };
    (reduce_all, $result_data:ident, $inp_name:ident, $init_val:expr) => {
        let val = $inp_name.as_raw_mut().par_iter().fold(|| $init_val, |acc, &x| acc._add(x)).reduce(|| $init_val,|a, b| a._add(b));
        $result_data.write(val);
    };
}

#[macro_export]
macro_rules! nansum_kernel {
    (
        $_:expr,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_last_stride:ident,
        $shape_len:ident
    ) => {
        for _ in 0..$loop_size2 {
            let mut tmp = $result_ptr[0isize];
            for i in 0..$loop_size as i64 {
                let a_val = $a_ptr[i * $a_last_stride];
                if a_val._is_nan() {
                    tmp = T::ZERO._add(tmp);
                } else {
                    tmp = a_val._add(tmp);
                }
            }
            $result_ptr.modify(0, tmp);
            for j in (0..$shape_len - 1).rev() {
                if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                    $iterator.prg[j as usize] += 1;
                    $a_ptr.offset($iterator.strides[j as usize]);
                    break;
                } else {
                    $iterator.prg[j as usize] = 0;
                    $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
                }
            }
        }
        $result_ptr.add(1);
    };
    (
        $_:expr,
        $outer_idx:ident,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_stride:ident,
        $shape_len:ident
    ) => {
        for _ in 0..$loop_size2 {
            for i in 0..$loop_size as i64 {
                let a_val = $a_ptr[i * $a_stride];
                if a_val._is_nan() {
                    $result_ptr.modify(i, T::ZERO._add($result_ptr[i]));
                } else {
                    $result_ptr.modify(i, a_val._add($result_ptr[i]));
                }
            }
            for j in ($shape_len..=$iterator.a_shape.len() as i64 - 1).rev() {
                if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                    $iterator.prg[j as usize] += 1;
                    $a_ptr.offset($iterator.strides[j as usize]);
                    break;
                } else {
                    $iterator.prg[j as usize] = 0;
                    $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
                }
            }
        }
        for j in (0..$shape_len - 1).rev() {
            if $iterator.a_prg[j as usize] < $iterator.a_shape[j as usize] {
                $iterator.a_prg[j as usize] += 1;
                $a_ptr.offset($iterator.strides[j as usize]);
                break;
            } else {
                $iterator.a_prg[j as usize] = 0;
                $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
            }
        }
        $result_ptr.add($loop_size);
        $iterator.reset_prg();
    };
    (reduce_all, $result_data:ident, $inp_name:ident, $init_val:expr) => {
        let val = $inp_name.as_raw_mut().par_iter().fold(|| $init_val, |acc, &x|
            if x._is_nan() {
                acc
            } else {
             acc._add(x)
            }).reduce(|| $init_val,|a, b| a._add(b));
        $result_data.write(val);
    };
}

#[macro_export]
macro_rules! prod_kernel {
    (
        $_:expr,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_last_stride:ident,
        $shape_len:ident
    ) => {
        for _ in 0..$loop_size2 {
            let mut tmp = $result_ptr[0isize];
            for i in 0..$loop_size as i64 {
                tmp = $a_ptr[i * $a_last_stride]._mul(tmp);
            }
            $result_ptr.modify(0, tmp);
            for j in (0..$shape_len - 1).rev() {
                if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                    $iterator.prg[j as usize] += 1;
                    $a_ptr.offset($iterator.strides[j as usize]);
                    break;
                } else {
                    $iterator.prg[j as usize] = 0;
                    $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
                }
            }
        }
        $result_ptr.add(1);
    };
    (
        $_:expr,
        $outer_idx:ident,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_stride:ident,
        $shape_len:ident
    ) => {
        for _ in 0..$loop_size2 {
            for i in 0..$loop_size as i64 {
                let a_val = $a_ptr[i * $a_stride];
                let result_val = $result_ptr[i];
                $result_ptr.modify(i, a_val._mul(result_val));
            }
            for j in ($shape_len..=$iterator.a_shape.len() as i64 - 1).rev() {
                if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                    $iterator.prg[j as usize] += 1;
                    $a_ptr.offset($iterator.strides[j as usize]);
                    break;
                } else {
                    $iterator.prg[j as usize] = 0;
                    $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
                }
            }
        }
        for j in (0..$shape_len - 1).rev() {
            if $iterator.a_prg[j as usize] < $iterator.a_shape[j as usize] {
                $iterator.a_prg[j as usize] += 1;
                $a_ptr.offset($iterator.strides[j as usize]);
                break;
            } else {
                $iterator.a_prg[j as usize] = 0;
                $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
            }
        }
        $result_ptr.add($loop_size);
        $iterator.reset_prg();
    };
    (reduce_all, $result_data:ident, $inp_name:ident, $init_val:expr) => {
        let val = $inp_name.as_raw_mut().par_iter().fold(|| $init_val, |acc, &x| acc._mul(x)).reduce(|| $init_val,|a, b| a._mul(b));
        $result_data.write(val);
    };
}

#[macro_export]
macro_rules! nanprod_kernel {
    (
        $_:expr,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_last_stride:ident,
        $shape_len:ident
    ) => {
        for _ in 0..$loop_size2 {
            let mut tmp = $result_ptr[0isize];
            for i in 0..$loop_size as i64 {
                let a_val = $a_ptr[i * $a_last_stride];
                if !a_val._is_nan() {
                    tmp = a_val._mul(tmp);
                }
            }
            $result_ptr.modify(0, tmp);
            for j in (0..$shape_len - 1).rev() {
                if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                    $iterator.prg[j as usize] += 1;
                    $a_ptr.offset($iterator.strides[j as usize]);
                    break;
                } else {
                    $iterator.prg[j as usize] = 0;
                    $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
                }
            }
        }
        $result_ptr.add(1);
    };
    (
        $_:expr,
        $outer_idx:ident,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_stride:ident,
        $shape_len:ident
    ) => {
        for _ in 0..$loop_size2 {
            for i in 0..$loop_size  as i64{
                let a_val = $a_ptr[i * $a_stride];
                let result_val = $result_ptr[i];
                if !a_val._is_nan() {
                    $result_ptr.modify(i, a_val._mul(result_val));
                }
            }
            for j in ($shape_len..=$iterator.a_shape.len() as i64 - 1).rev() {
                if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                    $iterator.prg[j as usize] += 1;
                    $a_ptr.offset($iterator.strides[j as usize]);
                    break;
                } else {
                    $iterator.prg[j as usize] = 0;
                    $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
                }
            }
        }
        for j in (0..$shape_len - 1).rev() {
            if $iterator.a_prg[j as usize] < $iterator.a_shape[j as usize] {
                $iterator.a_prg[j as usize] += 1;
                $a_ptr.offset($iterator.strides[j as usize]);
                break;
            } else {
                $iterator.a_prg[j as usize] = 0;
                $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
            }
        }
        $result_ptr.add($loop_size);
        $iterator.reset_prg();
    };
    (reduce_all, $result_data:ident, $inp_name:ident, $init_val:expr) => {
        let val = $inp_name.as_raw_mut().par_iter().fold(|| $init_val, |acc, &x|
             if x._is_nan(){
                acc
            } else {
                acc._mul(x)
            }).reduce(|| $init_val,|a, b| a._mul(b));
        $result_data.write(val);
    };
}

#[macro_export]
macro_rules! min_kernel {
    (
        $_:expr,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_last_stride:ident,
        $shape_len:ident
    ) => {
        for _ in 0..$loop_size2 {
            let mut tmp = $result_ptr[0isize];
            for i in 0..$loop_size as i64 {
                let a_val = $a_ptr[i * $a_last_stride];
                if a_val._lt(tmp) {
                    tmp = a_val;
                }
            }
            $result_ptr.modify(0, tmp);
            for j in (0..$shape_len - 1).rev() {
                if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                    $iterator.prg[j as usize] += 1;
                    $a_ptr.offset($iterator.strides[j as usize]);
                    break;
                } else {
                    $iterator.prg[j as usize] = 0;
                    $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
                }
            }
        }
        $result_ptr.add(1);
    };
    (
        $_:expr,
        $outer_idx:ident,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_stride:ident,
        $shape_len:ident
    ) => {
        for _ in 0..$loop_size2 {
            for i in 0..$loop_size as i64 {
                let a_val = $a_ptr[i * $a_stride];
                let result_val = $result_ptr[i];
                if a_val._lt(result_val) {
                    $result_ptr.modify(i, a_val);
                }
            }
            for j in ($shape_len..=$iterator.a_shape.len() as i64 - 1).rev() {
                if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                    $iterator.prg[j as usize] += 1;
                    $a_ptr.offset($iterator.strides[j as usize]);
                    break;
                } else {
                    $iterator.prg[j as usize] = 0;
                    $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
                }
            }
        }
        for j in (0..$shape_len - 1).rev() {
            if $iterator.a_prg[j as usize] < $iterator.a_shape[j as usize] {
                $iterator.a_prg[j as usize] += 1;
                $a_ptr.offset($iterator.strides[j as usize]);
                break;
            } else {
                $iterator.a_prg[j as usize] = 0;
                $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
            }
        }
        $result_ptr.add($loop_size);
        $iterator.reset_prg();
    };
    (reduce_all, $result_data:ident, $inp_name:ident, $init_val:expr) => {
        let val = $inp_name.as_raw_mut().par_iter().fold(|| $init_val, |acc, &x| if acc._lt(x) {acc} else {x}).reduce(|| $init_val,|a, b| if a._lt(b) {a} else {b});
        $result_data.write(val);
    };
}

#[macro_export]
macro_rules! max_kernel {
    (
        $_:expr,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_last_stride:ident,
        $shape_len:ident
    ) => {
        for _ in 0..$loop_size2 {
            let mut tmp = $result_ptr[0isize];
            for i in 0..$loop_size as i64 {
                let a_val = $a_ptr[i * $a_last_stride];
                if a_val._gt(tmp) {
                    tmp = a_val;
                }
            }
            $result_ptr.modify(0, tmp);
            for j in (0..$shape_len - 1).rev() {
                if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                    $iterator.prg[j as usize] += 1;
                    $a_ptr.offset($iterator.strides[j as usize]);
                    break;
                } else {
                    $iterator.prg[j as usize] = 0;
                    $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
                }
            }
        }
        $result_ptr.add(1);
    };
    (
        $_:expr,
        $outer_idx:ident,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_stride:ident,
        $shape_len:ident
    ) => {
        for _ in 0..$loop_size2 {
            for i in 0..$loop_size as i64 {
                let a_val = $a_ptr[i * $a_stride];
                let result_val = $result_ptr[i];
                if a_val._gt(result_val) {
                    $result_ptr.modify(i, a_val);
                }
            }
            for j in ($shape_len..=$iterator.a_shape.len() as i64 - 1).rev() {
                if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                    $iterator.prg[j as usize] += 1;
                    $a_ptr.offset($iterator.strides[j as usize]);
                    break;
                } else {
                    $iterator.prg[j as usize] = 0;
                    $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
                }
            }
        }
        for j in (0..$shape_len - 1).rev() {
            if $iterator.a_prg[j as usize] < $iterator.a_shape[j as usize] {
                $iterator.a_prg[j as usize] += 1;
                $a_ptr.offset($iterator.strides[j as usize]);
                break;
            } else {
                $iterator.a_prg[j as usize] = 0;
                $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
            }
        }
        $result_ptr.add($loop_size);
        $iterator.reset_prg();
    };
    (reduce_all, $result_data:ident, $inp_name:ident, $init_val:expr) => {
        let val = $inp_name.as_raw_mut().par_iter().fold(|| $init_val, |acc, &x| if acc._gt(x) {acc} else {x}).reduce(|| $init_val,|a, b| if a._gt(b) {a} else {b});
        $result_data.write(val);
    };
}

#[macro_export]
macro_rules! all_kernel {
    (
        $_:expr,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_last_stride:ident,
        $shape_len:ident
    ) => {
        for _ in 0..$loop_size2 {
            let mut tmp = $result_ptr[0isize];
            for i in 0..$loop_size as i64 {
                let a_val = $a_ptr[i * $a_last_stride];
                tmp = a_val._is_true() & tmp;
            }
            $result_ptr.modify(0, tmp);
            for j in (0..$shape_len - 1).rev() {
                if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                    $iterator.prg[j as usize] += 1;
                    $a_ptr.offset($iterator.strides[j as usize]);
                    break;
                } else {
                    $iterator.prg[j as usize] = 0;
                    $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
                }
            }
        }
        $result_ptr.add(1);
    };
    (
        $_:expr,
        $outer_idx:ident,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_stride:ident,
        $shape_len:ident
    ) => {
        for _ in 0..$loop_size2 {
            for i in 0..$loop_size as i64 {
                let a_val = $a_ptr[i * $a_stride];
                let result_val = $result_ptr[i];
                $result_ptr.modify(i, a_val._is_true() & result_val);
            }
            for j in ($shape_len..=$iterator.a_shape.len() as i64 - 1).rev() {
                if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                    $iterator.prg[j as usize] += 1;
                    $a_ptr.offset($iterator.strides[j as usize]);
                    break;
                } else {
                    $iterator.prg[j as usize] = 0;
                    $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
                }
            }
        }
        for j in (0..$shape_len - 1).rev() {
            if $iterator.a_prg[j as usize] < $iterator.a_shape[j as usize] {
                $iterator.a_prg[j as usize] += 1;
                $a_ptr.offset($iterator.strides[j as usize]);
                break;
            } else {
                $iterator.a_prg[j as usize] = 0;
                $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
            }
        }
        $result_ptr.add($loop_size);
        $iterator.reset_prg();
    };
    (reduce_all, $result_data:ident, $inp_name:ident, $init_val:expr) => {
        let val = $inp_name.as_raw_mut().par_iter()
        .fold(|| $init_val, |acc, &x| acc == x._is_true())
        .reduce(|| $init_val,|a, b| a == b);
        $result_data.write(val);
    };
}

#[macro_export]
macro_rules! any_kernel {
    (
        $_:expr,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_last_stride:ident,
        $shape_len:ident
    ) => {
        for _ in 0..$loop_size2 {
            let mut tmp = $result_ptr[0isize];
            for i in 0..$loop_size as i64 {
                let a_val = $a_ptr[i * $a_last_stride];
                tmp = a_val._is_true() | tmp;
            }
            $result_ptr.modify(0, tmp);
            for j in (0..$shape_len - 1).rev() {
                if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                    $iterator.prg[j as usize] += 1;
                    $a_ptr.offset($iterator.strides[j as usize]);
                    break;
                } else {
                    $iterator.prg[j as usize] = 0;
                    $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
                }
            }
        }
        $result_ptr.add(1);
    };
    (
        $_:expr,
        $outer_idx:ident,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_stride:ident,
        $shape_len:ident
    ) => {
        for _ in 0..$loop_size2 {
            for i in 0..$loop_size as i64 {
                let a_val = $a_ptr[i * $a_stride];
                let result_val = $result_ptr[i];
                $result_ptr.modify(i, a_val._is_true() | result_val);
            }
            for j in ($shape_len..=$iterator.a_shape.len() as i64 - 1).rev() {
                if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                    $iterator.prg[j as usize] += 1;
                    $a_ptr.offset($iterator.strides[j as usize]);
                    break;
                } else {
                    $iterator.prg[j as usize] = 0;
                    $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
                }
            }
        }
        for j in (0..$shape_len - 1).rev() {
            if $iterator.a_prg[j as usize] < $iterator.a_shape[j as usize] {
                $iterator.a_prg[j as usize] += 1;
                $a_ptr.offset($iterator.strides[j as usize]);
                break;
            } else {
                $iterator.a_prg[j as usize] = 0;
                $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
            }
        }
        $result_ptr.add($loop_size);
        $iterator.reset_prg();
    };
    (reduce_all, $result_data:ident, $inp_name:ident, $init_val:expr) => {
        let val = $inp_name.as_raw_mut().par_iter()
        .fold(|| $init_val, |acc, &x| acc || x._is_true())
        .reduce(|| $init_val,|a, b| a || b);
        $result_data.write(val);
    };
}

#[macro_export]
macro_rules! mean_kernel {
    (
        $init_val:expr,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_last_stride:ident,
        $shape_len:ident
    ) => {
        let mut init_val = $init_val;
        for _ in 0..$loop_size2 {
            for i in 0..$loop_size as i64 {
                let a_val = $a_ptr[i * $a_last_stride];
                init_val = init_val._add(a_val);
            }
            for j in (0..$shape_len - 1).rev() {
                if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                    $iterator.prg[j as usize] += 1;
                    $a_ptr.offset($iterator.strides[j as usize]);
                    break;
                } else {
                    $iterator.prg[j as usize] = 0;
                    $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
                }
            }
        }
        $result_ptr.modify(0, init_val._div(<T as FloatOut>::Output::__from($loop_size * $loop_size2)));
        $result_ptr.add(1);
    };
    (
        $init_val:expr,
        $outer_idx:ident,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_stride:ident,
        $shape_len:ident
    ) => {
        for _ in 0..$loop_size2 {
            for i in 0..$loop_size as i64 {
                let a_val = $a_ptr[i * $a_stride];
                let result_val = $result_ptr[i];
                $result_ptr.modify(i, a_val._add(result_val));
            }
            for j in ($shape_len..=$iterator.a_shape.len() as i64 - 1).rev() {
                if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                    $iterator.prg[j as usize] += 1;
                    $a_ptr.offset($iterator.strides[j as usize]);
                    break;
                } else {
                    $iterator.prg[j as usize] = 0;
                    $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
                }
            }
        }
        for i in 0..$loop_size as i64 {
            let result_val = $result_ptr[i];
            $result_ptr.modify(i, result_val._div(<T as FloatOut>::Output::__from($loop_size2)));
        }
        for j in (0..$shape_len - 1).rev() {
            if $iterator.a_prg[j as usize] < $iterator.a_shape[j as usize] {
                $iterator.a_prg[j as usize] += 1;
                $a_ptr.offset($iterator.strides[j as usize]);
                break;
            } else {
                $iterator.a_prg[j as usize] = 0;
                $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
            }
        }
        $result_ptr.add($loop_size);
        $iterator.reset_prg();
    };
    (reduce_all, $result_data:ident, $inp_name:ident, $init_val:expr) => {
        let val = $inp_name.as_raw_mut().par_iter().fold(|| $init_val, |acc, &x| acc._add(x)).reduce(|| $init_val,|a, b| a._add(b));
        $result_data.write(val._div(<T as FloatOut>::Output::__from($inp_name.size())));
    };
}

#[macro_export]
macro_rules! argmax_kernel {
    (
        $_:expr,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_last_stride:ident,
        $shape_len:ident
    ) => {
        let mut max_val = T::NEG_INF;
        let mut max_index = 0;
        for i in 0..$loop_size {
            let a_val = $a_ptr[i * $a_last_stride];
            if a_val._gt(max_val) {
                max_val = a_val;
                max_index = i;
            }
        }
        for j in (0..$shape_len - 1).rev() {
            if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                $iterator.prg[j as usize] += 1;
                $a_ptr.offset($iterator.strides[j as usize]);
                break;
            } else {
                $iterator.prg[j as usize] = 0;
                $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
            }
        }
        $result_ptr.write(max_index.into_scalar());
        $result_ptr.add(1);
    };
}

#[macro_export]
macro_rules! argmin_kernel {
    (
        $_:expr,
        $iterator:ident,
        $loop_size:ident,
        $loop_size2:ident,
        $result_ptr:ident,
        $a_ptr:ident,
        $a_last_stride:ident,
        $shape_len:ident
    ) => {
        let mut max_val = T::INF;
        let mut max_index = 0;
        for i in 0..$loop_size {
            let a_val = $a_ptr[i * $a_last_stride];
            if a_val._lt(max_val) {
                max_val = a_val;
                max_index = i;
            }
        }
        for j in (0..$shape_len - 1).rev() {
            if $iterator.prg[j as usize] < $iterator.a_shape[j as usize] {
                $iterator.prg[j as usize] += 1;
                $a_ptr.offset($iterator.strides[j as usize]);
                break;
            } else {
                $iterator.prg[j as usize] = 0;
                $a_ptr.offset(-$iterator.strides[j as usize] * $iterator.a_shape[j as usize]);
            }
        }
        $result_ptr.write(max_index.into_scalar());
        $result_ptr.add(1);
    };
}
