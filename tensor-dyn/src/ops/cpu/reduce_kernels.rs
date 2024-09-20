
/// argmax kernel
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

/// argmin kernel
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
