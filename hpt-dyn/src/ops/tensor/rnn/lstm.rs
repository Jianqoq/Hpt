use crate::{
    Tensor,
    ops::tensor::matmul::{
        common::matmul_prepare,
        matmul::{ addmm_prepacked_, matmul_prepacked_ },
    },
};
use hpt_common::{
    Pointer,
    error::base::TensorError,
    layout::layout::Layout,
    shape::shape_utils::mt_intervals,
    slice::slice_process,
};
use hpt_macros::select;
use matconv_simd::Zero;
use hpt_traits::tensor::{ CommonBounds, TensorInfo };
use hpt_types::{ dtype::{ DType, ToDType }, type_promote::FloatOutUnary };
use rayon::iter::{ IntoParallelIterator, ParallelIterator };

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    Forward,
    Reverse,
    Bidirectional,
}

impl Direction {
    fn from_str(s: &str) -> Self {
        match s {
            "forward" => Direction::Forward,
            "reverse" => Direction::Reverse,
            "bidirectional" => Direction::Bidirectional,
            _ => panic!("Invalid direction: {}", s),
        }
    }

    fn num_directions(&self) -> usize {
        match self {
            Direction::Forward => 1,
            Direction::Reverse => 1,
            Direction::Bidirectional => 2,
        }
    }
}

#[inline]
fn elementwise_activate<T: CommonBounds>(
    mut ptr: Pointer<T>,
    size: i64,
    activate_scalar: impl (Fn(T) -> T) + Send + Sync + Copy,
    activate_vec: impl (Fn(T::Vec) -> T::Vec) + Send + Sync + Copy
) {
    let vec_ptr = ptr.cast::<T::Vec>();
    use hpt_types::traits::VecTrait;
    for h in 0..size / (T::Vec::SIZE as i64) {
        vec_ptr
            .offset(h as isize)
            .write_unaligned(activate_vec(vec_ptr.offset(h as isize).read_unaligned()));
    }
    for h in (size / (T::Vec::SIZE as i64)) * (T::Vec::SIZE as i64)..size {
        let res = activate_scalar(ptr[h]);
        ptr[h] = res;
    }
}

#[inline]
fn elementwise_mul<T: CommonBounds>(mut lhs: Pointer<T>, rhs: Pointer<T>, size: i64) {
    let vec_ptr = lhs.cast::<T::Vec>();
    let rhs_vec_ptr = rhs.cast::<T::Vec>();
    use hpt_types::traits::VecTrait;
    use hpt_types::type_promote::NormalOut;
    for h in 0..size / (T::Vec::SIZE as i64) {
        vec_ptr.offset(h as isize).write_unaligned(
            vec_ptr
                .offset(h as isize)
                .read_unaligned()
                ._mul(rhs_vec_ptr.offset(h as isize).read_unaligned())
        );
    }
    for h in (size / (T::Vec::SIZE as i64)) * (T::Vec::SIZE as i64)..size {
        let res = lhs[h]._mul(rhs[h]);
        lhs[h] = res;
    }
}

fn lstm_post_process<T: CommonBounds>(
    y: Pointer<T>,
    c_t: Pointer<T>,
    gates: Pointer<T>,
    p: Option<Pointer<T>>,
    seq_lens: Option<Pointer<i64>>,
    batch_step: i64,
    time_step: i64,
    [y_b_stride, y_sq_stride]: [i64; 2],
    c_b_stride: i64,
    gates_b_stride: i64,
    hidden_size: i64,
    activate1: impl (Fn(T) -> T) + Send + Sync + Copy,
    activate1_vec: impl (Fn(T::Vec) -> T::Vec) + Send + Sync + Copy,
    activate2: impl (Fn(T) -> T) + Send + Sync + Copy,
    activate2_vec: impl (Fn(T::Vec) -> T::Vec) + Send + Sync + Copy
)
    where T: FloatOutUnary<Output = T>, T::Vec: FloatOutUnary<Output = T::Vec>
{
    use hpt_types::traits::VecTrait;
    use hpt_types::type_promote::NormalOut;
    let mut y = y + batch_step * y_b_stride + time_step * y_sq_stride;

    if let Some(seq_ptr) = seq_lens {
        let seq_len = seq_ptr.add(batch_step as usize).read();
        if time_step >= seq_len {
            let slice = unsafe { std::slice::from_raw_parts_mut(y.ptr, hidden_size as usize) };
            slice.fill(T::ZERO);
            return;
        }
    }

    let mut c_t = c_t + batch_step * c_b_stride;
    let i = gates + batch_step * gates_b_stride;
    let o = i + hidden_size;
    let f = i + 2 * hidden_size;
    let g = i + 3 * hidden_size;
    let i_vec_ptr = i.cast::<T::Vec>();
    let o_vec_ptr = o.cast::<T::Vec>();
    let f_vec_ptr = f.cast::<T::Vec>();
    let c_vec_ptr = c_t.cast::<T::Vec>();
    let y_vec_ptr = y.cast::<T::Vec>();

    match p {
        Some(p) => {
            let pi = p;
            let pf = p + hidden_size;
            let po = p + 2 * hidden_size;
            for h in 0..hidden_size as usize {
                let pi = pi[h];
                let pf = pf[h];
                let po = po[h];
                let c_t_res = c_t[h];
                let pi_mul_ct = pi._mul(c_t_res);
                let pf_mul_ct = pf._mul(c_t_res);
                let po_mul_ct = po._mul(c_t_res);
                let i_res = activate1(i[h]._add(pi_mul_ct));
                let o_res = activate1(o[h]._add(po_mul_ct));
                let f_res = activate1(f[h]._add(pf_mul_ct));
                let g_res = activate2(g[h]);
                let tmp = c_t_res._mul_add(f_res, i_res._mul(g_res));
                y[h] = o_res._mul(tmp._tanh());
                c_t[h] = tmp;
            }
        }
        None => {
            elementwise_activate(i, 3 * hidden_size, activate1, activate1_vec);
            elementwise_activate(g, hidden_size, activate2, activate2_vec);
            elementwise_mul(i, g, hidden_size);
            let num_vec = hidden_size / (T::Vec::SIZE as i64);
            for h in 0..num_vec {
                let i_res = i_vec_ptr.offset(h as isize).read_unaligned();
                let o_res = o_vec_ptr.offset(h as isize).read_unaligned();
                let f_res = f_vec_ptr.offset(h as isize).read_unaligned();
                let c_t_res = c_vec_ptr.offset(h as isize).read_unaligned();
                let tmp = c_t_res._mul_add(f_res, i_res);
                y_vec_ptr.offset(h as isize).write_unaligned(o_res._mul(tmp._tanh()));
                c_vec_ptr.offset(h as isize).write_unaligned(tmp);
            }
            for h in num_vec * (T::Vec::SIZE as i64)..hidden_size {
                let i_res = i[h];
                let o_res = o[h];
                let f_res = f[h];
                let c_t_res = c_t[h];
                let tmp = c_t_res._mul_add(f_res, i_res);
                y[h] = o_res._mul(tmp._tanh());
                c_t[h] = tmp;
            }
        }
    }
}

fn lstm_step<T: CommonBounds>(
    x: &Tensor, // [seq_len, batch_size, input_size]
    w_t: &Tensor, // [input_size, 4 * hidden_size]
    r_t: &Tensor, // [hidden_size, 4 * hidden_size]
    b: Option<&Tensor>, // [8 * hidden_size]
    p: Option<&Tensor>, // [3*hidden_size]
    initial_h: &Tensor, // [batch_size, hidden_size]
    initial_c: &Tensor, // [batch_size, hidden_size],
    seq_lens: Option<&Tensor>, // [batch_size]
    direction: Direction,
    hidden_size: i64,
    activate1: impl (Fn(T) -> T) + Send + Sync + Copy,
    activate1_vec: impl (Fn(T::Vec) -> T::Vec) + Send + Sync + Copy,
    activate2: impl (Fn(T) -> T) + Send + Sync + Copy,
    activate2_vec: impl (Fn(T::Vec) -> T::Vec) + Send + Sync + Copy
)
    -> Result<(Tensor, Tensor, Tensor), TensorError>
    where
        T: FloatOutUnary<Output = T> + hpt_matmul::MatmulMicroKernel + Zero,
        T::Vec: FloatOutUnary<Output = T::Vec>
{
    let seq_len = x.shape()[0];
    let batch_size = x.shape()[1];

    let seq_ptr = if let Some(seq_lens) = seq_lens {
        assert_eq!(seq_lens.shape().as_slice(), &[batch_size]);
        assert_eq!(seq_lens.dtype(), DType::I64);
        Some(seq_lens.ptr::<i64>())
    } else {
        None
    };

    let sum_bias = if let Some(b) = b {
        let wb = b.slice(&select![0:4*hidden_size])?;
        let rb = b.slice(&select![4*hidden_size:])?;
        Some(wb + rb)
    } else {
        None
    };

    let h_t = initial_h.clone(); // [batch_size, hidden_size]
    let c_t = initial_c.clone(); // [batch_size, hidden_size]

    let y = Tensor::empty(&[seq_len, batch_size, hidden_size], x.dtype, x.device.clone())?;
    let mut final_h = Tensor::empty(&[batch_size, hidden_size], x.dtype, x.device.clone())?;
    let mut final_c = Tensor::empty(&[batch_size, hidden_size], x.dtype, x.device.clone())?;

    let x_reshaped = x.reshape(&[seq_len * batch_size, -1])?;

    let tmp = (
        if let Some(sum_bias) = &sum_bias {
            x_reshaped.addmm(w_t, sum_bias)?
        } else {
            x_reshaped.matmul(w_t)?
        }
    ).reshape(&[seq_len, batch_size, 4 * hidden_size])?; // [seq_len, batch_size, 4 * hidden_size]

    let batch_size = x.shape()[1];

    let y_sq_stride = y.strides()[0] as i64;
    let y_b_stride = y.strides()[1] as i64;
    let c_b_stride = c_t.strides()[0] as i64;
    let batch_parallel = (batch_size as usize) > crate::physical_cores();
    if batch_parallel {
        let func = |
            start: i64,
            end: i64,
            y: Pointer<T>,
            y_layout: &Layout,
            prepacked_r: &hpt_matmul::PrePackedRhs
        | -> Result<(), TensorError> {
            let gates = Tensor::empty(&[end - start, 4 * hidden_size], x.dtype, x.device.clone())?;
            let h_t = initial_h.slice(&select![start:end, ..])?;
            let c_t = initial_c.slice(&select![start:end, ..])?;
            let tmp = tmp.slice(&select![:, start:end, ..])?;
            let mut h_t_ptr = h_t.ptr::<u8>();
            let mut h_t_layout = h_t.layout();
            let r_t_ptr = r_t.ptr::<u8>();
            let r_t_layout = r_t.layout();
            let gates_ptr = gates.ptr::<u8>();
            let gates_layout = gates.layout();
            let bias_layout = Layout::new(
                [tmp.shape()[1], tmp.shape()[2]],
                [tmp.strides()[1], tmp.strides()[2]]
            );

            let mut func = |t: i64| -> Result<(), TensorError> {
                let bias_ptr = tmp.ptr::<T>() + (t as usize) * (tmp.strides()[0] as usize);
                addmm_prepacked_(
                    h_t_ptr,
                    h_t_layout,
                    r_t_ptr,
                    r_t_layout,
                    bias_ptr.cast::<u8>(),
                    &bias_layout,
                    gates_ptr,
                    gates_layout,
                    1,
                    x.dtype,
                    Some(prepacked_r)
                )?;
                for b in 0..end - start {
                    lstm_post_process(
                        y,
                        c_t.ptr::<T>(),
                        gates.ptr::<T>(),
                        p.as_ref().map(|p| p.ptr::<T>()),
                        seq_ptr,
                        b,
                        t,
                        [y_b_stride, y_sq_stride],
                        c_b_stride,
                        gates.strides()[0],
                        hidden_size,
                        activate1,
                        activate1_vec,
                        activate2,
                        activate2_vec
                    );
                }
                h_t_ptr = (y + t * y_sq_stride).cast::<u8>();
                h_t_layout = &y_layout;
                Ok(())
            };
            match direction {
                Direction::Forward => {
                    for t in 0..seq_len {
                        func(t)?;
                    }
                }
                Direction::Reverse => {
                    for t in (0..seq_len).rev() {
                        func(t)?;
                    }
                }
                _ => unreachable!(),
            }
            final_h.slice(&select![start:end, ..])?.copy_from(&h_t);
            final_c.slice(&select![start:end, ..])?.copy_from(&c_t);
            Ok(())
        };
        let chunks = mt_intervals(batch_size as usize, crate::physical_cores());
        let prepacked_r = hpt_matmul::prepack_rhs(
            (r_t.ptr::<T>().ptr as *const T, r_t.size() as i64),
            4 * (hidden_size as usize),
            hidden_size as usize,
            [r_t.strides()[0], r_t.strides()[1]],
            true
        );
        chunks.into_par_iter().for_each(|(start, end)| {
            let (sliced_shape, sliced_strides, offset) = slice_process(
                &y.layout,
                &select![:, start as i64:end as i64, ..],
                1
            ).expect("slice error");
            let y_layout = Layout::new(
                [sliced_shape[1], sliced_shape[2]],
                [sliced_strides[1], sliced_strides[2]]
            );
            let y_ptr = y.ptr::<T>() + offset;
            func(start as i64, end as i64, y_ptr, &y_layout, &prepacked_r).expect(
                "lstm step error"
            );
        });
    } else {
        let gates = Tensor::empty(&[batch_size, 4 * hidden_size], x.dtype, x.device.clone())?;
        let prepacked_r = hpt_matmul::prepack_rhs(
            (r_t.ptr::<T>().ptr as *const T, r_t.size() as i64),
            4 * (hidden_size as usize),
            hidden_size as usize,
            [r_t.strides()[0], r_t.strides()[1]],
            false
        );
        let mut h_t_ptr = h_t.ptr::<u8>();
        let mut h_t_layout = h_t.layout();
        let r_t_ptr = r_t.ptr::<u8>();
        let r_t_layout = r_t.layout();
        let gates_ptr = gates.ptr::<u8>();
        let gates_layout = gates.layout();
        let y_layout = Layout::new([y.shape()[1], y.shape()[2]], [y.strides()[1], y.strides()[2]]);
        let mut func = |t: i64| -> Result<(), TensorError> {
            let bias_ptr = tmp.ptr::<T>() + (t as usize) * (tmp.strides()[0] as usize);
            let bias_layout = Layout::new(
                [tmp.shape()[1], tmp.shape()[2]],
                [tmp.strides()[1], tmp.strides()[2]]
            );
            addmm_prepacked_(
                h_t_ptr,
                h_t_layout,
                r_t_ptr,
                r_t_layout,
                bias_ptr.cast::<u8>(),
                &bias_layout,
                gates_ptr,
                gates_layout,
                1,
                x.dtype,
                Some(&prepacked_r)
            )?;
            let gates_b_stride = gates.strides()[0] as i64;
            for b in 0..batch_size {
                lstm_post_process(
                    y.ptr::<T>(),
                    c_t.ptr::<T>(),
                    gates.ptr::<T>(),
                    p.as_ref().map(|p| p.ptr::<T>()),
                    seq_ptr,
                    b,
                    t,
                    [y_b_stride, y_sq_stride],
                    c_b_stride,
                    gates_b_stride,
                    hidden_size,
                    activate1,
                    activate1_vec,
                    activate2,
                    activate2_vec
                );
            }

            h_t_ptr = (y.ptr::<T>() + (t as usize) * (y.strides()[0] as usize)).cast::<u8>();
            h_t_layout = &y_layout;
            Ok(())
        };
        match direction {
            Direction::Forward => {
                for t in 0..seq_len {
                    func(t)?;
                }
            }
            Direction::Reverse => {
                for t in (0..seq_len).rev() {
                    func(t)?;
                }
            }
            _ => unreachable!(),
        }
        final_h.copy_from(&h_t);
        final_c.copy_from(&c_t);
    }

    Ok((y, final_h, final_c))
}

fn lstm_step_onnx<T: CommonBounds>(
    x: &Tensor, // [seq_len, batch_size, input_size]
    w: &(hpt_matmul::PrePackedRhs, Layout), // [input_size, 4 * hidden_size]
    r: &(hpt_matmul::PrePackedRhs, Layout), // [hidden_size, 4 * hidden_size]
    summed_bias: Option<&Tensor>, // [8 * hidden_size]
    p: Option<&Tensor>, // [3*hidden_size]
    initial_h: &Tensor, // [batch_size, hidden_size]
    initial_c: &Tensor, // [batch_size, hidden_size],
    seq_lens: Option<&Tensor>, // [batch_size]
    direction: Direction,
    hidden_size: i64,
    activate1: impl (Fn(T) -> T) + Send + Sync + Copy,
    activate1_vec: impl (Fn(T::Vec) -> T::Vec) + Send + Sync + Copy,
    activate2: impl (Fn(T) -> T) + Send + Sync + Copy,
    activate2_vec: impl (Fn(T::Vec) -> T::Vec) + Send + Sync + Copy
)
    -> Result<(Tensor, Tensor, Tensor), TensorError>
    where
        T: FloatOutUnary<Output = T> + hpt_matmul::MatmulMicroKernel + Zero,
        T::Vec: FloatOutUnary<Output = T::Vec>
{
    let seq_len = x.shape()[0];
    let batch_size = x.shape()[1];

    let seq_ptr = if let Some(seq_lens) = seq_lens {
        assert_eq!(seq_lens.shape().as_slice(), &[batch_size]);
        assert_eq!(seq_lens.dtype(), DType::I64);
        Some(seq_lens.ptr::<i64>())
    } else {
        None
    };

    let h_t = initial_h.clone(); // [batch_size, hidden_size]
    let c_t = initial_c.clone(); // [batch_size, hidden_size]

    let y = Tensor::empty(&[seq_len, batch_size, hidden_size], x.dtype, x.device.clone())?;
    let mut final_h = Tensor::empty(&[batch_size, hidden_size], x.dtype, x.device.clone())?;
    let mut final_c = Tensor::empty(&[batch_size, hidden_size], x.dtype, x.device.clone())?;

    let x_reshaped = x.reshape(&[seq_len * batch_size, -1])?;

    let mut projection = matmul_prepare(
        &x_reshaped.device,
        &x_reshaped.layout,
        x_reshaped.dtype,
        &w.1,
        x_reshaped.dtype,
        None
    )?;
    if let Some(sum_bias) = summed_bias {
        addmm_prepacked_(
            x_reshaped.ptr::<u8>(),
            x_reshaped.layout(),
            Pointer::new(w.0.buffer.0.ptr, w.0.buffer.1.size() as i64),
            &w.1,
            sum_bias.ptr::<u8>(),
            &sum_bias.layout,
            projection.ptr::<u8>(),
            projection.layout(),
            crate::physical_cores(),
            x.dtype,
            Some(&w.0)
        )?;
    } else {
        matmul_prepacked_(
            x_reshaped.ptr::<u8>(),
            x_reshaped.layout(),
            Pointer::new(w.0.buffer.0.ptr, w.0.buffer.1.size() as i64),
            &w.1,
            projection.ptr::<u8>(),
            projection.layout(),
            crate::physical_cores(),
            x.dtype,
            Some(&w.0)
        )?;
    }
    projection = projection.reshape(&[seq_len, batch_size, 4 * hidden_size])?; // [seq_len, batch_size, 4 * hidden_size]

    let batch_size = x.shape()[1];

    let y_sq_stride = y.strides()[0] as i64;
    let y_b_stride = y.strides()[1] as i64;
    let c_b_stride = c_t.strides()[0] as i64;
    let batch_parallel = (batch_size as usize) >= crate::physical_cores();
    if batch_parallel {
        let func = |
            start: i64,
            end: i64,
            y: Pointer<T>,
            y_layout: &Layout
        | -> Result<(), TensorError> {
            let gates = Tensor::empty(&[end - start, 4 * hidden_size], x.dtype, x.device.clone())?;
            let h_t = initial_h.slice(&select![start:end, ..])?;
            let c_t = initial_c.slice(&select![start:end, ..])?;
            let tmp = projection.slice(&select![:, start:end, ..])?;
            let mut h_t_ptr = h_t.ptr::<u8>();
            let mut h_t_layout = h_t.layout();
            let gates_ptr = gates.ptr::<u8>();
            let gates_layout = gates.layout();
            let bias_layout = Layout::new(
                [tmp.shape()[1], tmp.shape()[2]],
                [tmp.strides()[1], tmp.strides()[2]]
            );

            let mut func = |t: i64| -> Result<(), TensorError> {
                let bias_ptr = tmp.ptr::<T>() + (t as usize) * (tmp.strides()[0] as usize);
                addmm_prepacked_(
                    h_t_ptr,
                    h_t_layout,
                    Pointer::new(r.0.buffer.0.ptr, r.0.buffer.1.size() as i64),
                    &r.1,
                    bias_ptr.cast::<u8>(),
                    &bias_layout,
                    gates_ptr,
                    gates_layout,
                    1,
                    x.dtype,
                    Some(&r.0)
                )?;
                for b in 0..end - start {
                    lstm_post_process(
                        y,
                        c_t.ptr::<T>(),
                        gates.ptr::<T>(),
                        p.as_ref().map(|p| p.ptr::<T>()),
                        seq_ptr,
                        b,
                        t,
                        [y_b_stride, y_sq_stride],
                        c_b_stride,
                        gates.strides()[0],
                        hidden_size,
                        activate1,
                        activate1_vec,
                        activate2,
                        activate2_vec
                    );
                }
                h_t_ptr = (y + t * y_sq_stride).cast::<u8>();
                h_t_layout = &y_layout;
                Ok(())
            };
            if direction == Direction::Forward {
                for t in 0..seq_len {
                    func(t)?;
                }
            } else {
                for t in (0..seq_len).rev() {
                    func(t)?;
                }
            }
            final_h.slice(&select![start:end, ..])?.copy_from_thr_specific(&h_t, 1);
            final_c.slice(&select![start:end, ..])?.copy_from_thr_specific(&c_t, 1);
            Ok(())
        };

        let chunks = mt_intervals(batch_size as usize, crate::physical_cores());

        spindle::for_each(crate::physical_cores(), chunks, |(start, end)| {
            let (sliced_shape, sliced_strides, offset) = slice_process(
                &y.layout,
                &select![:, start as i64:end as i64, ..],
                1
            ).expect("slice error");

            let y_layout = Layout::new(
                [sliced_shape[1], sliced_shape[2]],
                [sliced_strides[1], sliced_strides[2]]
            );

            let y_ptr = y.ptr::<T>() + offset;

            func(start as i64, end as i64, y_ptr, &y_layout).expect("lstm step error");
        });
    } else {
        let gates = Tensor::empty(&[batch_size, 4 * hidden_size], x.dtype, x.device.clone())?;
        let mut h_t_ptr = h_t.ptr::<u8>();
        let mut h_t_layout = h_t.layout();
        let gates_ptr = gates.ptr::<u8>();
        let gates_layout = gates.layout();
        let y_layout = Layout::new([y.shape()[1], y.shape()[2]], [y.strides()[1], y.strides()[2]]);
        let mut func = |t: i64| -> Result<(), TensorError> {
            let bias_ptr =
                projection.ptr::<T>() + (t as usize) * (projection.strides()[0] as usize);
            let bias_layout = Layout::new(
                [projection.shape()[1], projection.shape()[2]],
                [projection.strides()[1], projection.strides()[2]]
            );
            addmm_prepacked_(
                h_t_ptr,
                h_t_layout,
                Pointer::new(r.0.buffer.0.ptr, r.0.buffer.1.size() as i64),
                &r.1,
                bias_ptr.cast::<u8>(),
                &bias_layout,
                gates_ptr,
                gates_layout,
                1,
                x.dtype,
                Some(&r.0)
            )?;
            let gates_b_stride = gates.strides()[0] as i64;
            for b in 0..batch_size {
                lstm_post_process(
                    y.ptr::<T>(),
                    c_t.ptr::<T>(),
                    gates.ptr::<T>(),
                    p.as_ref().map(|p| p.ptr::<T>()),
                    seq_ptr,
                    b,
                    t,
                    [y_b_stride, y_sq_stride],
                    c_b_stride,
                    gates_b_stride,
                    hidden_size,
                    activate1,
                    activate1_vec,
                    activate2,
                    activate2_vec
                );
            }

            h_t_ptr = (y.ptr::<T>() + (t as usize) * (y.strides()[0] as usize)).cast::<u8>();
            h_t_layout = &y_layout;
            Ok(())
        };
        match direction {
            Direction::Forward => {
                for t in 0..seq_len {
                    func(t)?;
                }
            }
            Direction::Reverse => {
                for t in (0..seq_len).rev() {
                    func(t)?;
                }
            }
            _ => unreachable!(),
        }
        final_h.copy_from(&h_t);
        final_c.copy_from(&c_t);
    }

    Ok((y, final_h, final_c))
}

fn lstm_cell_ref<T>(
    x: &Tensor, // [seq_length, batch_size, input_size]
    w: &Tensor, // [num_directions, 4 * hidden_size, input_size]
    r: &Tensor, // [num_directions, 4 * hidden_size, hidden_size]
    b: Option<&Tensor>, // [num_directions, 8 * hidden_size]
    seq_lens: Option<&Tensor>, // [batch_size]
    init_h: Option<&Tensor>, // [num_directions, batch_size, hidden_size]
    init_c: Option<&Tensor>, // [num_directions, batch_size, hidden_size]
    p: Option<&Tensor>, // [num_directions, 3*hidden_size],
    direction: Direction,
    f: impl (Fn(T) -> T) + Send + Sync + Copy,
    f_vec: impl (Fn(T::Vec) -> T::Vec) + Send + Sync + Copy,
    g: impl (Fn(T) -> T) + Send + Sync + Copy,
    g_vec: impl (Fn(T::Vec) -> T::Vec) + Send + Sync + Copy
)
    -> Result<(Tensor, Tensor, Tensor), TensorError>
    where
        T: FloatOutUnary<Output = T> +
            ToDType +
            CommonBounds +
            hpt_matmul::MatmulMicroKernel +
            Zero,
        T::Vec: FloatOutUnary<Output = T::Vec>
{
    let seq_length = x.shape()[0];
    let batch_size = x.shape()[1];

    let num_directions = w.shape()[0];
    let hidden_size = r.shape()[2];

    assert_eq!(num_directions, direction.num_directions() as i64);

    let y = Tensor::empty(
        &[seq_length, num_directions, batch_size, hidden_size],
        x.dtype,
        x.device.clone()
    )?;
    let y_h = if let Some(h) = init_h {
        h.clone()
    } else {
        Tensor::zeros(&[num_directions, batch_size, hidden_size], x.dtype, x.device.clone())?
    };

    let y_c = if let Some(c) = init_c {
        c.clone()
    } else {
        Tensor::zeros(&[num_directions, batch_size, hidden_size], x.dtype, x.device.clone())?
    };

    for dir in 0..num_directions {
        let direction = match direction {
            Direction::Forward => Direction::Forward,
            Direction::Reverse => Direction::Reverse,
            Direction::Bidirectional => {
                if dir == 0 { Direction::Forward } else { Direction::Reverse }
            }
        };

        let h0 = y_h.slice(&select![dir:dir + 1, ..])?.squeeze(&[0])?;

        let c0 = y_c.slice(&select![dir:dir + 1, ..])?.squeeze(&[0])?;

        let b = if let Some(b) = b {
            Some(b.slice(&select![dir:dir + 1, ..])?.squeeze(&[0])?)
        } else {
            None
        };
        let r = r.slice(&select![dir:dir + 1, ..])?.squeeze(&[0])?;
        let w = w.slice(&select![dir:dir + 1, ..])?.squeeze(&[0])?;

        let w_t = w.t()?;
        let r_t = r.t()?;

        let (out, h_t, c_t) = lstm_step(
            x,
            &w_t,
            &r_t,
            b.as_ref(),
            p,
            &h0,
            &c0,
            seq_lens,
            direction,
            hidden_size,
            f,
            f_vec,
            g,
            g_vec
        )?;

        let mut y_local = y.slice(&select![:, dir:dir + 1, ..])?.squeeze(&[1])?;
        y_local.copy_from(&out);
        let mut y_h_local = y_h.slice(&select![dir:dir + 1, ..])?.squeeze(&[0])?;
        y_h_local.copy_from(&h_t);
        let mut y_c_local = y_c.slice(&select![dir:dir + 1, ..])?.squeeze(&[0])?;
        y_c_local.copy_from(&c_t);
    }
    Ok((y, y_h, y_c))
}

fn lstm_cell_onnx<T>(
    x: &Tensor, // [seq_length, batch_size, input_size]
    w: &[(hpt_matmul::PrePackedRhs, Layout)], // [num_directions, 4 * hidden_size, input_size]
    r: &[(hpt_matmul::PrePackedRhs, Layout)], // [num_directions, 4 * hidden_size, hidden_size]
    summed_bias: Option<&Tensor>, // [num_directions, 4 * hidden_size]
    seq_lens: Option<&Tensor>, // [batch_size]
    init_h: Option<&Tensor>, // [num_directions, batch_size, hidden_size]
    init_c: Option<&Tensor>, // [num_directions, batch_size, hidden_size]
    p: Option<&Tensor>, // [num_directions, 3*hidden_size],
    direction: Direction,
    f: impl (Fn(T) -> T) + Send + Sync + Copy,
    f_vec: impl (Fn(T::Vec) -> T::Vec) + Send + Sync + Copy,
    g: impl (Fn(T) -> T) + Send + Sync + Copy,
    g_vec: impl (Fn(T::Vec) -> T::Vec) + Send + Sync + Copy
)
    -> Result<(Tensor, Tensor, Tensor), TensorError>
    where
        T: FloatOutUnary<Output = T> +
            ToDType +
            CommonBounds +
            hpt_matmul::MatmulMicroKernel +
            Zero,
        T::Vec: FloatOutUnary<Output = T::Vec>
{
    let seq_length = x.shape()[0];
    let batch_size = x.shape()[1];

    let num_directions = match direction {
        Direction::Forward | Direction::Reverse => 1,
        Direction::Bidirectional => 2,
    };
    let hidden_size = r[0].1.shape()[0]; // r is transposed, shape is [hidden_size, 4 * hidden_size]

    assert_eq!(num_directions, direction.num_directions() as i64);

    let y = Tensor::empty(
        &[seq_length, num_directions, batch_size, hidden_size],
        x.dtype,
        x.device.clone()
    )?;
    let y_h = if let Some(h) = init_h {
        h.clone()
    } else {
        Tensor::zeros(&[num_directions, batch_size, hidden_size], x.dtype, x.device.clone())?
    };

    let y_c = if let Some(c) = init_c {
        c.clone()
    } else {
        Tensor::zeros(&[num_directions, batch_size, hidden_size], x.dtype, x.device.clone())?
    };

    for dir in 0..num_directions {
        let direction = match direction {
            Direction::Forward => Direction::Forward,
            Direction::Reverse => Direction::Reverse,
            Direction::Bidirectional => {
                if dir == 0 { Direction::Forward } else { Direction::Reverse }
            }
        };

        let h0 = y_h.slice(&select![dir:dir + 1, ..])?.squeeze(&[0])?;

        let c0 = y_c.slice(&select![dir:dir + 1, ..])?.squeeze(&[0])?;

        let b = if let Some(b) = summed_bias {
            Some(b.slice(&select![dir:dir + 1, ..])?.squeeze(&[0])?)
        } else {
            None
        };

        let (out, h_t, c_t) = lstm_step_onnx::<T>(
            x,
            &w[dir as usize],
            &r[dir as usize],
            b.as_ref(),
            p,
            &h0,
            &c0,
            seq_lens,
            direction,
            hidden_size,
            f,
            f_vec,
            g,
            g_vec
        )?;

        let mut y_local = y.slice(&select![:, dir:dir + 1, ..])?.squeeze(&[1])?;
        y_local.copy_from(&out);
        let mut y_h_local = y_h.slice(&select![dir:dir + 1, ..])?.squeeze(&[0])?;
        y_h_local.copy_from(&h_t);
        let mut y_c_local = y_c.slice(&select![dir:dir + 1, ..])?.squeeze(&[0])?;
        y_c_local.copy_from(&c_t);
    }
    Ok((y, y_h, y_c))
}

impl Tensor {
    pub fn lstm(
        &self,
        w: &Tensor,
        r: &Tensor,
        b: Option<&Tensor>,
        seq_lens: Option<&Tensor>,
        init_h: Option<&Tensor>,
        init_c: Option<&Tensor>,
        p: Option<&Tensor>,
        direction: &str
    ) -> Result<(Tensor, Tensor, Tensor), TensorError> {
        let direction = Direction::from_str(direction);
        let (y, y_h, y_c) = match self.dtype {
            DType::F32 =>
                lstm_cell_ref::<f32>(
                    self,
                    w,
                    r,
                    b,
                    seq_lens,
                    init_h,
                    init_c,
                    p,
                    direction,
                    |x| x._sigmoid(),
                    |x| x._sigmoid(),
                    |x| x._tanh(),
                    |x| x._tanh()
                )?,
            #[cfg(feature = "f16")]
            DType::F16 =>
                lstm_cell_ref::<half::f16>(
                    self,
                    w,
                    r,
                    b,
                    seq_lens,
                    init_h,
                    init_c,
                    p,
                    direction,
                    |x| x._sigmoid(),
                    |x| x._sigmoid(),
                    |x| x._tanh(),
                    |x| x._tanh()
                )?,
            #[cfg(feature = "bf16")]
            DType::BF16 =>
                lstm_cell_ref::<half::bf16>(
                    self,
                    w,
                    r,
                    b,
                    seq_lens,
                    init_h,
                    init_c,
                    p,
                    direction,
                    |x| x._sigmoid(),
                    |x| x._tanh()
                )?,
            _ => unimplemented!(),
        };
        Ok((y, y_h, y_c))
    }

    pub(crate) fn lstm_onnx(
        &self,
        w: &[(hpt_matmul::PrePackedRhs, Layout)],
        r: &[(hpt_matmul::PrePackedRhs, Layout)],
        summed_bias: Option<&Tensor>,
        seq_lens: Option<&Tensor>,
        init_h: Option<&Tensor>,
        init_c: Option<&Tensor>,
        p: Option<&Tensor>,
        direction: &str
    ) -> Result<(Tensor, Tensor, Tensor), TensorError> {
        let direction = Direction::from_str(direction);
        let (y, y_h, y_c) = match self.dtype {
            DType::F32 =>
                lstm_cell_onnx::<f32>(
                    self,
                    w,
                    r,
                    summed_bias,
                    seq_lens,
                    init_h,
                    init_c,
                    p,
                    direction,
                    |x| x._sigmoid(),
                    |x| x._sigmoid(),
                    |x| x._tanh(),
                    |x| x._tanh()
                )?,
            #[cfg(feature = "f16")]
            DType::F16 =>
                lstm_cell_onnx::<half::f16>(
                    self,
                    w,
                    r,
                    summed_bias,
                    seq_lens,
                    init_h,
                    init_c,
                    p,
                    direction,
                    |x| x._sigmoid(),
                    |x| x._sigmoid(),
                    |x| x._tanh(),
                    |x| x._tanh()
                )?,
            #[cfg(feature = "bf16")]
            DType::BF16 =>
                lstm_cell_ref::<half::bf16>(
                    self,
                    w,
                    r,
                    b,
                    seq_lens,
                    init_h,
                    init_c,
                    p,
                    direction,
                    |x| x._sigmoid(),
                    |x| x._tanh()
                )?,
            _ => unimplemented!(),
        };
        Ok((y, y_h, y_c))
    }
}
