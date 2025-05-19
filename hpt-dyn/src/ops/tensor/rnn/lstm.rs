use std::sync::Arc;

use crate::{
    ops::tensor::matmul::{ matmul::addmm_prepacked_, microkernel_trait::MatmulMicroKernel },
    Tensor,
};
use hpt_common::{
    error::base::TensorError,
    layout::layout::Layout,
    shape::shape_utils::mt_intervals,
    Pointer,
};
use hpt_macros::select;
use hpt_matmul::Zero;
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
    activate2: impl (Fn(T) -> T) + Send + Sync + Copy
)
    where T: FloatOutUnary<Output = T>, T::Vec: FloatOutUnary<Output = T::Vec>
{
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
            for h in 0..hidden_size as usize {
                let i_res = activate1(i[h]);
                let o_res = activate1(o[h]);
                let f_res = activate1(f[h]);
                let g_res = activate2(g[h]);
                let c_t_res = c_t[h];
                let tmp = c_t_res._mul_add(f_res, i_res._mul(g_res));
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
    activate2: impl (Fn(T) -> T) + Send + Sync + Copy
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
            y: &Tensor,
            prepacked_r: Arc<Vec<hpt_matmul::NewPrePackedRhs>>
        | -> Result<(), TensorError> {
            let global_now = std::time::Instant::now();
            let gates = Tensor::empty(
                &[end - start, 4 * hidden_size],
                x.dtype,
                x.device.clone()
            )?;
            let h_t = initial_h.slice(&select![start:end, ..])?;
            let c_t = initial_c.slice(&select![start:end, ..])?;
            let tmp = tmp.slice(&select![:, start:end, ..])?;
            let mut h_t_ptr = h_t.ptr::<u8>();
            let mut h_t_layout = h_t.layout();
            let r_t_ptr = r_t.ptr::<u8>();
            let r_t_layout = r_t.layout();
            let gates_ptr = gates.ptr::<u8>();
            let gates_layout = gates.layout();
            let bias_layout = Layout::new([tmp.shape()[1]], [tmp.strides()[1]]);
            let y_layout = Layout::new(
                [y.shape()[1], y.shape()[2]],
                [y.strides()[1], y.strides()[2]]
            );

            let mut total_mm = std::time::Duration::from_secs(0);
            let mut total_post = std::time::Duration::from_secs(0);
            let mut func = |t: i64| -> Result<(), TensorError> {
                let bias_ptr = tmp.ptr::<T>() + (t as usize) * (tmp.strides()[0] as usize);
                let now = std::time::Instant::now();
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
                    Some(prepacked_r.clone())
                )?;
                total_mm += now.elapsed();

                let now = std::time::Instant::now();
                for b in 0..end - start {
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
                        gates.strides()[0],
                        hidden_size,
                        activate1,
                        activate2
                    );
                }
                total_post += now.elapsed();
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
            let now = std::time::Instant::now();
            final_h.slice(&select![start:end, ..])?.copy_from(&h_t);
            final_c.slice(&select![start:end, ..])?.copy_from(&c_t);
            println!(
                "mm: {:?}, post: {:?}, copy: {:?}, total: {:?}",
                total_mm,
                total_post,
                now.elapsed(),
                global_now.elapsed()
            );
            Ok(())
        };
        let chunks = mt_intervals(batch_size as usize, crate::physical_cores());
        let same_chunk = chunks
            .iter()
            .fold(true, |acc, (start, end)| { acc && end - start == chunks[0].1 - chunks[0].0 });
        let prepacked_r = if same_chunk {
            let prepacked_r = hpt_matmul::prepack_rhs(
                4 * (hidden_size as usize),
                (chunks[0].1 - chunks[0].0) as usize,
                hidden_size as usize,
                r_t.ptr::<T>().ptr as *const T,
                [r_t.strides()[0], r_t.strides()[1]],
                1
            );
            Arc::new(prepacked_r)
        } else {
            panic!("LSTM step: different chunk size is not supported yet");
        };
        let now = std::time::Instant::now();
        chunks.into_par_iter().for_each(|(start, end)| {
            let y = y.slice(&select![:, start as i64:end as i64, ..]).expect("lstm step error");
            func(start as i64, end as i64, &y, prepacked_r.clone()).expect("lstm step error");
        });
        println!("time taken: {:?}", now.elapsed());
    } else {
        let gates = Tensor::empty(&[batch_size, 4 * hidden_size], x.dtype, x.device.clone())?;
        let num_threads = crate::physical_cores();
        let prepacked_r = hpt_matmul::prepack_rhs(
            4 * (hidden_size as usize),
            batch_size as usize,
            hidden_size as usize,
            r_t.ptr::<T>().ptr as *const T,
            [r_t.strides()[0], r_t.strides()[1]],
            num_threads
        );
        let mut h_t_ptr = h_t.ptr::<u8>();
        let mut h_t_layout = h_t.layout();
        let r_t_ptr = r_t.ptr::<u8>();
        let r_t_layout = r_t.layout();
        let gates_ptr = gates.ptr::<u8>();
        let gates_layout = gates.layout();
        let y_layout = Layout::new(
            [y.shape()[1], y.shape()[2]],
            [y.strides()[1], y.strides()[2]]
        );

        let prepacked_r = Arc::new(prepacked_r);
        let mut func = |t: i64| -> Result<(), TensorError> {
            let bias_ptr = tmp.ptr::<T>() + (t as usize) * (tmp.strides()[0] as usize);
            let bias_layout = Layout::new([tmp.shape()[1]], [tmp.strides()[1]]);
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
                Some(prepacked_r.clone())
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
                    activate2
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
    g: impl (Fn(T) -> T) + Send + Sync + Copy
)
    -> Result<(Tensor, Tensor, Tensor), TensorError>
    where
        T: FloatOutUnary<Output = T> +
            MatmulMicroKernel +
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
            g
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
            DType::F32 => {
                #[inline(always)]
                fn sigmoid(x: f32) -> f32 {
                    if x > 0.0 { 1.0 / (1.0 + (-x).exp()) } else { x.exp() / (1.0 + x.exp()) }
                }
                #[inline(always)]
                fn tanh(x: f32) -> f32 {
                    2.0 * sigmoid(2.0 * x) - 1.0
                }
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
                    sigmoid,
                    tanh
                )?
            }
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
