use crate::{
    Tensor,
    ops::tensor::matmul::{
        matmul::matmul_prepack_rhs, microkernel_trait::MatmulMicroKernel, utils::PrePackedRhs,
    },
};
use hpt_common::{Pointer, error::base::TensorError};
use hpt_macros::select;
use hpt_traits::tensor::{CommonBounds, TensorInfo};
use hpt_types::{
    dtype::{DType, ToDType},
    type_promote::FloatOutUnary,
};
use hpt_types::{traits::VecTrait, type_promote::NormalOut};

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
    activate1: impl Fn(T) -> T + Send + Sync + Copy,
    activate1_vec: impl Fn(T::Vec) -> T::Vec + Send + Sync + Copy,
    activate2: impl Fn(T) -> T + Send + Sync + Copy,
    activate2_vec: impl Fn(T::Vec) -> T::Vec + Send + Sync + Copy,
) where
    T: FloatOutUnary<Output = T>,
    T::Vec: FloatOutUnary<Output = T::Vec>,
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

    let num_vec = hidden_size as usize / T::Vec::SIZE;
    let rem = hidden_size as usize % T::Vec::SIZE;

    let i_vec_ptr = i.cast::<T::Vec>();
    let o_vec_ptr = o.cast::<T::Vec>();
    let f_vec_ptr = f.cast::<T::Vec>();
    let g_vec_ptr = g.cast::<T::Vec>();
    let y_vec_ptr = y.cast::<T::Vec>();
    let c_t_vec_ptr = c_t.cast::<T::Vec>();

    match p {
        Some(p) => {
            let pi = p;
            let pf = p + hidden_size;
            let po = p + 2 * hidden_size;
            let pi_vec_ptr = pi.cast::<T::Vec>();
            let pf_vec_ptr = pf.cast::<T::Vec>();
            let po_vec_ptr = po.cast::<T::Vec>();
            for h in 0..num_vec {
                let pi_vec = pi_vec_ptr.add(h).read_unaligned();
                let pf_vec = pf_vec_ptr.add(h).read_unaligned();
                let po_vec = po_vec_ptr.add(h).read_unaligned();
                let c_t_vec = c_t_vec_ptr.add(h).read_unaligned();
                let pi_mul_ct = pi_vec._mul(c_t_vec);
                let pf_mul_ct = pf_vec._mul(c_t_vec);
                let po_mul_ct = po_vec._mul(c_t_vec);
                let i_vec = activate1_vec(i_vec_ptr.add(h).read_unaligned()._add(pi_mul_ct));
                let o_vec = activate1_vec(o_vec_ptr.add(h).read_unaligned()._add(po_mul_ct));
                let f_vec = activate1_vec(f_vec_ptr.add(h).read_unaligned()._add(pf_mul_ct));
                let g_vec = activate2_vec(g_vec_ptr.add(h).read_unaligned());
                let tmp = c_t_vec._mul_add(f_vec, i_vec._mul(g_vec));
                y_vec_ptr.add(h).write_unaligned(o_vec._mul(tmp._tanh()));
                c_t_vec_ptr.add(h).write_unaligned(tmp);
            }
            if rem > 0 {
                for h in num_vec * T::Vec::SIZE..hidden_size as usize {
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
        }
        None => {
            for h in 0..num_vec {
                let i_vec = activate1_vec(i_vec_ptr.add(h).read_unaligned());
                let o_vec = activate1_vec(o_vec_ptr.add(h).read_unaligned());
                let f_vec = activate1_vec(f_vec_ptr.add(h).read_unaligned());
                let g_vec = activate2_vec(g_vec_ptr.add(h).read_unaligned());
                let c_t_vec = c_t_vec_ptr.add(h).read_unaligned();
                let tmp = c_t_vec._mul_add(f_vec, i_vec._mul(g_vec));
                y_vec_ptr.add(h).write_unaligned(o_vec._mul(tmp._tanh()));
                c_t_vec_ptr.add(h).write_unaligned(tmp);
            }
            if rem > 0 {
                for h in num_vec * T::Vec::SIZE..hidden_size as usize {
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
}

fn lstm_step<T: CommonBounds>(
    x: &Tensor,                // [seq_len, batch_size, input_size]
    w_t: &Tensor,              // [input_size, 4 * hidden_size]
    r_t: &Tensor,              // [hidden_size, 4 * hidden_size]
    r: hpt_matmul::PrePackedRhs,           // [4 * hidden_size, hidden_size]
    b: Option<&Tensor>,        // [8 * hidden_size]
    p: Option<&Tensor>,        // [3*hidden_size]
    initial_h: &Tensor,        // [batch_size, hidden_size]
    initial_c: &Tensor,        // [batch_size, hidden_size],
    seq_lens: Option<&Tensor>, // [batch_size]
    direction: Direction,
    hidden_size: i64,
    num_threads: usize,
    activate1: impl Fn(T) -> T + Send + Sync + Copy,
    activate1_vec: impl Fn(T::Vec) -> T::Vec + Send + Sync + Copy,
    activate2: impl Fn(T) -> T + Send + Sync + Copy,
    activate2_vec: impl Fn(T::Vec) -> T::Vec + Send + Sync + Copy,
) -> Result<(Tensor, Tensor, Tensor), TensorError>
where
    T: FloatOutUnary<Output = T>,
    T::Vec: FloatOutUnary<Output = T::Vec>,
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

    let mut h_t = initial_h.clone(); // [batch_size, hidden_size]
    let c_t = initial_c.clone(); // [batch_size, hidden_size]

    let y = Tensor::empty(
        &[seq_len, batch_size, hidden_size],
        x.dtype,
        x.device.clone(),
    )?;

    let x_reshaped = x.reshape(&[seq_len * batch_size, -1])?;

    let tmp = if let Some(sum_bias) = &sum_bias {
        // x_reshaped.addmm(w_t, sum_bias)?
        x_reshaped._addmm_f32_(w_t, &sum_bias, None, num_threads, None)?
    } else {
        x_reshaped.matmul(w_t)?
    };

    let gates = Tensor::empty(&[batch_size, 4 * hidden_size], x.dtype, x.device.clone())?;

    let batch_size = x.shape()[1];

    let y_sq_stride = y.strides()[0] as i64;
    let y_b_stride = y.strides()[1] as i64;
    let c_b_stride = c_t.strides()[0] as i64;
    let gates_b_stride = gates.strides()[0] as i64;

    let mut func = |t: i64| -> Result<(), TensorError> {
        // let gates = h_t._addmm(r_t, &tmp, Some(gates.clone()), num_threads, Some(r.clone()))?;

        let gates = h_t._addmm_f32_(r_t, &tmp, Some(gates.clone()), num_threads, Some(r.clone()))?;
        let sliced_y = y.slice(&select![t:t + 1, ..])?.squeeze(&[0])?;

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
                activate2_vec,
            );
        }

        h_t = sliced_y;
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

    Ok((y, h_t, c_t))
}

fn lstm_cell_ref<T>(
    x: &Tensor,                // [seq_length, batch_size, input_size]
    w: &Tensor,                // [num_directions, 4 * hidden_size, input_size]
    r: &Tensor,                // [num_directions, 4 * hidden_size, hidden_size]
    b: Option<&Tensor>,        // [num_directions, 8 * hidden_size]
    seq_lens: Option<&Tensor>, // [batch_size]
    init_h: Option<&Tensor>,   // [num_directions, batch_size, hidden_size]
    init_c: Option<&Tensor>,   // [num_directions, batch_size, hidden_size]
    p: Option<&Tensor>,        // [num_directions, 3*hidden_size],
    direction: Direction,
    f: impl Fn(T) -> T + Send + Sync + Copy,
    f_vec: impl Fn(T::Vec) -> T::Vec + Send + Sync + Copy,
    g: impl Fn(T) -> T + Send + Sync + Copy,
    g_vec: impl Fn(T::Vec) -> T::Vec + Send + Sync + Copy,
) -> Result<(Tensor, Tensor, Tensor), TensorError>
where
    T: FloatOutUnary<Output = T> + MatmulMicroKernel + ToDType + CommonBounds + hpt_matmul::MatmulMicroKernel,
    T::Vec: FloatOutUnary<Output = T::Vec>,
{
    let seq_length = x.shape()[0];
    let batch_size = x.shape()[1];

    let num_directions = w.shape()[0];
    let hidden_size = r.shape()[2];

    assert_eq!(num_directions, direction.num_directions() as i64);

    let y = Tensor::empty(
        &[seq_length, num_directions, batch_size, hidden_size],
        x.dtype,
        x.device.clone(),
    )?;
    let y_h = if let Some(h) = init_h {
        h.clone()
    } else {
        Tensor::zeros(
            &[num_directions, batch_size, hidden_size],
            x.dtype,
            x.device.clone(),
        )?
    };

    let y_c = if let Some(c) = init_c {
        c.clone()
    } else {
        Tensor::zeros(
            &[num_directions, batch_size, hidden_size],
            x.dtype,
            x.device.clone(),
        )?
    };

    for dir in 0..num_directions {
        let direction = match direction {
            Direction::Forward => Direction::Forward,
            Direction::Reverse => Direction::Reverse,
            Direction::Bidirectional => {
                if dir == 0 {
                    Direction::Forward
                } else {
                    Direction::Reverse
                }
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
        let num_threads = crate::physical_cores();
        let prepacked_r = hpt_matmul::matmul_prepack_rhs(
            r.ptr::<T>().ptr as *const T,
            0,
            1,
            r_t.strides()[1],
            r_t.strides()[0],
            h0.shape(),
            r_t.shape(),
            num_threads,
        );

        let (out, h_t, c_t) = lstm_step(
            x,
            &w_t,
            &r_t,
            prepacked_r,
            b.as_ref(),
            p,
            &h0,
            &c0,
            seq_lens,
            direction,
            hidden_size,
            num_threads,
            f,
            f_vec,
            g,
            g_vec,
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
        direction: &str,
    ) -> Result<(Tensor, Tensor, Tensor), TensorError> {
        let direction = Direction::from_str(direction);
        let (y, y_h, y_c) = match self.dtype {
            DType::F32 => lstm_cell_ref::<f32>(
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
                |x| x._tanh(),
            )?,
            #[cfg(feature = "f16")]
            DType::F16 => lstm_cell_ref::<half::f16>(
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
                |x| x._tanh(),
            )?,
            #[cfg(feature = "bf16")]
            DType::BF16 => lstm_cell_ref::<half::bf16>(
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
                |x| x._tanh(),
            )?,
            _ => unimplemented!(),
        };
        Ok((y, y_h, y_c))
    }
}
