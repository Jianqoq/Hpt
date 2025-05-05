use crate::Tensor;
use half::{bf16, f16};
use hpt_common::{error::base::TensorError, utils::simd_ref::MutVec};
use hpt_iterator::{
    TensorIterator,
    iterator_traits::{ParStridedIteratorSimd, ParStridedIteratorSimdZip},
};
use hpt_macros::select;
use hpt_traits::tensor::{CommonBounds, TensorInfo};
use hpt_types::traits::VecTrait;
use hpt_types::{dtype::DType, type_promote::FloatOutUnary};

fn lstm_step_ref(
    x: &Tensor,         // [seq_len, batch_size, input_size]
    w: &Tensor,         // [4 * hidden_size, input_size]
    r: &Tensor,         // [4 * hidden_size, hidden_size]
    b: Option<&Tensor>, // [8 * hidden_size]
    initial_h: &Tensor, // [batch_size, hidden_size]
    initial_c: &Tensor, // [batch_size, hidden_size]
) -> Result<(Tensor, Tensor, Tensor), TensorError> {
    let seq_len = x.shape()[0];
    let batch_size = x.shape()[1];

    let hidden_size = r.shape()[1];

    let wb = if let Some(b) = b {
        Some(b.slice(&select![0:4*hidden_size])?)
    } else {
        None
    };
    let rb = if let Some(b) = b {
        Some(b.slice(&select![4*hidden_size:])?)
    } else {
        None
    };

    let mut h_t = initial_h.clone();
    let mut c_t = initial_c.clone();

    let mut y = vec![];

    let x_reshaped = x.reshape(&[seq_len * batch_size, batch_size])?;

    let tmp = if let Some(wb) = &wb {
        x_reshaped.addmm(&w.t()?, wb)?
    } else {
        x_reshaped.matmul(&w.t()?)?
    };

    for _ in 0..seq_len {
        let gates = if let Some(rb) = &rb {
            &tmp + &h_t.matmul(&r.t()?)? + rb.clone()
        } else {
            &tmp + &h_t.matmul(&r.t()?)?
        };

        let i = gates.slice(&select![:, 0:hidden_size])?.sigmoid()?;
        let o = gates
            .slice(&select![:, hidden_size:2*hidden_size])?
            .sigmoid()?;
        let f = gates
            .slice(&select![:, 2*hidden_size:3*hidden_size])?
            .sigmoid()?;
        let g = gates.slice(&select![:, 3*hidden_size:])?.tanh()?;

        c_t = c_t.mul_add(&f, &(i * g))?;
        h_t = o * c_t.tanh()?;

        y.push(h_t.clone());
    }

    let y = y.iter().map(|t| t).collect::<Vec<_>>();
    Ok((Tensor::concat(y, 0, true)?, h_t, c_t))
}

fn lstm_step_optimized<T: CommonBounds>(
    x: &Tensor,         // [seq_len, batch_size, input_size]
    w: &Tensor,         // [4 * hidden_size, input_size]
    r: &Tensor,         // [4 * hidden_size, hidden_size]
    b: Option<&Tensor>, // [8 * hidden_size]
    initial_h: &Tensor, // [batch_size, hidden_size]
    initial_c: &Tensor, // [batch_size, hidden_size]
    f: impl Fn(&mut T) + Send + Sync + Copy,
    f_vec: impl Fn(MutVec<'_, T::Vec>) + Send + Sync + Copy,
    g: impl Fn(&mut T) + Send + Sync + Copy,
    g_vec: impl Fn(MutVec<'_, T::Vec>) + Send + Sync + Copy,
) -> Result<(Tensor, Tensor, Tensor), TensorError> {
    let seq_len = x.shape()[0];
    let batch_size = x.shape()[1];

    let hidden_size = r.shape()[1];

    let sum_bias = if let Some(b) = b {
        let wb = b.slice(&select![0:4*hidden_size])?;
        let rb = b.slice(&select![4*hidden_size:])?;
        Some(wb + rb)
    } else {
        None
    };

    let mut h_t = initial_h.clone(); // [batch_size, hidden_size]
    let mut c_t = initial_c.clone(); // [batch_size, hidden_size]

    let y = Tensor::empty(
        &[seq_len, batch_size, hidden_size],
        x.dtype,
        x.device.clone(),
    )?;

    let x_reshaped = x.reshape(&[seq_len * batch_size, -1])?;

    let tmp = if let Some(sum_bias) = &sum_bias {
        x_reshaped.addmm(&w.t()?, sum_bias)?
    } else {
        x_reshaped.matmul(&w.t()?)?
    };

    let mut gates = Tensor::empty(&[batch_size, 4 * hidden_size], x.dtype, x.device.clone())?;

    for t in 0..seq_len {
        // r.T = [hidden_size, 4 * hidden_size]
        gates = h_t.addmm_(&r.t()?, &tmp, &mut gates)?;

        let mut f_gates = gates.slice(&select![:, 0:3 * hidden_size])?;
        let mut g_gates = gates.slice(&select![:, 3 * hidden_size:])?;

        f_gates.par_iter_mut_simd().for_each(f, f_vec);
        g_gates.par_iter_mut_simd().for_each(g, g_vec);

        let i = gates.slice(&select![:, 0:hidden_size])?;
        let o = gates.slice(&select![:, hidden_size:2*hidden_size])?;
        let f = gates.slice(&select![:, 2*hidden_size:3*hidden_size])?;
        let g = gates.slice(&select![:, 3*hidden_size:])?;

        let mut sliced_y = y.slice(&select![t:t + 1, ..])?.squeeze(&[0])?;
        sliced_y
            .par_iter_mut_simd()
            .zip(c_t.par_iter_mut_simd())
            .zip(f.par_iter_simd())
            .zip(i.par_iter_simd())
            .zip(g.par_iter_simd())
            .zip(o.par_iter_simd())
            .for_each(
                |(((((y, c_t), f), i), g), o): (((((&mut f32, &mut f32), f32), f32), f32), f32)| {
                    let tmp = (*c_t).mul_add(f, i * g);
                    *y = o * tmp.tanh();
                    *c_t = tmp;
                },
                |(((((y, c_t), f), i), g), o)| {
                    let c_prev = c_t.read_unaligned();
                    let tmp = c_prev.mul_add(f, i * g);
                    y.write_unaligned(o * tmp._tanh());
                    c_t.write_unaligned(tmp);
                },
            );

        h_t = sliced_y;
    }

    Ok((y, h_t, c_t))
}

fn lstm_cell_ref<T: CommonBounds>(
    x: &Tensor,                // [seq_length, batch_size, input_size]
    w: &Tensor,                // [num_directions, 4 * hidden_size, input_size]
    r: &Tensor,                // [num_directions, 4 * hidden_size, hidden_size]
    b: Option<&Tensor>,        // [num_directions, 8 * hidden_size]
    seq_lens: Option<&Tensor>, // [batch_size]
    init_h: Option<&Tensor>,   // [num_directions, batch_size, hidden_size]
    init_c: Option<&Tensor>,   // [num_directions, batch_size, hidden_size]
    p: Option<&Tensor>,        // [num_directions, 3*hidden_size],
    f: impl Fn(&mut T) + Send + Sync + Copy,
    f_vec: impl Fn(MutVec<'_, T::Vec>) + Send + Sync + Copy,
    g: impl Fn(&mut T) + Send + Sync + Copy,
    g_vec: impl Fn(MutVec<'_, T::Vec>) + Send + Sync + Copy,
) -> Result<(Tensor, Tensor, Tensor), TensorError> {
    let seq_length = x.shape()[0];
    let batch_size = x.shape()[1];

    let num_directions = w.shape()[0];
    let hidden_size = r.shape()[2];

    let mut y = if num_directions == 1 {
        Tensor::empty(&[1], x.dtype, x.device.clone())?
    } else {
        Tensor::empty(
            &[seq_length, num_directions, batch_size, hidden_size],
            x.dtype,
            x.device.clone(),
        )?
    };
    let mut y_h = if let Some(h) = init_h {
        h.clone()
    } else {
        Tensor::zeros(
            &[num_directions, batch_size, hidden_size],
            x.dtype,
            x.device.clone(),
        )?
    };

    let mut y_c = if let Some(c) = init_c {
        c.clone()
    } else {
        Tensor::zeros(
            &[num_directions, batch_size, hidden_size],
            x.dtype,
            x.device.clone(),
        )?
    };

    // let now = std::time::Instant::now();
    for dir in 0..num_directions {
        let h0 = y_h.slice(&select![dir:dir + 1, ..])?.squeeze(&[0])?;

        let c0 = y_c.slice(&select![dir:dir + 1, ..])?.squeeze(&[0])?;

        let b = if let Some(b) = b {
            Some(b.slice(&select![dir:dir + 1, ..])?.squeeze(&[0])?)
        } else {
            None
        };
        let r = r.slice(&select![dir:dir + 1, ..])?.squeeze(&[0])?;
        let w = w.slice(&select![dir:dir + 1, ..])?.squeeze(&[0])?;

        let (out, h_t, c_t) =
            lstm_step_optimized(x, &w, &r, b.as_ref(), &h0, &c0, f, f_vec, g, g_vec)?;

        if num_directions == 1 {
            y = out.unsqueeze(&[1])?;
            y_h = h_t.unsqueeze(&[0])?;
            y_c = c_t.unsqueeze(&[0])?;
        } else {
            let mut y_local = y.slice(&select![:, dir:dir + 1, ..])?.squeeze(&[1])?;
            y_local.copy_from(&out);
            let mut y_h_local = y_h.slice(&select![dir:dir + 1, ..])?.squeeze(&[0])?;
            y_h_local.copy_from(&h_t);
            let mut y_c_local = y_c.slice(&select![dir:dir + 1, ..])?.squeeze(&[0])?;
            y_c_local.copy_from(&c_t);
        }
    }
    // let duration = now.elapsed();
    // println!("Time taken: {:?}", duration);
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
    ) -> Result<(Tensor, Tensor, Tensor), TensorError> {
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
                |x| *x = x._sigmoid(),
                |x| x.write_unaligned(x.read_unaligned()._sigmoid()),
                |x| *x = x._tanh(),
                |x| x.write_unaligned(x.read_unaligned()._tanh()),
            )?,
            DType::F16 => lstm_cell_ref::<f16>(
                self,
                w,
                r,
                b,
                seq_lens,
                init_h,
                init_c,
                p,
                |x| *x = x._sigmoid(),
                |x| x.write_unaligned(x.read_unaligned()._sigmoid()),
                |x| *x = x._tanh(),
                |x| x.write_unaligned(x.read_unaligned()._tanh()),
            )?,
            DType::BF16 => lstm_cell_ref::<bf16>(
                self,
                w,
                r,
                b,
                seq_lens,
                init_h,
                init_c,
                p,
                |x| *x = x._sigmoid(),
                |x| x.write_unaligned(x.read_unaligned()._sigmoid()),
                |x| *x = x._tanh(),
                |x| x.write_unaligned(x.read_unaligned()._tanh()),
            )?,
            _ => unimplemented!(),
        };
        Ok((y, y_h, y_c))
    }
}
