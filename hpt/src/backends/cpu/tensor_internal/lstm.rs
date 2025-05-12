use crate::Tensor;
use hpt_allocator::Cpu;
use hpt_common::{error::base::TensorError, Pointer};
use hpt_iterator::iterator_traits::{ParStridedIteratorSimd, ParStridedIteratorSimdZip};
use hpt_iterator::TensorIterator;
use hpt_macros::select;
use hpt_matmul::PrePackedRhs;
use hpt_traits::ops::binary::Matmul;
use hpt_traits::{
    ops::{creation::TensorCreator, shape_manipulate::ShapeManipulate, slice::Slice},
    tensor::TensorInfo,
};
use hpt_types::{
    dtype::{ToDType, TypeCommon},
    type_promote::FloatOutUnary,
};
use hpt_types::{traits::VecTrait, type_promote::NormalOut};
use spindle::current_num_threads;

type F32Vec = <f32 as TypeCommon>::Vec;

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

fn lstm_post_process(
    y: Pointer<f32>,
    c_t: Pointer<f32>,
    gates: Pointer<f32>,
    p: Option<Pointer<f32>>,
    seq_lens: Option<Pointer<i64>>,
    batch_step: i64,
    time_step: i64,
    [y_b_stride, y_sq_stride]: [i64; 2],
    c_b_stride: i64,
    gates_b_stride: i64,
    hidden_size: i64,
    activate1: impl Fn(f32) -> f32 + Send + Sync + Copy,
    activate1_vec: impl Fn(F32Vec) -> F32Vec + Send + Sync + Copy,
    activate2: impl Fn(f32) -> f32 + Send + Sync + Copy,
    activate2_vec: impl Fn(F32Vec) -> F32Vec + Send + Sync + Copy,
) where
    f32: FloatOutUnary<Output = f32>,
    F32Vec: FloatOutUnary<Output = F32Vec>,
{
    let mut y = y + batch_step * y_b_stride + time_step * y_sq_stride;

    if let Some(seq_ptr) = seq_lens {
        let seq_len = seq_ptr.add(batch_step as usize).read();
        if time_step >= seq_len {
            let slice = unsafe { std::slice::from_raw_parts_mut(y.ptr, hidden_size as usize) };
            slice.fill(f32::ZERO);
            return;
        }
    }

    let mut c_t = c_t + batch_step * c_b_stride;
    let i = gates + batch_step * gates_b_stride;
    let o = i + hidden_size;
    let f = i + 2 * hidden_size;
    let g = i + 3 * hidden_size;

    let num_vec = hidden_size as usize / F32Vec::SIZE;
    let rem = hidden_size as usize % F32Vec::SIZE;

    let i_vec_ptr = i.cast::<F32Vec>();
    let o_vec_ptr = o.cast::<F32Vec>();
    let f_vec_ptr = f.cast::<F32Vec>();
    let g_vec_ptr = g.cast::<F32Vec>();
    let y_vec_ptr = y.cast::<F32Vec>();
    let c_t_vec_ptr = c_t.cast::<F32Vec>();

    match p {
        Some(p) => {
            let pi = p;
            let pf = p + hidden_size;
            let po = p + 2 * hidden_size;
            let pi_vec_ptr = pi.cast::<F32Vec>();
            let pf_vec_ptr = pf.cast::<F32Vec>();
            let po_vec_ptr = po.cast::<F32Vec>();
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
                for h in num_vec * F32Vec::SIZE..hidden_size as usize {
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
                for h in num_vec * F32Vec::SIZE..hidden_size as usize {
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

fn lstm_step(
    x: &Tensor<f32>,                // [seq_len, batch_size, input_size]
    w_t: &Tensor<f32>,              // [input_size, 4 * hidden_size]
    r_t: &Tensor<f32>,              // [hidden_size, 4 * hidden_size]
    r: PrePackedRhs,                // [4 * hidden_size, hidden_size]
    b: Option<&Tensor<f32>>,        // [8 * hidden_size]
    p: Option<&Tensor<f32>>,        // [3*hidden_size]
    initial_h: &Tensor<f32>,        // [batch_size, hidden_size]
    initial_c: &Tensor<f32>,        // [batch_size, hidden_size],
    seq_lens: Option<&Tensor<i64>>, // [batch_size]
    direction: Direction,
    hidden_size: i64,
    num_threads: usize,
    activate1: impl Fn(f32) -> f32 + Send + Sync + Copy,
    activate1_vec: impl Fn(F32Vec) -> F32Vec + Send + Sync + Copy,
    activate2: impl Fn(f32) -> f32 + Send + Sync + Copy,
    activate2_vec: impl Fn(F32Vec) -> F32Vec + Send + Sync + Copy,
) -> Result<(Tensor<f32>, Tensor<f32>, Tensor<f32>), TensorError>
where
    f32: FloatOutUnary<Output = f32>,
    F32Vec: FloatOutUnary<Output = F32Vec>,
{
    let seq_len = x.shape()[0];
    let batch_size = x.shape()[1];

    let seq_ptr = if let Some(seq_lens) = seq_lens {
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

    let y = Tensor::empty(&[seq_len, batch_size, hidden_size])?;

    let x_reshaped = x.reshape(&[seq_len * batch_size, x.shape()[2]])?;

    let tmp = if let Some(sum_bias) = &sum_bias {
        x_reshaped._addmm(w_t, sum_bias, None, num_threads, None)?
    } else {
        x_reshaped.matmul(w_t)?
    };

    let gates = Tensor::<f32>::empty(&[batch_size, 4 * hidden_size])?;

    let batch_size = x.shape()[1];

    let y_sq_stride = y.strides()[0] as i64;
    let y_b_stride = y.strides()[1] as i64;
    let c_b_stride = c_t.strides()[0] as i64;
    let gates_b_stride = gates.strides()[0] as i64;

    let mut func = |t: i64| -> Result<(), TensorError> {
        let gates = h_t._addmm(r_t, &tmp, Some(gates.clone()), num_threads, Some(r.clone()))?;

        let sliced_y = y.slice(&select![t:t + 1, ..])?.squeeze(&[0])?;

        for b in 0..batch_size {
            lstm_post_process(
                y.ptr::<f32>(),
                c_t.ptr::<f32>(),
                gates.ptr::<f32>(),
                p.as_ref().map(|p| p.ptr::<f32>()),
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

fn lstm_cell_ref(
    x: &Tensor<f32>,                // [seq_length, batch_size, input_size]
    w: &Tensor<f32>,                // [num_directions, 4 * hidden_size, input_size]
    r: &Tensor<f32>,                // [num_directions, 4 * hidden_size, hidden_size]
    b: Option<&Tensor<f32>>,        // [num_directions, 8 * hidden_size]
    seq_lens: Option<&Tensor<i64>>, // [batch_size]
    init_h: Option<&Tensor<f32>>,   // [num_directions, batch_size, hidden_size]
    init_c: Option<&Tensor<f32>>,   // [num_directions, batch_size, hidden_size]
    p: Option<&Tensor<f32>>,        // [num_directions, 3*hidden_size],
    direction: Direction,
    f: impl Fn(f32) -> f32 + Send + Sync + Copy,
    f_vec: impl Fn(F32Vec) -> F32Vec + Send + Sync + Copy,
    g: impl Fn(f32) -> f32 + Send + Sync + Copy,
    g_vec: impl Fn(F32Vec) -> F32Vec + Send + Sync + Copy,
) -> Result<(Tensor<f32>, Tensor<f32>, Tensor<f32>), TensorError> {
    let seq_length = x.shape()[0];
    let batch_size = x.shape()[1];

    let num_directions = w.shape()[0];
    let hidden_size = r.shape()[2];

    assert_eq!(num_directions, direction.num_directions() as i64);

    let y = Tensor::empty(&[seq_length, num_directions, batch_size, hidden_size])?;
    let y_h = if let Some(h) = init_h {
        h.clone()
    } else {
        Tensor::<f32>::zeros(&[num_directions, batch_size, hidden_size])?
    };

    let y_c = if let Some(c) = init_c {
        c.clone()
    } else {
        Tensor::<f32>::zeros(&[num_directions, batch_size, hidden_size])?
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
        let num_threads = current_num_threads();
        let r_ptr = r.ptr::<f32>();
        let prepacked_r = hpt_matmul::matmul_prepack_rhs(
            r_ptr.ptr,
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
        y_local.copy_from(&out)?;
        let mut y_h_local = y_h.slice(&select![dir:dir + 1, ..])?.squeeze(&[0])?;
        y_h_local.copy_from(&h_t)?;
        let mut y_c_local = y_c.slice(&select![dir:dir + 1, ..])?.squeeze(&[0])?;
        y_c_local.copy_from(&c_t)?;
    }
    Ok((y, y_h, y_c))
}

impl Tensor<f32> {
    ///
    pub fn lstm(
        &self,
        w: &Tensor<f32>,
        r: &Tensor<f32>,
        b: Option<&Tensor<f32>>,
        seq_lens: Option<&Tensor<i64>>,
        init_h: Option<&Tensor<f32>>,
        init_c: Option<&Tensor<f32>>,
        p: Option<&Tensor<f32>>,
        direction: &str,
    ) -> Result<(Tensor<f32>, Tensor<f32>, Tensor<f32>), TensorError> {
        let direction = Direction::from_str(direction);
        lstm_cell_ref(
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
        )
    }

    #[track_caller]
    fn _addmm(
        &self,
        rhs: &Tensor<f32>,
        bias: &Tensor<f32>,
        out: Option<Tensor<f32>>,
        num_threads: usize,
        prepacked_rhs: Option<PrePackedRhs>,
    ) -> Result<Tensor<f32>, TensorError> {
        let bias_strides = bias.strides();
        if bias.ndim() > 2 {
            panic!("bias must be a 2D tensor");
        }
        let bias_cs = bias_strides[bias_strides.len() - 1];
        let bias_rs = if bias.ndim() == 1 {
            0i64
        } else {
            bias_strides[bias_strides.len() - 2]
        };
        let bias_ptr = bias.ptr::<f32>();
        let c = crate::backends::cpu::kernels::matmul::common::matmul_prepare::<
            f32,
            0,
            hpt_allocator::HptAllocator<Cpu>,
        >(
            &self.inner,
            &rhs.inner,
            out.map(|t| t.inner.as_ref().clone()),
        )?;
        let m = self.shape()[0] as usize;
        let n = rhs.shape()[1] as usize;
        let k = self.shape()[1] as usize;

        hpt_matmul::matmul_with_post::<f32, _, _>(
            self.ptr().ptr as *const f32,
            rhs.ptr().ptr as *const f32,
            c.ptr().ptr as *mut f32,
            m,
            n,
            k,
            self.strides()[self.ndim() - 2],
            rhs.strides()[rhs.ndim() - 2],
            c.strides()[c.ndim() - 2] as i64,
            self.strides()[self.ndim() - 1],
            rhs.strides()[rhs.ndim() - 1],
            num_threads,
            prepacked_rhs,
            move |inp, m, n| bias_ptr[(m as i64) * bias_rs + (n as i64) * bias_cs]._add(inp),
            move |inp, m, n| unsafe {
                let inp = std::mem::transmute(inp);
                let offset = (m as i64) * bias_rs + (n as i64) * bias_cs;
                std::mem::transmute((bias_ptr + offset).cast::<F32Vec>().read_unaligned() + inp)
            },
        );

        Ok(c.into())
    }

    fn copy_from(&mut self, other: &Tensor<f32>) -> Result<(), TensorError> {
        self.par_iter_mut_simd()
            .zip(other.par_iter_simd())
            .for_each(
                |(a, b)| {
                    *a = b;
                },
                |(a, b)| {
                    a.write_unaligned(b.read_unaligned());
                },
            );
        Ok(())
    }
}
