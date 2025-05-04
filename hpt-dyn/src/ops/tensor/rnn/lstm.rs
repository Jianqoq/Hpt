use crate::Tensor;
use half::{bf16, f16};
use hpt_common::error::base::TensorError;
use hpt_macros::select;
use hpt_traits::tensor::{CommonBounds, TensorInfo};
use hpt_types::dtype::DType;

fn lstm_cell_ref<T: CommonBounds>(
    x: &Tensor,                // [seq_length, batch_size, input_size]
    w: &Tensor,                // [num_directions, 4 * hidden_size, input_size]
    r: &Tensor,                // [num_directions, 4 * hidden_size, hidden_size]
    b: Option<&Tensor>,        // [num_directions, 8 * hidden_size]
    seq_lens: Option<&Tensor>, // [batch_size]
    init_h: Option<&Tensor>,   // [num_directions, batch_size, hidden_size]
    init_c: Option<&Tensor>,   // [num_directions, batch_size, hidden_size]
    p: Option<&Tensor>,        // [num_directions, 3*hidden_size]
) -> Result<(Tensor, Tensor, Tensor), TensorError> {
    let seq_length = x.shape()[0];
    let batch_size = x.shape()[1];
    let input_size = x.shape()[2];

    let num_directions = w.shape()[0];
    assert_eq!(w.shape()[1] % 4, 0);
    let hidden_size = w.shape()[1] / 4;

    let y = Tensor::empty(
        &[seq_length, num_directions, batch_size, hidden_size],
        x.dtype,
        x.device.clone(),
    )?;
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

    // 序列长度处理
    let seq_lengths = if let Some(lens) = seq_lens {
        assert_eq!(lens.dtype, DType::I64);
        lens.clone()
    } else {
        Tensor::full(&[1], seq_length as f64, DType::I64, x.device.clone())?
            .expand(&[batch_size])?
    };

    // 预计算所有时间步的X*W，类似ONNX的ComputeGemm操作
    // 将x从[seq_length, batch_size, input_size]重新形状为[seq_length*batch_size, input_size]
    let x_reshaped = x.reshape(&[seq_length * batch_size, input_size])?;

    // 对每个方向进行计算
    for dir in 0..num_directions {
        // 获取当前方向的权重
        let w_dir = w.slice(&select![dir:dir+1, :, :])?;
        let w_dir = w_dir.squeeze(&[0])?;

        // 获取循环权重
        let r_dir = r.slice(&select![dir:dir+1, :, :])?;
        let r_dir = r_dir.squeeze(&[0])?;

        // 处理偏置
        let (w_bias, r_bias) = if let Some(bias) = b {
            if bias.shape()[1] == 8 * hidden_size {
                // 分离输入偏置和循环偏置
                let wb = bias.slice(&select![dir:dir + 1, 0:4 * hidden_size])?;
                let rb = bias.slice(&select![dir:dir + 1, 4 * hidden_size:8 * hidden_size])?;
                (Some(wb.squeeze(&[0])?), Some(rb.squeeze(&[0])?))
            } else {
                // 共享偏置
                let b_dir = bias.slice(&select![dir:dir + 1, :])?;
                let b_dir = b_dir.squeeze(&[0])?;
                (Some(b_dir.clone()), Some(b_dir))
            }
        } else {
            (None, None)
        };

        let peep_weights = if let Some(peep) = p {
            let p_dir = peep.slice(&select![dir:dir+1, :])?;
            Some(p_dir.squeeze(&[0])?)
        } else {
            None
        };

        let xw_bias = if let Some(wb) = &w_bias {
            x_reshaped.addmm(&w_dir.t()?, &wb)?.reshape(&[
                seq_length,
                batch_size,
                4 * hidden_size,
            ])?
        } else {
            x_reshaped
                .matmul(&w_dir.t()?)?
                .reshape(&[seq_length, batch_size, 4 * hidden_size])?
        };
        let mut h_prev = y_h.slice(&select![dir:dir+1, :, :])?;
        let mut c_prev = y_c.slice(&select![dir:dir+1, :, :])?;
        h_prev = h_prev.squeeze(&[0])?;
        c_prev = c_prev.squeeze(&[0])?;

        // 按时间步计算
        for t in 0..seq_length {
            // 获取当前时间步的预计算输入
            let xt_w = xw_bias.slice(&select![t:t+1, :, :])?;
            let xt_w = xt_w.squeeze(&[0])?;

            // 应用循环偏置
            let ht_r_bias = if let Some(rb) = &r_bias {
                h_prev.addmm(&r_dir.t()?, &rb)?
            } else {
                h_prev.matmul(&r_dir.t()?)?
            };
            // 合并门控输入
            let gates = &xt_w + &ht_r_bias;

            // 分离各门
            let i_gate = gates.slice(&select![:, 0:hidden_size])?;
            let f_gate = gates.slice(&select![:, hidden_size:2 * hidden_size])?;
            let c_gate = gates.slice(&select![:, 2 * hidden_size:3 * hidden_size])?;
            let o_gate = gates.slice(&select![:, 3 * hidden_size:4 * hidden_size])?;

            let (i_peep, f_peep) = if let Some(p) = &peep_weights {
                let p_i = p.slice(&select![0:hidden_size])?;
                let p_f = p.slice(&select![hidden_size:2*hidden_size])?;
                (i_gate + &c_prev * &p_i, f_gate + &c_prev * &p_f)
            } else {
                (i_gate, f_gate)
            };

            // 应用激活函数 - 使用sigmoid和tanh
            let i = i_peep.sigmoid()?;
            let f = f_peep.sigmoid()?;
            let c = c_gate.tanh()?;

            // 更新细胞状态
            let c_next = &f * &c_prev + &i * &c;

            // 应用输出门的peephole
            let o_peep = if let Some(p) = &peep_weights {
                let p_o = p.slice(&select![2 * hidden_size:3 * hidden_size])?;
                o_gate + &c_next * &p_o
            } else {
                o_gate
            };

            let o = o_peep.sigmoid()?;

            // 计算新的隐藏状态
            let h_next = &o * &c_next.tanh()?;

            // 根据序列长度掩码应用更新
            // 在真实实现中，这可以通过张量操作批量完成以提高性能
            let h_next_masked = h_prev.clone();
            let c_next_masked = c_prev.clone();

            for b in 0..batch_size {
                let seq_len = seq_lengths.get::<i64>(&[b])?;
                if t < seq_len {
                    // 更新输出
                    let mut y_slice = y.slice(&select![t:t+1, dir:dir+1, b:b+1, :])?;
                    y_slice = y_slice.squeeze(&[0])?.squeeze(&[0])?.squeeze(&[0])?;
                    let h_next_b = h_next.slice(&select![b:b+1, :])?;
                    y_slice.copy_from(&h_next_b.squeeze(&[0])?);

                    // 更新隐藏状态和细胞状态
                    let mut h_next_slice = h_next_masked.slice(&select![b:b+1, ..])?;
                    let mut c_next_slice = c_next_masked.slice(&select![b:b+1, ..])?;
                    h_next_slice = h_next_slice.squeeze(&[0])?;
                    c_next_slice = c_next_slice.squeeze(&[0])?;

                    h_next_slice.copy_from(&h_next.slice(&select![b:b+1, ..])?);
                    c_next_slice.copy_from(&c_next.slice(&select![b:b+1, ..])?);
                }
            }

            h_prev = h_next_masked;
            c_prev = c_next_masked;
        }
        let mut y_h_dir = y_h.slice(&select![dir:dir+1, :, :])?;
        let mut y_c_dir = y_c.slice(&select![dir:dir+1, :, :])?;
        y_h_dir = y_h_dir.squeeze(&[0])?;
        y_c_dir = y_c_dir.squeeze(&[0])?;

        y_h.copy_from(&y_h_dir);
        y_c.copy_from(&y_c_dir);
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
    ) -> Result<(Tensor, Tensor, Tensor), TensorError> {
        let (y, y_h, y_c) = match self.dtype {
            DType::I8 => lstm_cell_ref::<i8>(self, w, r, b, seq_lens, init_h, init_c, p)?,
            DType::U8 => lstm_cell_ref::<u8>(self, w, r, b, seq_lens, init_h, init_c, p)?,
            DType::F32 => lstm_cell_ref::<f32>(self, w, r, b, seq_lens, init_h, init_c, p)?,
            DType::F16 => lstm_cell_ref::<f16>(self, w, r, b, seq_lens, init_h, init_c, p)?,
            DType::BF16 => lstm_cell_ref::<bf16>(self, w, r, b, seq_lens, init_h, init_c, p)?,
            _ => unimplemented!(),
        };
        Ok((y, y_h, y_c))
    }
}
