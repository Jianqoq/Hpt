use crate::Tensor;
use half::{bf16, f16};
use hpt_common::error::base::TensorError;
use hpt_macros::select;
use hpt_traits::tensor::{CommonBounds, TensorInfo};
use hpt_types::dtype::DType;

fn lstm_step_ref(
    x: &Tensor,         // [seq_len, batch_size, input_size]
    w: &Tensor,         // [4 * hidden_size, input_size]
    r: &Tensor,         // [4 * hidden_size, hidden_size]
    b: Option<&Tensor>, // [8 * hidden_size]
    initial_h: &Tensor, // [batch_size, hidden_size]
    initial_c: &Tensor, // [batch_size, hidden_size]
) -> Result<(Tensor, Tensor, Tensor), TensorError> {
    let seq_len = x.shape()[0];

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

    for t in 0..seq_len {
        let x_t = x.slice(&select![t:t + 1, ..])?.squeeze(&[0])?;

        let gates = if b.is_some() {
            let wb = wb.as_ref().unwrap();
            let rb = rb.as_ref().unwrap();
            let res0 = x_t.matmul(&w.t()?)? + wb.clone();
            let res1 = h_t.matmul(&r.t()?)? + rb.clone();
            res0 + res1
        } else {
            x_t.matmul(&w.t()?)? + h_t.matmul(&r.t()?)?
        };

        let i = gates.slice(&select![:, 0:hidden_size])?.sigmoid()?;
        let f = gates
            .slice(&select![:, hidden_size:2*hidden_size])?
            .sigmoid()?;
        let g = gates.slice(&select![:, 2*hidden_size:3*hidden_size])?.tanh()?;
        let o = gates.slice(&select![:, 3*hidden_size:])?.sigmoid()?;

        c_t = c_t * f + i * g;
        h_t = o * c_t.tanh()?;

        y.push(h_t.clone());
    }

    let y = y.iter().map(|t| t).collect::<Vec<_>>();
    Ok((Tensor::concat(y, 0, true)?, h_t, c_t))
}

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
        let h0 = if let Some(h) = init_h {
            h.slice(&select![dir:dir + 1, ..])?.squeeze(&[0])?
        } else {
            Tensor::zeros(&[batch_size, hidden_size], x.dtype, x.device.clone())?
        };

        let c0 = if let Some(c) = init_c {
            c.slice(&select![dir:dir + 1, ..])?.squeeze(&[0])?
        } else {
            Tensor::zeros(&[batch_size, hidden_size], x.dtype, x.device.clone())?
        };

        let b = if let Some(b) = b {
            Some(b.slice(&select![dir:dir + 1, ..])?.squeeze(&[0])?)
        } else {
            None
        };
        let r = r.slice(&select![dir:dir + 1, ..])?.squeeze(&[0])?;
        let w = w.slice(&select![dir:dir + 1, ..])?.squeeze(&[0])?;

        let (out, h_t, c_t) = lstm_step_ref(x, &w, &r, b.as_ref(), &h0, &c0)?;

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
