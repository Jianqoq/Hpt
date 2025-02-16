use candle_core::{IndexOp, Tensor};

struct LSTM {
    w_ii: Tensor,
    w_hi: Tensor,
    b_i: Tensor,

    w_if: Tensor,
    w_hf: Tensor,
    b_f: Tensor,

    w_ig: Tensor,
    w_hg: Tensor,
    b_g: Tensor,

    w_io: Tensor,
    w_ho: Tensor,
    b_o: Tensor,
}

impl LSTM {
    fn new(input_size: usize, hidden_size: usize) -> anyhow::Result<Self> {
        Ok(Self {
            w_ii: Tensor::randn(0f32, 1f32,&[hidden_size, input_size], &candle_core::Device::Cpu)?,
            w_hi: Tensor::randn(0f32, 1f32, &[hidden_size, hidden_size], &candle_core::Device::Cpu)?,
            b_i: Tensor::zeros(&[hidden_size], candle_core::DType::F32, &candle_core::Device::Cpu)?,

            w_if: Tensor::randn(0f32, 1f32, &[hidden_size, input_size], &candle_core::Device::Cpu)?,
            w_hf: Tensor::randn(0f32, 1f32, &[hidden_size, hidden_size], &candle_core::Device::Cpu)?,
            b_f: Tensor::zeros(&[hidden_size], candle_core::DType::F32, &candle_core::Device::Cpu)?,

            w_ig: Tensor::randn(0f32, 1f32, &[hidden_size, input_size], &candle_core::Device::Cpu)?,
            w_hg: Tensor::randn(0f32, 1f32, &[hidden_size, hidden_size], &candle_core::Device::Cpu)?,
            b_g: Tensor::zeros(&[hidden_size], candle_core::DType::F32, &candle_core::Device::Cpu)?,

            w_io: Tensor::randn(0f32, 1f32, &[hidden_size, input_size], &candle_core::Device::Cpu)?,
            w_ho: Tensor::randn(0f32, 1f32, &[hidden_size, hidden_size], &candle_core::Device::Cpu)?,
            b_o: Tensor::zeros(&[hidden_size], candle_core::DType::F32, &candle_core::Device::Cpu)?,
        })
    }
    fn forward(
        &self,
        x_t: &Tensor,
        h_t_1: &Tensor,
        c_t_1: &Tensor,
    ) -> anyhow::Result<(Tensor, Tensor)> {
        let x_t = x_t.contiguous()?;
        let i_t =
        candle_nn::ops::sigmoid(&((x_t.matmul(&self.w_ii.t()?)? + h_t_1.matmul(&self.w_hi.t()?)?)?.broadcast_add(&self.b_i))?)?;

        let f_t =
        candle_nn::ops::sigmoid(&((x_t.matmul(&self.w_if.t()?)? + h_t_1.matmul(&self.w_hf.t()?)?)?.broadcast_add(&self.b_f))?)?;

        let g_t =
            ((x_t.matmul(&self.w_ig.t()?)? + h_t_1.matmul(&self.w_hg.t()?)?)?.broadcast_add(&self.b_g))?.tanh()?;

        let o_t =
        candle_nn::ops::sigmoid(&((x_t.matmul(&self.w_io.t()?)? + h_t_1.matmul(&self.w_ho.t()?)?)?.broadcast_add(&self.b_o))?)?;

        
        let c_t = ((f_t * c_t_1.clone())? + (i_t * g_t)?)?;
        let h_t = o_t * c_t.tanh();

        Ok((h_t?, c_t))
    }
}

struct LSTMModel {
    hidden_size: usize,
    num_layers: usize,
    lstm_cells: Vec<LSTM>,
    output_layer: Option<LinearLayer>,
}

struct LinearLayer {
    weight: Tensor,
    bias: Tensor,
}

impl LinearLayer {
    fn new(in_features: usize, out_features: usize) -> anyhow::Result<Self> {
        let weight = Tensor::randn(0f32, 1f32, &[out_features, in_features], &candle_core::Device::Cpu)?;
        let bias = Tensor::zeros(&[out_features], candle_core::DType::F32, &candle_core::Device::Cpu)?;

        Ok(Self { weight, bias })
    }

    fn forward(&self, input: &Tensor) -> anyhow::Result<Tensor> {
        let output = input.matmul(&self.weight.t()?)?;
        Ok((output + self.bias.clone())?)
    }
}

impl LSTMModel {
    fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        output_size: Option<usize>,
    ) -> anyhow::Result<Self> {
        let mut lstm_cells = Vec::with_capacity(num_layers);

        lstm_cells.push(LSTM::new(input_size, hidden_size)?);

        for _ in 1..num_layers {
            lstm_cells.push(LSTM::new(hidden_size, hidden_size)?);
        }

        let output_layer = if let Some(out_size) = output_size {
            Some(LinearLayer::new(hidden_size, out_size)?)
        } else {
            None
        };
        Ok(Self {
            hidden_size,
            num_layers,
            lstm_cells,
            output_layer,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        init_states: Option<Vec<(Tensor, Tensor)>>,
    ) -> anyhow::Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        let batch_size = x.shape().dim(0)?;
        let seq_length = x.shape().dim(1)?;

        let mut states = if let Some(init) = init_states {
            init
        } else {
            let mut states = Vec::with_capacity(self.num_layers);
            for _ in 0..self.num_layers {
                states.push((
                    Tensor::zeros(&[batch_size, self.hidden_size], candle_core::DType::F32, &candle_core::Device::Cpu)?,
                    Tensor::zeros(&[batch_size, self.hidden_size], candle_core::DType::F32, &candle_core::Device::Cpu)?,
                ));
            }
            states
        };

        let mut outputs = Vec::with_capacity(seq_length as usize);

        let mut total_time = std::time::Duration::from_secs(0);
        for t in 0..seq_length {
            let mut layer_input = x.i((.., t..t+1, ..))?.squeeze(1)?;
            // let mut layer_input = x.slice(&match_selection![:, t:t+1, :])?.squeeze(1)?;
            for layer_idx in 0..self.num_layers {
                let lstm = &self.lstm_cells[layer_idx];
                let now = std::time::Instant::now();
                let (h_t, c_t) =
                    lstm.forward(&layer_input, &states[layer_idx].0, &states[layer_idx].1)?;
                println!("lstm Time taken: {:?}", now.elapsed());
                total_time += now.elapsed();
                states[layer_idx] = (h_t.clone(), c_t);

                layer_input = h_t;
            }
            // println!("next layer\n");
            let output_t = layer_input.unsqueeze(1)?;
            outputs.push(output_t);
        }
        println!("total time taken: {:?}", total_time);
        let output = Tensor::cat(&outputs, 1)?;
        // let output = Tensor::concat(outputs, 1, false)?;

        let final_output = if let Some(output_layer) = &self.output_layer {
            output_layer.forward(&output)?
        } else {
            output
        };

        Ok((final_output, states))
    }
}

fn main() -> anyhow::Result<()> {
    let model = LSTMModel::new(1024, 1024, 4, Some(20))?;

    let batch_size = 1024;
    let seq_length = 10;
    let input = Tensor::randn(0f32, 1f32, &[batch_size, seq_length, 1024], &candle_core::Device::Cpu)?;

    let start_time = std::time::Instant::now();
    for _ in 0..1 {
        let _ = model.forward(&input, None)?;
    }
    println!("Time taken: {:?}", start_time.elapsed() / 1);

    Ok(())
}
