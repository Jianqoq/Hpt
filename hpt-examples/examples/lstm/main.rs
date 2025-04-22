use hpt::ops::FloatUnaryOps;
use hpt::{
    common::TensorInfo,
    error::TensorError,
    iter::{ParStridedIteratorSimdZip, TensorIterator},
    ops::{Concat, Matmul, NormalBinOps, Random, ShapeManipulate, Slice, TensorCreator},
    types::math::FloatOutUnary,
    utils::select,
    Tensor,
};

struct LSTM {
    x_weights: Tensor<f32>,
    h_weights: Tensor<f32>,
    biases: Tensor<f32>,
}

impl LSTM {
    fn new(input_size: usize, hidden_size: usize) -> Result<Self, TensorError> {
        let biases = Tensor::concat(
            vec![
                Tensor::<f32>::zeros(&[hidden_size])?,
                Tensor::<f32>::zeros(&[hidden_size])?,
                Tensor::<f32>::zeros(&[hidden_size])?,
                Tensor::<f32>::zeros(&[hidden_size])?,
            ],
            0,
            false,
        )?;
        let x_weights = Tensor::concat(
            vec![
                Tensor::<f32>::randn(&[hidden_size, input_size])?,
                Tensor::<f32>::randn(&[hidden_size, input_size])?,
                Tensor::<f32>::randn(&[hidden_size, input_size])?,
                Tensor::<f32>::randn(&[hidden_size, input_size])?,
            ],
            0,
            false,
        )?;
        let h_weights = Tensor::concat(
            vec![
                Tensor::<f32>::randn(&[hidden_size, hidden_size])?,
                Tensor::<f32>::randn(&[hidden_size, hidden_size])?,
                Tensor::<f32>::randn(&[hidden_size, hidden_size])?,
                Tensor::<f32>::randn(&[hidden_size, hidden_size])?,
            ],
            0,
            false,
        )?;
        use hpt::ops::Contiguous;
        Ok(Self {
            x_weights: x_weights.t()?.contiguous()?,
            h_weights: h_weights.t()?.contiguous()?,
            biases,
        })
    }
    fn forward(
        &self,
        x_t: &Tensor<f32>,
        h_t_1: &Tensor<f32>,
        c_t_1: &Tensor<f32>,
    ) -> Result<(Tensor<f32>, Tensor<f32>), TensorError> {
        let x_proj: Tensor<f32> = x_t.matmul(&self.x_weights)?;
        let h_proj: Tensor<f32> = h_t_1.matmul(&self.h_weights)?;

        let gates: Tensor<f32> = &x_proj + &h_proj + &self.biases;

        let hidden_size = c_t_1.shape()[1] as i64;
        // let i_gate: Tensor<f32> = gates.slice(&select![:, 0:hidden_size])?;
        let f_gate: Tensor<f32> = gates.slice(&select![:, hidden_size:2*hidden_size])?;
        // let g_gate: Tensor<f32> = gates.slice(&select![:, 2*hidden_size:3*hidden_size])?;
        let o_gate: Tensor<f32> = gates.slice(&select![:, 3*hidden_size:4*hidden_size])?;
        // let i_t: Tensor<f32> = i_gate.sigmoid()?;
        let f_t: Tensor<f32> = f_gate.sigmoid()?;
        // let g_t: Tensor<f32> = g_gate.tanh()?;
        let o_t: Tensor<f32> = o_gate.sigmoid()?;

        // let c_t: Tensor<f32> = f_t * c_t_1 + i_t * g_t;
        // let h_t: Tensor<f32> = o_t * c_t.tanh()?;

        // Ok((h_t, c_t))
        Ok((o_t, f_t))
    }
}

struct LSTMModel {
    hidden_size: usize,
    num_layers: usize,
    lstm_cells: Vec<LSTM>,
    output_layer: Option<LinearLayer>,
}

struct LinearLayer {
    weight: Tensor<f32>,
    bias: Tensor<f32>,
}

impl LinearLayer {
    fn new(in_features: usize, out_features: usize) -> Result<Self, TensorError> {
        let weight = Tensor::<f32>::randn(&[out_features, in_features])?;
        let bias = Tensor::<f32>::zeros(&[out_features])?;

        Ok(Self { weight, bias })
    }

    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        let output = input.matmul(&self.weight.t()?)?;
        output.add_(&self.bias, output.clone())
    }
}

impl LSTMModel {
    fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        output_size: Option<usize>,
    ) -> Result<Self, TensorError> {
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
        x: &Tensor<f32>,
        init_states: Option<Vec<(Tensor<f32>, Tensor<f32>)>>,
    ) -> Result<(Tensor<f32>, Vec<(Tensor<f32>, Tensor<f32>)>), TensorError> {
        let batch_size = x.shape()[0];
        let seq_length = x.shape()[1];

        let mut states = if let Some(init) = init_states {
            init
        } else {
            let mut states = Vec::with_capacity(self.num_layers);
            for _ in 0..self.num_layers {
                states.push((
                    Tensor::<f32>::zeros(&[batch_size, self.hidden_size as i64])?,
                    Tensor::<f32>::zeros(&[batch_size, self.hidden_size as i64])?,
                ));
            }
            states
        };

        let mut outputs = Vec::with_capacity(seq_length as usize);

        let mut total = std::time::Duration::from_secs(0);
        for t in 0..seq_length {
            let mut layer_input = x.slice(&select![:, t:t+1, :])?.squeeze(1)?;
            let now = std::time::Instant::now();
            for layer_idx in 0..self.num_layers {
                let lstm = &self.lstm_cells[layer_idx];
                let (h_t, c_t) =
                    lstm.forward(&layer_input, &states[layer_idx].0, &states[layer_idx].1)?;
                states[layer_idx] = (h_t.clone(), c_t);

                layer_input = h_t;
            }
            total += now.elapsed();
            // println!("next layer\n");
            let output_t = layer_input.unsqueeze(1)?;
            outputs.push(output_t);
        }
        // println!("total time: {:?}", total);
        let output = Tensor::concat(outputs, 1, false)?;

        let final_output = if let Some(output_layer) = &self.output_layer {
            output_layer.forward(&output)?
        } else {
            output
        };

        Ok((final_output, states))
    }
}

fn main() -> anyhow::Result<()> {
    let model = LSTMModel::new(512, 512, 4, Some(20))?;

    let mut times = Vec::new();
    let mut batch_sizes = Vec::new();
    for b in 4..=4 {
        let batch_size = b * 32;
        let seq_length = 10;
        let input = Tensor::randn(&[batch_size, seq_length, 512])?;

        let start_time = std::time::Instant::now();
        for _ in 0..10 {
            let _ = model.forward(&input, None)?;
        }
        times.push((start_time.elapsed() / 10).as_secs_f32() * 1000.0);
        batch_sizes.push(batch_size);
        println!(
            "batch_size: {}, time: {}",
            batch_size,
            times.last().unwrap()
        );
    }
    println!("batch_sizes: {:?}", batch_sizes);
    println!("times: {:?}", times);
    Ok(())
}
