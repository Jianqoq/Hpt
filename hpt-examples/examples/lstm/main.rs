use hpt::types::vectors::traits::VecTrait;
use hpt::{
    common::TensorInfo, error::TensorError, ops::*, types::math::FloatOutUnary, utils::select,
    Tensor,
};

struct LSTMCell {
    w: Tensor<f32>,
    r: Tensor<f32>,
    biases: Tensor<f32>,
    input_size: usize,
    hidden_size: usize,
}

impl LSTMCell {
    fn new(input_size: usize, hidden_size: usize, direction: &str) -> Result<Self, TensorError> {
        let num_direction = if direction == "forward" {
            1
        } else if direction == "reverse" {
            1
        } else {
            2
        };
        let biases = Tensor::<f32>::zeros(&[num_direction, 8 * hidden_size])?;
        let w = Tensor::<f32>::randn(&[num_direction, 4 * hidden_size, input_size])?;
        let r = Tensor::<f32>::randn(&[num_direction, 4 * hidden_size, hidden_size])?;
        Ok(Self {
            w,
            r,
            biases,
            input_size,
            hidden_size,
        })
    }
    fn forward(
        &self,
        x: &Tensor<f32>,
        states: Option<(Tensor<f32>, Tensor<f32>)>,
        total: &mut std::time::Duration,
    ) -> Result<(Tensor<f32>, Tensor<f32>, Tensor<f32>), TensorError> {
        x.lstm(
            &self.w,
            &self.r,
            Some(&self.biases),
            None,
            None,
            None,
            None,
            "forward",
        )
    }
}

struct LSTM2 {
    num_layers: usize,
    lstm_cells: Vec<LSTMCell>,
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

impl LSTM2 {
    fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        output_size: Option<usize>,
        direction: &str,
    ) -> Result<Self, TensorError> {
        let mut lstm_cells = Vec::with_capacity(num_layers);

        lstm_cells.push(LSTMCell::new(input_size, hidden_size, direction)?);

        for _ in 1..num_layers {
            lstm_cells.push(LSTMCell::new(hidden_size, hidden_size, direction)?);
        }

        let output_layer = if let Some(out_size) = output_size {
            Some(LinearLayer::new(hidden_size, out_size)?)
        } else {
            None
        };
        Ok(Self {
            num_layers,
            lstm_cells,
            output_layer,
        })
    }

    fn forward(
        &self,
        x: &Tensor<f32>,
    ) -> Result<(Tensor<f32>, Vec<(Tensor<f32>, Tensor<f32>)>), TensorError> {
        let mut inp = x.clone();
        let mut new_states = Vec::with_capacity(self.num_layers);
        let mut total = std::time::Duration::from_secs(0);
        for layer_idx in 0..self.num_layers {
            let lstm = &self.lstm_cells[layer_idx];
            let (output, h, c) = lstm.forward(&inp, None, &mut total)?;
            new_states.push((h, c));
            inp = output;
        }
        // println!("total time: {:?}", total);
        let final_output = if let Some(output_layer) = &self.output_layer {
            output_layer.forward(&inp)?
        } else {
            inp
        };

        Ok((final_output, new_states))
    }
}

fn main() -> anyhow::Result<()> {
    let model = LSTM2::new(512, 512, 4, Some(20), "forward")?;

    let mut times = Vec::new();
    let mut batch_sizes = Vec::new();
    for b in 4..=4 {
        let batch_size = 1;
        let seq_length = 512;
        let input = Tensor::randn(&[seq_length, batch_size, 512])?;

        let start_time = std::time::Instant::now();
        for _ in 0..1 {
            let _ = model.forward(&input)?;
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
