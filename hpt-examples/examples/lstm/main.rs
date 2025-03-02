use hpt::{
    select, FloatOutUnary, Matmul, NormalBinOps, ParStridedIteratorSimdZip, Random,
    ShapeManipulate, Slice, Tensor, TensorCreator, TensorError, TensorInfo, TensorIterator,
};

struct LSTM {
    w_ii: Tensor<f32>,
    w_hi: Tensor<f32>,
    b_i: Tensor<f32>,

    w_if: Tensor<f32>,
    w_hf: Tensor<f32>,
    b_f: Tensor<f32>,

    w_ig: Tensor<f32>,
    w_hg: Tensor<f32>,
    b_g: Tensor<f32>,

    w_io: Tensor<f32>,
    w_ho: Tensor<f32>,
    b_o: Tensor<f32>,
}

impl LSTM {
    fn new(input_size: usize, hidden_size: usize) -> Result<Self, TensorError> {
        Ok(Self {
            w_ii: Tensor::<f32>::randn(&[hidden_size, input_size])?,
            w_hi: Tensor::<f32>::randn(&[hidden_size, hidden_size])?,
            b_i: Tensor::<f32>::zeros(&[hidden_size])?,

            w_if: Tensor::<f32>::randn(&[hidden_size, input_size])?,
            w_hf: Tensor::<f32>::randn(&[hidden_size, hidden_size])?,
            b_f: Tensor::<f32>::zeros(&[hidden_size])?,

            w_ig: Tensor::<f32>::randn(&[hidden_size, input_size])?,
            w_hg: Tensor::<f32>::randn(&[hidden_size, hidden_size])?,
            b_g: Tensor::<f32>::zeros(&[hidden_size])?,

            w_io: Tensor::<f32>::randn(&[hidden_size, input_size])?,
            w_ho: Tensor::<f32>::randn(&[hidden_size, hidden_size])?,
            b_o: Tensor::<f32>::zeros(&[hidden_size])?,
        })
    }
    fn forward(
        &self,
        x_t: &Tensor<f32>,
        h_t_1: &Tensor<f32>,
        c_t_1: &Tensor<f32>,
    ) -> Result<(Tensor<f32>, Tensor<f32>), TensorError> {
        let i_t = x_t.matmul(self.w_ii.t()?)? + h_t_1.matmul(self.w_hi.t()?)? + &self.b_i;
        // i_t.sigmoid_(i_t.clone())?;

        let f_t = x_t.matmul(self.w_if.t()?)? + h_t_1.matmul(self.w_hf.t()?)? + &self.b_f;
        // f_t.sigmoid_(f_t.clone())?;

        let g_t = x_t.matmul(self.w_ig.t()?)? + h_t_1.matmul(self.w_hg.t()?)? + &self.b_g;
        // g_t.tanh_(g_t.clone())?;

        let o_t = x_t.matmul(self.w_io.t()?)? + h_t_1.matmul(self.w_ho.t()?)? + &self.b_o;
        // o_t.sigmoid_(o_t.clone())?;

        let c_t = f_t
            .par_iter_simd()
            .zip(c_t_1.par_iter_simd())
            .zip(i_t.par_iter_simd())
            .zip(g_t.par_iter_simd())
            .strided_map_simd(
                |(res, (((x, y), z), w))| {
                    *res = x * y + z * w;
                },
                |(res, (((x, y), z), w))| {
                    res.write_unaligned(x * y + z * w);
                },
            )
            .collect::<Tensor<f32>>();

        let h_t = o_t
            .par_iter_simd()
            .zip(c_t.par_iter_simd())
            .strided_map_simd(
                |(res, (x, y))| {
                    *res = x * y.tanh();
                },
                |(res, (x, y))| {
                    res.write_unaligned(x * y._tanh());
                },
            )
            .collect::<Tensor<f32>>();

        Ok((h_t, c_t))
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

        let mut total_time = std::time::Duration::from_secs(0);
        for t in 0..seq_length {
            let mut layer_input = x.slice(&select![:, t:t+1, :])?.squeeze(1)?;
            for layer_idx in 0..self.num_layers {
                let lstm = &self.lstm_cells[layer_idx];
                let now = std::time::Instant::now();
                let (h_t, c_t) =
                    lstm.forward(&layer_input, &states[layer_idx].0, &states[layer_idx].1)?;
                // println!("lstm Time taken: {:?}", now.elapsed());
                total_time += now.elapsed();
                states[layer_idx] = (h_t.clone(), c_t);

                layer_input = h_t;
            }
            // println!("next layer\n");
            let output_t = layer_input.unsqueeze(1)?;
            outputs.push(output_t);
        }
        // println!("total time taken: {:?}", total_time);
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
    for b in 1..=32 {
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
