use tensor_dyn::{
    match_selection, FloatOutUnary, Matmul, NormalBinOps, ParStridedIteratorSimdZip, Random,
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
            w_ii: Tensor::randn(&[hidden_size, input_size])?,
            w_hi: Tensor::randn(&[hidden_size, hidden_size])?,
            b_i: Tensor::zeros(&[hidden_size])?,

            w_if: Tensor::randn(&[hidden_size, input_size])?,
            w_hf: Tensor::randn(&[hidden_size, hidden_size])?,
            b_f: Tensor::zeros(&[hidden_size])?,

            w_ig: Tensor::randn(&[hidden_size, input_size])?,
            w_hg: Tensor::randn(&[hidden_size, hidden_size])?,
            b_g: Tensor::zeros(&[hidden_size])?,

            w_io: Tensor::randn(&[hidden_size, input_size])?,
            w_ho: Tensor::randn(&[hidden_size, hidden_size])?,
            b_o: Tensor::zeros(&[hidden_size])?,
        })
    }
    fn forward(
        &self,
        x_t: &Tensor<f32>,
        h_t_1: &Tensor<f32>,
        c_t_1: &Tensor<f32>,
    ) -> Result<(Tensor<f32>, Tensor<f32>), TensorError> {
        let x_w_ii = x_t.matmul(self.w_ii.t()?)?;
        let h_t_w_hi = h_t_1.matmul(self.w_hi.t()?)?;
        let i_t = x_w_ii
            .par_iter_simd()
            .zip(h_t_w_hi.par_iter_simd())
            .zip(self.b_i.par_iter_simd())
            .strided_map_simd(
                |(res, ((x, y), z))| {
                    *res = (x + y + z)._sigmoid();
                },
                |(res, ((x, y), z))| {
                    res.write_unaligned((x + y + z)._sigmoid());
                },
            )
            .collect::<Tensor<f32>>();

        let x_w_if = x_t.matmul(self.w_if.t()?)?;
        let h_t_w_hf = h_t_1.matmul(self.w_hf.t()?)?;
        let f_t = x_w_if
            .par_iter_simd()
            .zip(h_t_w_hf.par_iter_simd())
            .zip(self.b_f.par_iter_simd())
            .strided_map_simd(
                |(res, ((x, y), z))| {
                    *res = (x + y + z)._sigmoid();
                },
                |(res, ((x, y), z))| {
                    res.write_unaligned((x + y + z)._sigmoid());
                },
            )
            .collect::<Tensor<f32>>();

        let x_w_ig = x_t.matmul(self.w_ig.t()?)?;
        let h_t_w_hg = h_t_1.matmul(self.w_hg.t()?)?;
        let g_t = x_w_ig
            .par_iter_simd()
            .zip(h_t_w_hg.par_iter_simd())
            .zip(self.b_g.par_iter_simd())
            .strided_map_simd(
                |(res, ((x, y), z))| {
                    *res = (x + y + z)._tanh();
                },
                |(res, ((x, y), z))| {
                    res.write_unaligned((x + y + z)._tanh());
                },
            )
            .collect::<Tensor<f32>>();

        let x_w_io = x_t.matmul(self.w_io.t()?)?;
        let h_t_w_ho = h_t_1.matmul(self.w_ho.t()?)?;
        let o_t = x_w_io
            .par_iter_simd()
            .zip(h_t_w_ho.par_iter_simd())
            .zip(self.b_o.par_iter_simd())
            .strided_map_simd(
                |(res, ((x, y), z))| {
                    *res = (x + y + z)._sigmoid();
                },
                |(res, ((x, y), z))| {
                    res.write_unaligned((x + y + z)._sigmoid());
                },
            )
            .collect::<Tensor<f32>>();

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
    output_layer: Option<LinearLayer>, // 可选的输出层
}

struct LinearLayer {
    weight: Tensor<f32>,
    bias: Tensor<f32>,
}

impl LinearLayer {
    fn new(in_features: usize, out_features: usize) -> Result<Self, TensorError> {
        // 初始化权重和偏置
        let weight = Tensor::randn(&[out_features, in_features])?;
        let bias = Tensor::zeros(&[out_features])?;

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

        // 创建第一层
        lstm_cells.push(LSTM::new(input_size, hidden_size)?);

        // 创建其余层
        for _ in 1..num_layers {
            lstm_cells.push(LSTM::new(hidden_size, hidden_size)?);
        }

        // 如果指定了输出大小，创建输出层
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
                    Tensor::zeros(&[batch_size, self.hidden_size as i64])?,
                    Tensor::zeros(&[batch_size, self.hidden_size as i64])?,
                ));
            }
            states
        };

        let mut outputs = Vec::with_capacity(seq_length as usize);

        for t in 0..seq_length {
            let mut layer_input = x.slice(&match_selection![:, t:t+1, :])?.squeeze(1)?;
            for layer_idx in 0..self.num_layers {
                let lstm = &self.lstm_cells[layer_idx];

                let (h_t, c_t) =
                    lstm.forward(&layer_input, &states[layer_idx].0, &states[layer_idx].1)?;

                states[layer_idx] = (h_t.clone(), c_t);

                layer_input = h_t;
            }
            let output_t = layer_input.unsqueeze(1)?;
            outputs.push(output_t);
        }

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
    // 创建模型
    let model = LSTMModel::new(10, 20, 2, Some(5))?;

    // 创建示例输入
    let batch_size = 3;
    let seq_length = 4;
    let input = Tensor::randn(&[batch_size, seq_length, 10])?;

    // 前向传播
    let (output, states) = model.forward(&input, None)?;

    // 打印输出形状
    println!("Input shape: {:?}", input.shape());
    println!("Output shape: {:?}", output.shape());
    println!("Number of states: {}", states.len());

    Ok(())
}
