use hpt::types::vectors::traits::VecTrait;
use hpt::{
    common::TensorInfo, error::TensorError, ops::*, types::math::FloatOutUnary, utils::select,
    Tensor,
};

struct LSTMCell {
    ih: Tensor<f32>,
    hh: Tensor<f32>,
    biases: Tensor<f32>,
    input_size: usize,
    hidden_size: usize,
}

impl LSTMCell {
    fn new(input_size: usize, hidden_size: usize) -> Result<Self, TensorError> {
        let biases = Tensor::<f32>::zeros(&[4 * hidden_size])?;
        let ih = Tensor::<f32>::randn(&[4 * hidden_size, input_size])?
            .t()?
            .contiguous()?;
        let hh = Tensor::<f32>::randn(&[4 * hidden_size, hidden_size])?
            .t()?
            .contiguous()?;
        Ok(Self {
            ih,
            hh,
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
        let (seq_len, batch_size, input_size) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let hidden_size = self.hidden_size as i64;
        let has_states = states.is_some();
        let (hs, mut c) = if let Some(states) = states {
            states
        } else {
            (
                Tensor::<f32>::zeros(&[seq_len, batch_size, hidden_size])?,
                Tensor::<f32>::zeros(&[batch_size, hidden_size])?,
            )
        };

        let flattened: Tensor<f32> = x.reshape(&[batch_size * seq_len, input_size])?;
        let now = std::time::Instant::now();
        let ih: Tensor<f32> =
            flattened
                .matmul(&self.ih)?
                .reshape(&[seq_len, batch_size, 4 * hidden_size])?; // [seq_len, batch_size, 4 * hidden_size]
        *total += now.elapsed();
        for i in 0..seq_len {
            let mut h = hs.slice(&select![i:i+1, :, :])?; // [batch_size, hidden_size]
            let ih_t: Tensor<f32> = ih.slice(&select![i:i+1, :, :])?.squeeze(0)?; // [batch_size, 4 * hidden_size]
            let now = std::time::Instant::now();
            let hh = if !has_states && i == 0 {
                ih_t
            } else {
                let h = if i == 0 {
                    h.clone()
                } else {
                    hs.slice(&select![i - 1:i, :, :])?.squeeze(0)?
                };
                h.addmm_post(
                    &self.hh,
                    &ih_t,
                    move |x, _, n| {
                        if n >= 2 * hidden_size as usize && n < 3 * hidden_size as usize {
                            x._tanh()
                        } else {
                            x._sigmoid()
                        }
                    },
                    move |x, _, n| {
                        if n >= 2 * hidden_size as usize && n < 3 * hidden_size as usize {
                            x._tanh()
                        } else {
                            x._sigmoid()
                        }
                    },
                )?
            }; // [batch_size, 4 * hidden_size]
            *total += now.elapsed();

            let i = hh.slice(&select![:, 0:hidden_size])?; // sigmoid， [batch_size, hidden_size]
            let f = hh.slice(&select![:, hidden_size:2*hidden_size])?; // sigmoid， [batch_size, hidden_size]
            let g = hh.slice(&select![:, 2*hidden_size:3*hidden_size])?; // tanh， [batch_size, hidden_size]
            let o = hh.slice(&select![:, 3*hidden_size:4*hidden_size])?; // sigmoid， [batch_size, hidden_size]

            // use hpt::iter::ParStridedIteratorSimd;
            // use hpt::iter::ParStridedIteratorSimdZip;
            // use hpt::iter::TensorIterator;
            // c.par_iter_mut_simd()
            //     .zip(i.par_iter_simd())
            //     .zip(f.par_iter_simd())
            //     .zip(g.par_iter_simd())
            //     .zip(h.par_iter_mut_simd())
            //     .zip(o.par_iter_simd())
            //     .for_each(
            //         |(((((c, i), f), g), h), o)| {
            //             let mul = i * g;
            //             let res = f.mul_add(*c, mul);
            //             let tanh = res._tanh();
            //             let o = o * tanh;
            //             *c = res;
            //             *h = o;
            //         },
            //         |(((((c, i), f), g), h), o)| {
            //             let mul = i * g;
            //             let res = f.mul_add(c.read_unaligned(), mul);
            //             let tanh = res._tanh();
            //             let o = o * tanh;
            //             c.write_unaligned(res);
            //             h.write_unaligned(o);
            //         },
            //     );

            // h.par_iter_mut_simd()
            //     .zip(c.par_iter_simd())
            //     .zip(o.par_iter_simd())
            //     .for_each(
            //         |((h, c), o)| {
            //             *h = o * c._tanh();
            //         },
            //         |((h, c), o)| {
            //             h.write_unaligned(o * c._tanh());
            //         },
            //     );
        }

        let last_h = hs.slice(&select![-1:,:,:])?;
        Ok((hs, last_h, c))
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
    ) -> Result<Self, TensorError> {
        let mut lstm_cells = Vec::with_capacity(num_layers);

        lstm_cells.push(LSTMCell::new(input_size, hidden_size)?);

        for _ in 1..num_layers {
            lstm_cells.push(LSTMCell::new(hidden_size, hidden_size)?);
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
    let model = LSTM2::new(512, 512, 4, Some(20))?;

    let mut times = Vec::new();
    let mut batch_sizes = Vec::new();
    for b in 4..=4 {
        let batch_size = b * 32;
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
