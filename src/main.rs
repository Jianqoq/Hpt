use hpt_core::{
    FloatOutUnary, Matmul, ParStridedIteratorSimdZip, Random, ShapeManipulate, Tensor,
    TensorCreator, TensorError, TensorIterator,
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
        let x_w_if = x_t.matmul(self.w_if.t()?)?;
        let x_w_ig = x_t.matmul(self.w_ig.t()?)?;
        let x_w_io = x_t.matmul(self.w_io.t()?)?;
        let h_t_w_hi = h_t_1.matmul(self.w_hi.t()?)?;
        let h_t_w_hf = h_t_1.matmul(self.w_hf.t()?)?;
        let h_t_w_hg = h_t_1.matmul(self.w_hg.t()?)?;
        let h_t_w_ho = h_t_1.matmul(self.w_ho.t()?)?;
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
fn main() -> Result<(), TensorError> {
    let batch_size = 16;
    let input_size = 1024;
    let output_size = 1024;
    let a = Tensor::<f32>::randn([batch_size, input_size])?;
    let h = Tensor::<f32>::zeros([batch_size, input_size])?;
    let c = Tensor::<f32>::zeros([batch_size, input_size])?;
    let lstm = LSTM::new(input_size as usize, output_size as usize)?;

    let now = std::time::Instant::now();
    for _ in 0..100 {
        let _ = lstm.forward(&a, &h, &c)?;
    }
    println!("{:?}", now.elapsed() / 100);
    Ok(())
}
