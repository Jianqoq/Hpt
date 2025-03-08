#![allow(unused)]
use std::collections::HashMap;

use hpt::iter::TensorIterator;
use hpt::ops::IndexReduce;
use hpt::ops::Matmul;
use hpt::ops::NormalUaryOps;
use hpt::ops::NormalizationOps;
use hpt::types::vectors::traits::VecTrait;
use hpt::{
    buitin_templates::cpu::binary_with_out,
    common::TensorInfo,
    error::TensorError,
    iter::ParStridedIteratorZip,
    ops::{Concat, NormalBinOps, Random, RandomInt, ShapeManipulate, Slice, TensorCreator},
    types::{
        math::{Eval, NormalOut},
        TypeCommon,
    },
    utils::select,
    Tensor,
};
use rayon::iter::ParallelIterator;

type F32Vec = <f32 as TypeCommon>::Vec;

struct Encoder {
    mha: MultiHeadAttention,
    layernorm: LayerNorm,
    feedforward: FeedForward,
    layernorm2: LayerNorm,
}

impl Encoder {
    pub fn new(embedding_dim: i64, hidden_size: i64, num_head: i64) -> Result<Self, TensorError> {
        Ok(Encoder {
            mha: MultiHeadAttention::new(embedding_dim, num_head, 0.0)?,
            layernorm: LayerNorm::new(&[embedding_dim], 1e-5)?,
            feedforward: FeedForward::new(embedding_dim, embedding_dim, hidden_size)?,
            layernorm2: LayerNorm::new(&[embedding_dim], 1e-5)?,
        })
    }
    pub fn forward(&self, word_vec: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        let x = self.mha.forward(&word_vec, &word_vec, &word_vec, None)?;
        let x = x.add_(word_vec, x.clone())?;
        let o = self.layernorm.forward(x)?;
        let ff_res = self.feedforward.forward(&o)?;
        let ff_res = ff_res.add_(o, ff_res.clone())?;
        self.layernorm2.forward(ff_res)
    }
}

struct Decoder {
    masked_mha: MultiHeadAttention,
    layernorm: LayerNorm,
    mha: MultiHeadAttention,
    layernorm2: LayerNorm,
    ff: FeedForward,
    layernorm3: LayerNorm,
    num_heads: i64,
}

impl Decoder {
    pub fn new(embedding_dim: i64, hidden_size: i64, num_head: i64) -> Result<Self, TensorError> {
        Ok(Decoder {
            masked_mha: MultiHeadAttention::new(embedding_dim, num_head, 0.0)?,
            layernorm: LayerNorm::new(&[embedding_dim], 1e-5)?,
            mha: MultiHeadAttention::new(embedding_dim, num_head, 0.0)?,
            layernorm2: LayerNorm::new(&[embedding_dim], 1e-5)?,
            ff: FeedForward::new(embedding_dim, embedding_dim, hidden_size)?,
            layernorm3: LayerNorm::new(&[embedding_dim], 1e-5)?,
            num_heads: num_head,
        })
    }

    pub fn forward(
        &self,
        seq_len: i64,
        word_vec: &Tensor<f32>,
        o2: &Tensor<f32>,
    ) -> Result<Tensor<f32>, TensorError> {
        let mask = Tensor::<f32>::tri(seq_len as usize, seq_len as usize, 0, false)?;
        let masked = self
            .masked_mha
            .forward(&word_vec, &word_vec, &word_vec, Some(&mask))?;
        let o = self.layernorm.forward(&masked + word_vec)?;
        let x = self.mha.forward(&o, &o2, &o2, None)?;
        let o2 = self.layernorm2.forward(x + o)?;
        let ff_res = self.ff.forward(&o2)?;
        self.layernorm3.forward(ff_res + o2)
    }
}

struct Embedding {
    weight: Tensor<f32>,
}

impl Embedding {
    fn new(num_embeddings: i64, embedding_dim: i64) -> Result<Self, TensorError> {
        let weight = Tensor::<f32>::randn([num_embeddings, embedding_dim])?;
        Ok(Self { weight })
    }

    fn forward(&self, input: &Tensor<i64>) -> Result<Tensor<f32>, TensorError> {
        let input_shape = input.shape();
        let embedding_dim = self.weight.shape()[1];

        let flat_input = input.flatten(None, None)?;
        let indices = flat_input.ptr();

        let mut output = Vec::with_capacity(flat_input.size() as usize);

        for idx in 0..flat_input.size() {
            let idx = indices[idx];
            let embedding = self
                .weight
                .slice(&[(idx, idx + 1, 1), (0, 0x7FFFFFFFFFFFFFFF, 1)])?;
            output.push(embedding);
        }

        let mut new_shape = input_shape.to_vec();
        new_shape.push(embedding_dim);

        Tensor::concat(output, 0, false)?.reshape(new_shape)
    }
}

struct PositionalEncoding;

impl PositionalEncoding {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        let shape = input.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let embedding_dim = shape[2];
        let i_mat = binary_with_out(
            &Tensor::<f32>::new([10000.0]),
            &Tensor::<f32>::arange_step(0.0, embedding_dim as f32, 2.0)?,
            |a, b| a._pow(b) / embedding_dim as f32,
            |a, b| a._pow(b) / F32Vec::splat(embedding_dim as f32),
            None::<Tensor<f32>>,
        )?;
        let pos = Tensor::<f32>::arange(0.0, seq_len as f32)?.unsqueeze(1)?;
        let pe = Tensor::<f32>::zeros([seq_len, embedding_dim])?;
        let mut pe_sin = pe.slice(&[(0, 0x7FFFFFFFFFFFFFFF, 1), (0, 2, 1)])?;
        let mut pe_cos = pe.slice(&[(0, 0x7FFFFFFFFFFFFFFF, 1), (1, 2, 1)])?;
        pe_sin
            .par_iter_mut()
            .zip(pos.par_iter())
            .zip(i_mat.par_iter())
            .for_each(|((res, pos), i_mat)| {
                *res = (pos / i_mat).sin();
            });
        pe_cos
            .par_iter_mut()
            .zip(pos.par_iter())
            .zip(i_mat.par_iter())
            .for_each(|((res, pos), i_mat)| {
                *res = (pos / i_mat).cos();
            });
        let expanded = pe
            .unsqueeze(0)?
            .expand([batch_size, seq_len, embedding_dim])?;
        Ok(expanded
            .par_iter()
            .zip(input.par_iter())
            .strided_map(|(res, (pe, input))| {
                *res = pe + input;
            })
            .collect::<Tensor<f32>>())
    }
}

struct MultiHeadAttention {
    embedding_dim: i64,
    num_heads: i64,
    dropout: f32,
    head_dim: i64,
    q_linear: Linear,
    k_linear: Linear,
    v_linear: Linear,
    out_linear: Linear,
    dropout_p: f32,
}

impl MultiHeadAttention {
    fn new(embedding_dim: i64, num_heads: i64, dropout: f32) -> Result<Self, TensorError> {
        let head_dim = embedding_dim / num_heads;
        Ok(Self {
            embedding_dim,
            num_heads,
            dropout,
            head_dim,
            q_linear: Linear::new(embedding_dim, embedding_dim)?,
            k_linear: Linear::new(embedding_dim, embedding_dim)?,
            v_linear: Linear::new(embedding_dim, embedding_dim)?,
            out_linear: Linear::new(embedding_dim, embedding_dim)?,
            dropout_p: dropout,
        })
    }

    fn forward(
        &self,
        query: &Tensor<f32>,
        key: &Tensor<f32>,
        value: &Tensor<f32>,
        attn_mask: Option<&Tensor<f32>>,
    ) -> Result<Tensor<f32>, TensorError> {
        let batch_size = query.shape()[0];
        let seq_len = query.shape()[1];
        let embedding_dim = query.shape()[2];
        let seq_len_k = key.shape()[1];
        let seq_len_v = value.shape()[1];

        let scaling = Tensor::<f32>::new([1.0 / (self.head_dim as f32).sqrt()]);

        let q = self.q_linear.forward(query)?;
        let k = self.k_linear.forward(key)?;
        let v = self.v_linear.forward(value)?;

        let q = q
            .reshape([batch_size, seq_len, self.num_heads, self.head_dim])?
            .swap_axes(1, 2)?;
        let k = k
            .reshape([batch_size, seq_len_k, self.num_heads, self.head_dim])?
            .swap_axes(1, 2)?;
        let v = v
            .reshape([batch_size, seq_len_v, self.num_heads, self.head_dim])?
            .swap_axes(1, 2)?;

        let r = q.matmul(k.t()?)?;
        let mut scores = r.mul_(scaling, r.clone())?;

        if let Some(attn_mask) = attn_mask {
            scores
                .par_iter_mut()
                .zip(attn_mask.par_iter())
                .for_each(|(res, mask)| {
                    if mask._is_true() {
                        *res = f32::NEG_INFINITY;
                    }
                });
        }

        let attn_weights = scores.softmax(-1)?;

        let output = attn_weights.matmul(v)?;

        let output = output
            .swap_axes(1, 2)?
            .reshape([batch_size, seq_len, embedding_dim])?;

        let output = self.out_linear.forward(&output)?;

        Ok(output)
    }
}

pub struct Linear {
    weight: Tensor<f32>,

    bias: Tensor<f32>,
}

impl Linear {
    pub fn new(in_features: i64, out_features: i64) -> Result<Self, TensorError> {
        Ok(Self {
            weight: Tensor::<f32>::randn([in_features, out_features])?,
            bias: Tensor::<f32>::zeros([out_features])?,
        })
    }
    pub fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        let out = x.matmul(&self.weight)?;
        Ok(out)
    }
}

pub struct LayerNorm {
    gamma: Tensor<f32>,
    beta: Tensor<f32>,
    eps: f32,
    normalized_shape: Vec<i64>,
}

impl LayerNorm {
    pub fn new(normalized_shape: &[i64], eps: f32) -> Result<Self, TensorError> {
        Ok(LayerNorm {
            gamma: Tensor::<f32>::empty(normalized_shape)?,
            beta: Tensor::<f32>::empty(normalized_shape)?,
            eps,
            normalized_shape: normalized_shape.to_vec(),
        })
    }

    pub fn forward(&self, x: Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        x.layernorm(
            &self.normalized_shape,
            Some(&self.gamma),
            Some(&self.beta),
            self.eps,
        )
    }
}

pub struct Relu;

impl Relu {
    pub fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        x.relu_(&mut x.clone())
    }
}

pub struct FeedForward {
    linear1: Linear,
    linear2: Linear,
    relu: Relu,
}

impl FeedForward {
    pub fn new(
        in_feature: i64,
        out_feature: i64,
        hidden_feature: i64,
    ) -> Result<Self, TensorError> {
        Ok(FeedForward {
            linear1: Linear::new(in_feature, hidden_feature)?,
            linear2: Linear::new(hidden_feature, out_feature)?,
            relu: Relu,
        })
    }

    pub fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        let x = self.linear1.forward(x)?;
        let x = self.linear2.forward(&x)?;
        self.relu.forward(&x)
    }
}

struct Transformer {
    word_embedding: Embedding,
    encoders: Vec<Encoder>,
    word_embedding2: Embedding,
    decoder: Vec<Decoder>,
    linear: Linear,
}

impl Transformer {
    pub fn new(
        corpus: i64,
        embedding_dim: i64,
        hidden_size: i64,
        num_head: i64,
        num_layers: usize,
    ) -> Result<Self, TensorError> {
        Ok(Transformer {
            word_embedding: Embedding::new(corpus, embedding_dim)?,
            encoders: {
                let mut ret = Vec::with_capacity(num_layers);
                for _ in 0..num_layers {
                    ret.push(Encoder::new(embedding_dim, hidden_size, num_head)?);
                }
                ret
            },
            word_embedding2: Embedding::new(corpus, embedding_dim)?,
            decoder: {
                let mut ret = Vec::with_capacity(num_layers);
                for _ in 0..num_layers {
                    ret.push(Decoder::new(embedding_dim, hidden_size, num_head)?);
                }
                ret
            },
            linear: Linear::new(embedding_dim, corpus)?,
        })
    }

    pub fn generate(
        &self,
        question: &Tensor<i64>,
        word_id: &HashMap<String, i64>,
    ) -> Result<Tensor<i64>, TensorError> {
        let word_vec = self.word_embedding.forward(&question)?;
        let mut encoder_output = PositionalEncoding.forward(&word_vec)?;
        for _ in 0..10 {
            for i in &self.encoders {
                encoder_output = i.forward(&encoder_output)?;
            }
        }
        let mut outputs = Vec::new();
        let start_token = Tensor::<i64>::new([[*word_id.get("_").unwrap()]]);
        for _ in 0..10 {
            let x = if outputs.is_empty() {
                start_token.clone()
            } else {
                let mut tensors = vec![start_token.clone()];
                tensors.extend_from_slice(&outputs);
                Tensor::<i64>::concat(tensors, 1, false)?
            };
            let word_vec = self.word_embedding2.forward(&x)?;
            let mut decoder_output = PositionalEncoding.forward(&word_vec)?;
            for i in &self.decoder {
                decoder_output =
                    i.forward(*x.shape().last().unwrap(), &decoder_output, &encoder_output)?;
            }
            let score = self.linear.forward(&decoder_output)?;
            let sliced = score.slice(&select!(:, -1:, :))?;
            let prob = sliced.softmax(-1)?;
            let next_token = prob.argmax(-1, false)?;
            outputs.push(next_token);
        }
        Tensor::<i64>::concat(outputs, 1, false)
    }
}

fn main() -> anyhow::Result<()> {
    let mh = Transformer::new(14, 2048, 516, 8, 2)?;
    let dummy_questions = Tensor::<i64>::randint(0, 14, [1, 1024])?;
    let mut word_id = HashMap::new();
    word_id.insert("_".to_string(), 5);
    let now = std::time::Instant::now();
    let output = mh.generate(&dummy_questions, &word_id)?;
    println!("Time: {:?}", now.elapsed());

    Ok(())
}
