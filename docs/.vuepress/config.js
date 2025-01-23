import { defineUserConfig } from '@vuepress/cli'
import { defaultTheme } from '@vuepress/theme-default'
import { viteBundler } from '@vuepress/bundler-vite'
import { mdEnhancePlugin } from "vuepress-plugin-md-enhance"

export default defineUserConfig({
  base: process.env.NODE_ENV === 'development'
    ? '/'
    : '/eTensor/',
  lang: 'zh-CN',
  title: 'Hpt',
  plugins: [
    mdEnhancePlugin({
      katex: true,
    }),
  ],
  bundler: viteBundler(),
  theme: defaultTheme({
    home: false,
    navbar: [
      {
        text: 'Home',
        link: '/',
      },
      {
        text: 'GitHub',
        link: 'https://github.com/Jianqoq/eTensor',
      },
    ],
    sidebar: {
      '/user_guide/': [
        {
          text: 'Docs',
          children: [
            {
              text: 'unary',
              collapsible: true,
              children: [
                { text: 'sin', link: '/user_guide/unary/sin.md' },
                { text: 'sin_', link: '/user_guide/unary/sin_.md' },
                { text: 'cos', link: '/user_guide/unary/cos.md' },
                { text: 'cos_', link: '/user_guide/unary/cos_.md' },
                { text: 'tan', link: '/user_guide/unary/tan.md' },
                { text: 'tan_', link: '/user_guide/unary/tan_.md' },
                { text: 'sinh', link: '/user_guide/unary/sinh.md' },
                { text: 'sinh_', link: '/user_guide/unary/sinh_.md' },
                { text: 'cosh', link: '/user_guide/unary/cosh.md' },
                { text: 'cosh_', link: '/user_guide/unary/cosh_.md' },
                { text: 'tanh', link: '/user_guide/unary/tanh.md' },
                { text: 'tanh_', link: '/user_guide/unary/tanh_.md' },
                { text: 'asinh', link: '/user_guide/unary/asinh.md' },
                { text: 'asinh_', link: '/user_guide/unary/asinh_.md' },
                { text: 'acosh', link: '/user_guide/unary/acosh.md' },
                { text: 'acosh_', link: '/user_guide/unary/acosh_.md' },
                { text: 'atanh', link: '/user_guide/unary/atanh.md' },
                { text: 'atanh_', link: '/user_guide/unary/atanh_.md' },
                { text: 'asin', link: '/user_guide/unary/asin.md' },
                { text: 'asin_', link: '/user_guide/unary/asin_.md' },
                { text: 'acos', link: '/user_guide/unary/acos.md' },
                { text: 'acos_', link: '/user_guide/unary/acos_.md' },
                { text: 'atan', link: '/user_guide/unary/atan.md' },
                { text: 'atan_', link: '/user_guide/unary/atan_.md' },
                { text: 'exp', link: '/user_guide/unary/exp.md' },
                { text: 'exp_', link: '/user_guide/unary/exp_.md' },
                { text: 'exp2', link: '/user_guide/unary/exp2.md' },
                { text: 'exp2_', link: '/user_guide/unary/exp2_.md' },
                { text: 'sqrt', link: '/user_guide/unary/sqrt.md' },
                { text: 'sqrt_', link: '/user_guide/unary/sqrt_.md' },
                { text: 'recip', link: '/user_guide/unary/recip.md' },
                { text: 'recip_', link: '/user_guide/unary/recip_.md' },
                { text: 'ln', link: '/user_guide/unary/ln.md' },
                { text: 'ln_', link: '/user_guide/unary/ln_.md' },
                { text: 'log2', link: '/user_guide/unary/log2.md' },
                { text: 'log2_', link: '/user_guide/unary/log2_.md' },
                { text: 'log10', link: '/user_guide/unary/log10.md' },
                { text: 'log10_', link: '/user_guide/unary/log10_.md' },
                { text: 'celu', link: '/user_guide/unary/celu.md' },
                { text: 'celu_', link: '/user_guide/unary/celu_.md' },
                { text: 'sigmoid', link: '/user_guide/unary/sigmoid.md' },
                { text: 'sigmoid_', link: '/user_guide/unary/sigmoid_.md' },
                { text: 'elu', link: '/user_guide/unary/elu.md' },
                { text: 'elu_', link: '/user_guide/unary/elu_.md' },
                { text: 'erf', link: '/user_guide/unary/erf.md' },
                { text: 'fast_hard_sigmoid', link: '/user_guide/unary/fast_hard_sigmoid.md' },
                { text: 'gelu', link: '/user_guide/unary/gelu.md' },
                { text: 'gelu_', link: '/user_guide/unary/gelu_.md' },
                { text: 'selu', link: '/user_guide/unary/selu.md' },
                { text: 'selu_', link: '/user_guide/unary/selu_.md' },
                { text: 'hard_sigmoid', link: '/user_guide/unary/hard_sigmoid.md' },
                { text: 'hard_sigmoid_', link: '/user_guide/unary/hard_sigmoid_.md' },
                { text: 'hard_swish', link: '/user_guide/unary/hard_swish.md' },
                { text: 'hard_swish_', link: '/user_guide/unary/hard_swish_.md' },
                { text: 'softplus', link: '/user_guide/unary/softplus.md' },
                { text: 'softplus_', link: '/user_guide/unary/softplus_.md' },
                { text: 'softsign', link: '/user_guide/unary/softsign.md' },
                { text: 'softsign_', link: '/user_guide/unary/softsign_.md' },
                { text: 'mish', link: '/user_guide/unary/mish.md' },
                { text: 'mish_', link: '/user_guide/unary/mish_.md' },
                { text: 'cbrt', link: '/user_guide/unary/cbrt.md' },
                { text: 'cbrt_', link: '/user_guide/unary/cbrt_.md' },
              ]
            },
            {
              text: 'binary',
              collapsible: true,
              children: [
                { text: 'add', link: '/user_guide/binary/add.md' },
                { text: 'add_', link: '/user_guide/binary/add_.md' },
                { text: 'sub', link: '/user_guide/binary/sub.md' },
                { text: 'sub_', link: '/user_guide/binary/sub_.md' },
                { text: 'mul', link: '/user_guide/binary/mul.md' },
                { text: 'mul_', link: '/user_guide/binary/mul_.md' },
                { text: 'div', link: '/user_guide/binary/div.md' },
                { text: 'div_', link: '/user_guide/binary/div_.md' },
              ]
            },
            {
              text: 'reduce',
              collapsible: true,
              children: [
                { text: 'logsumexp', link: '/user_guide/reduce/logsumexp.md' },
                { text: 'argmin', link: '/user_guide/reduce/argmin.md' },
                { text: 'argmax', link: '/user_guide/reduce/argmax.md' },
                { text: 'max', link: '/user_guide/reduce/max.md' },
                { text: 'min', link: '/user_guide/reduce/min.md' },
                { text: 'mean', link: '/user_guide/reduce/mean.md' },
                { text: 'sum', link: '/user_guide/reduce/sum.md' },
                { text: 'sum_', link: '/user_guide/reduce/sum_.md' },
                { text: 'nansum', link: '/user_guide/reduce/nansum.md' },
                { text: 'nansum_', link: '/user_guide/reduce/nansum_.md' },
                { text: 'prod', link: '/user_guide/reduce/prod.md' },
                { text: 'nanprod', link: '/user_guide/reduce/nanprod.md' },
                { text: 'sum_square', link: '/user_guide/reduce/sum_square.md' },
                { text: 'reducel1', link: '/user_guide/reduce/reducel1.md' },
                { text: 'reducel2', link: '/user_guide/reduce/reducel2.md' },
                { text: 'reducel3', link: '/user_guide/reduce/reducel3.md' },
                { text: 'all', link: '/user_guide/reduce/all.md' },
                { text: 'any', link: '/user_guide/reduce/any.md' },
              ]
            },
            {
              text: 'compare',
              collapsible: true,
              children: [
                { text: 'tensor_neq', link: '/user_guide/cmp/tensor_neq.md' },
                { text: 'tensor_eq', link: '/user_guide/cmp/tensor_eq.md' },
                { text: 'tensor_gt', link: '/user_guide/cmp/tensor_gt.md' },
                { text: 'tensor_lt', link: '/user_guide/cmp/tensor_lt.md' },
                { text: 'tensor_ge', link: '/user_guide/cmp/tensor_ge.md' },
                { text: 'tensor_le', link: '/user_guide/cmp/tensor_le.md' },
              ]
            },
            {
              text: 'linalg',
              collapsible: true,
              children: [
                { text: 'matmul', link: '/user_guide/linalg/matmul.md' },
                { text: 'tensordot', link: '/user_guide/linalg/tensordot.md' },
              ]
            },
            {
              text: 'random',
              collapsible: true,
              children: [
                { text: 'randn', link: '/user_guide/random/randn.md' },
                { text: 'randn_like', link: '/user_guide/random/randn_like.md' },
                { text: 'rand', link: '/user_guide/random/rand.md' },
                { text: 'rand_like', link: '/user_guide/random/rand_like.md' },
                { text: 'beta', link: '/user_guide/random/beta.md' },
                { text: 'beta_like', link: '/user_guide/random/beta_like.md' },
                { text: 'chisquare', link: '/user_guide/random/chisquare.md' },
                { text: 'chisquare_like', link: '/user_guide/random/chisquare_like.md' },
                { text: 'exponential', link: '/user_guide/random/exponential.md' },
                { text: 'exponential_like', link: '/user_guide/random/exponential_like.md' },
                { text: 'gamma', link: '/user_guide/random/gamma.md' },
                { text: 'gamma_like', link: '/user_guide/random/gamma_like.md' },
                { text: 'gumbel', link: '/user_guide/random/gumbel.md' },
                { text: 'gumbel_like', link: '/user_guide/random/gumbel_like.md' },
                { text: 'lognormal', link: '/user_guide/random/lognormal.md' },
                { text: 'lognormal_like', link: '/user_guide/random/lognormal_like.md' },
                { text: 'normal_gaussian', link: '/user_guide/random/normal_gaussian.md' },
                { text: 'normal_gaussian_like', link: '/user_guide/random/normal_gaussian_like.md' },
                { text: 'pareto', link: '/user_guide/random/pareto.md' },
                { text: 'pareto_like', link: '/user_guide/random/pareto_like.md' },
                { text: 'poisson', link: '/user_guide/random/poisson.md' },
                { text: 'poisson_like', link: '/user_guide/random/poisson_like.md' },
                { text: 'weibull', link: '/user_guide/random/weibull.md' },
                { text: 'weibull_like', link: '/user_guide/random/weibull_like.md' },
                { text: 'zipf', link: '/user_guide/random/zipf.md' },
                { text: 'zipf_like', link: '/user_guide/random/zipf_like.md' },
                { text: 'triangular', link: '/user_guide/random/triangular.md' },
                { text: 'triangular_like', link: '/user_guide/random/triangular_like.md' },
                { text: 'bernoulli', link: '/user_guide/random/bernoulli.md' },
                { text: 'randint', link: '/user_guide/random/randint.md' },
                { text: 'randint_like', link: '/user_guide/random/randint_like.md' },
              ]
            },
            {
              text: 'shape manipulate',
              collapsible: true,
              children: [
                { text: 'squeeze', link: '/user_guide/shape_manipulate/squeeze.md' },
                { text: 'unsqueeze', link: '/user_guide/shape_manipulate/unsqueeze.md' },
                { text: 'reshape', link: '/user_guide/shape_manipulate/reshape.md' },
                { text: 'transpose', link: '/user_guide/shape_manipulate/transpose.md' },
                { text: 'permute', link: '/user_guide/shape_manipulate/permute.md' },
                { text: 'permute_inv', link: '/user_guide/shape_manipulate/permute_inv.md' },
                { text: 'expand', link: '/user_guide/shape_manipulate/expand.md' },
                { text: 't', link: '/user_guide/shape_manipulate/t.md' },
                { text: 'mt', link: '/user_guide/shape_manipulate/mt.md' },
                { text: 'flip', link: '/user_guide/shape_manipulate/flip.md' },
                { text: 'fliplr', link: '/user_guide/shape_manipulate/fliplr.md' },
                { text: 'flipud', link: '/user_guide/shape_manipulate/flipud.md' },
                { text: 'tile', link: '/user_guide/shape_manipulate/tile.md' },
                { text: 'trim_zeros', link: '/user_guide/shape_manipulate/trim_zeros.md' },
                { text: 'repeat', link: '/user_guide/shape_manipulate/repeat.md' },
                { text: 'split', link: '/user_guide/shape_manipulate/split.md' },
                { text: 'dsplit', link: '/user_guide/shape_manipulate/dsplit.md' },
                { text: 'hsplit', link: '/user_guide/shape_manipulate/hsplit.md' },
                { text: 'vsplit', link: '/user_guide/shape_manipulate/vsplit.md' },
                { text: 'swap_axes', link: '/user_guide/shape_manipulate/swap_axes.md' },
                { text: 'flatten', link: '/user_guide/shape_manipulate/flatten.md' },
                { text: 'concat', link: '/user_guide/shape_manipulate/concat.md' },
                { text: 'vstack', link: '/user_guide/shape_manipulate/vstack.md' },
                { text: 'hstack', link: '/user_guide/shape_manipulate/hstack.md' },
                { text: 'dstack', link: '/user_guide/shape_manipulate/dstack.md' },
              ]
            },
            {
              text: 'creation',
              collapsible: true,
              children: [
                { text: 'empty', link: '/user_guide/creation/empty.md' },
                { text: 'zeros', link: '/user_guide/creation/zeros.md' },
                { text: 'ones', link: '/user_guide/creation/ones.md' },
                { text: 'empty_like', link: '/user_guide/creation/empty_like.md' },
                { text: 'zeros_like', link: '/user_guide/creation/zeros_like.md' },
                { text: 'ones_like', link: '/user_guide/creation/ones_like.md' },
                { text: 'full', link: '/user_guide/creation/full.md' },
                { text: 'full_like', link: '/user_guide/creation/full_like.md' },
                { text: 'arange', link: '/user_guide/creation/arange.md' },
                { text: 'arange_step', link: '/user_guide/creation/arange_step.md' },
                { text: 'eye', link: '/user_guide/creation/eye.md' },
                { text: 'linspace', link: '/user_guide/creation/linspace.md' },
                { text: 'logspace', link: '/user_guide/creation/logspace.md' },
                { text: 'tril', link: '/user_guide/creation/tril.md' },
                { text: 'triu', link: '/user_guide/creation/triu.md' },
                { text: 'identity', link: '/user_guide/creation/identity.md' },
              ]
            },
            {
              text: 'windows',
              collapsible: true,
              children: [
                { text: 'hamming_window', link: '/user_guide/windows/hamming_window.md' },
                { text: 'hann_window', link: '/user_guide/windows/hann_window.md' },
                { text: 'blackman_window', link: '/user_guide/windows/blackman_window.md' },
              ]
            },
            {
              text: 'iterator',
              collapsible: true,
              children: [
                { text: 'par_iter', link: '/user_guide/iterator/par_iter.md' },
                { text: 'par_iter_mut', link: '/user_guide/iterator/par_iter_mut.md' },
                { text: 'par_iter_simd', link: '/user_guide/iterator/par_iter_simd.md' },
                { text: 'par_iter_simd_mut', link: '/user_guide/iterator/par_iter_simd_mut.md' },
                { text: 'strided_map', link: '/user_guide/iterator/strided_map.md' },
                { text: 'strided_map_simd', link: '/user_guide/iterator/strided_map_simd.md' },
                { text: 'for_each', link: '/user_guide/iterator/for_each.md' },
                { text: 'collect', link: '/user_guide/iterator/collect.md' },
              ]
            },
            {
              text: 'custom type',
              collapsible: true,
              children: [
              ]
            },
            {
              text: 'slice',
              link: '/user_guide/slice/slice.md'
            }
          ]
        }
      ],
      '/dev_guide/': [
        {
          text: 'dev guide',
          children: [
            '/dev_guide/getting_started.md',
            '/dev_guide/contribution.md',
          ]
        }
      ],
    },
  }),
})