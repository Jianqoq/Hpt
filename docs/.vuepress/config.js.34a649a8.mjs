// docs/.vuepress/config.js
import { defineUserConfig } from "@vuepress/cli";
import { defaultTheme } from "@vuepress/theme-default";
import { viteBundler } from "@vuepress/bundler-vite";
import { markdownMathPlugin } from "@vuepress/plugin-markdown-math";
import { mdEnhancePlugin } from "vuepress-plugin-md-enhance";
var config_default = defineUserConfig({
  base: process.env.NODE_ENV === "development" ? "/" : "/Hpt/",
  lang: "zh-CN",
  title: "Hpt",
  plugins: [
    markdownMathPlugin({
      type: "katex"
    }),
    mdEnhancePlugin({
      mermaid: true,
      chartjs: true
    })
  ],
  markdown: {
    anchor: false
  },
  bundler: viteBundler(),
  theme: defaultTheme({
    home: false,
    navbar: [
      {
        text: "Home",
        link: "/"
      },
      {
        text: "GitHub",
        link: "https://github.com/Jianqoq/Hpt"
      },
      {
        text: "crate.io",
        link: "https://crates.io/crates/hpt"
      },
      {
        text: "Benchmarks",
        link: "/benchmarks/benchmarks.md"
      }
    ],
    sidebar: {
      "/user_guide/": [
        {
          text: "Docs",
          children: [
            {
              text: "unary",
              collapsible: true,
              children: [
                { text: "sin", link: "/user_guide/unary/sin.md" },
                { text: "sin_", link: "/user_guide/unary/sin_.md" },
                { text: "cos", link: "/user_guide/unary/cos.md" },
                { text: "cos_", link: "/user_guide/unary/cos_.md" },
                { text: "tan", link: "/user_guide/unary/tan.md" },
                { text: "tan_", link: "/user_guide/unary/tan_.md" },
                { text: "sinh", link: "/user_guide/unary/sinh.md" },
                { text: "sinh_", link: "/user_guide/unary/sinh_.md" },
                { text: "cosh", link: "/user_guide/unary/cosh.md" },
                { text: "cosh_", link: "/user_guide/unary/cosh_.md" },
                { text: "tanh", link: "/user_guide/unary/tanh.md" },
                { text: "tanh_", link: "/user_guide/unary/tanh_.md" },
                { text: "asinh", link: "/user_guide/unary/asinh.md" },
                { text: "asinh_", link: "/user_guide/unary/asinh_.md" },
                { text: "acosh", link: "/user_guide/unary/acosh.md" },
                { text: "acosh_", link: "/user_guide/unary/acosh_.md" },
                { text: "atanh", link: "/user_guide/unary/atanh.md" },
                { text: "atanh_", link: "/user_guide/unary/atanh_.md" },
                { text: "asin", link: "/user_guide/unary/asin.md" },
                { text: "asin_", link: "/user_guide/unary/asin_.md" },
                { text: "acos", link: "/user_guide/unary/acos.md" },
                { text: "acos_", link: "/user_guide/unary/acos_.md" },
                { text: "atan", link: "/user_guide/unary/atan.md" },
                { text: "atan_", link: "/user_guide/unary/atan_.md" },
                { text: "exp", link: "/user_guide/unary/exp.md" },
                { text: "exp_", link: "/user_guide/unary/exp_.md" },
                { text: "exp2", link: "/user_guide/unary/exp2.md" },
                { text: "exp2_", link: "/user_guide/unary/exp2_.md" },
                { text: "sqrt", link: "/user_guide/unary/sqrt.md" },
                { text: "sqrt_", link: "/user_guide/unary/sqrt_.md" },
                { text: "recip", link: "/user_guide/unary/recip.md" },
                { text: "recip_", link: "/user_guide/unary/recip_.md" },
                { text: "ln", link: "/user_guide/unary/ln.md" },
                { text: "ln_", link: "/user_guide/unary/ln_.md" },
                { text: "log2", link: "/user_guide/unary/log2.md" },
                { text: "log2_", link: "/user_guide/unary/log2_.md" },
                { text: "log10", link: "/user_guide/unary/log10.md" },
                { text: "log10_", link: "/user_guide/unary/log10_.md" },
                { text: "celu", link: "/user_guide/unary/celu.md" },
                { text: "celu_", link: "/user_guide/unary/celu_.md" },
                { text: "sigmoid", link: "/user_guide/unary/sigmoid.md" },
                { text: "sigmoid_", link: "/user_guide/unary/sigmoid_.md" },
                { text: "elu", link: "/user_guide/unary/elu.md" },
                { text: "elu_", link: "/user_guide/unary/elu_.md" },
                { text: "erf", link: "/user_guide/unary/erf.md" },
                { text: "gelu", link: "/user_guide/unary/gelu.md" },
                { text: "gelu_", link: "/user_guide/unary/gelu_.md" },
                { text: "selu", link: "/user_guide/unary/selu.md" },
                { text: "selu_", link: "/user_guide/unary/selu_.md" },
                { text: "hard_sigmoid", link: "/user_guide/unary/hard_sigmoid.md" },
                { text: "hard_sigmoid_", link: "/user_guide/unary/hard_sigmoid_.md" },
                { text: "hard_swish", link: "/user_guide/unary/hard_swish.md" },
                { text: "hard_swish_", link: "/user_guide/unary/hard_swish_.md" },
                { text: "softplus", link: "/user_guide/unary/softplus.md" },
                { text: "softplus_", link: "/user_guide/unary/softplus_.md" },
                { text: "softsign", link: "/user_guide/unary/softsign.md" },
                { text: "softsign_", link: "/user_guide/unary/softsign_.md" },
                { text: "mish", link: "/user_guide/unary/mish.md" },
                { text: "mish_", link: "/user_guide/unary/mish_.md" },
                { text: "cbrt", link: "/user_guide/unary/cbrt.md" },
                { text: "cbrt_", link: "/user_guide/unary/cbrt_.md" },
                { text: "sincos", link: "/user_guide/unary/sincos.md" },
                { text: "sincos_", link: "/user_guide/unary/sincos_.md" },
                { text: "exp10", link: "/user_guide/unary/exp10.md" },
                { text: "exp10_", link: "/user_guide/unary/exp10_.md" }
              ]
            },
            {
              text: "binary",
              collapsible: true,
              children: [
                { text: "add", link: "/user_guide/binary/add.md" },
                { text: "add_", link: "/user_guide/binary/add_.md" },
                { text: "sub", link: "/user_guide/binary/sub.md" },
                { text: "sub_", link: "/user_guide/binary/sub_.md" },
                { text: "mul", link: "/user_guide/binary/mul.md" },
                { text: "mul_", link: "/user_guide/binary/mul_.md" },
                { text: "div", link: "/user_guide/binary/div.md" },
                { text: "div_", link: "/user_guide/binary/div_.md" },
                { text: "rem", link: "/user_guide/binary/rem.md" },
                { text: "rem_", link: "/user_guide/binary/rem_.md" },
                { text: "pow", link: "/user_guide/binary/pow.md" },
                { text: "pow_", link: "/user_guide/binary/pow_.md" },
                { text: "hypot", link: "/user_guide/binary/hypot.md" },
                { text: "hypot_", link: "/user_guide/binary/hypot_.md" }
              ]
            },
            {
              text: "reduce",
              collapsible: true,
              children: [
                { text: "logsumexp", link: "/user_guide/reduce/logsumexp.md" },
                { text: "argmin", link: "/user_guide/reduce/argmin.md" },
                { text: "argmax", link: "/user_guide/reduce/argmax.md" },
                { text: "max", link: "/user_guide/reduce/max.md" },
                { text: "min", link: "/user_guide/reduce/min.md" },
                { text: "mean", link: "/user_guide/reduce/mean.md" },
                { text: "sum", link: "/user_guide/reduce/sum.md" },
                { text: "sum_", link: "/user_guide/reduce/sum_.md" },
                { text: "nansum", link: "/user_guide/reduce/nansum.md" },
                { text: "nansum_", link: "/user_guide/reduce/nansum_.md" },
                { text: "prod", link: "/user_guide/reduce/prod.md" },
                { text: "nanprod", link: "/user_guide/reduce/nanprod.md" },
                { text: "sum_square", link: "/user_guide/reduce/sum_square.md" },
                { text: "reducel1", link: "/user_guide/reduce/reducel1.md" },
                { text: "reducel2", link: "/user_guide/reduce/reducel2.md" },
                { text: "reducel3", link: "/user_guide/reduce/reducel3.md" },
                { text: "all", link: "/user_guide/reduce/all.md" },
                { text: "any", link: "/user_guide/reduce/any.md" }
              ]
            },
            {
              text: "conv",
              collapsible: true,
              children: [
                { text: "batchnorm_conv2d", link: "/user_guide/conv/batchnorm_conv2d.md" },
                { text: "conv2d_group", link: "/user_guide/conv/conv2d_group.md" },
                { text: "conv2d_transpose", link: "/user_guide/conv/conv2d_transpose.md" },
                { text: "conv2d", link: "/user_guide/conv/conv2d.md" },
                { text: "dwconv2d", link: "/user_guide/conv/dwconv2d.md" }
              ]
            },
            {
              text: "pooling",
              collapsible: true,
              children: [
                { text: "maxpool2d", link: "/user_guide/pooling/maxpool2d.md" },
                { text: "avgpool2d", link: "/user_guide/pooling/avgpool2d.md" },
                { text: "adaptive_maxpool2d", link: "/user_guide/pooling/adaptive_maxpool2d.md" },
                { text: "adaptive_avgpool2d", link: "/user_guide/pooling/adaptive_avgpool2d.md" }
              ]
            },
            {
              text: "compare",
              collapsible: true,
              children: [
                { text: "tensor_neq", link: "/user_guide/cmp/tensor_neq.md" },
                { text: "tensor_eq", link: "/user_guide/cmp/tensor_eq.md" },
                { text: "tensor_gt", link: "/user_guide/cmp/tensor_gt.md" },
                { text: "tensor_lt", link: "/user_guide/cmp/tensor_lt.md" },
                { text: "tensor_ge", link: "/user_guide/cmp/tensor_ge.md" },
                { text: "tensor_le", link: "/user_guide/cmp/tensor_le.md" }
              ]
            },
            {
              text: "advanced",
              collapsible: true,
              children: [
                { text: "scatter", link: "/user_guide/advanced/scatter.md" },
                { text: "hardmax", link: "/user_guide/advanced/hardmax.md" },
                { text: "tensor_where", link: "/user_guide/advanced/tensor_where.md" },
                { text: "topk", link: "/user_guide/advanced/topk.md" },
                { text: "onehot", link: "/user_guide/advanced/onehot.md" }
              ]
            },
            {
              text: "normalization",
              collapsible: true,
              children: [
                { text: "log_softmax", link: "/user_guide/normalization/log_softmax.md" },
                { text: "layernorm", link: "/user_guide/normalization/layernorm.md" },
                { text: "softmax", link: "/user_guide/normalization/softmax.md" }
              ]
            },
            {
              text: "cumulative",
              collapsible: true,
              children: [
                { text: "cumsum", link: "/user_guide/cumulative/cumsum.md" },
                { text: "cumprod", link: "/user_guide/cumulative/cumprod.md" }
              ]
            },
            {
              text: "regularization",
              collapsible: true,
              children: [
                { text: "dropout", link: "/user_guide/regularization/dropout.md" },
                { text: "shrinkage", link: "/user_guide/regularization/shrinkage.md" }
              ]
            },
            {
              text: "linalg",
              collapsible: true,
              children: [
                { text: "matmul", link: "/user_guide/linalg/matmul.md" },
                { text: "tensordot", link: "/user_guide/linalg/tensordot.md" }
              ]
            },
            {
              text: "random",
              collapsible: true,
              children: [
                { text: "randn", link: "/user_guide/random/randn.md" },
                { text: "randn_like", link: "/user_guide/random/randn_like.md" },
                { text: "rand", link: "/user_guide/random/rand.md" },
                { text: "rand_like", link: "/user_guide/random/rand_like.md" },
                { text: "beta", link: "/user_guide/random/beta.md" },
                { text: "beta_like", link: "/user_guide/random/beta_like.md" },
                { text: "chisquare", link: "/user_guide/random/chisquare.md" },
                { text: "chisquare_like", link: "/user_guide/random/chisquare_like.md" },
                { text: "exponential", link: "/user_guide/random/exponential.md" },
                { text: "exponential_like", link: "/user_guide/random/exponential_like.md" },
                { text: "gamma", link: "/user_guide/random/gamma.md" },
                { text: "gamma_like", link: "/user_guide/random/gamma_like.md" },
                { text: "gumbel", link: "/user_guide/random/gumbel.md" },
                { text: "gumbel_like", link: "/user_guide/random/gumbel_like.md" },
                { text: "lognormal", link: "/user_guide/random/lognormal.md" },
                { text: "lognormal_like", link: "/user_guide/random/lognormal_like.md" },
                { text: "normal_gaussian", link: "/user_guide/random/normal_gaussian.md" },
                { text: "normal_gaussian_like", link: "/user_guide/random/normal_gaussian_like.md" },
                { text: "pareto", link: "/user_guide/random/pareto.md" },
                { text: "pareto_like", link: "/user_guide/random/pareto_like.md" },
                { text: "poisson", link: "/user_guide/random/poisson.md" },
                { text: "poisson_like", link: "/user_guide/random/poisson_like.md" },
                { text: "weibull", link: "/user_guide/random/weibull.md" },
                { text: "weibull_like", link: "/user_guide/random/weibull_like.md" },
                { text: "zipf", link: "/user_guide/random/zipf.md" },
                { text: "zipf_like", link: "/user_guide/random/zipf_like.md" },
                { text: "triangular", link: "/user_guide/random/triangular.md" },
                { text: "triangular_like", link: "/user_guide/random/triangular_like.md" },
                { text: "bernoulli", link: "/user_guide/random/bernoulli.md" },
                { text: "randint", link: "/user_guide/random/randint.md" },
                { text: "randint_like", link: "/user_guide/random/randint_like.md" }
              ]
            },
            {
              text: "shape manipulate",
              collapsible: true,
              children: [
                { text: "squeeze", link: "/user_guide/shape_manipulate/squeeze.md" },
                { text: "unsqueeze", link: "/user_guide/shape_manipulate/unsqueeze.md" },
                { text: "reshape", link: "/user_guide/shape_manipulate/reshape.md" },
                { text: "transpose", link: "/user_guide/shape_manipulate/transpose.md" },
                { text: "permute", link: "/user_guide/shape_manipulate/permute.md" },
                { text: "permute_inv", link: "/user_guide/shape_manipulate/permute_inv.md" },
                { text: "expand", link: "/user_guide/shape_manipulate/expand.md" },
                { text: "t", link: "/user_guide/shape_manipulate/t.md" },
                { text: "mt", link: "/user_guide/shape_manipulate/mt.md" },
                { text: "flip", link: "/user_guide/shape_manipulate/flip.md" },
                { text: "fliplr", link: "/user_guide/shape_manipulate/fliplr.md" },
                { text: "flipud", link: "/user_guide/shape_manipulate/flipud.md" },
                { text: "tile", link: "/user_guide/shape_manipulate/tile.md" },
                { text: "trim_zeros", link: "/user_guide/shape_manipulate/trim_zeros.md" },
                { text: "repeat", link: "/user_guide/shape_manipulate/repeat.md" },
                { text: "split", link: "/user_guide/shape_manipulate/split.md" },
                { text: "dsplit", link: "/user_guide/shape_manipulate/dsplit.md" },
                { text: "hsplit", link: "/user_guide/shape_manipulate/hsplit.md" },
                { text: "vsplit", link: "/user_guide/shape_manipulate/vsplit.md" },
                { text: "swap_axes", link: "/user_guide/shape_manipulate/swap_axes.md" },
                { text: "flatten", link: "/user_guide/shape_manipulate/flatten.md" },
                { text: "concat", link: "/user_guide/shape_manipulate/concat.md" },
                { text: "vstack", link: "/user_guide/shape_manipulate/vstack.md" },
                { text: "hstack", link: "/user_guide/shape_manipulate/hstack.md" },
                { text: "dstack", link: "/user_guide/shape_manipulate/dstack.md" }
              ]
            },
            {
              text: "creation",
              collapsible: true,
              children: [
                { text: "empty", link: "/user_guide/creation/empty.md" },
                { text: "zeros", link: "/user_guide/creation/zeros.md" },
                { text: "ones", link: "/user_guide/creation/ones.md" },
                { text: "empty_like", link: "/user_guide/creation/empty_like.md" },
                { text: "zeros_like", link: "/user_guide/creation/zeros_like.md" },
                { text: "ones_like", link: "/user_guide/creation/ones_like.md" },
                { text: "full", link: "/user_guide/creation/full.md" },
                { text: "full_like", link: "/user_guide/creation/full_like.md" },
                { text: "arange", link: "/user_guide/creation/arange.md" },
                { text: "arange_step", link: "/user_guide/creation/arange_step.md" },
                { text: "eye", link: "/user_guide/creation/eye.md" },
                { text: "linspace", link: "/user_guide/creation/linspace.md" },
                { text: "logspace", link: "/user_guide/creation/logspace.md" },
                { text: "geomspace", link: "/user_guide/creation/geomspace.md" },
                { text: "tri", link: "/user_guide/creation/tri.md" },
                { text: "tril", link: "/user_guide/creation/tril.md" },
                { text: "triu", link: "/user_guide/creation/triu.md" },
                { text: "identity", link: "/user_guide/creation/identity.md" }
              ]
            },
            {
              text: "windows",
              collapsible: true,
              children: [
                { text: "hamming_window", link: "/user_guide/windows/hamming_window.md" },
                { text: "hann_window", link: "/user_guide/windows/hann_window.md" },
                { text: "blackman_window", link: "/user_guide/windows/blackman_window.md" }
              ]
            },
            {
              text: "iterator",
              collapsible: true,
              children: [
                { text: "par_iter", link: "/user_guide/iterator/par_iter.md" },
                { text: "par_iter_mut", link: "/user_guide/iterator/par_iter_mut.md" },
                { text: "par_iter_simd", link: "/user_guide/iterator/par_iter_simd.md" },
                { text: "par_iter_simd_mut", link: "/user_guide/iterator/par_iter_simd_mut.md" },
                { text: "strided_map", link: "/user_guide/iterator/strided_map.md" },
                { text: "strided_map_simd", link: "/user_guide/iterator/strided_map_simd.md" },
                { text: "collect", link: "/user_guide/iterator/collect.md" }
              ]
            },
            {
              text: "utils",
              collapsible: true,
              children: [
                { text: "set_display_elements", link: "/user_guide/utils/set_display_elements.md" },
                { text: "resize_cpu_lru_cache", link: "/user_guide/utils/resize_cpu_lru_cache.md" },
                { text: "resize_cuda_lru_cache", link: "/user_guide/utils/resize_cuda_lru_cache.md" },
                { text: "set_seed", link: "/user_guide/utils/set_seed.md" },
                { text: "num_threads", link: "/user_guide/utils/num_threads.md" }
              ]
            },
            {
              text: "custom type",
              link: "/user_guide/custom_type/custom_type.md"
            },
            {
              text: "custom allocator",
              link: "/user_guide/custom_allocator/custom_allocator.md"
            },
            {
              text: "slice",
              link: "/user_guide/slice/slice.md"
            },
            {
              text: "save/load",
              link: "/user_guide/save_load/save_load.md"
            }
          ]
        }
      ],
      "/dev_guide/": [
        {
          text: "Dev Guide",
          children: [
            {
              text: "allocation",
              link: "/dev_guide/allocation/allocator.md"
            },
            {
              text: "New Type",
              link: "/dev_guide/new_type.md"
            },
            {
              text: "type promote",
              link: "/dev_guide/type_promote.md"
            },
            {
              text: "pointer",
              link: "/dev_guide/pointer/pointer.md"
            },
            {
              text: "test cases",
              link: "/dev_guide/test_rules.md"
            },
            {
              text: "iterator",
              link: "/dev_guide/iterator/iterator.md"
            },
            {
              text: "New op",
              link: "/dev_guide/adding_new_op.md"
            },
            {
              text: "New arch support",
              link: "/dev_guide/adding_new_arch.md"
            }
          ]
        }
      ],
      "/benchmarks/": [
        {
          text: "Benchmarks",
          children: [
            {
              text: "unary",
              link: "/benchmarks/unary.md"
            },
            {
              text: "binary",
              link: "/benchmarks/binary.md"
            },
            {
              text: "reduce",
              link: "/benchmarks/reduce.md"
            },
            {
              text: "conv",
              link: "/benchmarks/conv.md"
            },
            {
              text: "pooling",
              link: "/benchmarks/pooling.md"
            },
            {
              text: "normalization",
              link: "/benchmarks/normalization.md"
            },
            {
              text: "matmul",
              link: "/benchmarks/matmul.md"
            },
            {
              text: "fft",
              link: "/benchmarks/fft.md"
            },
            {
              text: "nn",
              collapsible: true,
              children: [
                { text: "resnet", link: "/benchmarks/nn/resnet.md" },
                { text: "lstm", link: "/benchmarks/nn/lstm.md" }
              ]
            }
          ]
        }
      ]
    }
  })
});
export {
  config_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiZG9jcy8udnVlcHJlc3MvY29uZmlnLmpzIl0sCiAgInNvdXJjZXNDb250ZW50IjogWyJjb25zdCBfX3ZpdGVfaW5qZWN0ZWRfb3JpZ2luYWxfZGlybmFtZSA9IFwiQzovVXNlcnMvMTIzL0hwdC9kb2NzLy52dWVwcmVzc1wiO2NvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9maWxlbmFtZSA9IFwiQzpcXFxcVXNlcnNcXFxcMTIzXFxcXEhwdFxcXFxkb2NzXFxcXC52dWVwcmVzc1xcXFxjb25maWcuanNcIjtjb25zdCBfX3ZpdGVfaW5qZWN0ZWRfb3JpZ2luYWxfaW1wb3J0X21ldGFfdXJsID0gXCJmaWxlOi8vL0M6L1VzZXJzLzEyMy9IcHQvZG9jcy8udnVlcHJlc3MvY29uZmlnLmpzXCI7aW1wb3J0IHsgZGVmaW5lVXNlckNvbmZpZyB9IGZyb20gJ0B2dWVwcmVzcy9jbGknXHJcbmltcG9ydCB7IGRlZmF1bHRUaGVtZSB9IGZyb20gJ0B2dWVwcmVzcy90aGVtZS1kZWZhdWx0J1xyXG5pbXBvcnQgeyB2aXRlQnVuZGxlciB9IGZyb20gJ0B2dWVwcmVzcy9idW5kbGVyLXZpdGUnXHJcbmltcG9ydCB7IG1hcmtkb3duTWF0aFBsdWdpbiB9IGZyb20gJ0B2dWVwcmVzcy9wbHVnaW4tbWFya2Rvd24tbWF0aCdcclxuaW1wb3J0IHsgbWRFbmhhbmNlUGx1Z2luIH0gZnJvbSBcInZ1ZXByZXNzLXBsdWdpbi1tZC1lbmhhbmNlXCI7XHJcblxyXG5leHBvcnQgZGVmYXVsdCBkZWZpbmVVc2VyQ29uZmlnKHtcclxuICBiYXNlOiBwcm9jZXNzLmVudi5OT0RFX0VOViA9PT0gJ2RldmVsb3BtZW50J1xyXG4gICAgPyAnLydcclxuICAgIDogJy9IcHQvJyxcclxuICBsYW5nOiAnemgtQ04nLFxyXG4gIHRpdGxlOiAnSHB0JyxcclxuICBwbHVnaW5zOiBbXHJcbiAgICBtYXJrZG93bk1hdGhQbHVnaW4oe1xyXG4gICAgICB0eXBlOiAna2F0ZXgnXHJcbiAgICB9KSxcclxuICAgIG1kRW5oYW5jZVBsdWdpbih7XHJcbiAgICAgIG1lcm1haWQ6IHRydWUsXHJcbiAgICAgIGNoYXJ0anM6IHRydWUsXHJcbiAgICB9KSxcclxuICBdLFxyXG4gIG1hcmtkb3duOiB7XHJcbiAgICBhbmNob3I6IGZhbHNlLFxyXG4gIH0sXHJcbiAgYnVuZGxlcjogdml0ZUJ1bmRsZXIoKSxcclxuICB0aGVtZTogZGVmYXVsdFRoZW1lKHtcclxuICAgIGhvbWU6IGZhbHNlLFxyXG4gICAgbmF2YmFyOiBbXHJcbiAgICAgIHtcclxuICAgICAgICB0ZXh0OiAnSG9tZScsXHJcbiAgICAgICAgbGluazogJy8nLFxyXG4gICAgICB9LFxyXG4gICAgICB7XHJcbiAgICAgICAgdGV4dDogJ0dpdEh1YicsXHJcbiAgICAgICAgbGluazogJ2h0dHBzOi8vZ2l0aHViLmNvbS9KaWFucW9xL0hwdCcsXHJcbiAgICAgIH0sXHJcbiAgICAgIHtcclxuICAgICAgICB0ZXh0OiAnY3JhdGUuaW8nLFxyXG4gICAgICAgIGxpbms6ICdodHRwczovL2NyYXRlcy5pby9jcmF0ZXMvaHB0JyxcclxuICAgICAgfSxcclxuICAgICAge1xyXG4gICAgICAgIHRleHQ6ICdCZW5jaG1hcmtzJyxcclxuICAgICAgICBsaW5rOiAnL2JlbmNobWFya3MvYmVuY2htYXJrcy5tZCcsXHJcbiAgICAgIH1cclxuICAgIF0sXHJcblxyXG4gICAgc2lkZWJhcjoge1xyXG4gICAgICAnL3VzZXJfZ3VpZGUvJzogW1xyXG4gICAgICAgIHtcclxuICAgICAgICAgIHRleHQ6ICdEb2NzJyxcclxuICAgICAgICAgIGNoaWxkcmVuOiBbXHJcbiAgICAgICAgICAgIHtcclxuICAgICAgICAgICAgICB0ZXh0OiAndW5hcnknLFxyXG4gICAgICAgICAgICAgIGNvbGxhcHNpYmxlOiB0cnVlLFxyXG4gICAgICAgICAgICAgIGNoaWxkcmVuOiBbXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdzaW4nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvc2luLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnc2luXycsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9zaW5fLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnY29zJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L2Nvcy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2Nvc18nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvY29zXy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3RhbicsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS90YW4ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICd0YW5fJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L3Rhbl8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdzaW5oJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L3NpbmgubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdzaW5oXycsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9zaW5oXy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2Nvc2gnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvY29zaC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2Nvc2hfJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L2Nvc2hfLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAndGFuaCcsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS90YW5oLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAndGFuaF8nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvdGFuaF8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdhc2luaCcsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9hc2luaC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2FzaW5oXycsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9hc2luaF8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdhY29zaCcsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9hY29zaC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2Fjb3NoXycsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9hY29zaF8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdhdGFuaCcsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9hdGFuaC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2F0YW5oXycsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9hdGFuaF8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdhc2luJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L2FzaW4ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdhc2luXycsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9hc2luXy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2Fjb3MnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvYWNvcy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2Fjb3NfJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L2Fjb3NfLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnYXRhbicsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9hdGFuLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnYXRhbl8nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvYXRhbl8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdleHAnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvZXhwLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnZXhwXycsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9leHBfLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnZXhwMicsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9leHAyLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnZXhwMl8nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvZXhwMl8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdzcXJ0JywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L3NxcnQubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdzcXJ0XycsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9zcXJ0Xy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3JlY2lwJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L3JlY2lwLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAncmVjaXBfJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L3JlY2lwXy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2xuJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L2xuLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnbG5fJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L2xuXy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2xvZzInLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvbG9nMi5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2xvZzJfJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L2xvZzJfLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnbG9nMTAnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvbG9nMTAubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdsb2cxMF8nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvbG9nMTBfLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnY2VsdScsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9jZWx1Lm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnY2VsdV8nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvY2VsdV8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdzaWdtb2lkJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L3NpZ21vaWQubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdzaWdtb2lkXycsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9zaWdtb2lkXy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2VsdScsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9lbHUubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdlbHVfJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L2VsdV8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdlcmYnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvZXJmLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnZ2VsdScsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9nZWx1Lm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnZ2VsdV8nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvZ2VsdV8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdzZWx1JywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L3NlbHUubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdzZWx1XycsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9zZWx1Xy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2hhcmRfc2lnbW9pZCcsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9oYXJkX3NpZ21vaWQubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdoYXJkX3NpZ21vaWRfJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L2hhcmRfc2lnbW9pZF8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdoYXJkX3N3aXNoJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L2hhcmRfc3dpc2gubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdoYXJkX3N3aXNoXycsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9oYXJkX3N3aXNoXy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3NvZnRwbHVzJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L3NvZnRwbHVzLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnc29mdHBsdXNfJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L3NvZnRwbHVzXy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3NvZnRzaWduJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L3NvZnRzaWduLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnc29mdHNpZ25fJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L3NvZnRzaWduXy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ21pc2gnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvbWlzaC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ21pc2hfJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L21pc2hfLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnY2JydCcsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9jYnJ0Lm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnY2JydF8nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvY2JydF8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdzaW5jb3MnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvc2luY29zLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnc2luY29zXycsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9zaW5jb3NfLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnZXhwMTAnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvZXhwMTAubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdleHAxMF8nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvZXhwMTBfLm1kJyB9LFxyXG4gICAgICAgICAgICAgIF1cclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgIHRleHQ6ICdiaW5hcnknLFxyXG4gICAgICAgICAgICAgIGNvbGxhcHNpYmxlOiB0cnVlLFxyXG4gICAgICAgICAgICAgIGNoaWxkcmVuOiBbXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdhZGQnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvYmluYXJ5L2FkZC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2FkZF8nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvYmluYXJ5L2FkZF8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdzdWInLCBsaW5rOiAnL3VzZXJfZ3VpZGUvYmluYXJ5L3N1Yi5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3N1Yl8nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvYmluYXJ5L3N1Yl8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdtdWwnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvYmluYXJ5L211bC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ211bF8nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvYmluYXJ5L211bF8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdkaXYnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvYmluYXJ5L2Rpdi5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2Rpdl8nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvYmluYXJ5L2Rpdl8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdyZW0nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvYmluYXJ5L3JlbS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3JlbV8nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvYmluYXJ5L3JlbV8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdwb3cnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvYmluYXJ5L3Bvdy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3Bvd18nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvYmluYXJ5L3Bvd18ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdoeXBvdCcsIGxpbms6ICcvdXNlcl9ndWlkZS9iaW5hcnkvaHlwb3QubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdoeXBvdF8nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvYmluYXJ5L2h5cG90Xy5tZCcgfSxcclxuICAgICAgICAgICAgICBdXHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIHtcclxuICAgICAgICAgICAgICB0ZXh0OiAncmVkdWNlJyxcclxuICAgICAgICAgICAgICBjb2xsYXBzaWJsZTogdHJ1ZSxcclxuICAgICAgICAgICAgICBjaGlsZHJlbjogW1xyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnbG9nc3VtZXhwJywgbGluazogJy91c2VyX2d1aWRlL3JlZHVjZS9sb2dzdW1leHAubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdhcmdtaW4nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmVkdWNlL2FyZ21pbi5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2FyZ21heCcsIGxpbms6ICcvdXNlcl9ndWlkZS9yZWR1Y2UvYXJnbWF4Lm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnbWF4JywgbGluazogJy91c2VyX2d1aWRlL3JlZHVjZS9tYXgubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdtaW4nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmVkdWNlL21pbi5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ21lYW4nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmVkdWNlL21lYW4ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdzdW0nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmVkdWNlL3N1bS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3N1bV8nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmVkdWNlL3N1bV8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICduYW5zdW0nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmVkdWNlL25hbnN1bS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ25hbnN1bV8nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmVkdWNlL25hbnN1bV8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdwcm9kJywgbGluazogJy91c2VyX2d1aWRlL3JlZHVjZS9wcm9kLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnbmFucHJvZCcsIGxpbms6ICcvdXNlcl9ndWlkZS9yZWR1Y2UvbmFucHJvZC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3N1bV9zcXVhcmUnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmVkdWNlL3N1bV9zcXVhcmUubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdyZWR1Y2VsMScsIGxpbms6ICcvdXNlcl9ndWlkZS9yZWR1Y2UvcmVkdWNlbDEubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdyZWR1Y2VsMicsIGxpbms6ICcvdXNlcl9ndWlkZS9yZWR1Y2UvcmVkdWNlbDIubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdyZWR1Y2VsMycsIGxpbms6ICcvdXNlcl9ndWlkZS9yZWR1Y2UvcmVkdWNlbDMubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdhbGwnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmVkdWNlL2FsbC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2FueScsIGxpbms6ICcvdXNlcl9ndWlkZS9yZWR1Y2UvYW55Lm1kJyB9LFxyXG4gICAgICAgICAgICAgIF1cclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgIHRleHQ6IFwiY29udlwiLFxyXG4gICAgICAgICAgICAgIGNvbGxhcHNpYmxlOiB0cnVlLFxyXG4gICAgICAgICAgICAgIGNoaWxkcmVuOiBbXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdiYXRjaG5vcm1fY29udjJkJywgbGluazogJy91c2VyX2d1aWRlL2NvbnYvYmF0Y2hub3JtX2NvbnYyZC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2NvbnYyZF9ncm91cCcsIGxpbms6ICcvdXNlcl9ndWlkZS9jb252L2NvbnYyZF9ncm91cC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2NvbnYyZF90cmFuc3Bvc2UnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvY29udi9jb252MmRfdHJhbnNwb3NlLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnY29udjJkJywgbGluazogJy91c2VyX2d1aWRlL2NvbnYvY29udjJkLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnZHdjb252MmQnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvY29udi9kd2NvbnYyZC5tZCcgfSxcclxuICAgICAgICAgICAgICBdXHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIHtcclxuICAgICAgICAgICAgICB0ZXh0OiBcInBvb2xpbmdcIixcclxuICAgICAgICAgICAgICBjb2xsYXBzaWJsZTogdHJ1ZSxcclxuICAgICAgICAgICAgICBjaGlsZHJlbjogW1xyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnbWF4cG9vbDJkJywgbGluazogJy91c2VyX2d1aWRlL3Bvb2xpbmcvbWF4cG9vbDJkLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnYXZncG9vbDJkJywgbGluazogJy91c2VyX2d1aWRlL3Bvb2xpbmcvYXZncG9vbDJkLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnYWRhcHRpdmVfbWF4cG9vbDJkJywgbGluazogJy91c2VyX2d1aWRlL3Bvb2xpbmcvYWRhcHRpdmVfbWF4cG9vbDJkLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnYWRhcHRpdmVfYXZncG9vbDJkJywgbGluazogJy91c2VyX2d1aWRlL3Bvb2xpbmcvYWRhcHRpdmVfYXZncG9vbDJkLm1kJyB9LFxyXG4gICAgICAgICAgICAgIF1cclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgIHRleHQ6ICdjb21wYXJlJyxcclxuICAgICAgICAgICAgICBjb2xsYXBzaWJsZTogdHJ1ZSxcclxuICAgICAgICAgICAgICBjaGlsZHJlbjogW1xyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAndGVuc29yX25lcScsIGxpbms6ICcvdXNlcl9ndWlkZS9jbXAvdGVuc29yX25lcS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3RlbnNvcl9lcScsIGxpbms6ICcvdXNlcl9ndWlkZS9jbXAvdGVuc29yX2VxLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAndGVuc29yX2d0JywgbGluazogJy91c2VyX2d1aWRlL2NtcC90ZW5zb3JfZ3QubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICd0ZW5zb3JfbHQnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvY21wL3RlbnNvcl9sdC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3RlbnNvcl9nZScsIGxpbms6ICcvdXNlcl9ndWlkZS9jbXAvdGVuc29yX2dlLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAndGVuc29yX2xlJywgbGluazogJy91c2VyX2d1aWRlL2NtcC90ZW5zb3JfbGUubWQnIH0sXHJcbiAgICAgICAgICAgICAgXVxyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICB7XHJcbiAgICAgICAgICAgICAgdGV4dDogJ2FkdmFuY2VkJyxcclxuICAgICAgICAgICAgICBjb2xsYXBzaWJsZTogdHJ1ZSxcclxuICAgICAgICAgICAgICBjaGlsZHJlbjogW1xyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnc2NhdHRlcicsIGxpbms6ICcvdXNlcl9ndWlkZS9hZHZhbmNlZC9zY2F0dGVyLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnaGFyZG1heCcsIGxpbms6ICcvdXNlcl9ndWlkZS9hZHZhbmNlZC9oYXJkbWF4Lm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAndGVuc29yX3doZXJlJywgbGluazogJy91c2VyX2d1aWRlL2FkdmFuY2VkL3RlbnNvcl93aGVyZS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3RvcGsnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvYWR2YW5jZWQvdG9way5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ29uZWhvdCcsIGxpbms6ICcvdXNlcl9ndWlkZS9hZHZhbmNlZC9vbmVob3QubWQnIH0sXHJcbiAgICAgICAgICAgICAgXVxyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICB7XHJcbiAgICAgICAgICAgICAgdGV4dDogJ25vcm1hbGl6YXRpb24nLFxyXG4gICAgICAgICAgICAgIGNvbGxhcHNpYmxlOiB0cnVlLFxyXG4gICAgICAgICAgICAgIGNoaWxkcmVuOiBbXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdsb2dfc29mdG1heCcsIGxpbms6ICcvdXNlcl9ndWlkZS9ub3JtYWxpemF0aW9uL2xvZ19zb2Z0bWF4Lm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnbGF5ZXJub3JtJywgbGluazogJy91c2VyX2d1aWRlL25vcm1hbGl6YXRpb24vbGF5ZXJub3JtLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnc29mdG1heCcsIGxpbms6ICcvdXNlcl9ndWlkZS9ub3JtYWxpemF0aW9uL3NvZnRtYXgubWQnIH0sXHJcbiAgICAgICAgICAgICAgXVxyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICB7XHJcbiAgICAgICAgICAgICAgdGV4dDogJ2N1bXVsYXRpdmUnLFxyXG4gICAgICAgICAgICAgIGNvbGxhcHNpYmxlOiB0cnVlLFxyXG4gICAgICAgICAgICAgIGNoaWxkcmVuOiBbXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdjdW1zdW0nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvY3VtdWxhdGl2ZS9jdW1zdW0ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdjdW1wcm9kJywgbGluazogJy91c2VyX2d1aWRlL2N1bXVsYXRpdmUvY3VtcHJvZC5tZCcgfSxcclxuICAgICAgICAgICAgICBdXHJcbiAgICAgICAgICAgIH0sIFxyXG4gICAgICAgICAgICB7XHJcbiAgICAgICAgICAgICAgdGV4dDogJ3JlZ3VsYXJpemF0aW9uJyxcclxuICAgICAgICAgICAgICBjb2xsYXBzaWJsZTogdHJ1ZSxcclxuICAgICAgICAgICAgICBjaGlsZHJlbjogW1xyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnZHJvcG91dCcsIGxpbms6ICcvdXNlcl9ndWlkZS9yZWd1bGFyaXphdGlvbi9kcm9wb3V0Lm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnc2hyaW5rYWdlJywgbGluazogJy91c2VyX2d1aWRlL3JlZ3VsYXJpemF0aW9uL3Nocmlua2FnZS5tZCcgfSxcclxuICAgICAgICAgICAgICBdXHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIHtcclxuICAgICAgICAgICAgICB0ZXh0OiAnbGluYWxnJyxcclxuICAgICAgICAgICAgICBjb2xsYXBzaWJsZTogdHJ1ZSxcclxuICAgICAgICAgICAgICBjaGlsZHJlbjogW1xyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnbWF0bXVsJywgbGluazogJy91c2VyX2d1aWRlL2xpbmFsZy9tYXRtdWwubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICd0ZW5zb3Jkb3QnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvbGluYWxnL3RlbnNvcmRvdC5tZCcgfSxcclxuICAgICAgICAgICAgICBdXHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIHtcclxuICAgICAgICAgICAgICB0ZXh0OiAncmFuZG9tJyxcclxuICAgICAgICAgICAgICBjb2xsYXBzaWJsZTogdHJ1ZSxcclxuICAgICAgICAgICAgICBjaGlsZHJlbjogW1xyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAncmFuZG4nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmFuZG9tL3JhbmRuLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAncmFuZG5fbGlrZScsIGxpbms6ICcvdXNlcl9ndWlkZS9yYW5kb20vcmFuZG5fbGlrZS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3JhbmQnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmFuZG9tL3JhbmQubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdyYW5kX2xpa2UnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmFuZG9tL3JhbmRfbGlrZS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2JldGEnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmFuZG9tL2JldGEubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdiZXRhX2xpa2UnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmFuZG9tL2JldGFfbGlrZS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2NoaXNxdWFyZScsIGxpbms6ICcvdXNlcl9ndWlkZS9yYW5kb20vY2hpc3F1YXJlLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnY2hpc3F1YXJlX2xpa2UnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmFuZG9tL2NoaXNxdWFyZV9saWtlLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnZXhwb25lbnRpYWwnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmFuZG9tL2V4cG9uZW50aWFsLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnZXhwb25lbnRpYWxfbGlrZScsIGxpbms6ICcvdXNlcl9ndWlkZS9yYW5kb20vZXhwb25lbnRpYWxfbGlrZS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2dhbW1hJywgbGluazogJy91c2VyX2d1aWRlL3JhbmRvbS9nYW1tYS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2dhbW1hX2xpa2UnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmFuZG9tL2dhbW1hX2xpa2UubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdndW1iZWwnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmFuZG9tL2d1bWJlbC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2d1bWJlbF9saWtlJywgbGluazogJy91c2VyX2d1aWRlL3JhbmRvbS9ndW1iZWxfbGlrZS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2xvZ25vcm1hbCcsIGxpbms6ICcvdXNlcl9ndWlkZS9yYW5kb20vbG9nbm9ybWFsLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnbG9nbm9ybWFsX2xpa2UnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmFuZG9tL2xvZ25vcm1hbF9saWtlLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnbm9ybWFsX2dhdXNzaWFuJywgbGluazogJy91c2VyX2d1aWRlL3JhbmRvbS9ub3JtYWxfZ2F1c3NpYW4ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdub3JtYWxfZ2F1c3NpYW5fbGlrZScsIGxpbms6ICcvdXNlcl9ndWlkZS9yYW5kb20vbm9ybWFsX2dhdXNzaWFuX2xpa2UubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdwYXJldG8nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmFuZG9tL3BhcmV0by5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3BhcmV0b19saWtlJywgbGluazogJy91c2VyX2d1aWRlL3JhbmRvbS9wYXJldG9fbGlrZS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3BvaXNzb24nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmFuZG9tL3BvaXNzb24ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdwb2lzc29uX2xpa2UnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmFuZG9tL3BvaXNzb25fbGlrZS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3dlaWJ1bGwnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmFuZG9tL3dlaWJ1bGwubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICd3ZWlidWxsX2xpa2UnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmFuZG9tL3dlaWJ1bGxfbGlrZS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3ppcGYnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmFuZG9tL3ppcGYubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICd6aXBmX2xpa2UnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmFuZG9tL3ppcGZfbGlrZS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3RyaWFuZ3VsYXInLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmFuZG9tL3RyaWFuZ3VsYXIubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICd0cmlhbmd1bGFyX2xpa2UnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmFuZG9tL3RyaWFuZ3VsYXJfbGlrZS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2Jlcm5vdWxsaScsIGxpbms6ICcvdXNlcl9ndWlkZS9yYW5kb20vYmVybm91bGxpLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAncmFuZGludCcsIGxpbms6ICcvdXNlcl9ndWlkZS9yYW5kb20vcmFuZGludC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3JhbmRpbnRfbGlrZScsIGxpbms6ICcvdXNlcl9ndWlkZS9yYW5kb20vcmFuZGludF9saWtlLm1kJyB9LFxyXG4gICAgICAgICAgICAgIF1cclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgIHRleHQ6ICdzaGFwZSBtYW5pcHVsYXRlJyxcclxuICAgICAgICAgICAgICBjb2xsYXBzaWJsZTogdHJ1ZSxcclxuICAgICAgICAgICAgICBjaGlsZHJlbjogW1xyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnc3F1ZWV6ZScsIGxpbms6ICcvdXNlcl9ndWlkZS9zaGFwZV9tYW5pcHVsYXRlL3NxdWVlemUubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICd1bnNxdWVlemUnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvc2hhcGVfbWFuaXB1bGF0ZS91bnNxdWVlemUubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdyZXNoYXBlJywgbGluazogJy91c2VyX2d1aWRlL3NoYXBlX21hbmlwdWxhdGUvcmVzaGFwZS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3RyYW5zcG9zZScsIGxpbms6ICcvdXNlcl9ndWlkZS9zaGFwZV9tYW5pcHVsYXRlL3RyYW5zcG9zZS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3Blcm11dGUnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvc2hhcGVfbWFuaXB1bGF0ZS9wZXJtdXRlLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAncGVybXV0ZV9pbnYnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvc2hhcGVfbWFuaXB1bGF0ZS9wZXJtdXRlX2ludi5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2V4cGFuZCcsIGxpbms6ICcvdXNlcl9ndWlkZS9zaGFwZV9tYW5pcHVsYXRlL2V4cGFuZC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3QnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvc2hhcGVfbWFuaXB1bGF0ZS90Lm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnbXQnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvc2hhcGVfbWFuaXB1bGF0ZS9tdC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2ZsaXAnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvc2hhcGVfbWFuaXB1bGF0ZS9mbGlwLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnZmxpcGxyJywgbGluazogJy91c2VyX2d1aWRlL3NoYXBlX21hbmlwdWxhdGUvZmxpcGxyLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnZmxpcHVkJywgbGluazogJy91c2VyX2d1aWRlL3NoYXBlX21hbmlwdWxhdGUvZmxpcHVkLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAndGlsZScsIGxpbms6ICcvdXNlcl9ndWlkZS9zaGFwZV9tYW5pcHVsYXRlL3RpbGUubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICd0cmltX3plcm9zJywgbGluazogJy91c2VyX2d1aWRlL3NoYXBlX21hbmlwdWxhdGUvdHJpbV96ZXJvcy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3JlcGVhdCcsIGxpbms6ICcvdXNlcl9ndWlkZS9zaGFwZV9tYW5pcHVsYXRlL3JlcGVhdC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3NwbGl0JywgbGluazogJy91c2VyX2d1aWRlL3NoYXBlX21hbmlwdWxhdGUvc3BsaXQubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdkc3BsaXQnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvc2hhcGVfbWFuaXB1bGF0ZS9kc3BsaXQubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdoc3BsaXQnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvc2hhcGVfbWFuaXB1bGF0ZS9oc3BsaXQubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICd2c3BsaXQnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvc2hhcGVfbWFuaXB1bGF0ZS92c3BsaXQubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdzd2FwX2F4ZXMnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvc2hhcGVfbWFuaXB1bGF0ZS9zd2FwX2F4ZXMubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdmbGF0dGVuJywgbGluazogJy91c2VyX2d1aWRlL3NoYXBlX21hbmlwdWxhdGUvZmxhdHRlbi5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2NvbmNhdCcsIGxpbms6ICcvdXNlcl9ndWlkZS9zaGFwZV9tYW5pcHVsYXRlL2NvbmNhdC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3ZzdGFjaycsIGxpbms6ICcvdXNlcl9ndWlkZS9zaGFwZV9tYW5pcHVsYXRlL3ZzdGFjay5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2hzdGFjaycsIGxpbms6ICcvdXNlcl9ndWlkZS9zaGFwZV9tYW5pcHVsYXRlL2hzdGFjay5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2RzdGFjaycsIGxpbms6ICcvdXNlcl9ndWlkZS9zaGFwZV9tYW5pcHVsYXRlL2RzdGFjay5tZCcgfSxcclxuICAgICAgICAgICAgICBdXHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIHtcclxuICAgICAgICAgICAgICB0ZXh0OiAnY3JlYXRpb24nLFxyXG4gICAgICAgICAgICAgIGNvbGxhcHNpYmxlOiB0cnVlLFxyXG4gICAgICAgICAgICAgIGNoaWxkcmVuOiBbXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdlbXB0eScsIGxpbms6ICcvdXNlcl9ndWlkZS9jcmVhdGlvbi9lbXB0eS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3plcm9zJywgbGluazogJy91c2VyX2d1aWRlL2NyZWF0aW9uL3plcm9zLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnb25lcycsIGxpbms6ICcvdXNlcl9ndWlkZS9jcmVhdGlvbi9vbmVzLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnZW1wdHlfbGlrZScsIGxpbms6ICcvdXNlcl9ndWlkZS9jcmVhdGlvbi9lbXB0eV9saWtlLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnemVyb3NfbGlrZScsIGxpbms6ICcvdXNlcl9ndWlkZS9jcmVhdGlvbi96ZXJvc19saWtlLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnb25lc19saWtlJywgbGluazogJy91c2VyX2d1aWRlL2NyZWF0aW9uL29uZXNfbGlrZS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2Z1bGwnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvY3JlYXRpb24vZnVsbC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2Z1bGxfbGlrZScsIGxpbms6ICcvdXNlcl9ndWlkZS9jcmVhdGlvbi9mdWxsX2xpa2UubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdhcmFuZ2UnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvY3JlYXRpb24vYXJhbmdlLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnYXJhbmdlX3N0ZXAnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvY3JlYXRpb24vYXJhbmdlX3N0ZXAubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdleWUnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvY3JlYXRpb24vZXllLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnbGluc3BhY2UnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvY3JlYXRpb24vbGluc3BhY2UubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdsb2dzcGFjZScsIGxpbms6ICcvdXNlcl9ndWlkZS9jcmVhdGlvbi9sb2dzcGFjZS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2dlb21zcGFjZScsIGxpbms6ICcvdXNlcl9ndWlkZS9jcmVhdGlvbi9nZW9tc3BhY2UubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICd0cmknLCBsaW5rOiAnL3VzZXJfZ3VpZGUvY3JlYXRpb24vdHJpLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAndHJpbCcsIGxpbms6ICcvdXNlcl9ndWlkZS9jcmVhdGlvbi90cmlsLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAndHJpdScsIGxpbms6ICcvdXNlcl9ndWlkZS9jcmVhdGlvbi90cml1Lm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnaWRlbnRpdHknLCBsaW5rOiAnL3VzZXJfZ3VpZGUvY3JlYXRpb24vaWRlbnRpdHkubWQnIH0sXHJcbiAgICAgICAgICAgICAgXVxyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICB7XHJcbiAgICAgICAgICAgICAgdGV4dDogJ3dpbmRvd3MnLFxyXG4gICAgICAgICAgICAgIGNvbGxhcHNpYmxlOiB0cnVlLFxyXG4gICAgICAgICAgICAgIGNoaWxkcmVuOiBbXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdoYW1taW5nX3dpbmRvdycsIGxpbms6ICcvdXNlcl9ndWlkZS93aW5kb3dzL2hhbW1pbmdfd2luZG93Lm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnaGFubl93aW5kb3cnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvd2luZG93cy9oYW5uX3dpbmRvdy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2JsYWNrbWFuX3dpbmRvdycsIGxpbms6ICcvdXNlcl9ndWlkZS93aW5kb3dzL2JsYWNrbWFuX3dpbmRvdy5tZCcgfSxcclxuICAgICAgICAgICAgICBdXHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIHtcclxuICAgICAgICAgICAgICB0ZXh0OiAnaXRlcmF0b3InLFxyXG4gICAgICAgICAgICAgIGNvbGxhcHNpYmxlOiB0cnVlLFxyXG4gICAgICAgICAgICAgIGNoaWxkcmVuOiBbXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdwYXJfaXRlcicsIGxpbms6ICcvdXNlcl9ndWlkZS9pdGVyYXRvci9wYXJfaXRlci5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3Bhcl9pdGVyX211dCcsIGxpbms6ICcvdXNlcl9ndWlkZS9pdGVyYXRvci9wYXJfaXRlcl9tdXQubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdwYXJfaXRlcl9zaW1kJywgbGluazogJy91c2VyX2d1aWRlL2l0ZXJhdG9yL3Bhcl9pdGVyX3NpbWQubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdwYXJfaXRlcl9zaW1kX211dCcsIGxpbms6ICcvdXNlcl9ndWlkZS9pdGVyYXRvci9wYXJfaXRlcl9zaW1kX211dC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3N0cmlkZWRfbWFwJywgbGluazogJy91c2VyX2d1aWRlL2l0ZXJhdG9yL3N0cmlkZWRfbWFwLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnc3RyaWRlZF9tYXBfc2ltZCcsIGxpbms6ICcvdXNlcl9ndWlkZS9pdGVyYXRvci9zdHJpZGVkX21hcF9zaW1kLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnY29sbGVjdCcsIGxpbms6ICcvdXNlcl9ndWlkZS9pdGVyYXRvci9jb2xsZWN0Lm1kJyB9LFxyXG4gICAgICAgICAgICAgIF1cclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgIHRleHQ6ICd1dGlscycsXHJcbiAgICAgICAgICAgICAgY29sbGFwc2libGU6IHRydWUsXHJcbiAgICAgICAgICAgICAgY2hpbGRyZW46IFtcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3NldF9kaXNwbGF5X2VsZW1lbnRzJywgbGluazogJy91c2VyX2d1aWRlL3V0aWxzL3NldF9kaXNwbGF5X2VsZW1lbnRzLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAncmVzaXplX2NwdV9scnVfY2FjaGUnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdXRpbHMvcmVzaXplX2NwdV9scnVfY2FjaGUubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdyZXNpemVfY3VkYV9scnVfY2FjaGUnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdXRpbHMvcmVzaXplX2N1ZGFfbHJ1X2NhY2hlLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnc2V0X3NlZWQnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdXRpbHMvc2V0X3NlZWQubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6IFwibnVtX3RocmVhZHNcIiwgbGluazogJy91c2VyX2d1aWRlL3V0aWxzL251bV90aHJlYWRzLm1kJyB9LFxyXG4gICAgICAgICAgICAgIF1cclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgIHRleHQ6ICdjdXN0b20gdHlwZScsXHJcbiAgICAgICAgICAgICAgbGluazogJy91c2VyX2d1aWRlL2N1c3RvbV90eXBlL2N1c3RvbV90eXBlLm1kJ1xyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICB7XHJcbiAgICAgICAgICAgICAgdGV4dDogJ2N1c3RvbSBhbGxvY2F0b3InLFxyXG4gICAgICAgICAgICAgIGxpbms6ICcvdXNlcl9ndWlkZS9jdXN0b21fYWxsb2NhdG9yL2N1c3RvbV9hbGxvY2F0b3IubWQnXHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIHtcclxuICAgICAgICAgICAgICB0ZXh0OiAnc2xpY2UnLFxyXG4gICAgICAgICAgICAgIGxpbms6ICcvdXNlcl9ndWlkZS9zbGljZS9zbGljZS5tZCdcclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgIHRleHQ6ICdzYXZlL2xvYWQnLFxyXG4gICAgICAgICAgICAgIGxpbms6ICcvdXNlcl9ndWlkZS9zYXZlX2xvYWQvc2F2ZV9sb2FkLm1kJ1xyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICBdXHJcbiAgICAgICAgfVxyXG4gICAgICBdLFxyXG5cclxuICAgICAgJy9kZXZfZ3VpZGUvJzogW1xyXG4gICAgICAgIHtcclxuICAgICAgICAgIHRleHQ6ICdEZXYgR3VpZGUnLFxyXG4gICAgICAgICAgY2hpbGRyZW46IFtcclxuICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgIHRleHQ6ICdhbGxvY2F0aW9uJyxcclxuICAgICAgICAgICAgICBsaW5rOiAnL2Rldl9ndWlkZS9hbGxvY2F0aW9uL2FsbG9jYXRvci5tZCdcclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgIHRleHQ6ICdOZXcgVHlwZScsXHJcbiAgICAgICAgICAgICAgbGluazogJy9kZXZfZ3VpZGUvbmV3X3R5cGUubWQnXHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIHtcclxuICAgICAgICAgICAgICB0ZXh0OiAndHlwZSBwcm9tb3RlJyxcclxuICAgICAgICAgICAgICBsaW5rOiAnL2Rldl9ndWlkZS90eXBlX3Byb21vdGUubWQnXHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIHtcclxuICAgICAgICAgICAgICB0ZXh0OiAncG9pbnRlcicsXHJcbiAgICAgICAgICAgICAgbGluazogJy9kZXZfZ3VpZGUvcG9pbnRlci9wb2ludGVyLm1kJ1xyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICB7XHJcbiAgICAgICAgICAgICAgdGV4dDogJ3Rlc3QgY2FzZXMnLFxyXG4gICAgICAgICAgICAgIGxpbms6ICcvZGV2X2d1aWRlL3Rlc3RfcnVsZXMubWQnXHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIHtcclxuICAgICAgICAgICAgICB0ZXh0OiAnaXRlcmF0b3InLFxyXG4gICAgICAgICAgICAgIGxpbms6ICcvZGV2X2d1aWRlL2l0ZXJhdG9yL2l0ZXJhdG9yLm1kJ1xyXG4gICAgICAgICAgICB9LCB7XHJcbiAgICAgICAgICAgICAgdGV4dDogJ05ldyBvcCcsXHJcbiAgICAgICAgICAgICAgbGluazogJy9kZXZfZ3VpZGUvYWRkaW5nX25ld19vcC5tZCdcclxuICAgICAgICAgICAgfSwge1xyXG4gICAgICAgICAgICAgIHRleHQ6ICdOZXcgYXJjaCBzdXBwb3J0JyxcclxuICAgICAgICAgICAgICBsaW5rOiAnL2Rldl9ndWlkZS9hZGRpbmdfbmV3X2FyY2gubWQnXHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgIF1cclxuICAgICAgICB9XHJcbiAgICAgIF0sXHJcbiAgICAgICcvYmVuY2htYXJrcy8nOiBbXHJcbiAgICAgICAge1xyXG4gICAgICAgICAgdGV4dDogJ0JlbmNobWFya3MnLFxyXG4gICAgICAgICAgY2hpbGRyZW46IFtcclxuICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgIHRleHQ6ICd1bmFyeScsXHJcbiAgICAgICAgICAgICAgbGluazogJy9iZW5jaG1hcmtzL3VuYXJ5Lm1kJ1xyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICB7XHJcbiAgICAgICAgICAgICAgdGV4dDogJ2JpbmFyeScsXHJcbiAgICAgICAgICAgICAgbGluazogJy9iZW5jaG1hcmtzL2JpbmFyeS5tZCdcclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgIHRleHQ6ICdyZWR1Y2UnLFxyXG4gICAgICAgICAgICAgIGxpbms6ICcvYmVuY2htYXJrcy9yZWR1Y2UubWQnXHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIHtcclxuICAgICAgICAgICAgICB0ZXh0OiAnY29udicsXHJcbiAgICAgICAgICAgICAgbGluazogJy9iZW5jaG1hcmtzL2NvbnYubWQnXHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIHtcclxuICAgICAgICAgICAgICB0ZXh0OiAncG9vbGluZycsXHJcbiAgICAgICAgICAgICAgbGluazogJy9iZW5jaG1hcmtzL3Bvb2xpbmcubWQnXHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIHtcclxuICAgICAgICAgICAgICB0ZXh0OiAnbm9ybWFsaXphdGlvbicsXHJcbiAgICAgICAgICAgICAgbGluazogJy9iZW5jaG1hcmtzL25vcm1hbGl6YXRpb24ubWQnXHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIHtcclxuICAgICAgICAgICAgICB0ZXh0OiAnbWF0bXVsJyxcclxuICAgICAgICAgICAgICBsaW5rOiAnL2JlbmNobWFya3MvbWF0bXVsLm1kJ1xyXG4gICAgICAgICAgICB9LCB7XHJcbiAgICAgICAgICAgICAgdGV4dDogJ2ZmdCcsXHJcbiAgICAgICAgICAgICAgbGluazogJy9iZW5jaG1hcmtzL2ZmdC5tZCdcclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgIHRleHQ6ICdubicsXHJcbiAgICAgICAgICAgICAgY29sbGFwc2libGU6IHRydWUsXHJcbiAgICAgICAgICAgICAgY2hpbGRyZW46IFtcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3Jlc25ldCcsIGxpbms6ICcvYmVuY2htYXJrcy9ubi9yZXNuZXQubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdsc3RtJywgbGluazogJy9iZW5jaG1hcmtzL25uL2xzdG0ubWQnIH1cclxuICAgICAgICAgICAgICBdXHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgIF1cclxuICAgICAgICB9XHJcbiAgICAgIF1cclxuXHJcbiAgICB9LFxyXG4gIH0pLFxyXG59KSJdLAogICJtYXBwaW5ncyI6ICI7QUFBNFEsU0FBUyx3QkFBd0I7QUFDN1MsU0FBUyxvQkFBb0I7QUFDN0IsU0FBUyxtQkFBbUI7QUFDNUIsU0FBUywwQkFBMEI7QUFDbkMsU0FBUyx1QkFBdUI7QUFFaEMsSUFBTyxpQkFBUSxpQkFBaUI7QUFBQSxFQUM5QixNQUFNLFFBQVEsSUFBSSxhQUFhLGdCQUMzQixNQUNBO0FBQUEsRUFDSixNQUFNO0FBQUEsRUFDTixPQUFPO0FBQUEsRUFDUCxTQUFTO0FBQUEsSUFDUCxtQkFBbUI7QUFBQSxNQUNqQixNQUFNO0FBQUEsSUFDUixDQUFDO0FBQUEsSUFDRCxnQkFBZ0I7QUFBQSxNQUNkLFNBQVM7QUFBQSxNQUNULFNBQVM7QUFBQSxJQUNYLENBQUM7QUFBQSxFQUNIO0FBQUEsRUFDQSxVQUFVO0FBQUEsSUFDUixRQUFRO0FBQUEsRUFDVjtBQUFBLEVBQ0EsU0FBUyxZQUFZO0FBQUEsRUFDckIsT0FBTyxhQUFhO0FBQUEsSUFDbEIsTUFBTTtBQUFBLElBQ04sUUFBUTtBQUFBLE1BQ047QUFBQSxRQUNFLE1BQU07QUFBQSxRQUNOLE1BQU07QUFBQSxNQUNSO0FBQUEsTUFDQTtBQUFBLFFBQ0UsTUFBTTtBQUFBLFFBQ04sTUFBTTtBQUFBLE1BQ1I7QUFBQSxNQUNBO0FBQUEsUUFDRSxNQUFNO0FBQUEsUUFDTixNQUFNO0FBQUEsTUFDUjtBQUFBLE1BQ0E7QUFBQSxRQUNFLE1BQU07QUFBQSxRQUNOLE1BQU07QUFBQSxNQUNSO0FBQUEsSUFDRjtBQUFBLElBRUEsU0FBUztBQUFBLE1BQ1AsZ0JBQWdCO0FBQUEsUUFDZDtBQUFBLFVBQ0UsTUFBTTtBQUFBLFVBQ04sVUFBVTtBQUFBLFlBQ1I7QUFBQSxjQUNFLE1BQU07QUFBQSxjQUNOLGFBQWE7QUFBQSxjQUNiLFVBQVU7QUFBQSxnQkFDUixFQUFFLE1BQU0sT0FBTyxNQUFNLDJCQUEyQjtBQUFBLGdCQUNoRCxFQUFFLE1BQU0sUUFBUSxNQUFNLDRCQUE0QjtBQUFBLGdCQUNsRCxFQUFFLE1BQU0sT0FBTyxNQUFNLDJCQUEyQjtBQUFBLGdCQUNoRCxFQUFFLE1BQU0sUUFBUSxNQUFNLDRCQUE0QjtBQUFBLGdCQUNsRCxFQUFFLE1BQU0sT0FBTyxNQUFNLDJCQUEyQjtBQUFBLGdCQUNoRCxFQUFFLE1BQU0sUUFBUSxNQUFNLDRCQUE0QjtBQUFBLGdCQUNsRCxFQUFFLE1BQU0sUUFBUSxNQUFNLDRCQUE0QjtBQUFBLGdCQUNsRCxFQUFFLE1BQU0sU0FBUyxNQUFNLDZCQUE2QjtBQUFBLGdCQUNwRCxFQUFFLE1BQU0sUUFBUSxNQUFNLDRCQUE0QjtBQUFBLGdCQUNsRCxFQUFFLE1BQU0sU0FBUyxNQUFNLDZCQUE2QjtBQUFBLGdCQUNwRCxFQUFFLE1BQU0sUUFBUSxNQUFNLDRCQUE0QjtBQUFBLGdCQUNsRCxFQUFFLE1BQU0sU0FBUyxNQUFNLDZCQUE2QjtBQUFBLGdCQUNwRCxFQUFFLE1BQU0sU0FBUyxNQUFNLDZCQUE2QjtBQUFBLGdCQUNwRCxFQUFFLE1BQU0sVUFBVSxNQUFNLDhCQUE4QjtBQUFBLGdCQUN0RCxFQUFFLE1BQU0sU0FBUyxNQUFNLDZCQUE2QjtBQUFBLGdCQUNwRCxFQUFFLE1BQU0sVUFBVSxNQUFNLDhCQUE4QjtBQUFBLGdCQUN0RCxFQUFFLE1BQU0sU0FBUyxNQUFNLDZCQUE2QjtBQUFBLGdCQUNwRCxFQUFFLE1BQU0sVUFBVSxNQUFNLDhCQUE4QjtBQUFBLGdCQUN0RCxFQUFFLE1BQU0sUUFBUSxNQUFNLDRCQUE0QjtBQUFBLGdCQUNsRCxFQUFFLE1BQU0sU0FBUyxNQUFNLDZCQUE2QjtBQUFBLGdCQUNwRCxFQUFFLE1BQU0sUUFBUSxNQUFNLDRCQUE0QjtBQUFBLGdCQUNsRCxFQUFFLE1BQU0sU0FBUyxNQUFNLDZCQUE2QjtBQUFBLGdCQUNwRCxFQUFFLE1BQU0sUUFBUSxNQUFNLDRCQUE0QjtBQUFBLGdCQUNsRCxFQUFFLE1BQU0sU0FBUyxNQUFNLDZCQUE2QjtBQUFBLGdCQUNwRCxFQUFFLE1BQU0sT0FBTyxNQUFNLDJCQUEyQjtBQUFBLGdCQUNoRCxFQUFFLE1BQU0sUUFBUSxNQUFNLDRCQUE0QjtBQUFBLGdCQUNsRCxFQUFFLE1BQU0sUUFBUSxNQUFNLDRCQUE0QjtBQUFBLGdCQUNsRCxFQUFFLE1BQU0sU0FBUyxNQUFNLDZCQUE2QjtBQUFBLGdCQUNwRCxFQUFFLE1BQU0sUUFBUSxNQUFNLDRCQUE0QjtBQUFBLGdCQUNsRCxFQUFFLE1BQU0sU0FBUyxNQUFNLDZCQUE2QjtBQUFBLGdCQUNwRCxFQUFFLE1BQU0sU0FBUyxNQUFNLDZCQUE2QjtBQUFBLGdCQUNwRCxFQUFFLE1BQU0sVUFBVSxNQUFNLDhCQUE4QjtBQUFBLGdCQUN0RCxFQUFFLE1BQU0sTUFBTSxNQUFNLDBCQUEwQjtBQUFBLGdCQUM5QyxFQUFFLE1BQU0sT0FBTyxNQUFNLDJCQUEyQjtBQUFBLGdCQUNoRCxFQUFFLE1BQU0sUUFBUSxNQUFNLDRCQUE0QjtBQUFBLGdCQUNsRCxFQUFFLE1BQU0sU0FBUyxNQUFNLDZCQUE2QjtBQUFBLGdCQUNwRCxFQUFFLE1BQU0sU0FBUyxNQUFNLDZCQUE2QjtBQUFBLGdCQUNwRCxFQUFFLE1BQU0sVUFBVSxNQUFNLDhCQUE4QjtBQUFBLGdCQUN0RCxFQUFFLE1BQU0sUUFBUSxNQUFNLDRCQUE0QjtBQUFBLGdCQUNsRCxFQUFFLE1BQU0sU0FB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RTtBQUFBLFlBQ0Y7QUFBQSxZQUNBO0FBQUEsY0FDRSxNQUFNO0FBQUEsY0FDTixNQUFNO0FBQUEsWUFDUjtBQUFBLFlBQ0E7QUFBQSxjQUNFLE1BQU07QUFBQSxjQUNOLE1BQU07QUFBQSxZQUNSO0FBQUEsWUFDQTtBQUFBLGNBQ0UsTUFBTTtBQUFBLGNBQ04sTUFBTTtBQUFBLFlBQ1I7QUFBQSxZQUNBO0FBQUEsY0FDRSxNQUFNO0FBQUEsY0FDTixNQUFNO0FBQUEsWUFDUjtBQUFBLFVBQ0Y7QUFBQSxRQUNGO0FBQUEsTUFDRjtBQUFBLE1BRUEsZUFBZTtBQUFBLFFBQ2I7QUFBQSxVQUNFLE1BQU07QUFBQSxVQUNOLFVBQVU7QUFBQSxZQUNSO0FBQUEsY0FDRSxNQUFNO0FBQUEsY0FDTixNQUFNO0FBQUEsWUFDUjtBQUFBLFlBQ0E7QUFBQSxjQUNFLE1BQU07QUFBQSxjQUNOLE1BQU07QUFBQSxZQUNSO0FBQUEsWUFDQTtBQUFBLGNBQ0UsTUFBTTtBQUFBLGNBQ04sTUFBTTtBQUFBLFlBQ1I7QUFBQSxZQUNBO0FBQUEsY0FDRSxNQUFNO0FBQUEsY0FDTixNQUFNO0FBQUEsWUFDUjtBQUFBLFlBQ0E7QUFBQSxjQUNFLE1BQU07QUFBQSxjQUNOLE1BQU07QUFBQSxZQUNSO0FBQUEsWUFDQTtBQUFBLGNBQ0UsTUFBTTtBQUFBLGNBQ04sTUFBTTtBQUFBLFlBQ1I7QUFBQSxZQUFHO0FBQUEsY0FDRCxNQUFNO0FBQUEsY0FDTixNQUFNO0FBQUEsWUFDUjtBQUFBLFlBQUc7QUFBQSxjQUNELE1BQU07QUFBQSxjQUNOLE1BQU07QUFBQSxZQUNSO0FBQUEsVUFDRjtBQUFBLFFBQ0Y7QUFBQSxNQUNGO0FBQUEsTUFDQSxnQkFBZ0I7QUFBQSxRQUNkO0FBQUEsVUFDRSxNQUFNO0FBQUEsVUFDTixVQUFVO0FBQUEsWUFDUjtBQUFBLGNBQ0UsTUFBTTtBQUFBLGNBQ04sTUFBTTtBQUFBLFlBQ1I7QUFBQSxZQUNBO0FBQUEsY0FDRSxNQUFNO0FBQUEsY0FDTixNQUFNO0FBQUEsWUFDUjtBQUFBLFlBQ0E7QUFBQSxjQUNFLE1BQU07QUFBQSxjQUNOLE1BQU07QUFBQSxZQUNSO0FBQUEsWUFDQTtBQUFBLGNBQ0UsTUFBTTtBQUFBLGNBQ04sTUFBTTtBQUFBLFlBQ1I7QUFBQSxZQUNBO0FBQUEsY0FDRSxNQUFNO0FBQUEsY0FDTixNQUFNO0FBQUEsWUFDUjtBQUFBLFlBQ0E7QUFBQSxjQUNFLE1BQU07QUFBQSxjQUNOLE1BQU07QUFBQSxZQUNSO0FBQUEsWUFDQTtBQUFBLGNBQ0UsTUFBTTtBQUFBLGNBQ04sTUFBTTtBQUFBLFlBQ1I7QUFBQSxZQUFHO0FBQUEsY0FDRCxNQUFNO0FBQUEsY0FDTixNQUFNO0FBQUEsWUFDUjtBQUFBLFlBQ0E7QUFBQSxjQUNFLE1BQU07QUFBQSxjQUNOLGFBQWE7QUFBQSxjQUNiLFVBQVU7QUFBQSxnQkFDUixFQUFFLE1BQU0sVUFBVSxNQUFNLDJCQUEyQjtBQUFBLGdCQUNuRCxFQUFFLE1BQU0sUUFBUSxNQUFNLHlCQUF5QjtBQUFBLGNBQ2pEO0FBQUEsWUFDRjtBQUFBLFVBQ0Y7QUFBQSxRQUNGO0FBQUEsTUFDRjtBQUFBLElBRUY7QUFBQSxFQUNGLENBQUM7QUFDSCxDQUFDOyIsCiAgIm5hbWVzIjogW10KfQo=
