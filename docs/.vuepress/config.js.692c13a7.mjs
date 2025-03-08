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
                { text: "div_", link: "/user_guide/binary/div_.md" }
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
                { text: "shrinkage", link: "/user_guide/advanced/shrinkage.md" },
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
                { text: "dropout", link: "/user_guide/normalization/dropout.md" },
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
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsiZG9jcy8udnVlcHJlc3MvY29uZmlnLmpzIl0sCiAgInNvdXJjZXNDb250ZW50IjogWyJjb25zdCBfX3ZpdGVfaW5qZWN0ZWRfb3JpZ2luYWxfZGlybmFtZSA9IFwiQzovVXNlcnMvMTIzL0hwdC9kb2NzLy52dWVwcmVzc1wiO2NvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9maWxlbmFtZSA9IFwiQzpcXFxcVXNlcnNcXFxcMTIzXFxcXEhwdFxcXFxkb2NzXFxcXC52dWVwcmVzc1xcXFxjb25maWcuanNcIjtjb25zdCBfX3ZpdGVfaW5qZWN0ZWRfb3JpZ2luYWxfaW1wb3J0X21ldGFfdXJsID0gXCJmaWxlOi8vL0M6L1VzZXJzLzEyMy9IcHQvZG9jcy8udnVlcHJlc3MvY29uZmlnLmpzXCI7aW1wb3J0IHsgZGVmaW5lVXNlckNvbmZpZyB9IGZyb20gJ0B2dWVwcmVzcy9jbGknXHJcbmltcG9ydCB7IGRlZmF1bHRUaGVtZSB9IGZyb20gJ0B2dWVwcmVzcy90aGVtZS1kZWZhdWx0J1xyXG5pbXBvcnQgeyB2aXRlQnVuZGxlciB9IGZyb20gJ0B2dWVwcmVzcy9idW5kbGVyLXZpdGUnXHJcbmltcG9ydCB7IG1hcmtkb3duTWF0aFBsdWdpbiB9IGZyb20gJ0B2dWVwcmVzcy9wbHVnaW4tbWFya2Rvd24tbWF0aCdcclxuaW1wb3J0IHsgbWRFbmhhbmNlUGx1Z2luIH0gZnJvbSBcInZ1ZXByZXNzLXBsdWdpbi1tZC1lbmhhbmNlXCI7XHJcblxyXG5leHBvcnQgZGVmYXVsdCBkZWZpbmVVc2VyQ29uZmlnKHtcclxuICBiYXNlOiBwcm9jZXNzLmVudi5OT0RFX0VOViA9PT0gJ2RldmVsb3BtZW50J1xyXG4gICAgPyAnLydcclxuICAgIDogJy9IcHQvJyxcclxuICBsYW5nOiAnemgtQ04nLFxyXG4gIHRpdGxlOiAnSHB0JyxcclxuICBwbHVnaW5zOiBbXHJcbiAgICBtYXJrZG93bk1hdGhQbHVnaW4oe1xyXG4gICAgICB0eXBlOiAna2F0ZXgnXHJcbiAgICB9KSxcclxuICAgIG1kRW5oYW5jZVBsdWdpbih7XHJcbiAgICAgIG1lcm1haWQ6IHRydWUsXHJcbiAgICAgIGNoYXJ0anM6IHRydWUsXHJcbiAgICB9KSxcclxuICBdLFxyXG4gIG1hcmtkb3duOiB7XHJcbiAgICBhbmNob3I6IGZhbHNlLFxyXG4gIH0sXHJcbiAgYnVuZGxlcjogdml0ZUJ1bmRsZXIoKSxcclxuICB0aGVtZTogZGVmYXVsdFRoZW1lKHtcclxuICAgIGhvbWU6IGZhbHNlLFxyXG4gICAgbmF2YmFyOiBbXHJcbiAgICAgIHtcclxuICAgICAgICB0ZXh0OiAnSG9tZScsXHJcbiAgICAgICAgbGluazogJy8nLFxyXG4gICAgICB9LFxyXG4gICAgICB7XHJcbiAgICAgICAgdGV4dDogJ0dpdEh1YicsXHJcbiAgICAgICAgbGluazogJ2h0dHBzOi8vZ2l0aHViLmNvbS9KaWFucW9xL0hwdCcsXHJcbiAgICAgIH0sXHJcbiAgICAgIHtcclxuICAgICAgICB0ZXh0OiAnY3JhdGUuaW8nLFxyXG4gICAgICAgIGxpbms6ICdodHRwczovL2NyYXRlcy5pby9jcmF0ZXMvaHB0JyxcclxuICAgICAgfSxcclxuICAgICAge1xyXG4gICAgICAgIHRleHQ6ICdCZW5jaG1hcmtzJyxcclxuICAgICAgICBsaW5rOiAnL2JlbmNobWFya3MvYmVuY2htYXJrcy5tZCcsXHJcbiAgICAgIH1cclxuICAgIF0sXHJcblxyXG4gICAgc2lkZWJhcjoge1xyXG4gICAgICAnL3VzZXJfZ3VpZGUvJzogW1xyXG4gICAgICAgIHtcclxuICAgICAgICAgIHRleHQ6ICdEb2NzJyxcclxuICAgICAgICAgIGNoaWxkcmVuOiBbXHJcbiAgICAgICAgICAgIHtcclxuICAgICAgICAgICAgICB0ZXh0OiAndW5hcnknLFxyXG4gICAgICAgICAgICAgIGNvbGxhcHNpYmxlOiB0cnVlLFxyXG4gICAgICAgICAgICAgIGNoaWxkcmVuOiBbXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdzaW4nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvc2luLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnc2luXycsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9zaW5fLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnY29zJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L2Nvcy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2Nvc18nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvY29zXy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3RhbicsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS90YW4ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICd0YW5fJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L3Rhbl8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdzaW5oJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L3NpbmgubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdzaW5oXycsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9zaW5oXy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2Nvc2gnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvY29zaC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2Nvc2hfJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L2Nvc2hfLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAndGFuaCcsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS90YW5oLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAndGFuaF8nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvdGFuaF8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdhc2luaCcsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9hc2luaC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2FzaW5oXycsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9hc2luaF8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdhY29zaCcsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9hY29zaC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2Fjb3NoXycsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9hY29zaF8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdhdGFuaCcsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9hdGFuaC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2F0YW5oXycsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9hdGFuaF8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdhc2luJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L2FzaW4ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdhc2luXycsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9hc2luXy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2Fjb3MnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvYWNvcy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2Fjb3NfJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L2Fjb3NfLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnYXRhbicsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9hdGFuLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnYXRhbl8nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvYXRhbl8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdleHAnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvZXhwLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnZXhwXycsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9leHBfLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnZXhwMicsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9leHAyLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnZXhwMl8nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvZXhwMl8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdzcXJ0JywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L3NxcnQubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdzcXJ0XycsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9zcXJ0Xy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3JlY2lwJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L3JlY2lwLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAncmVjaXBfJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L3JlY2lwXy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2xuJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L2xuLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnbG5fJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L2xuXy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2xvZzInLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvbG9nMi5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2xvZzJfJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L2xvZzJfLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnbG9nMTAnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvbG9nMTAubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdsb2cxMF8nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvbG9nMTBfLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnY2VsdScsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9jZWx1Lm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnY2VsdV8nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvY2VsdV8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdzaWdtb2lkJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L3NpZ21vaWQubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdzaWdtb2lkXycsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9zaWdtb2lkXy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2VsdScsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9lbHUubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdlbHVfJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L2VsdV8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdlcmYnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvZXJmLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnZ2VsdScsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9nZWx1Lm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnZ2VsdV8nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvZ2VsdV8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdzZWx1JywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L3NlbHUubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdzZWx1XycsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9zZWx1Xy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2hhcmRfc2lnbW9pZCcsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9oYXJkX3NpZ21vaWQubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdoYXJkX3NpZ21vaWRfJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L2hhcmRfc2lnbW9pZF8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdoYXJkX3N3aXNoJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L2hhcmRfc3dpc2gubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdoYXJkX3N3aXNoXycsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9oYXJkX3N3aXNoXy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3NvZnRwbHVzJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L3NvZnRwbHVzLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnc29mdHBsdXNfJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L3NvZnRwbHVzXy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3NvZnRzaWduJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L3NvZnRzaWduLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnc29mdHNpZ25fJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L3NvZnRzaWduXy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ21pc2gnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvbWlzaC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ21pc2hfJywgbGluazogJy91c2VyX2d1aWRlL3VuYXJ5L21pc2hfLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnY2JydCcsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9jYnJ0Lm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnY2JydF8nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvY2JydF8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdzaW5jb3MnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvc2luY29zLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnc2luY29zXycsIGxpbms6ICcvdXNlcl9ndWlkZS91bmFyeS9zaW5jb3NfLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnZXhwMTAnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvZXhwMTAubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdleHAxMF8nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvdW5hcnkvZXhwMTBfLm1kJyB9LFxyXG4gICAgICAgICAgICAgIF1cclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgIHRleHQ6ICdiaW5hcnknLFxyXG4gICAgICAgICAgICAgIGNvbGxhcHNpYmxlOiB0cnVlLFxyXG4gICAgICAgICAgICAgIGNoaWxkcmVuOiBbXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdhZGQnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvYmluYXJ5L2FkZC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2FkZF8nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvYmluYXJ5L2FkZF8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdzdWInLCBsaW5rOiAnL3VzZXJfZ3VpZGUvYmluYXJ5L3N1Yi5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3N1Yl8nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvYmluYXJ5L3N1Yl8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdtdWwnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvYmluYXJ5L211bC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ211bF8nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvYmluYXJ5L211bF8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdkaXYnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvYmluYXJ5L2Rpdi5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2Rpdl8nLCBsaW5rOiAnL3VzZXJfZ3VpZGUvYmluYXJ5L2Rpdl8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgXVxyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICB7XHJcbiAgICAgICAgICAgICAgdGV4dDogJ3JlZHVjZScsXHJcbiAgICAgICAgICAgICAgY29sbGFwc2libGU6IHRydWUsXHJcbiAgICAgICAgICAgICAgY2hpbGRyZW46IFtcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2xvZ3N1bWV4cCcsIGxpbms6ICcvdXNlcl9ndWlkZS9yZWR1Y2UvbG9nc3VtZXhwLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnYXJnbWluJywgbGluazogJy91c2VyX2d1aWRlL3JlZHVjZS9hcmdtaW4ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdhcmdtYXgnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmVkdWNlL2FyZ21heC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ21heCcsIGxpbms6ICcvdXNlcl9ndWlkZS9yZWR1Y2UvbWF4Lm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnbWluJywgbGluazogJy91c2VyX2d1aWRlL3JlZHVjZS9taW4ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdtZWFuJywgbGluazogJy91c2VyX2d1aWRlL3JlZHVjZS9tZWFuLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnc3VtJywgbGluazogJy91c2VyX2d1aWRlL3JlZHVjZS9zdW0ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdzdW1fJywgbGluazogJy91c2VyX2d1aWRlL3JlZHVjZS9zdW1fLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnbmFuc3VtJywgbGluazogJy91c2VyX2d1aWRlL3JlZHVjZS9uYW5zdW0ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICduYW5zdW1fJywgbGluazogJy91c2VyX2d1aWRlL3JlZHVjZS9uYW5zdW1fLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAncHJvZCcsIGxpbms6ICcvdXNlcl9ndWlkZS9yZWR1Y2UvcHJvZC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ25hbnByb2QnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmVkdWNlL25hbnByb2QubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdzdW1fc3F1YXJlJywgbGluazogJy91c2VyX2d1aWRlL3JlZHVjZS9zdW1fc3F1YXJlLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAncmVkdWNlbDEnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmVkdWNlL3JlZHVjZWwxLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAncmVkdWNlbDInLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmVkdWNlL3JlZHVjZWwyLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAncmVkdWNlbDMnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmVkdWNlL3JlZHVjZWwzLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnYWxsJywgbGluazogJy91c2VyX2d1aWRlL3JlZHVjZS9hbGwubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdhbnknLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmVkdWNlL2FueS5tZCcgfSxcclxuICAgICAgICAgICAgICBdXHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIHtcclxuICAgICAgICAgICAgICB0ZXh0OiBcImNvbnZcIixcclxuICAgICAgICAgICAgICBjb2xsYXBzaWJsZTogdHJ1ZSxcclxuICAgICAgICAgICAgICBjaGlsZHJlbjogW1xyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnYmF0Y2hub3JtX2NvbnYyZCcsIGxpbms6ICcvdXNlcl9ndWlkZS9jb252L2JhdGNobm9ybV9jb252MmQubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdjb252MmRfZ3JvdXAnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvY29udi9jb252MmRfZ3JvdXAubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdjb252MmRfdHJhbnNwb3NlJywgbGluazogJy91c2VyX2d1aWRlL2NvbnYvY29udjJkX3RyYW5zcG9zZS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2NvbnYyZCcsIGxpbms6ICcvdXNlcl9ndWlkZS9jb252L2NvbnYyZC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2R3Y29udjJkJywgbGluazogJy91c2VyX2d1aWRlL2NvbnYvZHdjb252MmQubWQnIH0sXHJcbiAgICAgICAgICAgICAgXVxyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICB7XHJcbiAgICAgICAgICAgICAgdGV4dDogXCJwb29saW5nXCIsXHJcbiAgICAgICAgICAgICAgY29sbGFwc2libGU6IHRydWUsXHJcbiAgICAgICAgICAgICAgY2hpbGRyZW46IFtcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ21heHBvb2wyZCcsIGxpbms6ICcvdXNlcl9ndWlkZS9wb29saW5nL21heHBvb2wyZC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2F2Z3Bvb2wyZCcsIGxpbms6ICcvdXNlcl9ndWlkZS9wb29saW5nL2F2Z3Bvb2wyZC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2FkYXB0aXZlX21heHBvb2wyZCcsIGxpbms6ICcvdXNlcl9ndWlkZS9wb29saW5nL2FkYXB0aXZlX21heHBvb2wyZC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2FkYXB0aXZlX2F2Z3Bvb2wyZCcsIGxpbms6ICcvdXNlcl9ndWlkZS9wb29saW5nL2FkYXB0aXZlX2F2Z3Bvb2wyZC5tZCcgfSxcclxuICAgICAgICAgICAgICBdXHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIHtcclxuICAgICAgICAgICAgICB0ZXh0OiAnY29tcGFyZScsXHJcbiAgICAgICAgICAgICAgY29sbGFwc2libGU6IHRydWUsXHJcbiAgICAgICAgICAgICAgY2hpbGRyZW46IFtcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3RlbnNvcl9uZXEnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvY21wL3RlbnNvcl9uZXEubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICd0ZW5zb3JfZXEnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvY21wL3RlbnNvcl9lcS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3RlbnNvcl9ndCcsIGxpbms6ICcvdXNlcl9ndWlkZS9jbXAvdGVuc29yX2d0Lm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAndGVuc29yX2x0JywgbGluazogJy91c2VyX2d1aWRlL2NtcC90ZW5zb3JfbHQubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICd0ZW5zb3JfZ2UnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvY21wL3RlbnNvcl9nZS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3RlbnNvcl9sZScsIGxpbms6ICcvdXNlcl9ndWlkZS9jbXAvdGVuc29yX2xlLm1kJyB9LFxyXG4gICAgICAgICAgICAgIF1cclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgIHRleHQ6ICdhZHZhbmNlZCcsXHJcbiAgICAgICAgICAgICAgY29sbGFwc2libGU6IHRydWUsXHJcbiAgICAgICAgICAgICAgY2hpbGRyZW46IFtcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3NjYXR0ZXInLCBsaW5rOiAnL3VzZXJfZ3VpZGUvYWR2YW5jZWQvc2NhdHRlci5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3Nocmlua2FnZScsIGxpbms6ICcvdXNlcl9ndWlkZS9hZHZhbmNlZC9zaHJpbmthZ2UubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdoYXJkbWF4JywgbGluazogJy91c2VyX2d1aWRlL2FkdmFuY2VkL2hhcmRtYXgubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICd0ZW5zb3Jfd2hlcmUnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvYWR2YW5jZWQvdGVuc29yX3doZXJlLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAndG9waycsIGxpbms6ICcvdXNlcl9ndWlkZS9hZHZhbmNlZC90b3BrLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnb25laG90JywgbGluazogJy91c2VyX2d1aWRlL2FkdmFuY2VkL29uZWhvdC5tZCcgfSxcclxuICAgICAgICAgICAgICBdXHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIHtcclxuICAgICAgICAgICAgICB0ZXh0OiAnbm9ybWFsaXphdGlvbicsXHJcbiAgICAgICAgICAgICAgY29sbGFwc2libGU6IHRydWUsXHJcbiAgICAgICAgICAgICAgY2hpbGRyZW46IFtcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2Ryb3BvdXQnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvbm9ybWFsaXphdGlvbi9kcm9wb3V0Lm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnbG9nX3NvZnRtYXgnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvbm9ybWFsaXphdGlvbi9sb2dfc29mdG1heC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2xheWVybm9ybScsIGxpbms6ICcvdXNlcl9ndWlkZS9ub3JtYWxpemF0aW9uL2xheWVybm9ybS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3NvZnRtYXgnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvbm9ybWFsaXphdGlvbi9zb2Z0bWF4Lm1kJyB9LFxyXG4gICAgICAgICAgICAgIF1cclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgIHRleHQ6ICdjdW11bGF0aXZlJyxcclxuICAgICAgICAgICAgICBjb2xsYXBzaWJsZTogdHJ1ZSxcclxuICAgICAgICAgICAgICBjaGlsZHJlbjogW1xyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnY3Vtc3VtJywgbGluazogJy91c2VyX2d1aWRlL2N1bXVsYXRpdmUvY3Vtc3VtLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnY3VtcHJvZCcsIGxpbms6ICcvdXNlcl9ndWlkZS9jdW11bGF0aXZlL2N1bXByb2QubWQnIH0sXHJcbiAgICAgICAgICAgICAgXVxyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICB7XHJcbiAgICAgICAgICAgICAgdGV4dDogJ2xpbmFsZycsXHJcbiAgICAgICAgICAgICAgY29sbGFwc2libGU6IHRydWUsXHJcbiAgICAgICAgICAgICAgY2hpbGRyZW46IFtcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ21hdG11bCcsIGxpbms6ICcvdXNlcl9ndWlkZS9saW5hbGcvbWF0bXVsLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAndGVuc29yZG90JywgbGluazogJy91c2VyX2d1aWRlL2xpbmFsZy90ZW5zb3Jkb3QubWQnIH0sXHJcbiAgICAgICAgICAgICAgXVxyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICB7XHJcbiAgICAgICAgICAgICAgdGV4dDogJ3JhbmRvbScsXHJcbiAgICAgICAgICAgICAgY29sbGFwc2libGU6IHRydWUsXHJcbiAgICAgICAgICAgICAgY2hpbGRyZW46IFtcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3JhbmRuJywgbGluazogJy91c2VyX2d1aWRlL3JhbmRvbS9yYW5kbi5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3JhbmRuX2xpa2UnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmFuZG9tL3JhbmRuX2xpa2UubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdyYW5kJywgbGluazogJy91c2VyX2d1aWRlL3JhbmRvbS9yYW5kLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAncmFuZF9saWtlJywgbGluazogJy91c2VyX2d1aWRlL3JhbmRvbS9yYW5kX2xpa2UubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdiZXRhJywgbGluazogJy91c2VyX2d1aWRlL3JhbmRvbS9iZXRhLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnYmV0YV9saWtlJywgbGluazogJy91c2VyX2d1aWRlL3JhbmRvbS9iZXRhX2xpa2UubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdjaGlzcXVhcmUnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmFuZG9tL2NoaXNxdWFyZS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2NoaXNxdWFyZV9saWtlJywgbGluazogJy91c2VyX2d1aWRlL3JhbmRvbS9jaGlzcXVhcmVfbGlrZS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2V4cG9uZW50aWFsJywgbGluazogJy91c2VyX2d1aWRlL3JhbmRvbS9leHBvbmVudGlhbC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2V4cG9uZW50aWFsX2xpa2UnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmFuZG9tL2V4cG9uZW50aWFsX2xpa2UubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdnYW1tYScsIGxpbms6ICcvdXNlcl9ndWlkZS9yYW5kb20vZ2FtbWEubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdnYW1tYV9saWtlJywgbGluazogJy91c2VyX2d1aWRlL3JhbmRvbS9nYW1tYV9saWtlLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnZ3VtYmVsJywgbGluazogJy91c2VyX2d1aWRlL3JhbmRvbS9ndW1iZWwubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdndW1iZWxfbGlrZScsIGxpbms6ICcvdXNlcl9ndWlkZS9yYW5kb20vZ3VtYmVsX2xpa2UubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdsb2dub3JtYWwnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmFuZG9tL2xvZ25vcm1hbC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2xvZ25vcm1hbF9saWtlJywgbGluazogJy91c2VyX2d1aWRlL3JhbmRvbS9sb2dub3JtYWxfbGlrZS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ25vcm1hbF9nYXVzc2lhbicsIGxpbms6ICcvdXNlcl9ndWlkZS9yYW5kb20vbm9ybWFsX2dhdXNzaWFuLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnbm9ybWFsX2dhdXNzaWFuX2xpa2UnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmFuZG9tL25vcm1hbF9nYXVzc2lhbl9saWtlLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAncGFyZXRvJywgbGluazogJy91c2VyX2d1aWRlL3JhbmRvbS9wYXJldG8ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdwYXJldG9fbGlrZScsIGxpbms6ICcvdXNlcl9ndWlkZS9yYW5kb20vcGFyZXRvX2xpa2UubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdwb2lzc29uJywgbGluazogJy91c2VyX2d1aWRlL3JhbmRvbS9wb2lzc29uLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAncG9pc3Nvbl9saWtlJywgbGluazogJy91c2VyX2d1aWRlL3JhbmRvbS9wb2lzc29uX2xpa2UubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICd3ZWlidWxsJywgbGluazogJy91c2VyX2d1aWRlL3JhbmRvbS93ZWlidWxsLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnd2VpYnVsbF9saWtlJywgbGluazogJy91c2VyX2d1aWRlL3JhbmRvbS93ZWlidWxsX2xpa2UubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICd6aXBmJywgbGluazogJy91c2VyX2d1aWRlL3JhbmRvbS96aXBmLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnemlwZl9saWtlJywgbGluazogJy91c2VyX2d1aWRlL3JhbmRvbS96aXBmX2xpa2UubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICd0cmlhbmd1bGFyJywgbGluazogJy91c2VyX2d1aWRlL3JhbmRvbS90cmlhbmd1bGFyLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAndHJpYW5ndWxhcl9saWtlJywgbGluazogJy91c2VyX2d1aWRlL3JhbmRvbS90cmlhbmd1bGFyX2xpa2UubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdiZXJub3VsbGknLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmFuZG9tL2Jlcm5vdWxsaS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3JhbmRpbnQnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmFuZG9tL3JhbmRpbnQubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdyYW5kaW50X2xpa2UnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvcmFuZG9tL3JhbmRpbnRfbGlrZS5tZCcgfSxcclxuICAgICAgICAgICAgICBdXHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIHtcclxuICAgICAgICAgICAgICB0ZXh0OiAnc2hhcGUgbWFuaXB1bGF0ZScsXHJcbiAgICAgICAgICAgICAgY29sbGFwc2libGU6IHRydWUsXHJcbiAgICAgICAgICAgICAgY2hpbGRyZW46IFtcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3NxdWVlemUnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvc2hhcGVfbWFuaXB1bGF0ZS9zcXVlZXplLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAndW5zcXVlZXplJywgbGluazogJy91c2VyX2d1aWRlL3NoYXBlX21hbmlwdWxhdGUvdW5zcXVlZXplLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAncmVzaGFwZScsIGxpbms6ICcvdXNlcl9ndWlkZS9zaGFwZV9tYW5pcHVsYXRlL3Jlc2hhcGUubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICd0cmFuc3Bvc2UnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvc2hhcGVfbWFuaXB1bGF0ZS90cmFuc3Bvc2UubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdwZXJtdXRlJywgbGluazogJy91c2VyX2d1aWRlL3NoYXBlX21hbmlwdWxhdGUvcGVybXV0ZS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3Blcm11dGVfaW52JywgbGluazogJy91c2VyX2d1aWRlL3NoYXBlX21hbmlwdWxhdGUvcGVybXV0ZV9pbnYubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdleHBhbmQnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvc2hhcGVfbWFuaXB1bGF0ZS9leHBhbmQubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICd0JywgbGluazogJy91c2VyX2d1aWRlL3NoYXBlX21hbmlwdWxhdGUvdC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ210JywgbGluazogJy91c2VyX2d1aWRlL3NoYXBlX21hbmlwdWxhdGUvbXQubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdmbGlwJywgbGluazogJy91c2VyX2d1aWRlL3NoYXBlX21hbmlwdWxhdGUvZmxpcC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2ZsaXBscicsIGxpbms6ICcvdXNlcl9ndWlkZS9zaGFwZV9tYW5pcHVsYXRlL2ZsaXBsci5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2ZsaXB1ZCcsIGxpbms6ICcvdXNlcl9ndWlkZS9zaGFwZV9tYW5pcHVsYXRlL2ZsaXB1ZC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3RpbGUnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvc2hhcGVfbWFuaXB1bGF0ZS90aWxlLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAndHJpbV96ZXJvcycsIGxpbms6ICcvdXNlcl9ndWlkZS9zaGFwZV9tYW5pcHVsYXRlL3RyaW1femVyb3MubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdyZXBlYXQnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvc2hhcGVfbWFuaXB1bGF0ZS9yZXBlYXQubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdzcGxpdCcsIGxpbms6ICcvdXNlcl9ndWlkZS9zaGFwZV9tYW5pcHVsYXRlL3NwbGl0Lm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnZHNwbGl0JywgbGluazogJy91c2VyX2d1aWRlL3NoYXBlX21hbmlwdWxhdGUvZHNwbGl0Lm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnaHNwbGl0JywgbGluazogJy91c2VyX2d1aWRlL3NoYXBlX21hbmlwdWxhdGUvaHNwbGl0Lm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAndnNwbGl0JywgbGluazogJy91c2VyX2d1aWRlL3NoYXBlX21hbmlwdWxhdGUvdnNwbGl0Lm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnc3dhcF9heGVzJywgbGluazogJy91c2VyX2d1aWRlL3NoYXBlX21hbmlwdWxhdGUvc3dhcF9heGVzLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnZmxhdHRlbicsIGxpbms6ICcvdXNlcl9ndWlkZS9zaGFwZV9tYW5pcHVsYXRlL2ZsYXR0ZW4ubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdjb25jYXQnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvc2hhcGVfbWFuaXB1bGF0ZS9jb25jYXQubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICd2c3RhY2snLCBsaW5rOiAnL3VzZXJfZ3VpZGUvc2hhcGVfbWFuaXB1bGF0ZS92c3RhY2subWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdoc3RhY2snLCBsaW5rOiAnL3VzZXJfZ3VpZGUvc2hhcGVfbWFuaXB1bGF0ZS9oc3RhY2subWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdkc3RhY2snLCBsaW5rOiAnL3VzZXJfZ3VpZGUvc2hhcGVfbWFuaXB1bGF0ZS9kc3RhY2subWQnIH0sXHJcbiAgICAgICAgICAgICAgXVxyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICB7XHJcbiAgICAgICAgICAgICAgdGV4dDogJ2NyZWF0aW9uJyxcclxuICAgICAgICAgICAgICBjb2xsYXBzaWJsZTogdHJ1ZSxcclxuICAgICAgICAgICAgICBjaGlsZHJlbjogW1xyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnZW1wdHknLCBsaW5rOiAnL3VzZXJfZ3VpZGUvY3JlYXRpb24vZW1wdHkubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICd6ZXJvcycsIGxpbms6ICcvdXNlcl9ndWlkZS9jcmVhdGlvbi96ZXJvcy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ29uZXMnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvY3JlYXRpb24vb25lcy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2VtcHR5X2xpa2UnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvY3JlYXRpb24vZW1wdHlfbGlrZS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3plcm9zX2xpa2UnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvY3JlYXRpb24vemVyb3NfbGlrZS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ29uZXNfbGlrZScsIGxpbms6ICcvdXNlcl9ndWlkZS9jcmVhdGlvbi9vbmVzX2xpa2UubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdmdWxsJywgbGluazogJy91c2VyX2d1aWRlL2NyZWF0aW9uL2Z1bGwubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdmdWxsX2xpa2UnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvY3JlYXRpb24vZnVsbF9saWtlLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnYXJhbmdlJywgbGluazogJy91c2VyX2d1aWRlL2NyZWF0aW9uL2FyYW5nZS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2FyYW5nZV9zdGVwJywgbGluazogJy91c2VyX2d1aWRlL2NyZWF0aW9uL2FyYW5nZV9zdGVwLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnZXllJywgbGluazogJy91c2VyX2d1aWRlL2NyZWF0aW9uL2V5ZS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2xpbnNwYWNlJywgbGluazogJy91c2VyX2d1aWRlL2NyZWF0aW9uL2xpbnNwYWNlLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnbG9nc3BhY2UnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvY3JlYXRpb24vbG9nc3BhY2UubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdnZW9tc3BhY2UnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvY3JlYXRpb24vZ2VvbXNwYWNlLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAndHJpJywgbGluazogJy91c2VyX2d1aWRlL2NyZWF0aW9uL3RyaS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3RyaWwnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvY3JlYXRpb24vdHJpbC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3RyaXUnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvY3JlYXRpb24vdHJpdS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2lkZW50aXR5JywgbGluazogJy91c2VyX2d1aWRlL2NyZWF0aW9uL2lkZW50aXR5Lm1kJyB9LFxyXG4gICAgICAgICAgICAgIF1cclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgIHRleHQ6ICd3aW5kb3dzJyxcclxuICAgICAgICAgICAgICBjb2xsYXBzaWJsZTogdHJ1ZSxcclxuICAgICAgICAgICAgICBjaGlsZHJlbjogW1xyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnaGFtbWluZ193aW5kb3cnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvd2luZG93cy9oYW1taW5nX3dpbmRvdy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2hhbm5fd2luZG93JywgbGluazogJy91c2VyX2d1aWRlL3dpbmRvd3MvaGFubl93aW5kb3cubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdibGFja21hbl93aW5kb3cnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvd2luZG93cy9ibGFja21hbl93aW5kb3cubWQnIH0sXHJcbiAgICAgICAgICAgICAgXVxyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICB7XHJcbiAgICAgICAgICAgICAgdGV4dDogJ2l0ZXJhdG9yJyxcclxuICAgICAgICAgICAgICBjb2xsYXBzaWJsZTogdHJ1ZSxcclxuICAgICAgICAgICAgICBjaGlsZHJlbjogW1xyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAncGFyX2l0ZXInLCBsaW5rOiAnL3VzZXJfZ3VpZGUvaXRlcmF0b3IvcGFyX2l0ZXIubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdwYXJfaXRlcl9tdXQnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvaXRlcmF0b3IvcGFyX2l0ZXJfbXV0Lm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAncGFyX2l0ZXJfc2ltZCcsIGxpbms6ICcvdXNlcl9ndWlkZS9pdGVyYXRvci9wYXJfaXRlcl9zaW1kLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAncGFyX2l0ZXJfc2ltZF9tdXQnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvaXRlcmF0b3IvcGFyX2l0ZXJfc2ltZF9tdXQubWQnIH0sXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdzdHJpZGVkX21hcCcsIGxpbms6ICcvdXNlcl9ndWlkZS9pdGVyYXRvci9zdHJpZGVkX21hcC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3N0cmlkZWRfbWFwX3NpbWQnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvaXRlcmF0b3Ivc3RyaWRlZF9tYXBfc2ltZC5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ2NvbGxlY3QnLCBsaW5rOiAnL3VzZXJfZ3VpZGUvaXRlcmF0b3IvY29sbGVjdC5tZCcgfSxcclxuICAgICAgICAgICAgICBdXHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIHtcclxuICAgICAgICAgICAgICB0ZXh0OiAndXRpbHMnLFxyXG4gICAgICAgICAgICAgIGNvbGxhcHNpYmxlOiB0cnVlLFxyXG4gICAgICAgICAgICAgIGNoaWxkcmVuOiBbXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdzZXRfZGlzcGxheV9lbGVtZW50cycsIGxpbms6ICcvdXNlcl9ndWlkZS91dGlscy9zZXRfZGlzcGxheV9lbGVtZW50cy5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3Jlc2l6ZV9jcHVfbHJ1X2NhY2hlJywgbGluazogJy91c2VyX2d1aWRlL3V0aWxzL3Jlc2l6ZV9jcHVfbHJ1X2NhY2hlLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAncmVzaXplX2N1ZGFfbHJ1X2NhY2hlJywgbGluazogJy91c2VyX2d1aWRlL3V0aWxzL3Jlc2l6ZV9jdWRhX2xydV9jYWNoZS5tZCcgfSxcclxuICAgICAgICAgICAgICAgIHsgdGV4dDogJ3NldF9zZWVkJywgbGluazogJy91c2VyX2d1aWRlL3V0aWxzL3NldF9zZWVkLm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiBcIm51bV90aHJlYWRzXCIsIGxpbms6ICcvdXNlcl9ndWlkZS91dGlscy9udW1fdGhyZWFkcy5tZCcgfSxcclxuICAgICAgICAgICAgICBdXHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIHtcclxuICAgICAgICAgICAgICB0ZXh0OiAnY3VzdG9tIHR5cGUnLFxyXG4gICAgICAgICAgICAgIGxpbms6ICcvdXNlcl9ndWlkZS9jdXN0b21fdHlwZS9jdXN0b21fdHlwZS5tZCdcclxuICAgICAgICAgICAgfSwgXHJcbiAgICAgICAgICAgIHtcclxuICAgICAgICAgICAgICB0ZXh0OiAnY3VzdG9tIGFsbG9jYXRvcicsXHJcbiAgICAgICAgICAgICAgbGluazogJy91c2VyX2d1aWRlL2N1c3RvbV9hbGxvY2F0b3IvY3VzdG9tX2FsbG9jYXRvci5tZCdcclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgIHRleHQ6ICdzbGljZScsXHJcbiAgICAgICAgICAgICAgbGluazogJy91c2VyX2d1aWRlL3NsaWNlL3NsaWNlLm1kJ1xyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICB7XHJcbiAgICAgICAgICAgICAgdGV4dDogJ3NhdmUvbG9hZCcsXHJcbiAgICAgICAgICAgICAgbGluazogJy91c2VyX2d1aWRlL3NhdmVfbG9hZC9zYXZlX2xvYWQubWQnXHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgIF1cclxuICAgICAgICB9XHJcbiAgICAgIF0sXHJcblxyXG4gICAgICAnL2Rldl9ndWlkZS8nOiBbXHJcbiAgICAgICAge1xyXG4gICAgICAgICAgdGV4dDogJ0RldiBHdWlkZScsXHJcbiAgICAgICAgICBjaGlsZHJlbjogW1xyXG4gICAgICAgICAgICB7XHJcbiAgICAgICAgICAgICAgdGV4dDogJ2FsbG9jYXRpb24nLFxyXG4gICAgICAgICAgICAgIGxpbms6ICcvZGV2X2d1aWRlL2FsbG9jYXRpb24vYWxsb2NhdG9yLm1kJ1xyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICB7XHJcbiAgICAgICAgICAgICAgdGV4dDogJ05ldyBUeXBlJyxcclxuICAgICAgICAgICAgICBsaW5rOiAnL2Rldl9ndWlkZS9uZXdfdHlwZS5tZCdcclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgIHRleHQ6ICd0eXBlIHByb21vdGUnLFxyXG4gICAgICAgICAgICAgIGxpbms6ICcvZGV2X2d1aWRlL3R5cGVfcHJvbW90ZS5tZCdcclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgIHRleHQ6ICdwb2ludGVyJyxcclxuICAgICAgICAgICAgICBsaW5rOiAnL2Rldl9ndWlkZS9wb2ludGVyL3BvaW50ZXIubWQnXHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIHtcclxuICAgICAgICAgICAgICB0ZXh0OiAndGVzdCBjYXNlcycsXHJcbiAgICAgICAgICAgICAgbGluazogJy9kZXZfZ3VpZGUvdGVzdF9ydWxlcy5tZCdcclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgIHRleHQ6ICdpdGVyYXRvcicsXHJcbiAgICAgICAgICAgICAgbGluazogJy9kZXZfZ3VpZGUvaXRlcmF0b3IvaXRlcmF0b3IubWQnXHJcbiAgICAgICAgICAgIH0sIHtcclxuICAgICAgICAgICAgICB0ZXh0OiAnTmV3IG9wJyxcclxuICAgICAgICAgICAgICBsaW5rOiAnL2Rldl9ndWlkZS9hZGRpbmdfbmV3X29wLm1kJ1xyXG4gICAgICAgICAgICB9LCB7XHJcbiAgICAgICAgICAgICAgdGV4dDogJ05ldyBhcmNoIHN1cHBvcnQnLFxyXG4gICAgICAgICAgICAgIGxpbms6ICcvZGV2X2d1aWRlL2FkZGluZ19uZXdfYXJjaC5tZCdcclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgXVxyXG4gICAgICAgIH1cclxuICAgICAgXSxcclxuICAgICAgJy9iZW5jaG1hcmtzLyc6IFtcclxuICAgICAgICB7XHJcbiAgICAgICAgICB0ZXh0OiAnQmVuY2htYXJrcycsXHJcbiAgICAgICAgICBjaGlsZHJlbjogW1xyXG4gICAgICAgICAgICB7XHJcbiAgICAgICAgICAgICAgdGV4dDogJ3VuYXJ5JyxcclxuICAgICAgICAgICAgICBsaW5rOiAnL2JlbmNobWFya3MvdW5hcnkubWQnXHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIHtcclxuICAgICAgICAgICAgICB0ZXh0OiAnYmluYXJ5JyxcclxuICAgICAgICAgICAgICBsaW5rOiAnL2JlbmNobWFya3MvYmluYXJ5Lm1kJ1xyXG4gICAgICAgICAgICB9LFxyXG4gICAgICAgICAgICB7XHJcbiAgICAgICAgICAgICAgdGV4dDogJ3JlZHVjZScsXHJcbiAgICAgICAgICAgICAgbGluazogJy9iZW5jaG1hcmtzL3JlZHVjZS5tZCdcclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgIHRleHQ6ICdjb252JyxcclxuICAgICAgICAgICAgICBsaW5rOiAnL2JlbmNobWFya3MvY29udi5tZCdcclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgIHRleHQ6ICdwb29saW5nJyxcclxuICAgICAgICAgICAgICBsaW5rOiAnL2JlbmNobWFya3MvcG9vbGluZy5tZCdcclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgIHRleHQ6ICdub3JtYWxpemF0aW9uJyxcclxuICAgICAgICAgICAgICBsaW5rOiAnL2JlbmNobWFya3Mvbm9ybWFsaXphdGlvbi5tZCdcclxuICAgICAgICAgICAgfSxcclxuICAgICAgICAgICAge1xyXG4gICAgICAgICAgICAgIHRleHQ6ICdtYXRtdWwnLFxyXG4gICAgICAgICAgICAgIGxpbms6ICcvYmVuY2htYXJrcy9tYXRtdWwubWQnXHJcbiAgICAgICAgICAgIH0sXHJcbiAgICAgICAgICAgIHtcclxuICAgICAgICAgICAgICB0ZXh0OiAnbm4nLFxyXG4gICAgICAgICAgICAgIGNvbGxhcHNpYmxlOiB0cnVlLFxyXG4gICAgICAgICAgICAgIGNoaWxkcmVuOiBbXHJcbiAgICAgICAgICAgICAgICB7IHRleHQ6ICdyZXNuZXQnLCBsaW5rOiAnL2JlbmNobWFya3Mvbm4vcmVzbmV0Lm1kJyB9LFxyXG4gICAgICAgICAgICAgICAgeyB0ZXh0OiAnbHN0bScsIGxpbms6ICcvYmVuY2htYXJrcy9ubi9sc3RtLm1kJyB9XHJcbiAgICAgICAgICAgICAgXVxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICBdXHJcbiAgICAgICAgfVxyXG4gICAgICBdXHJcblxyXG4gICAgfSxcclxuICB9KSxcclxufSkiXSwKICAibWFwcGluZ3MiOiAiO0FBQTRRLFNBQVMsd0JBQXdCO0FBQzdTLFNBQVMsb0JBQW9CO0FBQzdCLFNBQVMsbUJBQW1CO0FBQzVCLFNBQVMsMEJBQTBCO0FBQ25DLFNBQVMsdUJBQXVCO0FBRWhDLElBQU8saUJBQVEsaUJBQWlCO0FBQUEsRUFDOUIsTUFBTSxRQUFRLElBQUksYUFBYSxnQkFDM0IsTUFDQTtBQUFBLEVBQ0osTUFBTTtBQUFBLEVBQ04sT0FBTztBQUFBLEVBQ1AsU0FBUztBQUFBLElBQ1AsbUJBQW1CO0FBQUEsTUFDakIsTUFBTTtBQUFBLElBQ1IsQ0FBQztBQUFBLElBQ0QsZ0JBQWdCO0FBQUEsTUFDZCxTQUFTO0FBQUEsTUFDVCxTQUFTO0FBQUEsSUFDWCxDQUFDO0FBQUEsRUFDSDtBQUFBLEVBQ0EsVUFBVTtBQUFBLElBQ1IsUUFBUTtBQUFBLEVBQ1Y7QUFBQSxFQUNBLFNBQVMsWUFBWTtBQUFBLEVBQ3JCLE9BQU8sYUFBYTtBQUFBLElBQ2xCLE1BQU07QUFBQSxJQUNOLFFBQVE7QUFBQSxNQUNOO0FBQUEsUUFDRSxNQUFNO0FBQUEsUUFDTixNQUFNO0FBQUEsTUFDUjtBQUFBLE1BQ0E7QUFBQSxRQUNFLE1BQU07QUFBQSxRQUNOLE1BQU07QUFBQSxNQUNSO0FBQUEsTUFDQTtBQUFBLFFBQ0UsTUFBTTtBQUFBLFFBQ04sTUFBTTtBQUFBLE1BQ1I7QUFBQSxNQUNBO0FBQUEsUUFDRSxNQUFNO0FBQUEsUUFDTixNQUFNO0FBQUEsTUFDUjtBQUFBLElBQ0Y7QUFBQSxJQUVBLFNBQVM7QUFBQSxNQUNQLGdCQUFnQjtBQUFBLFFBQ2Q7QUFBQSxVQUNFLE1BQU07QUFBQSxVQUNOLFVBQVU7QUFBQSxZQUNSO0FBQUEsY0FDRSxNQUFNO0FBQUEsY0FDTixhQUFhO0FBQUEsY0FDYixVQUFVO0FBQUEsZ0JBQ1IsRUFBRSxNQUFNLE9BQU8sTUFBTSwyQkFBMkI7QUFBQSxnQkFDaEQsRUFBRSxNQUFNLFFBQVEsTUFBTSw0QkFBNEI7QUFBQSxnQkFDbEQsRUFBRSxNQUFNLE9BQU8sTUFBTSwyQkFBMkI7QUFBQSxnQkFDaEQsRUFBRSxNQUFNLFFBQVEsTUFBTSw0QkFBNEI7QUFBQSxnQkFDbEQsRUFBRSxNQUFNLE9BQU8sTUFBTSwyQkFBMkI7QUFBQSxnQkFDaEQsRUFBRSxNQUFNLFFBQVEsTUFBTSw0QkFBNEI7QUFBQSxnQkFDbEQsRUFBRSxNQUFNLFFBQVEsTUFBTSw0QkFBNEI7QUFBQSxnQkFDbEQsRUFBRSxNQUFNLFNBQVMsTUFBTSw2QkFBNkI7QUFBQSxnQkFDcEQsRUFBRSxNQUFNLFFBQVEsTUFBTSw0QkFBNEI7QUFBQSxnQkFDbEQsRUFBRSxNQUFNLFNBQVMsTUFBTSw2QkFBNkI7QUFBQSxnQkFDcEQsRUFBRSxNQUFNLFFBQVEsTUFBTSw0QkFBNEI7QUFBQSxnQkFDbEQsRUFBRSxNQUFNLFNBQVMsTUFBTSw2QkFBNkI7QUFBQSxnQkFDcEQsRUFBRSxNQUFNLFNBQVMsTUFBTSw2QkFBNkI7QUFBQSxnQkFDcEQsRUFBRSxNQUFNLFVBQVUsTUFBTSw4QkFBOEI7QUFBQSxnQkFDdEQsRUFBRSxNQUFNLFNBQVMsTUFBTSw2QkFBNkI7QUFBQSxnQkFDcEQsRUFBRSxNQUFNLFVBQVUsTUFBTSw4QkFBOEI7QUFBQSxnQkFDdEQsRUFBRSxNQUFNLFNBQVMsTUFBTSw2QkFBNkI7QUFBQSxnQkFDcEQsRUFBRSxNQUFNLFVBQVUsTUFBTSw4QkFBOEI7QUFBQSxnQkFDdEQsRUFBRSxNQUFNLFFBQVEsTUFBTSw0QkFBNEI7QUFBQSxnQkFDbEQsRUFBRSxNQUFNLFNBQVMsTUFBTSw2QkFBNkI7QUFBQSxnQkFDcEQsRUFBRSxNQUFNLFFBQVEsTUFBTSw0QkFBNEI7QUFBQSxnQkFDbEQsRUFBRSxNQUFNLFNBQVMsTUFBTSw2QkFBNkI7QUFBQSxnQkFDcEQsRUFBRSxNQUFNLFFBQVEsTUFBTSw0QkFBNEI7QUFBQSxnQkFDbEQsRUFBRSxNQUFNLFNBQVMsTUFBTSw2QkFBNkI7QUFBQSxnQkFDcEQsRUFBRSxNQUFNLE9BQU8sTUFBTSwyQkFBMkI7QUFBQSxnQkFDaEQsRUFBRSxNQUFNLFFBQVEsTUFBTSw0QkFBNEI7QUFBQSxnQkFDbEQsRUFBRSxNQUFNLFFBQVEsTUFBTSw0QkFBNEI7QUFBQSxnQkFDbEQsRUFBRSxNQUFNLFNBQVMsTUFBTSw2QkFBNkI7QUFBQSxnQkFDcEQsRUFBRSxNQUFNLFFBQVEsTUFBTSw0QkFBNEI7QUFBQSxnQkFDbEQsRUFBRSxNQUFNLFNBQVMsTUFBTSw2QkFBNkI7QUFBQSxnQkFDcEQsRUFBRSxNQUFNLFNBQVMsTUFBTSw2QkFBNkI7QUFBQSxnQkFDcEQsRUFBRSxNQUFNLFVBQVUsTUFBTSw4QkFBOEI7QUFBQSxnQkFDdEQsRUFBRSxNQUFNLE1BQU0sTUFBTSwwQkFBMEI7QUFBQSxnQkFDOUMsRUFBRSxNQUFNLE9BQU8sTUFBTSwyQkFBMkI7QUFBQSxnQkFDaEQsRUFBRSxNQUFNLFFBQVEsTUFBTSw0QkFBNEI7QUFBQSxnQkFDbEQsRUFBRSxNQUFNLFNBQVMsTUFBTSw2QkFBNkI7QUFBQSxnQkFDcEQsRUFBRSxNQUFNLFNBQVMsTUFBTSw2QkFBNkI7QUFBQSxnQkFDcEQsRUFBRSxNQUFNLFVBQVUsTUFBTSw4QkFBOEI7QUFBQSxnQkFDdEQsRUFBRSxNQUFNLFFBQVEsTUFBTSw0QkFBNEI7QUFBQSxnQkFDbEQsRUFBRSxNQUFNLFNBQVMsTUFBTSw2QkFBNkI7QUFBQSxnQkFDcEQsRUFBRSxNQUFNLFdBQVcsTUFBTSwrQkFBK0I7QUFBQSxnQkFDeEQsRUFBRSxNQUFNLFlBQVksTUFBTSxnQ0FBZ0M7QUFBQSxnQkFDMUQsRUFBRSxNQUFNLE9BQU8sTUFBTSwyQkFBMkI7QUFBQSxnQkFDaEQsRUFBRSxNQUFNLFFBQVEsTUFBTSw0QkFBNEI7QUFBQSxnQkFDbEQsRUFBRSxNQUFNLE9BQU8sTUFBTSwyQkFBMkI7QUFBQSxnQkFDaEQsRUFBRSxNQUFNLFFBQVEsTUFBTSw0QkFBNEI7QUFBQSxnQkFDbEQsRUFBRSxNQUFNLFNBQVMsTUFBTSw2QkFBNkI7QUFBQSxnQkFDcEQsRUFBRSxNQUFNLFFBQVEsTUFBTSw0QkFBNEI7QUFBQSxnQkFDbEQsRUFBRSxNQUFNLFNBQVMsTUFBTSw2QkFBNkI7QUFBQSxnQkFDcEQsRUFBRSxNQUFNLGdCQUFnQixNQUFNLG9DQUFvQztBQUFBLGdCQUNsRSxFQUFFLE1BQU0saUJBQWlCLE1BQU0scUNBQXFDO0FBQUEsZ0JBQ3BFLEVBQUUsTUFBTSxjQUFjLE1BQU0sa0NBQWtDO0FBQUEsZ0JBQzlELEVBQUUsTUFBTSxlQUFlLE1BQU0sbUNBQW1DO0FBQUEsZ0JBQ2hFLEVBQUUsTUFBTSxZQUFZLE1BQU0sZ0NBQWdDO0FBQUEsZ0JBQzFELEVBQUUsTUFBTSxhQUFhLE1BQU0saUNBQWlDO0FBQUEsZ0JBQzVELEVBQUUsTUFBTSxZQUFZLE1BQU0sZ0NBQWdDO0FBQUEsZ0JBQzFELEVBQUUsTUFBTSxhQUFhLE1BQU0saUNBQWlDO0FBQUEsZ0JBQzVELEVBQUUsTUFBTSxRQUFRLE1BQU0sNEJBQTRCO0FBQUEsZ0JBQ2xELEVBQUUsTUFBTSxTQUFTLE1BQU0sNkJBQTZCO0FBQUEsZ0JBQ3BELEVBQUUsTUFBTSxRQUFRLE1BQU0sNEJBQTRCO0FBQUEsZ0JBQ2xELEVBQUUsTUFBTSxTQUFTLE1BQU0sNkJBQTZCO0FBQUEsZ0JBQ3BELEVBQUUsTUFBTSxVQUFVLE1BQU0sOEJBQThCO0FBQUEsZ0JBQ3RELEVBQUUsTUFBTSxXQUFXLE1BQU0sK0JBQStCO0FBQUEsZ0JBQ3hELEVBQUUsTUFBTSxTQUFTLE1BQU0sNkJBQTZCO0FBQUEsZ0JBQ3BELEVBQUUsTUFBTSxVQUFVLE1BQU0sOEJBQThCO0FBQUEsY0FDeEQ7QUFBQSxZQUNGO0FBQUEsWUFDQTtBQUFBLGNBQ0UsTUFBTTtBQUFBLGNBQ04sYUFBYTtBQUFBLGNBQ2IsVUFBVTtBQUFBLGdCQUNSLEVBQUUsTUFBTSxPQUFPLE1BQU0sNEJBQTRCO0FBQUEsZ0JBQ2pELEVBQUUsTUFBTSxRQUFRLE1BQU0sNkJBQTZCO0FBQUEsZ0JBQ25ELEVBQUUsTUFBTSxPQUFPLE1BQU0sNEJBQTRCO0FBQUEsZ0JBQ2pELEVBQUUsTUFBTSxRQUFRLE1BQU0sNkJBQTZCO0FBQUEsZ0JBQ25ELEVBQUUsTUFBTSxPQUFPLE1BQU0sNEJBQTRCO0FBQUEsZ0JBQ2pELEVBQUUsTUFBTSxRQUFRLE1BQU0sNkJBQTZCO0FBQUEsZ0JBQ25ELEVBQUUsTUFBTSxPQUFPLE1BQU0sNEJBQTRCO0FBQUEsZ0JBQ2pELEVBQUUsTUFBTSxRQUFRLE1BQU0sNkJBQTZCO0FBQUEsY0FDckQ7QUFBQSxZQUNGO0FBQUEsWUFDQTtBQUFBLGNBQ0UsTUFBTTtBQUFBLGNBQ04sYUFBYTtBQUFBLGNBQ2IsVUFBVTtBQUFBLGdCQUNSLEVBQUUsTUFBTSxhQUFhLE1BQU0sa0NBQWtDO0FBQUEsZ0JBQzdELEVBQUUsTUFBTSxVQUFVLE1BQU0sK0JBQStCO0FBQUEsZ0JBQ3ZELEVBQUUsTUFBTSxVQUFVLE1BQU0sK0JBQStCO0FBQUEsZ0JBQ3ZELEVBQUUsTUFBTSxPQUFPLE1BQU0sNEJBQTRCO0FBQUEsZ0JBQ2pELEVBQUUsTUFBTSxPQUFPLE1BQU0sNEJBQTRCO0FBQUEsZ0JBQ2pELEVBQUUsTUFBTSxRQUFRLE1BQU0sNkJBQTZCO0FBQUEsZ0JBQ25ELEVBQUUsTUFBTSxPQUFPLE1BQU0sNEJBQTRCO0FBQUEsZ0JBQ2pELEVBQUUsTUFBTSxRQUFRLE1BQU0sNkJBQTZCO0FBQUEsZ0JBQ25ELEVBQUUsTUFBTSxVQUFVLE1BQU0sK0JBQStCO0FBQUEsZ0JBQ3ZELEVBQUUsTUFBTSxXQUFXLE1BQU0sZ0NBQWdDO0FBQUEsZ0JBQ3pELEVBQUUsTUFBTSxRQUFRLE1BQU0sNkJBQTZCO0FBQUEsZ0JBQ25ELEVBQUUsTUFBTSxXQUFXLE1BQU0sZ0NBQWdDO0FBQUEsZ0JBQ3pELEVBQUUsTUFBTSxjQUFjLE1BQU0sbUNBQW1DO0FBQUEsZ0JBQy9ELEVBQUUsTUFBTSxZQUFZLE1BQU0saUNBQWlDO0FBQUEsZ0JBQzNELEVBQUUsTUFBTSxZQUFZLE1BQU0saUNBQWlDO0FBQUEsZ0JBQzNELEVBQUUsTUFBTSxZQUFZLE1BQU0saUNBQWlDO0FBQUEsZ0JBQzNELEVBQUUsTUFBTSxPQUFPLE1BQU0sNEJBQTRCO0FBQUEsZ0JBQ2pELEVBQUUsTUFBTSxPQUFPLE1BQU0sNEJBQTRCO0FBQUEsY0FDbkQ7QUFBQSxZQUNGO0FBQUEsWUFDQTtBQUFBLGNBQ0UsTUFBTTtBQUFBLGNBQ04sYUFBYTtBQUFBLGNBQ2IsVUFBVTtBQUFBLGdCQUNSLEVBQUUsTUFBTSxvQkFBb0IsTUFBTSx1Q0FBdUM7QUFBQSxnQkFDekUsRUFBRSxNQUFNLGdCQUFnQixNQUFNLG1DQUFtQztBQUFBLGdCQUNqRSxFQUFFLE1BQU0sb0JBQW9CLE1BQU0sdUNBQXVDO0FBQUEsZ0JBQ3pFLEVBQUUsTUFBTSxVQUFVLE1BQU0sNkJBQTZCO0FBQUEsZ0JBQ3JELEVBQUUsTUFBTSxZQUFZLE1BQU0sK0JBQStCO0FBQUEsY0FDM0Q7QUFBQSxZQUNGO0FBQUEsWUFDQTtBQUFBLGNBQ0UsTUFBTTtBQUFBLGNBQ04sYUFBYTtBQUFBLGNBQ2IsVUFBVTtBQUFBLGdCQUNSLEVBQUUsTUFBTSxhQUFhLE1BQU0sbUNBQW1DO0FBQUEsZ0JBQzlELEVBQUUsTUFBTSxhQUFhLE1BQU0sbUNBQW1DO0FBQUEsZ0JBQzlELEVBQUUsTUFBTSxzQkFBc0IsTUFBTSw0Q0FBNEM7QUFBQSxnQkFDaEYsRUFBRSxNQUFNLHNCQUFzQixNQUFNLDRDQUE0QztBQUFBLGNBQ2xGO0FBQUEsWUFDRjtBQUFBLFlBQ0E7QUFBQSxjQUNFLE1BQU07QUFBQSxjQUNOLGFBQWE7QUFBQSxjQUNiLFVBQVU7QUFBQSxnQkFDUixFQUFFLE1BQU0sY0FBYyxNQUFNLGdDQUFnQztBQUFBLGdCQUM1RCxFQUFFLE1BQU0sYUFBYSxNQUFNLCtCQUErQjtBQUFBLGdCQUMxRCxFQUFFLE1BQU0sYUFBYSxNQUFNLCtCQUErQjtBQUFBLGdCQUMxRCxFQUFFLE1BQU0sYUFBYSxNQUFNLCtCQUErQjtBQUFBLGdCQUMxRCxFQUFFLE1BQU0sYUFBYSxNQUFNLCtCQUErQjtBQUFBLGdCQUMxRCxFQUFFLE1BQU0sYUFBYSxNQUFNLCtCQUErQjtBQUFBLGNBQzVEO0FBQUEsWUFDRjtBQUFBLFlBQ0E7QUFBQSxjQUNFLE1BQU07QUFBQSxjQUNOLGFBQWE7QUFBQSxjQUNiLFVBQVU7QUFBQSxnQkFDUixFQUFFLE1BQU0sV0FBVyxNQUFNLGtDQUFrQztBQUFBLGdCQUMzRCxFQUFFLE1BQU0sYUFBYSxNQUFNLG9DQUFvQztBQUFBLGdCQUMvRCxFQUFFLE1BQU0sV0FBVyxNQUFNLGtDQUFrQztBQUFBLGdCQUMzRCxFQUFFLE1BQU0sZ0JBQWdCLE1BQU0sdUNBQXVDO0FBQUEsZ0JBQ3JFLEVBQUUsTUFBTSxRQUFRLE1BQU0sK0JBQStCO0FBQUEsZ0JBQ3JELEVBQUUsTUFBTSxVQUFVLE1BQU0saUNBQWlDO0FBQUEsY0FDM0Q7QUFBQSxZQUNGO0FBQUEsWUFDQTtBQUFBLGNBQ0UsTUFBTTtBQUFBLGNBQ04sYUFBYTtBQUFBLGNBQ2IsVUFBVTtBQUFBLGdCQUNSLEVBQUUsTUFBTSxXQUFXLE1BQU0sdUNBQXVDO0FBQUEsZ0JBQ2hFLEVBQUUsTUFBTSxlQUFlLE1BQU0sMkNBQTJDO0FBQUEsZ0JBQ3hFLEVBQUUsTUFBTSxhQUFhLE1BQU0seUNBQXlDO0FBQUEsZ0JBQ3BFLEVBQUUsTUFBTSxXQUFXLE1BQU0sdUNBQXVDO0FBQUEsY0FDbEU7QUFBQSxZQUNGO0FBQUEsWUFDQTtBQUFBLGNBQ0UsTUFBTTtBQUFBLGNBQ04sYUFBYTtBQUFBLGNBQ2IsVUFBVTtBQUFBLGdCQUNSLEVBQUUsTUFBTSxVQUFVLE1BQU0sbUNBQW1DO0FBQUEsZ0JBQzNELEVBQUUsTUFBTSxXQUFXLE1BQU0sb0NBQW9DO0FBQUEsY0FDL0Q7QUFBQSxZQUNGO0FBQUEsWUFDQTtBQUFBLGNBQ0UsTUFBTTtBQUFBLGNBQ04sYUFBYTtBQUFBLGNBQ2IsVUFBVTtBQUFBLGdCQUNSLEVBQUUsTUFBTSxVQUFVLE1BQU0sK0JBQStCO0FBQUEsZ0JBQ3ZELEVBQUUsTUFBTSxhQUFhLE1BQU0sa0NBQWtDO0FBQUEsY0FDL0Q7QUFBQSxZQUNGO0FBQUEsWUFDQTtBQUFBLGNBQ0UsTUFBTTtBQUFBLGNBQ04sYUFBYTtBQUFBLGNBQ2IsVUFBVTtBQUFBLGdCQUNSLEVBQUUsTUFBTSxTQUFTLE1BQU0sOEJBQThCO0FBQUEsZ0JBQ3JELEVBQUUsTUFBTSxjQUFjLE1BQU0sbUNBQW1DO0FBQUEsZ0JBQy9ELEVBQUUsTUFBTSxRQUFRLE1BQU0sNkJBQTZCO0FBQUEsZ0JBQ25ELEVBQUUsTUFBTSxhQUFhLE1BQU0sa0NBQWtDO0FBQUEsZ0JBQzdELEVBQUUsTUFBTSxRQUFRLE1BQU0sNkJBQTZCO0FBQUEsZ0JBQ25ELEVBQUUsTUFBTSxhQUFhLE1BQU0sa0NBQWtDO0FBQUEsZ0JBQzdELEVBQUUsTUFBTSxhQUFhLE1BQU0sa0NBQWtDO0FBQUEsZ0JBQzdELEVBQUUsTUFBTSxrQkFBa0IsTUFBTSx1Q0FBdUM7QUFBQSxnQkFDdkUsRUFBRSxNQUFNLGVBQWUsTUFBTSxvQ0FBb0M7QUFBQSxnQkFDakUsRUFBRSxNQUFNLG9CQUFvQixNQUFNLHlDQUF5QztBQUFBLGdCQUMzRSxFQUFFLE1BQU0sU0FBUyxNQUFNLDhCQUE4QjtBQUFBLGdCQUNyRCxFQUFFLE1BQU0sY0FBYyxNQUFNLG1DQUFtQztBQUFBLGdCQUMvRCxFQUFFLE1BQU0sVUFBVSxNQUFNLCtCQUErQjtBQUFBLGdCQUN2RCxFQUFFLE1BQU0sZUFBZSxNQUFNLG9DQUFvQztBQUFBLGdCQUNqRSxFQUFFLE1BQU0sYUFBYSxNQUFNLGtDQUFrQztBQUFBLGdCQUM3RCxFQUFFLE1BQU0sa0JBQWtCLE1BQU0sdUNBQXVDO0FBQUEsZ0JBQ3ZFLEVBQUUsTUFBTSxtQkFBbUIsTUFBTSx3Q0FBd0M7QUFBQSxnQkFDekUsRUFBRSxNQUFNLHdCQUF3QixNQUFNLDZDQUE2QztBQUFBLGdCQUNuRixFQUFFLE1BQU0sVUFBVSxNQUFNLCtCQUErQjtBQUFBLGdCQUN2RCxFQUFFLE1BQU0sZUFBZSxNQUFNLG9DQUFvQztBQUFBLGdCQUNqRSxFQUFFLE1BQU0sV0FBVyxNQUFNLGdDQUFnQztBQUFBLGdCQUN6RCxFQUFFLE1BQU0sZ0JBQWdCLE1BQU0scUNBQXFDO0FBQUEsZ0JBQ25FLEVBQUUsTUFBTSxXQUFXLE1BQU0sZ0NBQWdDO0FBQUEsZ0JBQ3pELEVBQUUsTUFBTSxnQkFBZ0IsTUFBTSxxQ0FBcUM7QUFBQSxnQkFDbkUsRUFBRSxNQUFNLFFBQVEsTUFBTSw2QkFBNkI7QUFBQSxnQkFDbkQsRUFBRSxNQUFNLGFBQWEsTUFBTSxrQ0FBa0M7QUFBQSxnQkFDN0QsRUFBRSxNQUFNLGNBQWMsTUFBTSxtQ0FBbUM7QUFBQSxnQkFDL0QsRUFBRSxNQUFNLG1CQUFtQixNQUFNLHdDQUF3QztBQUFBLGdCQUN6RSxFQUFFLE1BQU0sYUFBYSxNQUFNLGtDQUFrQztBQUFBLGdCQUM3RCxFQUFFLE1BQU0sV0FBVyxNQUFNLGdDQUFnQztBQUFBLGdCQUN6RCxFQUFFLE1BQU0sZ0JBQWdCLE1BQU0scUNBQXFDO0FBQUEsY0FDckU7QUFBQSxZQUNGO0FBQUEsWUFDQTtBQUFBLGNBQ0UsTUFBTTtBQUFBLGNBQ04sYUFBYTtBQUFBLGNBQ2IsVUFBVTtBQUFBLGdCQUNSLEVBQUUsTUFBTSxXQUFXLE1BQU0sMENBQTBDO0FBQUEsZ0JBQ25FLEVBQUUsTUFBTSxhQUFhLE1BQU0sNENBQTRDO0FBQUEsZ0JBQ3ZFLEVBQUUsTUFBTSxXQUFXLE1BQU0sMENBQTBDO0FBQUEsZ0JBQ25FLEVBQUUsTUFBTSxhQUFhLE1BQU0sNENBQTRDO0FBQUEsZ0JBQ3ZFLEVBQUUsTUFBTSxXQUFXLE1BQU0sMENBQTBDO0FBQUEsZ0JBQ25FLEVBQUUsTUFBTSxlQUFlLE1BQU0sOENBQThDO0FBQUEsZ0JBQzNFLEVBQUUsTUFBTSxVQUFVLE1BQU0seUNBQXlDO0FBQUEsZ0JBQ2pFLEVBQUUsTUFBTSxLQUFLLE1BQU0sb0NBQW9DO0FBQUEsZ0JBQ3ZELEVBQUUsTUFBTSxNQUFNLE1BQU0scUNBQXFDO0FBQUEsZ0JBQ3pELEVBQUUsTUFBTSxRQUFRLE1BQU0sdUNBQXVDO0FBQUEsZ0JBQzdELEVBQUUsTUFBTSxVQUFVLE1BQU0seUNBQXlDO0FBQUEsZ0JBQ2pFLEVBQUUsTUFBTSxVQUFVLE1BQU0seUNBQXlDO0FBQUEsZ0JBQ2pFLEVBQUUsTUFBTSxRQUFRLE1BQU0sdUNBQXVDO0FBQUEsZ0JBQzdELEVBQUUsTUFBTSxjQUFjLE1BQU0sNkNBQTZDO0FBQUEsZ0JBQ3pFLEVBQUUsTUFBTSxVQUFVLE1BQU0seUNBQXlDO0FBQUEsZ0JBQ2pFLEVBQUUsTUFBTSxTQUFTLE1BQU0sd0NBQXdDO0FBQUEsZ0JBQy9ELEVBQUUsTUFBTSxVQUFVLE1BQU0seUNBQXlDO0FBQUEsZ0JBQ2pFLEVBQUUsTUFBTSxVQUFVLE1BQU0seUNBQXlDO0FBQUEsZ0JBQ2pFLEVBQUUsTUFBTSxVQUFVLE1BQU0seUNBQXlDO0FBQUEsZ0JBQ2pFLEVBQUUsTUFBTSxhQUFhLE1BQU0sNENBQTRDO0FBQUEsZ0JBQ3ZFLEVBQUUsTUFBTSxXQUFXLE1BQU0sMENBQTBDO0FBQUEsZ0JBQ25FLEVBQUUsTUFBTSxVQUFVLE1BQU0seUNBQXlDO0FBQUEsZ0JBQ2pFLEVBQUUsTUFBTSxVQUFVLE1BQU0seUNBQXlDO0FBQUEsZ0JBQ2pFLEVBQUUsTUFBTSxVQUFVLE1BQU0seUNBQXlDO0FBQUEsZ0JBQ2pFLEVBQUUsTUFBTSxVQUFVLE1BQU0seUNBQXlDO0FBQUEsY0FDbkU7QUFBQSxZQUNGO0FBQUEsWUFDQTtBQUFBLGNBQ0UsTUFBTTtBQUFBLGNBQ04sYUFBYTtBQUFBLGNBQ2IsVUFBVTtBQUFBLGdCQUNSLEVBQUUsTUFBTSxTQUFTLE1BQU0sZ0NBQWdDO0FBQUEsZ0JBQ3ZELEVBQUUsTUFBTSxTQUFTLE1BQU0sZ0NBQWdDO0FBQUEsZ0JBQ3ZELEVBQUUsTUFBTSxRQUFRLE1BQU0sK0JBQStCO0FBQUEsZ0JBQ3JELEVBQUUsTUFBTSxjQUFjLE1BQU0scUNBQXFDO0FBQUEsZ0JBQ2pFLEVBQUUsTUFBTSxjQUFjLE1BQU0scUNBQXFDO0FBQUEsZ0JBQ2pFLEVBQUUsTUFBTSxhQUFhLE1BQU0sb0NBQW9DO0FBQUEsZ0JBQy9ELEVBQUUsTUFBTSxRQUFRLE1BQU0sK0JBQStCO0FBQUEsZ0JBQ3JELEVBQUUsTUFBTSxhQUFhLE1BQU0sb0NBQW9DO0FBQUEsZ0JBQy9ELEVBQUUsTUFBTSxVQUFVLE1BQU0saUNBQWlDO0FBQUEsZ0JBQ3pELEVBQUUsTUFBTSxlQUFlLE1BQU0sc0NBQXNDO0FBQUEsZ0JBQ25FLEVBQUUsTUFBTSxPQUFPLE1BQU0sOEJBQThCO0FBQUEsZ0JBQ25ELEVBQUUsTUFBTSxZQUFZLE1BQU0sbUNBQW1DO0FBQUEsZ0JBQzdELEVBQUUsTUFBTSxZQUFZLE1BQU0sbUNBQW1DO0FBQUEsZ0JBQzdELEVBQUUsTUFBTSxhQUFhLE1BQU0sb0NBQW9DO0FBQUEsZ0JBQy9ELEVBQUUsTUFBTSxPQUFPLE1BQU0sOEJBQThCO0FBQUEsZ0JBQ25ELEVBQUUsTUFBTSxRQUFRLE1BQU0sK0JBQStCO0FBQUEsZ0JBQ3JELEVBQUUsTUFBTSxRQUFRLE1BQU0sK0JBQStCO0FBQUEsZ0JBQ3JELEVBQUUsTUFBTSxZQUFZLE1BQU0sbUNBQW1DO0FBQUEsY0FDL0Q7QUFBQSxZQUNGO0FBQUEsWUFDQTtBQUFBLGNBQ0UsTUFBTTtBQUFBLGNBQ04sYUFBYTtBQUFBLGNBQ2IsVUFBVTtBQUFBLGdCQUNSLEVBQUUsTUFBTSxrQkFBa0IsTUFBTSx3Q0FBd0M7QUFBQSxnQkFDeEUsRUFBRSxNQUFNLGVBQWUsTUFBTSxxQ0FBcUM7QUFBQSxnQkFDbEUsRUFBRSxNQUFNLG1CQUFtQixNQUFNLHlDQUF5QztBQUFBLGNBQzVFO0FBQUEsWUFDRjtBQUFBLFlBQ0E7QUFBQSxjQUNFLE1BQU07QUFBQSxjQUNOLGFBQWE7QUFBQSxjQUNiLFVBQVU7QUFBQSxnQkFDUixFQUFFLE1BQU0sWUFBWSxNQUFNLG1DQUFtQztBQUFBLGdCQUM3RCxFQUFFLE1BQU0sZ0JBQWdCLE1BQU0sdUNBQXVDO0FBQUEsZ0JBQ3JFLEVBQUUsTUFBTSxpQkFBaUIsTUFBTSx3Q0FBd0M7QUFBQSxnQkFDdkUsRUFBRSxNQUFNLHFCQUFxQixNQUFNLDRDQUE0QztBQUFBLGdCQUMvRSxFQUFFLE1BQU0sZUFBZSxNQUFNLHNDQUFzQztBQUFBLGdCQUNuRSxFQUFFLE1BQU0sb0JBQW9CLE1BQU0sMkNBQTJDO0FBQUEsZ0JBQzdFLEVBQUUsTUFBTSxXQUFXLE1BQU0sa0NBQWtDO0FBQUEsY0FDN0Q7QUFBQSxZQUNGO0FBQUEsWUFDQTtBQUFBLGNBQ0UsTUFBTTtBQUFBLGNBQ04sYUFBYTtBQUFBLGNBQ2IsVUFBVTtBQUFBLGdCQUNSLEVBQUUsTUFBTSx3QkFBd0IsTUFBTSw0Q0FBNEM7QUFBQSxnQkFDbEYsRUFBRSxNQUFNLHdCQUF3QixNQUFNLDRDQUE0QztBQUFBLGdCQUNsRixFQUFFLE1BQU0seUJBQXlCLE1BQU0sNkNBQTZDO0FBQUEsZ0JBQ3BGLEVBQUUsTUFBTSxZQUFZLE1BQU0sZ0NBQWdDO0FBQUEsZ0JBQzFELEVBQUUsTUFBTSxlQUFlLE1BQU0sbUNBQW1DO0FBQUEsY0FDbEU7QUFBQSxZQUNGO0FBQUEsWUFDQTtBQUFBLGNBQ0UsTUFBTTtBQUFBLGNBQ04sTUFBTTtBQUFBLFlBQ1I7QUFBQSxZQUNBO0FBQUEsY0FDRSxNQUFNO0FBQUEsY0FDTixNQUFNO0FBQUEsWUFDUjtBQUFBLFlBQ0E7QUFBQSxjQUNFLE1BQU07QUFBQSxjQUNOLE1BQU07QUFBQSxZQUNSO0FBQUEsWUFDQTtBQUFBLGNBQ0UsTUFBTTtBQUFBLGNBQ04sTUFBTTtBQUFBLFlBQ1I7QUFBQSxVQUNGO0FBQUEsUUFDRjtBQUFBLE1BQ0Y7QUFBQSxNQUVBLGVBQWU7QUFBQSxRQUNiO0FBQUEsVUFDRSxNQUFNO0FBQUEsVUFDTixVQUFVO0FBQUEsWUFDUjtBQUFBLGNBQ0UsTUFBTTtBQUFBLGNBQ04sTUFBTTtBQUFBLFlBQ1I7QUFBQSxZQUNBO0FBQUEsY0FDRSxNQUFNO0FBQUEsY0FDTixNQUFNO0FBQUEsWUFDUjtBQUFBLFlBQ0E7QUFBQSxjQUNFLE1BQU07QUFBQSxjQUNOLE1BQU07QUFBQSxZQUNSO0FBQUEsWUFDQTtBQUFBLGNBQ0UsTUFBTTtBQUFBLGNBQ04sTUFBTTtBQUFBLFlBQ1I7QUFBQSxZQUNBO0FBQUEsY0FDRSxNQUFNO0FBQUEsY0FDTixNQUFNO0FBQUEsWUFDUjtBQUFBLFlBQ0E7QUFBQSxjQUNFLE1BQU07QUFBQSxjQUNOLE1BQU07QUFBQSxZQUNSO0FBQUEsWUFBRztBQUFBLGNBQ0QsTUFBTTtBQUFBLGNBQ04sTUFBTTtBQUFBLFlBQ1I7QUFBQSxZQUFHO0FBQUEsY0FDRCxNQUFNO0FBQUEsY0FDTixNQUFNO0FBQUEsWUFDUjtBQUFBLFVBQ0Y7QUFBQSxRQUNGO0FBQUEsTUFDRjtBQUFBLE1BQ0EsZ0JBQWdCO0FBQUEsUUFDZDtBQUFBLFVBQ0UsTUFBTTtBQUFBLFVBQ04sVUFBVTtBQUFBLFlBQ1I7QUFBQSxjQUNFLE1BQU07QUFBQSxjQUNOLE1BQU07QUFBQSxZQUNSO0FBQUEsWUFDQTtBQUFBLGNBQ0UsTUFBTTtBQUFBLGNBQ04sTUFBTTtBQUFBLFlBQ1I7QUFBQSxZQUNBO0FBQUEsY0FDRSxNQUFNO0FBQUEsY0FDTixNQUFNO0FBQUEsWUFDUjtBQUFBLFlBQ0E7QUFBQSxjQUNFLE1BQU07QUFBQSxjQUNOLE1BQU07QUFBQSxZQUNSO0FBQUEsWUFDQTtBQUFBLGNBQ0UsTUFBTTtBQUFBLGNBQ04sTUFBTTtBQUFBLFlBQ1I7QUFBQSxZQUNBO0FBQUEsY0FDRSxNQUFNO0FBQUEsY0FDTixNQUFNO0FBQUEsWUFDUjtBQUFBLFlBQ0E7QUFBQSxjQUNFLE1BQU07QUFBQSxjQUNOLE1BQU07QUFBQSxZQUNSO0FBQUEsWUFDQTtBQUFBLGNBQ0UsTUFBTTtBQUFBLGNBQ04sYUFBYTtBQUFBLGNBQ2IsVUFBVTtBQUFBLGdCQUNSLEVBQUUsTUFBTSxVQUFVLE1BQU0sMkJBQTJCO0FBQUEsZ0JBQ25ELEVBQUUsTUFBTSxRQUFRLE1BQU0seUJBQXlCO0FBQUEsY0FDakQ7QUFBQSxZQUNGO0FBQUEsVUFDRjtBQUFBLFFBQ0Y7QUFBQSxNQUNGO0FBQUEsSUFFRjtBQUFBLEVBQ0YsQ0FBQztBQUNILENBQUM7IiwKICAibmFtZXMiOiBbXQp9Cg==
