import { defineUserConfig } from '@vuepress/cli'
import { defaultTheme } from '@vuepress/theme-default'

export default defineUserConfig({
  base: process.env.NODE_ENV === 'development'
    ? '/'
    : '/eTensor/',
  lang: 'zh-CN',
  title: 'Hpt',

  theme: defaultTheme({
    home: false,
    navbar: [
      {
        text: 'home',
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
          ]
        }
      ],
      '/dev_guide/': [
        {
          text: '开发指南',
          children: [
            '/dev_guide/getting_started.md',
            '/dev_guide/contribution.md',
          ]
        }
      ],
    },
  }),
})