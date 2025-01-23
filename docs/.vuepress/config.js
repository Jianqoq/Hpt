import { defineUserConfig } from '@vuepress/cli'
import { defaultTheme } from '@vuepress/theme-default'

export default defineUserConfig({
  base: process.env.NODE_ENV === 'development'
    ? '/'
    : '/eTensor/',
  lang: 'zh-CN',
  title: 'HPT 文档',
  description: '这是项目文档',

  theme: defaultTheme({
    // 导航栏配置
    navbar: [
      {
        text: '首页',
        link: '/',
      },
      {
        text: '指南',
        link: '/guide/',
      },
      {
        text: 'GitHub',
        link: 'https://github.com/Jianqoq/eTensor',
      },
    ],
    
    // 侧边栏配置
    sidebar: {
      '/guide/': [
        {
          text: '指南',
          children: ['/guide/README.md'],
        },
      ],
    },
  }),
})