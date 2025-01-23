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

    sidebar: {
      '/guide/': [
        {
          text: '指南',
          children: [
            '/guide/README.md',
          ],
        },
      ],
    },

    // 启用编辑链接
    editLink: true,
    editLinkText: '在 GitHub 上编辑此页',
    docsRepo: 'Jianqoq/eTensor',
    docsBranch: 'main',
    docsDir: 'docs',

    // 显示最后更新时间
    lastUpdated: true,
    lastUpdatedText: '最后更新',

    // 显示贡献者
    contributors: true,
    contributorsText: '贡献者',
  }),
})