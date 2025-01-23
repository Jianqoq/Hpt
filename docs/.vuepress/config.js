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
    navbar: [
      {
        text: '首页',
        link: '/',
      },
    ],
  }),
})