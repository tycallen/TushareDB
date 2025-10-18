import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueJsx from '@vitejs/plugin-vue-jsx'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    vueJsx(),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  },
  server: { // 添加 server 配置
    proxy: {
      '/api': { // 当请求以 /api 开头时
        target: 'http://localhost:8000', // 代理到后端服务
        changeOrigin: true, // 改变源，确保后端认为请求来自其自身
        // 移除 rewrite 规则，因为前端的请求路径已经包含了 /api，后端也期望 /api
      }
    }
  }
})
