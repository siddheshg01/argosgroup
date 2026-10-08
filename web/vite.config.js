import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  base: './',
  plugins: [react()],
  server: { proxy: { '/api': 'http://127.0.0.1:8000' } },
  build: { outDir: '../', emptyOutDir: false },
  test: { environment: 'jsdom', globals: true, setupFiles: './src/test-setup.js' },
})
