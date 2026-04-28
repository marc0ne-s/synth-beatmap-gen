import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  clearScreen: false,
  server: {
    port: 3000,
    strictPort: true,
    watch: {
      ignored: ['**/src-tauri/**']
    }
  },
  envPrefix: ['VITE_', 'TAURI_'],
  build: {
    target: process.env.TAURI_ENV_PLATFORM ? 'chrome105' : 'esnext',
    minify: !process.env.TAURI_ENV_PLATFORM ? 'esbuild' : false,
    sourcemap: !!process.env.TAURI_ENV_PLATFORM
  }
})