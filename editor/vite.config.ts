import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  clearScreen: false,
  server: {
    port: 3000,
    strictPort: true,
  },
  envPrefix: ['VITE_'],
  build: {
    target: 'esnext',
    minify: 'esbuild',
  }
})
