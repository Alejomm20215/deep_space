import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [react()],
    resolve: {
        alias: {
            '@': path.resolve(__dirname, './src'),
        },
    },
    server: {
        proxy: {
            '/api': {
                target: process.env.VITE_API_TARGET || 'http://localhost:8000',
                changeOrigin: true,
            },
            '/ws': {
                target: process.env.VITE_WS_TARGET || 'ws://localhost:8000',
                ws: true,
            },
            '/outputs': {
                target: process.env.VITE_API_TARGET || 'http://localhost:8000',
                changeOrigin: true,
            }
        }
    }
})
