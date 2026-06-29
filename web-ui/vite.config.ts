import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';

const API_TARGET = process.env.LLGRAPH_API_PROXY ?? 'http://127.0.0.1:8765';

function checkApiBackendPlugin(): Plugin {
  return {
    name: 'llgraph-check-api-backend',
    configureServer() {
      fetch(`${API_TARGET}/api/health`)
        .then((res) => {
          if (!res.ok) {
            throw new Error(String(res.status));
          }
        })
        .catch(() => {
          console.warn(
            [
              '',
              '⚠️  llgraph API 未就绪（' + API_TARGET + '）',
              '   Vite 会把 /api 代理到 8765；请先启动后端：',
              '   · 推荐: cd .. && ./scripts/web-dev.sh dev',
              '   · 或另开终端: llgraph web --port 8765',
              '',
            ].join('\n'),
          );
        });
    },
  };
}

export default defineConfig({
  plugins: [react(), checkApiBackendPlugin()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: API_TARGET,
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes, req) => {
            if (req.url?.includes('/events')) {
              proxyRes.headers['cache-control'] = 'no-cache, no-transform';
              proxyRes.headers['x-accel-buffering'] = 'no';
            }
          });
        },
      },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
});
