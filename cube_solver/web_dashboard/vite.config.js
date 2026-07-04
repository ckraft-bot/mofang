import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      '/ws': {
        target: 'ws://127.0.0.1:8000',
        ws: true,
      },
      '/scan': {
        target: 'http://127.0.0.1:8000',
      },
      '/solve': {
        target: 'http://127.0.0.1:8000',
      },
      '/health': {
        target: 'http://127.0.0.1:8000',
      },
    },
  },
});
