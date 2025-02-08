import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
    // Load env file based on mode
    const env = loadEnv(mode, process.cwd(), '');
    const FRONTEND_PORT = parseInt(env.VITE_FRONTEND_PORT || '3001');
    const BACKEND_URL = env.VITE_BACKEND_URL || 'http://localhost:8001';

    return {
        plugins: [react()],
        server: {
            port: FRONTEND_PORT,
            proxy: {
                '/api': {
                    target: BACKEND_URL,
                    changeOrigin: true,
                },
            },
        },
        css: {
            modules: {
                localsConvention: 'camelCase',
            },
        },
        resolve: {
            extensions: ['.js', '.jsx', '.json'],
        },
    };
}); 